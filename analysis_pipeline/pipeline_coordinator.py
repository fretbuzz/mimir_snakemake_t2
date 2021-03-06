import gc
import pandas as pd
import pyximport
import ast
from matplotlib import pyplot as plt
from single_experiment_pipeline import determine_attacks_to_times
from statistical_analysis import single_rate_stats_pipeline
import generate_aggregate_report as generate_aggregate_report
pyximport.install() # to leverage cpython
import math
import operator
import copy
import multiprocessing
import pyximport
pyximport.install()
import pickle
from statistical_analysis_perSvc import exfil_detection_model

# this function determines how much time to is available for injection attacks in each experiment.
# it takes into account when the physical attack starts (b/c need to split into training/testing set
# temporally before the physical attack starts) and the goal percentage of split that we are aiming for.
# I think we're going to aim for the desired split in each experiment, but we WON'T try  to compensate
# not meeting one experiment's goal by modifying how we handle another experiment.
def determine_injection_times(exps_info, goal_train_test_split, goal_attack_NoAttack_split_training,
                              goal_attack_NoAttack_split_testing):
    #time_splits = []
    exp_injection_info = []
    end_of_train_portions = []

    print "exps_info",exps_info
    for exp_info in exps_info:
        print "exp_info", exp_info['total_experiment_length'], "float(goal_train_test_split)", float(goal_train_test_split)
        time_split = ((exp_info['total_experiment_length']) * float(goal_train_test_split))

        # we want physical to be in testing phase...
        if time_split > exp_info['exfil_startEnd_times'][0][0]:
            time_split = exp_info['exfil_startEnd_times'][0][0]

        end_of_train_portions.append(time_split)
        print "end_of_train_portions", end_of_train_portions
        ## now to find how much time to spending injecting during training and testing...
        ## okay, let's do testing first b/c it should be relatively straightforward...
        testing_time = exp_info['total_experiment_length'] - time_split

        physical_attack_time = 0
        for start_end_tuple in exp_info['exfil_startEnd_times']:
            physical_attack_time += start_end_tuple[1] - start_end_tuple[0]
        testing_time_for_attack_injection = (testing_time) * goal_attack_NoAttack_split_testing - physical_attack_time

        #testing_time_without_physical_attack = testing_time - physical_attack_time
        print "physical_attack_time",physical_attack_time, "testing_time", testing_time, testing_time_for_attack_injection
        testing_time_for_attack_injection = max(testing_time_for_attack_injection,0)

        # now let's find the time to inject during training... this'll be a percentage of the time between
        # system startup and the training/testing split point...
        #training_time_after_startup = time_split # - exp_info["startup_time"]
        training_time_for_attack_injection = time_split * goal_attack_NoAttack_split_training

        exp_injection_info.append({'testing': testing_time_for_attack_injection,
                                   "training": training_time_for_attack_injection})
        #time_splits.append(time_split)
    print "exp_injection_info", exp_injection_info
    #exit(34)
    return exp_injection_info,end_of_train_portions

# this function loops through multiple experiments (or even just a single experiment), accumulates the relevant
# feature dataframes, and then performs LASSO regression to determine a concise graphical model that can detect
# the injected synthetic attacks
class multi_experiment_pipeline(object):
    def __init__(self, function_list, base_output_name, ROC_curve_p, time_each_synthetic_exfil, goal_train_test_split,
                 goal_attack_NoAttack_split_training, training_window_size, size_of_neighbor_training_window, calc_vals,
                 skip_model_part, return_new_model_function, calculate_z_scores_p=True, avg_exfil_per_min=None, exfil_per_min_variance=None,
                 avg_pkt_size=None, pkt_size_variance=None, skip_graph_injection=False, get_endresult_from_memory=False,
                 goal_attack_NoAttack_split_testing=0.0, calc_ide=False,
                 perform_svcpair_sec_component=True, only_perform_cilium_component=True,
                 drop_pairwise_features=False, max_path_length=15, max_dns_porportion=1.0, drop_infra_from_graph=False,
                 ide_window_size=10, debug_basename=None, pretrained_sav2=None, auto_open_pdfs=True,
                 skip_heatmap_p=True, no_labeled_data=False, time_fraction_fp_increase=0.05,
                 use_ts_lower=True, use_logistic=False):

        self.single_rate_stats_pipelines = {}
        self.use_ts_lower = use_ts_lower
        self.auto_open_pdfs = auto_open_pdfs
        self.pretrained_min_pipeline = pretrained_sav2
        self.ROC_curve_p = ROC_curve_p
        self.training_window_size = training_window_size
        self.size_of_neighbor_training_window = size_of_neighbor_training_window
        self.calculate_z_scores_p = calculate_z_scores_p
        self.avg_exfil_per_min = avg_exfil_per_min
        self.exfil_per_min_variance = exfil_per_min_variance
        self.avg_pkt_size = avg_pkt_size
        self.pkt_size_variance = pkt_size_variance
        self.skip_graph_injection = skip_graph_injection
        self.calc_ide = calc_ide
        #self.only_ide = only_ide
        self.perform_svcpair_sec_component = perform_svcpair_sec_component
        #self.only_perform_cilium_component = only_perform_cilium_component  # never does anything anymore...
        #self.cilium_component_time = svcpair_sec_component_time
        self.drop_pairwise_features = drop_pairwise_features
        self.drop_infra_from_graph = drop_infra_from_graph
        self.ide_window_size = ide_window_size
        self.debug_basename = debug_basename
        self.list_of_optimal_fone_scores_at_exfil_rates = []
        self.rate_to_timegran_to_methods_to_attacks_found_dfs = {}
        self.rate_to_timegran_list_of_methods_to_attacks_found_training_df = {}
        self.rates_to_experiment_info = {}
        self.base_output_name = base_output_name
        self.test_results_df_loc = base_output_name + 'test_results_df_loc.txt'
        self.training_results_df_loc = base_output_name + 'train_results_df_loc.txt'
        self.rates_to_experiment_info_loc = base_output_name + 'rates_to_experiment_info_loc.txt'
        self.rates_to_outtraffic_info = base_output_name + 'outtraffic_bytese.txt'
        self.where_to_save_this_obj = base_output_name + '_multi_experiment_pipeline'
        self.where_to_save_minrate_statspipelines = base_output_name + '_min_rate_statspipeline'
        self.where_to_save_persvc_ensemble_model = base_output_name + '_persvc_ensemble_model'
        self.rate_to_time_gran_to_xs = {}
        self.rate_to_time_gran_to_ys = {}
        self.rate_to_time_gran_to_outtraffic = {}
        self.get_endresult_from_memory = get_endresult_from_memory
        self.calc_vals = calc_vals
        self.skip_model_part = skip_model_part
        self.function_list = function_list
        self.goal_train_test_split = goal_train_test_split
        self.goal_attack_NoAttack_split_training = goal_attack_NoAttack_split_training
        self.time_each_synthetic_exfil = time_each_synthetic_exfil
        self.goal_attack_NoAttack_split_testing = goal_attack_NoAttack_split_testing
        self.max_path_length = max_path_length
        self.max_dns_porportion = max_dns_porportion
        self.exps_exfil_paths = None
        self.end_of_train_portions = None
        self.training_exfil_paths = None
        self.testing_exfil_paths = None
        self.list_time_gran_to_mod_zscore_df = []
        self.list_time_gran_to_mod_zscore_df_training = []
        self.list_time_gran_to_mod_zscore_df_testing = []
        self.list_time_gran_to_zscore_dataframe = []
        self.list_time_gran_to_feature_dataframe = []
        self.starts_of_testing = []
        self.rate_to_timegran_to_statistical_pipeline = {}
        self.names = []
        self.skip_heatmap_p = skip_heatmap_p
        self.no_labeled_data = no_labeled_data
        self.rate_to_time_gran_to_predicted_test = {}
        self.time_fraction_fp = time_fraction_fp_increase
        self.use_logistic = use_logistic
        self.rate_to_tg_to_cm = {}
        self.cilium_allowed_svc_comm = None
        self.where_to_save_cilium_model = base_output_name + '_cilium_model'

        #self.use_new_model_func = return_new_model_function
        self.rate_to_persvc_model = {}
        self.rate_to_type_of_model_to_time_gran_to_predicted_test = {}
        self.rate_to_type_of_model_to_time_gran_to_cm = {}

    # note: this going to be used to load the pipeline object prior to doing all of this work...
    def loader(self, filename):
        with open(self.where_to_save_this_obj, 'rb') as pickle_input_file:
            multi_pipeline_object = pickle.load(pickle_input_file)
        return multi_pipeline_object

    def save(self):
        with open(self.where_to_save_this_obj, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        # can load like this:

    def generate_and_assign_exfil_paths(self):
        self.exps_exfil_paths, self.end_of_train_portions, self.training_exfil_paths, self.testing_exfil_paths, \
        self.exps_initiator_info = determine_and_assign_exfil_paths(self.calc_vals or self.calc_ide, self.skip_model_part,
                                                                    self.function_list, self.goal_train_test_split,
                                                                    self.goal_attack_NoAttack_split_training,
                                                                    self.time_each_synthetic_exfil,
                                                                    self.goal_attack_NoAttack_split_testing,
                                                                    self.max_path_length, self.max_dns_porportion)

    def run_pipelines(self, pretrained_model_object = None, no_tsl=False, svcpair_model=None,
                      per_svc_exfil_model_p=False, load_old_pipelines=False, persvc_ensemble_model=None,
                      load_all_old_models=False):
        if no_tsl:
            self.use_ts_lower = False
        else:
            self.use_ts_lower = True

        #if per_svc_exfil_model_p:
        #    self.use_new_model_func  = per_svc_exfil_model_p

        self.cilium_allowed_svc_comm = svcpair_model

        self.pretrained_min_pipeline = pretrained_model_object
        print "self.get_endresult_from_memory", self.get_endresult_from_memory
        if not self.get_endresult_from_memory:
            with multiprocessing.Manager() as manager:
                min_rate_statspipelines = None

                self.generate_and_assign_exfil_paths()
                #if not self.only_ide:
                ###initial_cilium_is_none_p = self.cilium_allowed_svc_comm is None
                rate_to_list_time_gran_to_mod_zscore_df = manager.dict()
                rate_to_cur_base_output_name = manager.dict()
                rate_to_list_time_gran_to_zscore_dataframe = manager.dict()
                rate_to_list_time_gran_to_feature_dataframe = manager.dict()
                rate_to_list_time_gran_to_mod_zscore_df_training = manager.dict()
                rate_to_list_time_gran_to_mod_zscore_df_testing = manager.dict()
                rate_to_starts_of_testing, rate_to_cilium_allowed_svc_comm = manager.dict(), manager.dict()
                rate_to_time_gran_to_cilium_alerts = manager.dict()

                jobs = []
                for rate_counter in range(0, len(self.avg_exfil_per_min)):
                    # (i), (ii) <--- I think that they both belong here...
                    #self.run_single_pipeline(rate_counter, self.calc_vals, self.skip_graph_injection,
                    #                         calc_ide=self.calc_ide,
                    #                         no_labeled_data=self.no_labeled_data,
                    #                         pretrained_cilium_model=self.cilium_allowed_svc_comm)

                    args = (rate_counter, self.avg_exfil_per_min, self.exfil_per_min_variance,
                            self.base_output_name, self.exps_exfil_paths,
                            self.exps_initiator_info, self.calculate_z_scores_p,
                            self.end_of_train_portions, self.training_exfil_paths,
                            self.testing_exfil_paths, self.skip_model_part,
                            self.ROC_curve_p, self.avg_pkt_size,
                            self.pkt_size_variance, self.drop_pairwise_features,
                            self.perform_svcpair_sec_component,
                            self.ide_window_size, self.drop_infra_from_graph, self.function_list,
                            self.pretrained_min_pipeline,
                            self.calc_vals, self.skip_graph_injection, self.calc_ide,
                            self.cilium_allowed_svc_comm, self.no_labeled_data,
                            rate_to_list_time_gran_to_mod_zscore_df,
                            rate_to_cur_base_output_name,
                            rate_to_list_time_gran_to_zscore_dataframe,
                            rate_to_list_time_gran_to_feature_dataframe,
                            rate_to_list_time_gran_to_mod_zscore_df_training,
                            rate_to_list_time_gran_to_mod_zscore_df_testing,
                            rate_to_starts_of_testing, rate_to_cilium_allowed_svc_comm,
                            rate_to_time_gran_to_cilium_alerts)
                    p = multiprocessing.Process(
                        target=inject_comm_graphs_at_single_exfil_rate,
                        args=args)
                    p.start()
                    jobs.append(p)

                for job in jobs:
                    job.join()

                #"rate_to_list_time_gran_to_mod_zscore_df.keys()", rate_to_list_time_gran_to_mod_zscore_df.keys()

                recipes_used = [recipe.base_exp_name for recipe in self.function_list]
                for counter, recipe in enumerate(recipes_used):
                    name = '_'.join(recipe.split('_')[1:])
                    self.names.append(name)

                for rate_counter in range(0, len(self.avg_exfil_per_min)):
                    cur_avg_exfil_per_min = self.avg_exfil_per_min[rate_counter]
                    self.single_pipeline_after_injection(rate_to_list_time_gran_to_mod_zscore_df[cur_avg_exfil_per_min],
                                                    rate_to_list_time_gran_to_zscore_dataframe[cur_avg_exfil_per_min],
                                                    rate_to_list_time_gran_to_feature_dataframe[cur_avg_exfil_per_min],
                                                    rate_to_list_time_gran_to_mod_zscore_df_training[cur_avg_exfil_per_min],
                                                    rate_to_list_time_gran_to_mod_zscore_df_testing[cur_avg_exfil_per_min],
                                                    rate_to_starts_of_testing[cur_avg_exfil_per_min],
                                                    rate_to_cilium_allowed_svc_comm[cur_avg_exfil_per_min],
                                                    rate_to_cur_base_output_name[cur_avg_exfil_per_min], rate_counter,
                                                    self.avg_exfil_per_min, (load_old_pipelines or load_all_old_models))

                    # TODO: apply the new model here?? (only for eval purposes tho... not for training...)
                    # I think we need to pass both the new and old models, honestly....
                    if persvc_ensemble_model:
                        list_time_gran_to_mod_zscore_df = rate_to_list_time_gran_to_mod_zscore_df_testing[cur_avg_exfil_per_min]
                        time_gran_to_aggregate_mod_score_dfs = aggregate_dfs(list_time_gran_to_mod_zscore_df)

                        persvc_ensemble_model_cur = copy.deepcopy(persvc_ensemble_model)

                        self.rate_to_persvc_model = {}
                        persvc_ensemble_model_cur.apply_to_new_data( time_gran_to_aggregate_mod_score_dfs,
                                                                     self.base_output_name, self.names,
                                                                     self.avg_exfil_per_min[rate_counter],
                                                                     self.avg_pkt_size[rate_counter],
                                                                     self.exfil_per_min_variance[rate_counter],
                                                                     self.pkt_size_variance[rate_counter],
                                                                     rate_to_time_gran_to_cilium_alerts[cur_avg_exfil_per_min])

                        self.rate_to_persvc_model[cur_avg_exfil_per_min] = persvc_ensemble_model_cur
                        self.rate_to_type_of_model_to_time_gran_to_predicted_test[cur_avg_exfil_per_min] = \
                            persvc_ensemble_model_cur.type_of_model_to_time_gran_to_predicted_test
                        self.rate_to_type_of_model_to_time_gran_to_cm[cur_avg_exfil_per_min] = \
                            persvc_ensemble_model_cur.type_of_model_to_time_gran_to_cm

                        using_pretrained_model = not (not self.pretrained_min_pipeline)
                        persvc_ensemble_model_cur.generate_reports(self.auto_open_pdfs, self.skip_heatmap_p, using_pretrained_model)


            if not self.pretrained_min_pipeline:

                # TODO: put new ensemble model here (it'll call a new member function...)
                persvc_ensemble_model = self.train_persvc_ensemble_model(rate_to_time_gran_to_cilium_alerts)
                with open(self.where_to_save_persvc_ensemble_model, 'w') as f:
                    pickle.dump(persvc_ensemble_model, f)

                # TODO: remove eventually
                #exit(11)

                if load_all_old_models:
                    with open(self.where_to_save_minrate_statspipelines, 'r') as f:
                        min_rate_statspipelines_ts = pickle.load(f)

                    with open(self.where_to_save_minrate_statspipelines + 'multi', 'r') as z:
                        min_rate_statspipelines_agg = pickle.load(z)
                else:
                    min_rate_statspipelines_ts = self.decrease_exfil_of_model()
                    with open(self.where_to_save_minrate_statspipelines, 'w') as f:
                        pickle.dump(min_rate_statspipelines_ts, f)

                    ## let's generate both reports... we can then use a param to pick between them...
                    min_rate_statspipelines_agg = self.train_multi_exfilrate_model()
                    with open(self.where_to_save_minrate_statspipelines + 'multi', 'w') as z:
                        pickle.dump(min_rate_statspipelines_agg, z)


                if self.use_ts_lower:
                    min_rate_statspipelines = min_rate_statspipelines_ts
                else:
                    min_rate_statspipelines = min_rate_statspipelines_agg

                ## save the cilium model, if the model is being trained...
                with open(self.where_to_save_cilium_model, 'w') as g:
                    pickle.dump(self.cilium_allowed_svc_comm, g)

            if not self.no_labeled_data:
                self.generate_aggregate_report()
        else:
            #if self.use_new_model_func:
            with open(self.where_to_save_persvc_ensemble_model, 'r') as f:
                persvc_ensemble_model = pickle.load(f)

            if self.use_ts_lower:
                #print "self.where_to_save_minrate_statspipelines", self.where_to_save_minrate_statspipelines
                with open(self.where_to_save_minrate_statspipelines, 'r') as f:
                    min_rate_statspipelines = pickle.load(f)
            else:
                with open(self.where_to_save_minrate_statspipelines  + 'multi', 'r') as f:
                    min_rate_statspipelines = pickle.load(f)

            try:
                with open(self.where_to_save_cilium_model, 'r') as f:
                    self.cilium_allowed_svc_comm = pickle.load(f)
            except:
                self.cilium_allowed_svc_comm = None
            #min_rate_statspipelines.create_the_report(self.auto_open_pdfs)

        ## should return performance table for second val instead of None...
        ## okay, so for non-eval this should be @ same injection rate for train and test (marked i)
        # for eval, this should simply be over eval (whether physical or strictly injected) (marked ii)

        ## TODO: modify the part below to handle the new model using eval data... (ideally ONLY the retiurn statements...)
        ## and in fact, only really the 2nd term in the retrun statements...

        '''
        self.rate_to_type_of_model_to_time_gran_to_predicted_test
        self.type_of_model_to_time_gran_to_cm
        '''

        if self.no_labeled_data and self.skip_model_part:
            return min_rate_statspipelines, self.rate_to_time_gran_to_predicted_test[min(self.avg_exfil_per_min)], \
                   None, persvc_ensemble_model, self.rate_to_type_of_model_to_time_gran_to_predicted_test[min(self.avg_exfil_per_min)]

        self.rate_to_tg_to_cm = {}
        for rate,single_rate_statsp in self.single_rate_stats_pipelines.iteritems():
            self.rate_to_tg_to_cm[rate] = single_rate_statsp.time_gran_to_cm

        ## okay, now write it out...

        with open(self.base_output_name + '_rate_to_tg_to_cm.pickle', 'w') as f:
            f.write(pickle.dumps(self.rate_to_tg_to_cm))

        # need to turn rate_to_type_of_model_to_time_gran_to_cm into
        type_of_model_to_rate_to_tg_to_cm = {}
        if len(self.rate_to_type_of_model_to_time_gran_to_cm.keys()) > 0:
            types_of_models = self.rate_to_type_of_model_to_time_gran_to_cm[ self.rate_to_type_of_model_to_time_gran_to_cm.keys()[0] ].keys()
            for type_of_model in types_of_models:
                type_of_model_to_rate_to_tg_to_cm[type_of_model] = {}

                for rate, type_of_model_to_time_gran_to_cm in self.rate_to_type_of_model_to_time_gran_to_cm.iteritems():
                    time_gran_to_cm = type_of_model_to_time_gran_to_cm[type_of_model]

                    type_of_model_to_rate_to_tg_to_cm[type_of_model][rate] = time_gran_to_cm

        return min_rate_statspipelines, self.rate_to_tg_to_cm, self.cilium_allowed_svc_comm, persvc_ensemble_model, type_of_model_to_rate_to_tg_to_cm

    def generate_aggregate_report(self):
        cur_exp_name = self.function_list[0].base_exp_name
        generate_aggregate_report.generate_aggregate_report(self.rate_to_timegran_to_methods_to_attacks_found_dfs,
                                                            self.rate_to_timegran_list_of_methods_to_attacks_found_training_df,
                                                            self.base_output_name, self.rates_to_experiment_info,
                                                            self.rate_to_time_gran_to_outtraffic, self.auto_open_pdfs,
                                                            cur_exp_name)
        # XAB

    def single_pipeline_after_injection(self, list_time_gran_to_mod_zscore_df, list_time_gran_to_zscore_dataframe,
                                        list_time_gran_to_feature_dataframe, list_time_gran_to_mod_zscore_df_training,
                                        list_time_gran_to_mod_zscore_df_testing, starts_of_testing, cilium_allowed_svc_comm,
                                        cur_base_output_name, rate_counter, avg_exfil_per_min, load_old_pipelines):

        if cilium_allowed_svc_comm is not None:
            self.cilium_allowed_svc_comm = cilium_allowed_svc_comm
            self.perform_svcpair_sec_component = False

        # step (2) :  store aggregated DFs for reference purposes
        print "about_to_do_list_time_gran_to_mod_zscore_df"
        time_gran_to_aggregate_mod_score_dfs = aggregate_dfs(list_time_gran_to_mod_zscore_df)
        print "about_to_do_list_time_gran_to_feature_dataframe"
        time_gran_to_aggreg_feature_dfs = aggregate_dfs(list_time_gran_to_feature_dataframe)

        for time_gran, aggregate_feature_df in time_gran_to_aggreg_feature_dfs.iteritems():
            aggregate_feature_df.to_csv(
                cur_base_output_name + 'aggregate_feature_df_at_time_gran_of_' + str(time_gran) + '_sec.csv',
                na_rep='?')
        for time_gran, aggregate_feature_df in time_gran_to_aggregate_mod_score_dfs.iteritems():
            aggregate_feature_df.to_csv(
                cur_base_output_name + 'modz_feat_df_at_time_gran_of_' + str(time_gran) + '_sec.csv',
                na_rep='?')

        recipes_used = [recipe.base_exp_name for recipe in self.function_list]

        ## Note: the rest of this function exists to support the use-case where there was a self.pretrained_min_pipeline
        ## value specified (I probably want to decouple this more...)

        ## ## start stuff that could be loaded... ## ##

        stats_pipeline_loc = self.base_output_name + '_single_rate_stats_pipelines_' + \
                             str(self.avg_exfil_per_min[rate_counter]) + ':' + str(self.exfil_per_min_variance[rate_counter])
        print "load_old_pipelines", load_old_pipelines
        #exit(2)

        # does this make sense??? I am not sure, tbh...
        if load_old_pipelines:
            with open(stats_pipeline_loc, 'r') as f:
                stats_pipelines = pickle.loads(f.read())
        else:
            stats_pipelines = single_rate_stats_pipeline(time_gran_to_aggregate_mod_score_dfs, self.ROC_curve_p,
                                                         cur_base_output_name, recipes_used, self.skip_model_part,
                                                         self.avg_exfil_per_min[rate_counter],
                                                         self.avg_pkt_size[rate_counter],
                                                         self.exfil_per_min_variance[rate_counter],
                                                         self.pkt_size_variance[rate_counter],
                                                         self.no_labeled_data)

            stats_pipelines.run_statistical_pipeline(self.drop_pairwise_features, self.pretrained_min_pipeline,
                                                     skip_heatmap_p=self.skip_heatmap_p, logistic_p=self.use_logistic)

            if not self.no_labeled_data:
                stats_pipelines.create_the_report(self.auto_open_pdfs, use_ts_lower=self.use_ts_lower)

            with open(stats_pipeline_loc, 'w') as f:
                pickle.dump(stats_pipelines, f)

        ## ## end stuff to load... ## ##

        self.single_rate_stats_pipelines[self.avg_exfil_per_min[rate_counter]] = stats_pipelines

        list_of_optimal_fone_scores_at_this_exfil_rates, Xs, Ys, Xts, Yts, trained_models, list_of_attacks_found_dfs, \
        list_of_attacks_found_training_df, experiment_info, time_gran_to_outtraffic, timegran_to_statistical_pipeline = \
            stats_pipelines.generate_return_values()

        optimal_fones = list_of_optimal_fone_scores_at_this_exfil_rates
        timegran_to_methods_to_attacks_found_dfs = list_of_attacks_found_dfs
        timegran_to_methods_toattacks_found_training_df = list_of_attacks_found_training_df

        self.rates_to_experiment_info[self.avg_exfil_per_min[rate_counter]] = experiment_info
        self.rate_to_timegran_to_methods_to_attacks_found_dfs[
            self.avg_exfil_per_min[rate_counter]] = timegran_to_methods_to_attacks_found_dfs
        self.rate_to_timegran_list_of_methods_to_attacks_found_training_df[
            self.avg_exfil_per_min[rate_counter]] = timegran_to_methods_toattacks_found_training_df
        self.list_of_optimal_fone_scores_at_exfil_rates.append(optimal_fones)

        if self.avg_exfil_per_min[rate_counter] not in self.rate_to_time_gran_to_xs:
            self.rate_to_time_gran_to_outtraffic[self.avg_exfil_per_min[rate_counter]] = []
        self.rate_to_time_gran_to_outtraffic[self.avg_exfil_per_min[rate_counter]].append(time_gran_to_outtraffic)

        cur_exfil_rate = avg_exfil_per_min[rate_counter]
        self.rate_to_timegran_to_statistical_pipeline[cur_exfil_rate] = timegran_to_statistical_pipeline
        self.rate_to_time_gran_to_predicted_test[cur_exfil_rate] = stats_pipelines.time_gran_to_predicted_test


    def train_persvc_ensemble_model(self, rate_to_time_gran_to_cilium_alerts):
        print "starting train_persvc_ensemble_model"


        time_gran_to_new_df = combine_different_exfil_rate_dfs(self.rate_to_timegran_to_statistical_pipeline)
        cur_base_output_name = self.base_output_name + 'new_models_'

        #print "time_gran_to_new_df", time_gran_to_new_df

        exfil_model_object = exfil_detection_model(time_gran_to_new_df, self.ROC_curve_p, cur_base_output_name,
                                                   self.names, self.skip_model_part, ' multirate_varies', 'multirate_varies',
                                                   'multirate_varies', 'multirate_varies', False)
        exfil_model_object.train_pergran_models()

        using_pretrained_model = not( not self.pretrained_min_pipeline)
        exfil_model_object.generate_reports(self.auto_open_pdfs, self.skip_heatmap_p, using_pretrained_model)

        return exfil_model_object

    def train_multi_exfilrate_model(self):
        ## using this is the current plan I think: self.rate_to_timegran_to_statistical_pipeline
        ## steps: (1) go through, get all the vals with injected exfil paths
        ##        (2) extract non-injected values
        ##        (3) create new-combined dataset that I can send to the stats part

        ## Q: would this be better??? single_rate_stats_pipelines
        ## A: yes. Xs and stuff are easy to get.

        # Okay, so we want to call the function below. so the goal of the above part is the make this new DF that can
        # be used in this function.
        '''
        sav2_object = single_rate_stats_pipeline(timegran_to_df_max_exfil, self.ROC_curve_p, cur_base_output_name,
                                         self.names, self.skip_model_part, 'varies', 'varies', 'varies',
                                         'varies', False)
        '''

        time_gran_to_new_df = combine_different_exfil_rate_dfs(self.rate_to_timegran_to_statistical_pipeline)

        for timegran in time_gran_to_new_df.keys():
            time_gran_to_new_df[timegran].to_csv(path_or_buf=self.base_output_name + str(timegran) + '_multi_exfilrate_vals.csv')

        cur_base_output_name = self.base_output_name + 'multi_rate_exfil_report'
        new_sav2_object = single_rate_stats_pipeline(time_gran_to_new_df, self.ROC_curve_p, cur_base_output_name,
                                         self.names, self.skip_model_part, ' multirate_varies', 'multirate_varies',
                                        'multirate_varies', 'multirate_varies', False)

        new_sav2_object.run_statistical_pipeline(self.drop_pairwise_features, self.pretrained_min_pipeline,
                                                 skip_heatmap_p=self.skip_heatmap_p, logistic_p=self.use_logistic)
        new_sav2_object.create_the_report(self.auto_open_pdfs)

        return new_sav2_object

    def lower_per_path_exfil_rates(self, timegran):
        exfil_rates = sorted(self.avg_exfil_per_min )
        # step 1: find the feature dataframe corresponding to the largest exfil rate
        feature_df_max_exfil = copy.deepcopy(self.rate_to_timegran_to_statistical_pipeline[max(exfil_rates)][timegran].aggregate_mod_score_df)
        path_to_cur_rate = {}
        exfil_paths_series = feature_df_max_exfil['exfil_path']

        # step 2: iterate through exfil paths and rates (in decreasing order)
        exfil_paths = set()
        for exfil_path in exfil_paths_series:
            exfil_paths.add(tuple((exfil_path,)))
            path_to_cur_rate[tuple((exfil_path,))] = max(exfil_rates)
        exfil_paths = list(exfil_paths)

        ## do I still want to modify lowering via:: find_operating_point_given_fps(self, timegran, fp_limit)??
        ## b/c if I do... I probably wanna change the logic so that the rate counter goes from big to small
        ## and then new_exfil_rate_statspipeline to involve th call to find_operating_point_given_fps to get the new operating point...
        ## (plus the calculation of the fps that are permissible)
        ## update: it does indeed go from big to smaller (b/c that is how self.avg_exfil_per_min is organized...)
        ## okay, then it is time to commence operation: find_operating_point_given_fps!!
        ## should be straightforward, but might add a parameter...

        fp_limit = (float(self.time_fraction_fp) * len(self.rate_to_timegran_to_statistical_pipeline[self.avg_exfil_per_min[0]][
                            timegran].test_predictions))
        ## initial FP limit is # of FPs at optimal F1 operating point of highst exfil rate systtem
        for rate_counter in range(0, len(self.avg_exfil_per_min)):
            #print "cur_fp_limit", fp_limit
            new_exfil_rate_statspipeline = self.rate_to_timegran_to_statistical_pipeline[self.avg_exfil_per_min[rate_counter]][timegran]
            new_rate_cm = new_exfil_rate_statspipeline.find_optimal_cm_given_fps(fp_limit)
            ## do we actually need to do this??? yes we do. as shown by performance at the 60 second granularity...
            #new_rate_cm = new_exfil_rate_statspipeline.method_to_cm_df_test['ensemble']

            ## note: I am still assuming that the paths are independent... otherwise I'd retrain after every completion
            ## of the loop through the possible exfil_paths...

            for counter, exfil_path in enumerate(exfil_paths):
                exfil_path = exfil_path[0]
                #print "counter", counter, type(exfil_path), exfil_path
                # step 3: all exfil paths in the feature_df_max_exfil have 'max' detection capabilities ATM...
                # if I can decrease the rate w/o decreasing the TPR, then I should do so.
                ## Step 3a: find performance of old rat
                if exfil_path == ('0',) or exfil_path == 0 or exfil_path == '0':
                    continue
                exfil_path_key = tuple(ast.literal_eval(exfil_path.replace('\\','')))
                #print self.rate_to_timegran_to_statistical_pipeline[path_to_cur_rate[tuple((exfil_path,))]][timegran].method_to_cm_df_train['ensemble']
                #print self.rate_to_timegran_to_statistical_pipeline[path_to_cur_rate[tuple((exfil_path,))]][timegran].method_to_cm_df_train['ensemble']

                old_test_dfs = self.rate_to_timegran_to_statistical_pipeline[path_to_cur_rate[tuple((exfil_path,))]][timegran].method_to_cm_df_test['ensemble']
                cur_exfil_path_performance = old_test_dfs.loc[[exfil_path_key]]
                cur_exfil_path_tpr = float(cur_exfil_path_performance['tp']) / (cur_exfil_path_performance['tp'] + cur_exfil_path_performance['fn'])
                cur_exfil_path_tpr = cur_exfil_path_tpr._values[0]

                ## Step 3b: find performance of new rate
                new_exfil_path_performance = new_rate_cm.loc[[exfil_path_key]]
                new_exfil_path_tpr = float(new_exfil_path_performance['tp']) / (new_exfil_path_performance['tp'] + new_exfil_path_performance['fn'])
                new_exfil_path_tpr = new_exfil_path_tpr._values[0]

                ## Step 3c: if new performance just as good, switch
                ##### note: may due to make modifications b/c increasing TPR could be a result of increasing FPR too
                if new_exfil_path_tpr >= cur_exfil_path_tpr and cur_exfil_path_tpr != 0.0:
                    feature_df_max_exfil[feature_df_max_exfil['exfil_path'] == exfil_path] = \
                        new_exfil_rate_statspipeline.aggregate_mod_score_df[ new_exfil_rate_statspipeline.aggregate_mod_score_df['exfil_path'] == exfil_path]

            ## increase fp limit  based on some  percentage of time length that has been passed as a system parameter...

        return feature_df_max_exfil

    def decrease_exfil_of_model(self):
        timegran_to_df_max_exfil = {}
        for timegran in self.rate_to_time_gran_to_outtraffic[ self.rate_to_time_gran_to_outtraffic.keys()[0] ][0].keys():
            timegran_to_df_max_exfil[timegran] = self.lower_per_path_exfil_rates(timegran)

        #t_rs = self.single_rate_stats_pipelines[self.avg_exfil_per_min[0]].timegran_to_robust_scaler

        cur_base_output_name = self.base_output_name + '_lower_per_path_exfil_report_'
        sav2_object = single_rate_stats_pipeline(timegran_to_df_max_exfil, self.ROC_curve_p, cur_base_output_name,
                                                 self.names, self.skip_model_part, 'varies', 'varies', 'varies',
                                                 'varies', False)

        sav2_object.run_statistical_pipeline(self.drop_pairwise_features, self.pretrained_min_pipeline,
                                             skip_heatmap_p=self.skip_heatmap_p, logistic_p=self.use_logistic)
        sav2_object.create_the_report(self.auto_open_pdfs)

        return sav2_object


def combine_different_exfil_rate_dfs(rate_to_timegran_to_statistical_pipeline):
    '''
    This function takes our set of feature dfs, all of which have an associated time granularity and exfiltration
    rate. The datasets of the different exfiltration rates needed to be combined (when the time granularities match).
    '''


    time_gran_to_new_df = {}
    # counter = 0
    for rate, timegran_to_statistical_pipeline in rate_to_timegran_to_statistical_pipeline.iteritems():
        # counter += 1
        # if counter ==4:
        #    break
        for timegran, stats_pipeline in timegran_to_statistical_pipeline.iteritems():
            if type(timegran) == tuple:
                continue
            if timegran not in time_gran_to_new_df:
                time_gran_to_new_df[timegran] = copy.deepcopy(stats_pipeline.orig_aggregate_mod_score_df)
                continue
            # append relevant part to the time_gran_to_new_df and flip is_test (b/c don't need is_test anymore)
            cur_df = stats_pipeline.orig_aggregate_mod_score_df
            ## (a) :: get only those with injected
            attack_portions = cur_df.loc[cur_df['labels'] == 1]
            ## (b) :: append onto dataframe
            time_gran_to_new_df[timegran] = time_gran_to_new_df[timegran].append(attack_portions, ignore_index=True)
            ## (c) :: switch is_test to all zeros (REMOVE IF I MAKE THE SWITCH PERMENANT) (<<- ignore this...)
            time_gran_to_new_df[timegran][
                'is_test'] = 0  ## actually going to keep this... so both models can be useful....
    return time_gran_to_new_df

def inject_comm_graphs_at_single_exfil_rate(rate_counter, avg_exfil_per_min, exfil_per_min_variance,
                                            base_output_name, exps_exfil_paths,
                                            exps_initiator_info, calculate_z_scores_p, end_of_train_portions,
                                            training_exfil_paths, testing_exfil_paths, skip_model_part,
                                            ROC_curve_p, avg_pkt_size,
                                            pkt_size_variance, drop_pairwise_features, perform_svcpair_sec_component,
                                            ide_window_size, drop_infra_from_graph, function_list, pretrained_min_pipeline,
                                            calc_vals, skip_graph_injection, calc_ide, pretrained_cilium_model, no_labeled_data,
                                            rate_to_list_time_gran_to_mod_zscore_df, rate_to_cur_base_output_name,
                                            rate_to_list_time_gran_to_zscore_dataframe,
                                            rate_to_list_time_gran_to_feature_dataframe,
                                            rate_to_list_time_gran_to_mod_zscore_df_training,
                                            rate_to_list_time_gran_to_mod_zscore_df_testing,
                                            rate_to_starts_of_testing, rate_to_cilium_allowed_svc_comm,
                                            rate_to_time_gran_to_cilium_alerts):
    ## okay, how about we make this a new function??
    prefix_for_inject_params = 'avg_exfil_' + str(avg_exfil_per_min[rate_counter]) + ':' + str(
        exfil_per_min_variance[rate_counter]) + '_'  # +  '_avg_pkt_' + str(self.avg_pkt_size[rate_counter]) + ':' + str(
    # self.pkt_size_variance[rate_counter]) + '_'
    cur_base_output_name = base_output_name + prefix_for_inject_params
    cur_exfil_rate = avg_exfil_per_min[rate_counter]

    out_q = multiprocessing.Queue()
    cur_function_list = [copy.deepcopy(i) for i in function_list]

    ###
    #if self.pretrained_min_pipeline:
    #    pretrained_min_pipeline = self.pretrained_min_pipeline
    #else:
    #    pretrained_min_pipeline = None
    ##

    args = [rate_counter, base_output_name, cur_function_list, exps_exfil_paths, exps_initiator_info,
            calculate_z_scores_p, calc_vals, end_of_train_portions, training_exfil_paths,
            testing_exfil_paths, skip_model_part, out_q,
            ROC_curve_p, avg_exfil_per_min, exfil_per_min_variance, avg_pkt_size,
            pkt_size_variance, skip_graph_injection, calc_ide,
            drop_pairwise_features, perform_svcpair_sec_component, None,
            ide_window_size, drop_infra_from_graph, prefix_for_inject_params, pretrained_min_pipeline,
            pretrained_cilium_model, no_labeled_data]
    p = multiprocessing.Process(
        target=pipeline_one_exfil_rate,
        args=args)
    p.start()

    exfil_rate = avg_exfil_per_min[rate_counter]
    #print "inject_comm_graphs_at_single_exfil_rate waiting for results on this exfil rate: " + str(exfil_rate)

    rate_to_list_time_gran_to_mod_zscore_df[exfil_rate] = out_q.get()
    rate_to_list_time_gran_to_zscore_dataframe[exfil_rate] = out_q.get()
    rate_to_list_time_gran_to_feature_dataframe[exfil_rate] = out_q.get()
    rate_to_list_time_gran_to_mod_zscore_df_training[exfil_rate] = out_q.get()
    rate_to_list_time_gran_to_mod_zscore_df_testing[exfil_rate] = out_q.get()
    rate_to_starts_of_testing[exfil_rate] = out_q.get()
    rate_to_cilium_allowed_svc_comm[exfil_rate] = out_q.get()
    rate_to_time_gran_to_cilium_alerts[exfil_rate] = out_q.get()
    rate_to_cur_base_output_name[exfil_rate] = cur_base_output_name

    #print "rate_to_list_time_gran_to_mod_zscore_df_at_end", rate_to_list_time_gran_to_mod_zscore_df

    p.join()

def pipeline_one_exfil_rate(rate_counter, base_output_name, function_list, exps_exfil_paths, exps_initiator_info,
                            calculate_z_scores_p, calc_vals, end_of_train_portions, training_exfil_paths,
                            testing_exfil_paths, skip_model_part, out_q, ROC_curve_p, avg_exfil_per_min,
                            exfil_per_min_variance, avg_pkt_size, pkt_size_variance, skip_graph_injection, calc_ide,
                            drop_pairwise_features, perform_svcpair_component,
                            cilium_component_time_not_used, ide_window_size, drop_infra_from_graph, prefix_for_inject_params,
                            pretrained_min_pipeline=None, pretrained_svcpair_model=None, no_labeled_data=False):
    ## step (1) : iterate through individual experiments...
    ##  # 1a. list of inputs [done]
    ##  # 1b. acculate DFs
    list_time_gran_to_mod_zscore_df = []
    list_time_gran_to_mod_zscore_df_training = []
    list_time_gran_to_mod_zscore_df_testing = []
    list_time_gran_to_zscore_dataframe = []
    list_time_gran_to_feature_dataframe = []
    list_to_time_gran_to_cilium_alerts = []
    starts_of_testing = []

    cilium_allowed_svc_comm = None
    for counter,experiment_object in enumerate(function_list):
        #print "exps_exfil_paths[counter]_to_func",exps_exfil_paths[counter], exps_initiator_info

        experiment_object.alert_file = experiment_object.orig_alert_file + prefix_for_inject_params
        experiment_object.basegraph_name = experiment_object.orig_basegraph_name + prefix_for_inject_params
        experiment_object.exp_name = experiment_object.orig_exp_name + prefix_for_inject_params

        experiment_object.calc_vals = calc_vals
        experiment_object.calc_zscore_p = calculate_z_scores_p or calc_vals
        experiment_object.skip_graph_injection = skip_graph_injection
        experiment_object.result_dir = base_output_name

        #if pretrained_transformers and time_gran in pretrained_transformers:
        #    pretrained_transformer = pretrained_transformers[time_gran]
        #else:
        #    pretrained_transformer = None

        time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, time_gran_to_feature_dataframe, _, start_of_testing, \
             = experiment_object.calculate_values(end_of_training=end_of_train_portions[counter],
                                           synthetic_exfil_paths_train=training_exfil_paths[counter],
                                           synthetic_exfil_paths_test=testing_exfil_paths[counter],
                                           avg_exfil_per_min=avg_exfil_per_min[rate_counter],
                                           exfil_per_min_variance=exfil_per_min_variance[rate_counter],
                                           avg_pkt_size=avg_pkt_size[rate_counter],
                                           pkt_size_variance=pkt_size_variance[rate_counter],
                                           calc_ide=calc_ide,
                                           ide_window_size=ide_window_size,
                                           drop_infra_from_graph=drop_infra_from_graph,
                                           pretrained_min_pipeline=pretrained_min_pipeline, no_labeled_data=no_labeled_data)

        if perform_svcpair_component and pretrained_svcpair_model is None: #
            cilium_allowed_svc_comm = experiment_object.run_cilium_component(start_of_testing, base_output_name,
                                                                             experiment_object.interval_to_filenames)
            perform_svcpair_component = False

        if perform_svcpair_component:
            time_gran_to_cilium_alerts = experiment_object.calc_cilium_performance(avg_exfil_per_min[rate_counter],
                    exfil_per_min_variance[rate_counter], avg_pkt_size[rate_counter], pkt_size_variance[rate_counter],
                    pretrained_svcpair_model)
            list_to_time_gran_to_cilium_alerts.append(time_gran_to_cilium_alerts)
            # okay, so now I probably need to do something with these alerts...
            # and then actually do something with all of this stuff...
            #print 'at rate', avg_exfil_per_min[rate_counter], "cilium_performance", time_gran_to_cilium_alerts

            for time_gran, cilium_alerts in time_gran_to_cilium_alerts.iteritems():
                length_alerts = len(time_gran_to_mod_zscore_df[time_gran].index.values)
                cilium_alerts = cilium_alerts[:length_alerts]
                #print len(time_gran_to_mod_zscore_df[time_gran].index.values), len(cilium_alerts), length_alerts
                time_gran_to_mod_zscore_df[time_gran]['cilium_for_first_sec_' + str(start_of_testing)] = cilium_alerts


        #print "exps_exfil_pathas[time_gran_to_mod_zscore_df]", time_gran_to_mod_zscore_df
        #print time_gran_to_mod_zscore_df[time_gran_to_mod_zscore_df.keys()[0]].columns.values
        list_time_gran_to_mod_zscore_df.append(time_gran_to_mod_zscore_df)
        list_time_gran_to_zscore_dataframe.append(time_gran_to_zscore_dataframe)
        list_time_gran_to_feature_dataframe.append(time_gran_to_feature_dataframe)
        list_time_gran_to_mod_zscore_df_training.append(generate_time_gran_sub_dataframes(time_gran_to_mod_zscore_df, 'is_test', 0))
        list_time_gran_to_mod_zscore_df_testing.append(generate_time_gran_sub_dataframes(time_gran_to_mod_zscore_df, 'is_test', 1))
        starts_of_testing.append(start_of_testing)
        print "about to garbage collect..."
        gc.collect()

    print "going to put items into the output qs..."
    out_q.put(list_time_gran_to_mod_zscore_df)
    out_q.put(list_time_gran_to_zscore_dataframe)
    out_q.put(list_time_gran_to_feature_dataframe)
    out_q.put(list_time_gran_to_mod_zscore_df_training)
    out_q.put(list_time_gran_to_mod_zscore_df_testing)
    out_q.put(starts_of_testing)
    out_q.put(cilium_allowed_svc_comm)
    out_q.put(list_to_time_gran_to_cilium_alerts)


def determine_and_assign_exfil_paths(calc_vals, skip_model_part, function_list, goal_train_test_split,
                                     goal_attack_NoAttack_split_training, time_each_synthetic_exfil,
                                     goal_attack_NoAttack_split_testing, max_path_length, dns_porportion):
    if calc_vals and not skip_model_part:
        #print function_list
        exp_infos = []
        for experiment_object in function_list:
            #print "calc_vals", calc_vals
            total_experiment_length, exfil_startEnd_times = experiment_object.get_exp_info()
            #print "func_exp_info", total_experiment_length, exfil_startEnd_times
            exp_infos.append({"total_experiment_length":total_experiment_length, "exfil_startEnd_times":exfil_startEnd_times})

        ## get the exfil_paths that were generated using the mulval component...
        ## this'll require passing a parameter to the single-experiment pipeline and then getting the set of paths
        exps_exfil_paths = []
        exps_initiator_info = []
        total_training_injections_possible, total_testing_injections_possible, _, _ = \
            determine_injection_amnts(exp_infos, goal_train_test_split, goal_attack_NoAttack_split_training,
                                      time_each_synthetic_exfil, float("inf"), goal_attack_NoAttack_split_testing)
        if total_training_injections_possible == 0 or total_testing_injections_possible == 0:
            # this happens during eval portion or when there's a bunch of physical exfiltration events occuring
            max_number_of_paths = max(total_testing_injections_possible, total_training_injections_possible)
        else:
            max_number_of_paths = min(total_training_injections_possible, total_testing_injections_possible)

        orig_max_number_of_paths=  max_number_of_paths
        for experiment_object in function_list:
            #print "experiment_object", experiment_object
            synthetic_exfil_paths, initiator_info_for_paths = \
                experiment_object.generate_synthetic_exfil_paths(max_number_of_paths=max_number_of_paths,
                                                                 max_path_length=max_path_length,
                                                                 dns_porportion=dns_porportion)
            max_number_of_paths = None
            exps_exfil_paths.append(synthetic_exfil_paths)
            exps_initiator_info.append(initiator_info_for_paths)

        print "orig_max_number_of_paths", orig_max_number_of_paths
        print "orig_exps_exfil_paths",exps_exfil_paths
        #print exps_exfil_paths
        for counter,exp_path in enumerate(exps_exfil_paths[0]):
            print counter,exp_path,len(exp_path)
        #exit(344)
        training_exfil_paths, testing_exfil_paths, end_of_train_portions = assign_exfil_paths_to_experiments(exp_infos,
                                                                                                             goal_train_test_split,
                                                                                                             goal_attack_NoAttack_split_training,
                                                                                                             time_each_synthetic_exfil,
                                                                                                             exps_exfil_paths,
                                                                                                             goal_attack_NoAttack_split_testing)
        print "end_of_train_portions", end_of_train_portions
        print total_training_injections_possible, total_testing_injections_possible
        possible_exps_exfil_paths = []
        for exp_exfil_paths in exps_exfil_paths:
            for exp_exfil_path in exp_exfil_paths:
                if exp_exfil_path not in possible_exps_exfil_paths:
                    possible_exps_exfil_paths.append(exp_exfil_path)
        print "possible_exps_exfil_paths:"
        for possible_exp_exfil_path in possible_exps_exfil_paths:
            print possible_exp_exfil_path
        print "training_exfil_paths:"
        for cur_training_exfil_paths in training_exfil_paths:
            print "cur_training_exfil_paths", cur_training_exfil_paths, len(cur_training_exfil_paths)
        print "testing_exfil_paths:"
        for cur_testing_exfil_paths in testing_exfil_paths:
            print "cur_testing_exfil_paths", cur_testing_exfil_paths, len(cur_testing_exfil_paths)
        print "look_here"
        #exit(122) ### TODO::: <--- remove!!!

    else:
        exps_exfil_paths = []
        end_of_train_portions = []
        training_exfil_paths = []
        testing_exfil_paths = []
        exps_initiator_info= []
        for func in function_list:
            # just fill these w/ nothing so that the function doesn't think that it needs to calculate them (b/c it doesn't)
            exps_exfil_paths.append([])
            end_of_train_portions.append(0)
            training_exfil_paths.append([])
            testing_exfil_paths.append([])
            exps_initiator_info.append([])

    return exps_exfil_paths, end_of_train_portions, training_exfil_paths, testing_exfil_paths, exps_initiator_info

def aggregate_dfs(list_time_gran_to_mod_zscore_df):
    time_gran_to_aggregate_mod_score_dfs = {}
    #print "list_time_gran_to_mod_zscore_df",list_time_gran_to_mod_zscore_df
    for time_gran_to_mod_zscore_df in list_time_gran_to_mod_zscore_df:
        #print "time_gran_to_mod_zscore_df",time_gran_to_mod_zscore_df
        for time_gran, mod_zscore_df in time_gran_to_mod_zscore_df.iteritems():
            if time_gran not in time_gran_to_aggregate_mod_score_dfs.keys():
                time_gran_to_aggregate_mod_score_dfs[time_gran] = mod_zscore_df
                #print "post_initializing_aggregate_dataframe", len(time_gran_to_aggregate_mod_score_dfs[time_gran]), \
                #    type(time_gran_to_aggregate_mod_score_dfs[time_gran]), time_gran

            else:
                time_gran_to_aggregate_mod_score_dfs[time_gran] = \
                    time_gran_to_aggregate_mod_score_dfs[time_gran].append(mod_zscore_df, sort=True)

    return time_gran_to_aggregate_mod_score_dfs#, time_gran_to_aggregate_mod_score_dfs_training, time_gran_to_aggregate_mod_score_dfs_testing

# this function determines which experiments should have which synthetic exfil paths injected into them
def assign_exfil_paths_to_experiments(exp_infos, goal_train_test_split, goal_attack_NoAttack_split_training,
                                      time_each_synthetic_exfil, exps_exfil_paths, goal_attack_NoAttack_split_testing):

    flat_exps_exfil_paths = [tuple(exfil_path) for exp_exfil_paths in exps_exfil_paths for exfil_path in exp_exfil_paths]
    #print "flat_exps_exfil_paths",flat_exps_exfil_paths
    possible_exfil_paths = list(set(flat_exps_exfil_paths))

    total_training_injections_possible,total_testing_injections_possible,possible_exfil_path_injections,end_of_train_portions = \
        determine_injection_amnts(exp_infos, goal_train_test_split, goal_attack_NoAttack_split_training,
                                  time_each_synthetic_exfil, possible_exfil_paths, goal_attack_NoAttack_split_testing)

    exfil_path_to_occurences = {}
    for possible_exfil_path in possible_exfil_paths:
        exfil_path_to_occurences[tuple(possible_exfil_path)] = flat_exps_exfil_paths.count(tuple(possible_exfil_path))
    # fourth, actually perform exfil_path assignments to each experiment
    ## if different exfil paths were allowed for each experiment, this would be rather difficult. However,
    ## at the moment, we'll implicitly that all exfil paths are allowed by each experiment. This'll keep the assignment code
    ## very simple. (b/c theoretically we could have a linear proogramming assignment problem on our hands here...)
    #### 4.a. determine how many times we could inject all the exfil paths
    training_exfil_paths = []
    testing_exfil_paths = []

    #print "total_testing_injections_possible",total_testing_injections_possible
    #print "total_training_injections_possible", total_training_injections_possible
    #print "possible_exfil_paths",possible_exfil_paths

    #if
    testing_number_times_inject_all_paths = math.floor(total_testing_injections_possible / float(len(possible_exfil_paths)))
    if total_training_injections_possible == 0: # this happens when runnning the eval poriton...
        training_number_times_inject_all_paths = 0
    else:
        training_number_times_inject_all_paths = math.floor(total_training_injections_possible / float(len(possible_exfil_paths)))
        if training_number_times_inject_all_paths < 1.0:
            print "can't inject all exfil paths in training set... "
            exit(33)

    if total_training_injections_possible == 0: ## happens when there's lots of physical attacks
        if testing_number_times_inject_all_paths < 1.0:
            print "can't inject all exfil paths in testing set..."
            exit(34)

    exfil_paths_to_test_injection_counts = {}
    exfil_paths_to_train_injection_counts = {}
    for exfil_path in possible_exfil_paths:
        exfil_paths_to_test_injection_counts[tuple(exfil_path)] = testing_number_times_inject_all_paths
        exfil_paths_to_train_injection_counts[tuple(exfil_path)] = training_number_times_inject_all_paths
    for possible_exfil_path_injection in possible_exfil_path_injections:
        ## note: this ^^ variable contains the number of times can inject training/testing exfil paths here...
        current_training_exfil_paths = []
        training_times_to_inject_this_exp = possible_exfil_path_injection['training']
        #print "(initial)training_times_to_inject_this_exp",training_times_to_inject_this_exp
        while training_times_to_inject_this_exp > 0:
            path = max(exfil_paths_to_train_injection_counts.iteritems(), key=operator.itemgetter(1))[0]
            #print "current_max_path", path
            if exfil_paths_to_train_injection_counts[path] > 0:
                current_training_exfil_paths.append(list(path))
                training_times_to_inject_this_exp -= 1
                exfil_paths_to_train_injection_counts[path] -= 1
            else:
                # note: this isn't actually a problem b/c we rounded down when assigning the # of injection counts for each path
                break
        training_exfil_paths.append(current_training_exfil_paths)

        current_testing_exfil_paths = []
        testing_times_to_inject_this_exp = possible_exfil_path_injection['testing']
        while testing_times_to_inject_this_exp > 0:
            path = max(exfil_paths_to_test_injection_counts.iteritems(), key=operator.itemgetter(1))[0]
            #print "current_max_testing_path", path
            if exfil_paths_to_test_injection_counts[path] > 0:
                current_testing_exfil_paths.append(list(path))
                testing_times_to_inject_this_exp -= 1
                exfil_paths_to_test_injection_counts[path] -= 1
            else:
                # note: this isn't actually a problem b/c we rounded down when assigning the # of injection counts for each path
                break
        testing_exfil_paths.append(current_testing_exfil_paths)

    #print "training_exfil_paths", training_exfil_paths
    #print "testing_exfil_paths", testing_exfil_paths

    remaining_testing_injections = sum(exfil_paths_to_test_injection_counts.values())
    remaining_training_injections = sum(exfil_paths_to_train_injection_counts.values())
    print "float(len(possible_exfil_paths))",float(len(possible_exfil_paths))
    print "remaining_testing_injections", remaining_testing_injections, "remaining_training_injections", remaining_training_injections
    print "testing_number_times_inject_all_paths", testing_number_times_inject_all_paths, \
        "training_number_times_inject_all_paths", training_number_times_inject_all_paths
    print "total_training_injections_possible", total_training_injections_possible, "total_testing_injections_possible", total_testing_injections_possible

    return training_exfil_paths,testing_exfil_paths,end_of_train_portions


def generate_time_gran_sub_dataframes(time_gran_to_df_dataframe, column_name, column_value):
    time_gran_to_sub_dataframe = {}
    for time_gran, dataframe in time_gran_to_df_dataframe.iteritems():
        sub_dataframe = dataframe[dataframe[column_name] == column_value]
        time_gran_to_sub_dataframe[time_gran] = sub_dataframe
    return time_gran_to_sub_dataframe

def determine_injection_amnts(exp_infos, goal_train_test_split, goal_attack_NoAttack_split_training,
                              time_each_synthetic_exfil, possible_exfil_paths, goal_attack_NoAttack_split_testing):
    ## now perform the actual assignment portion...
    # first, find the amt of time available for attack injections in each experiments training/testing phase...
    inject_times,end_of_train_portions = determine_injection_times(exp_infos, goal_train_test_split,
                                                                   goal_attack_NoAttack_split_training,
                                                                   goal_attack_NoAttack_split_testing)
    # second, find how many exfil_paths can be injected into each experiments training/testing
    possible_exfil_path_injections = []
    total_training_injections_possible = 0
    total_testing_injections_possible = 0
    for inject_time in inject_times:
        training_exfil_path_injections = min(math.floor(inject_time['training'] / time_each_synthetic_exfil),possible_exfil_paths)
        total_training_injections_possible += training_exfil_path_injections
        testing_exfil_path_injections = min(math.floor(inject_time['testing'] /  time_each_synthetic_exfil),possible_exfil_paths)
        total_testing_injections_possible += testing_exfil_path_injections
        possible_exfil_path_injections.append({"testing": testing_exfil_path_injections,
                                               "training": training_exfil_path_injections})
    #print "possible_exfil_path_injections", possible_exfil_path_injections
    #exit(34)
    return total_training_injections_possible,total_testing_injections_possible,possible_exfil_path_injections,end_of_train_portions

# generates a df indicating how long each logical exfil path occurs during each experiment, and returns a handle DF
# for use in the generated report.
def generate_exfil_path_occurence_df(list_of_time_gran_to_mod_zscore_df, experiment_names):
    experiments_to_exfil_path_time_dicts = []
    for time_gran_to_mod_zscore_df in list_of_time_gran_to_mod_zscore_df:
        #print time_gran_to_mod_zscore_df.keys()
        min_time_gran = min(time_gran_to_mod_zscore_df.keys())
        #print time_gran_to_mod_zscore_df[min_time_gran]
        # I *hope* this solves the list is unhashable problem....
        time_gran_to_mod_zscore_df[min_time_gran]['exfil_path'] = \
            [tuple(i) if type(i) == list  else i for i in time_gran_to_mod_zscore_df[min_time_gran]['exfil_path']]
        #print time_gran_to_mod_zscore_df[min_time_gran]['exfil_path']
        #print time_gran_to_mod_zscore_df[min_time_gran]['exfil_path']
        #print time_gran_to_mod_zscore_df[min_time_gran]['exfil_path'].values
        #print time_gran_to_mod_zscore_df[min_time_gran]['exfil_path'].value_counts()
        logical_exfil_paths_freq = time_gran_to_mod_zscore_df[min_time_gran]['exfil_path'].value_counts().to_dict()
        #exit(233)
        for path, occur in logical_exfil_paths_freq.iteritems():
            logical_exfil_paths_freq[path] = occur * min_time_gran
        experiments_to_exfil_path_time_dicts.append(logical_exfil_paths_freq)
    path_occurence_df = pd.DataFrame(experiments_to_exfil_path_time_dicts, index=experiment_names)
    return path_occurence_df

def graph_fone_versus_exfil_rate(optimal_fone_scores, exfil_weights_frac, exfil_pkts_frac, time_grans):

    time_gran_to_fone_list = {}
    time_gran_to_exfil_param_list = {}
    for exfil_counter, optimal_fones in enumerate(optimal_fone_scores):
        for timegran_counter, optimal_score in enumerate(optimal_fones):
            if time_grans[timegran_counter] in time_gran_to_fone_list:
                time_gran_to_fone_list[time_grans[timegran_counter]].append(optimal_score)
            else:
                time_gran_to_fone_list[time_grans[timegran_counter]] = [optimal_score]

            #print "time_gran_to_exfil_param_list", time_gran_to_exfil_param_list
            if time_grans[timegran_counter] in time_gran_to_exfil_param_list:
                time_gran_to_exfil_param_list[time_grans[timegran_counter]].append(
                    [(exfil_weights_frac[exfil_counter], exfil_pkts_frac[exfil_counter])])
            else:
                time_gran_to_exfil_param_list[time_grans[timegran_counter]] = \
                    [(exfil_weights_frac[exfil_counter], exfil_pkts_frac[exfil_counter])]
        #print counter,optimal_fones
    # and then plot...
    for time_gran in time_gran_to_fone_list.keys():
        plt.xlabel('f1')
        plt.ylabel('exfil_rate')
        plt.plot([i[0] for i in time_gran_to_exfil_param_list[time_gran]], time_gran_to_fone_list[time_gran])
        plt.draw()
        plt.savefig('temp_outputs/fone_vs_exfil_rate.png')


if __name__ == "__main__":
    time_gran_to_attack_labels = {1: [0, 0, 1, 1, 0, 0, 0, 0, 0, 0], 2: [0, 1, 0, 0, 0]}
    #print "INITIAL time_gran_to_attack_labels", time_gran_to_attack_labels
    synthetic_exfil_paths = [['a', 'b'], ['b', 'c']]
    time_of_synethic_exfil = 2
    startup_time_before_injection = 4
    time_gran_to_attack_labels, time_gran_to_attack_ranges, time_gran_to_physical_attack_ranges = \
        determine_attacks_to_times(time_gran_to_attack_labels, synthetic_exfil_paths,
                                   time_of_synethic_exfil, startup_time_before_injection)
    #print "time_gran_to_attack_labels", time_gran_to_attack_labels
    #print "time_gran_to_attack_ranges", time_gran_to_attack_ranges
    #print "time_gran_to_physical_attack_ranges", time_gran_to_physical_attack_ranges