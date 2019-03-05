import gc

import pandas as pd
import pyximport
from matplotlib import pyplot as plt

import generate_alerts
from analysis_pipeline.single_experiment_pipeline import determine_attacks_to_times
from analysis_pipeline.statistical_analysis import statistically_analyze_graph_features

pyximport.install() # to leverage cpython
import math
from sklearn.linear_model import LassoCV, LogisticRegressionCV
import operator
import copy
import multiprocessing
import pyximport
pyximport.install()

def generate_rocs(time_gran_to_anom_score_df, alert_file, sub_path):
    for time_gran, df_with_anom_features in time_gran_to_anom_score_df.iteritems():
        cur_alert_function,features_to_use = generate_alerts.determine_alert_function(df_with_anom_features)
        generate_alerts.generate_all_anom_ROCs(df_with_anom_features, time_gran, alert_file, sub_path, cur_alert_function,
                               features_to_use)

# this function determines how much time to is available for injection attacks in each experiment.
# it takes into account when the physical attack starts (b/c need to split into training/testing set
# temporally before the physical attack starts) and the goal percentage of split that we are aiming for.
# I think we're going to aim for the desired split in each experiment, but we WON'T try  to compensate
# not meeting one experiment's goal by modifying how we handle another experiment.
def determine_injection_times(exps_info, goal_train_test_split, goal_attack_NoAttack_split, ignore_physical_attacks_p):
    #time_splits = []
    exp_injection_info = []
    end_of_train_portions = []

    print "exps_info",exps_info
    for exp_info in exps_info:
        print "exp_info", exp_info['total_experiment_length'], "float(goal_train_test_split)", float(goal_train_test_split)
        time_split = ((exp_info['total_experiment_length'] - exp_info['startup_time']) * float(goal_train_test_split)) + exp_info['startup_time']
        if time_split > exp_info['exfil_start_time']:
            time_split = exp_info['exfil_start_time']
        end_of_train_portions.append(time_split)
        ## now to find how much time to spending injecting during training and testing...
        ## okay, let's do testing first b/c it should be relatively straightforward...
        testing_time = exp_info['total_experiment_length'] - time_split
        physical_attack_time = exp_info['exfil_end_time'] - exp_info['exfil_start_time']
        if ignore_physical_attacks_p:
            testing_time_for_attack_injection = (testing_time - physical_attack_time) * goal_attack_NoAttack_split
        else:
            testing_time_for_attack_injection = (testing_time) * goal_attack_NoAttack_split -physical_attack_time

        #testing_time_without_physical_attack = testing_time - physical_attack_time
        print "physical_attack_time",physical_attack_time, "testing_time", testing_time, testing_time_for_attack_injection
        testing_time_for_attack_injection = max(testing_time_for_attack_injection,0)

        # now let's find the time to inject during training... this'll be a percentage of the time between
        # system startup and the training/testing split point...
        training_time_after_startup = time_split - exp_info["startup_time"]
        training_time_for_attack_injection = training_time_after_startup * goal_attack_NoAttack_split

        exp_injection_info.append({'testing': testing_time_for_attack_injection,
                                   "training": training_time_for_attack_injection})
        #time_splits.append(time_split)
    print "exp_injection_info", exp_injection_info
    #exit(34)
    return exp_injection_info,end_of_train_portions

# this function loops through multiple experiments (or even just a single experiment), accumulates the relevant
# feature dataframes, and then performs LASSO regression to determine a concise graphical model that can detect
# the injected synthetic attacks
def multi_experiment_pipeline(function_list, base_output_name, ROC_curve_p, time_each_synthetic_exfil,
                              goal_train_test_split, goal_attack_NoAttack_split, training_window_size,
                              size_of_neighbor_training_window, calc_vals, skip_model_part, ignore_physical_attacks_p,
                              calculate_z_scores_p=True,avg_exfil_per_min=None, exfil_per_min_variance=None,
                              avg_pkt_size=None, pkt_size_variance=None):

    # step(0): need to find out the  meta-data for each experiment so we can coordinate the
    # synthetic attack injections between experiments
    exps_exfil_paths, end_of_train_portions, training_exfil_paths, testing_exfil_paths, exps_initiator_info = \
            determine_and_assign_exfil_paths(calc_vals, skip_model_part, function_list, goal_train_test_split,
                                             goal_attack_NoAttack_split, ignore_physical_attacks_p, time_each_synthetic_exfil)


    list_of_optimal_fone_scores_at_exfil_rates = []
    for rate_counter in range(0,len(avg_pkt_size)):
        out_q = multiprocessing.Queue()
        cur_function_list = [copy.deepcopy(i) for i in function_list]
        args = [rate_counter,
                base_output_name, cur_function_list, exps_exfil_paths, exps_initiator_info,
                calculate_z_scores_p, calc_vals, end_of_train_portions,training_exfil_paths,
                testing_exfil_paths, ignore_physical_attacks_p, skip_model_part, out_q,
                ROC_curve_p, avg_exfil_per_min, exfil_per_min_variance, avg_pkt_size, pkt_size_variance]
        p = multiprocessing.Process(
            target=pipeline_one_exfil_rate,
            args=args)
        p.start()
        Xs = out_q.get()
        Ys = out_q.get()
        Xts = out_q.get()
        Yts = out_q.get()
        optimal_fones = out_q.get()
        trained_models = out_q.get()
        p.join()


        list_of_optimal_fone_scores_at_exfil_rates.append(optimal_fones)

    # todo: graph f_one versus exfil rates...
    avg_exfil_size_per_path=  None # TODO
    avg_exfil_pkts_per_path = None # TODO
    graph_fone_versus_exfil_rate(list_of_optimal_fone_scores_at_exfil_rates, avg_exfil_size_per_path,
                                 avg_exfil_pkts_per_path, Xs.keys())


def pipeline_one_exfil_rate(rate_counter,
                            base_output_name, function_list, exps_exfil_paths, exps_initiator_info,
                            calculate_z_scores_p, calc_vals, end_of_train_portions, training_exfil_paths,
                            testing_exfil_paths, ignore_physical_attacks_p, skip_model_part, out_q,
                            ROC_curve_p, avg_exfil_per_min, exfil_per_min_variance, avg_pkt_size, pkt_size_variance):
    ## step (1) : iterate through individual experiments...
    ##  # 1a. list of inputs [done]
    ##  # 1b. acculate DFs
    prefix_for_inject_params = 'avg_exfil_' + str(avg_exfil_per_min[rate_counter]) + ':' + str(exfil_per_min_variance[rate_counter]) +\
        '_avg_pkt_' + str(avg_pkt_size[rate_counter])  + ':' + str(pkt_size_variance[rate_counter]) + '_'
    cur_base_output_name = base_output_name + prefix_for_inject_params
    list_time_gran_to_mod_zscore_df = []
    list_time_gran_to_mod_zscore_df_training = []
    list_time_gran_to_mod_zscore_df_testing = []
    list_time_gran_to_zscore_dataframe = []
    list_time_gran_to_feature_dataframe = []
    starts_of_testing = []

    for counter,experiment_object in enumerate(function_list):
        print "exps_exfil_paths[counter]_to_func",exps_exfil_paths[counter], exps_initiator_info

        experiment_object.alert_file = experiment_object.orig_alert_file + prefix_for_inject_params
        experiment_object.basegraph_name = experiment_object.orig_basegraph_name + prefix_for_inject_params
        experiment_object.exp_name = experiment_object.orig_exp_name + prefix_for_inject_params
        experiment_object.calc_zscore_p = calculate_z_scores_p or calc_vals



        time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, time_gran_to_feature_dataframe, _, start_of_testing = \
        experiment_object.calculate_values(end_of_training=end_of_train_portions[counter],
                                           synthetic_exfil_paths_train=training_exfil_paths[counter],
                                           synthetic_exfil_paths_test=testing_exfil_paths[counter],
                                           avg_exfil_per_min=avg_exfil_per_min[counter],
                                           exfil_per_min_variance=exfil_per_min_variance[counter],
                                           avg_pkt_size=avg_pkt_size[counter], pkt_size_variance=pkt_size_variance[counter])

        print "exps_exfil_pathas[time_gran_to_mod_zscore_df]", time_gran_to_mod_zscore_df
        print time_gran_to_mod_zscore_df[time_gran_to_mod_zscore_df.keys()[0]].columns.values
        list_time_gran_to_mod_zscore_df.append(time_gran_to_mod_zscore_df)
        list_time_gran_to_zscore_dataframe.append(time_gran_to_zscore_dataframe)
        list_time_gran_to_feature_dataframe.append(time_gran_to_feature_dataframe)
        list_time_gran_to_mod_zscore_df_training.append(generate_time_gran_sub_dataframes(time_gran_to_mod_zscore_df, 'is_test', 0))
        list_time_gran_to_mod_zscore_df_testing.append(generate_time_gran_sub_dataframes(time_gran_to_mod_zscore_df, 'is_test', 1))
        starts_of_testing.append(start_of_testing)
        gc.collect()

    # step (2) :  store aggregated DFs for reference purposes
    print "about_to_do_list_time_gran_to_mod_zscore_df"
    time_gran_to_aggregate_mod_score_dfs = aggregate_dfs(list_time_gran_to_mod_zscore_df)
    print "about_to_do_list_time_gran_to_feature_dataframe"
    time_gran_to_aggreg_feature_dfs = aggregate_dfs(list_time_gran_to_feature_dataframe)

    for time_gran, aggregate_feature_df in time_gran_to_aggreg_feature_dfs.iteritems():
        aggregate_feature_df.to_csv(cur_base_output_name + 'aggregate_feature_df_at_time_gran_of_' + str(time_gran) + '_sec.csv',
                                    na_rep='?')
    for time_gran, aggregate_feature_df in time_gran_to_aggregate_mod_score_dfs.iteritems():
        aggregate_feature_df.to_csv(cur_base_output_name + 'modz_feat_df_at_time_gran_of_' + str(time_gran) + '_sec.csv',
                                    na_rep='?')

    recipes_used = [recipe.base_exp_name for recipe in function_list]
    names = []
    for counter,recipe in enumerate(recipes_used):
        name = '_'.join(recipe.split('_')[1:])
        names.append(name)

    path_occurence_training_df = generate_exfil_path_occurence_df(list_time_gran_to_mod_zscore_df_training, names)
    path_occurence_testing_df = generate_exfil_path_occurence_df(list_time_gran_to_mod_zscore_df_testing, names)

    #time_gran_to_aggreg_feature_dfs
    ## okay, so now us the time to get a little tricky with everything... we gotta generate seperate reports for the different
    ## modls used...

    #'''
    '''
    clf = LassoCV(cv=3, max_iter=8000)
    list_of_optimal_fone_scores_at_this_exfil_rates, Xs,Ys,Xts,Yts, trained_models = \
        statistically_analyze_graph_features(time_gran_to_aggreg_feature_dfs, ROC_curve_p,
                                             cur_base_output_name + 'lasso_raw_',
                                             names, starts_of_testing, path_occurence_training_df,
                                             path_occurence_testing_df, recipes_used, skip_model_part, clf,
                                             ignore_physical_attacks_p,
                                             avg_exfil_per_min[rate_counter],
                                             avg_pkt_size[rate_counter],
                                             exfil_per_min_variance[rate_counter],
                                             pkt_size_variance[rate_counter])

    clf = LogisticRegressionCV(penalty="l1", cv=10, max_iter=10000, solver='saga')
    _, _, _, _, _, _ = statistically_analyze_graph_features(time_gran_to_aggreg_feature_dfs, ROC_curve_p,
                                                            cur_base_output_name + 'logistic_l1_raw_lass_feat_sel_',
                                                            names, starts_of_testing, path_occurence_training_df,
                                                            path_occurence_testing_df, recipes_used, skip_model_part, clf,
                                                            ignore_physical_attacks_p, avg_exfil_per_min[rate_counter],
                                                             avg_pkt_size[rate_counter],
                                                             exfil_per_min_variance[rate_counter],
                                                             pkt_size_variance[rate_counter])

    '''
    clf = LassoCV(cv=3, max_iter=80000)
    list_of_optimal_fone_scores_at_this_exfil_rates, Xs,Ys,Xts,Yts, trained_models = \
        statistically_analyze_graph_features(time_gran_to_aggregate_mod_score_dfs, ROC_curve_p,
                                             cur_base_output_name + 'lasso_mod_z_',
                                             names, starts_of_testing, path_occurence_training_df,
                                             path_occurence_testing_df, recipes_used, skip_model_part, clf,
                                             ignore_physical_attacks_p,
                                             avg_exfil_per_min[rate_counter],
                                             avg_pkt_size[rate_counter],
                                             exfil_per_min_variance[rate_counter],
                                             pkt_size_variance[rate_counter])


    '''
    statistically_analyze_graph_features(time_gran_to_aggreg_feature_dfs, ROC_curve_p, base_output_name + 'lasso_raw_',
                                         names, starts_of_testing, path_occurence_training_df,
                                         path_occurence_testing_df, recipes_used, skip_model_part, clf,
                                         ignore_physical_attacks_p)
    #'''
    #''' # appears to be strictly worse than lasso regression...
    # lass_feat_sel
    clf = LogisticRegressionCV(penalty="l1", cv=10, max_iter=10000, solver='saga')
    _, _, _, _, _, _ = statistically_analyze_graph_features(time_gran_to_aggregate_mod_score_dfs, ROC_curve_p,
                                                            cur_base_output_name + 'logistic_l1_mod_z_lass_feat_sel_',
                                                            names, starts_of_testing, path_occurence_training_df,
                                                            path_occurence_testing_df, recipes_used, skip_model_part, clf,
                                                            ignore_physical_attacks_p, avg_exfil_per_min[rate_counter],
                                                             avg_pkt_size[rate_counter],
                                                             exfil_per_min_variance[rate_counter],
                                                             pkt_size_variance[rate_counter])

    #'''
    '''
    clf = LogisticRegressionCV(penalty="l2", cv=3, max_iter=10000)
    statistically_analyze_graph_features(time_gran_to_aggregate_mod_score_dfs, ROC_curve_p,
                                         base_output_name + 'logistic_l2_mod_z_',
                                         names, starts_of_testing, path_occurence_training_df,
                                         path_occurence_testing_df, recipes_used, skip_model_part, clf,
                                         ignore_physical_attacks_p)

    statistically_analyze_graph_features(time_gran_to_aggreg_feature_dfs, ROC_curve_p, base_output_name + 'logistic_l2_raw_',
                                         names, starts_of_testing, path_occurence_training_df,
                                         path_occurence_testing_df, recipes_used, skip_model_part, clf,
                                         ignore_physical_attacks_p)
    #'''
    ''' # if i want to see logistic regression, i would typically use lasso for feature selection, which
    ## is what I do above, b/c the l1 regularization isn't strong enough...
    clf = LogisticRegressionCV(penalty="l1", cv=10, max_iter=10000, solver='saga')
    statistically_analyze_graph_features(time_gran_to_aggregate_mod_score_dfs, ROC_curve_p,
                                         cur_base_output_name + 'logistic_l1_mod_z_',
                                         names, starts_of_testing, path_occurence_training_df,
                                         path_occurence_testing_df, recipes_used, skip_model_part, clf,
                                         ignore_physical_attacks_p, fraction_of_edge_weights[rate_counter],
                                         fraction_of_edge_pkts[rate_counter])
    '''
    '''
    statistically_analyze_graph_features(time_gran_to_aggreg_feature_dfs, ROC_curve_p,
                                         cur_base_output_name + 'logistic_l1_raw_',
                                         names, starts_of_testing, path_occurence_training_df,
                                         path_occurence_testing_df, recipes_used, skip_model_part, clf,
                                         ignore_physical_attacks_p, fraction_of_edge_weights[rate_counter],
                                         fraction_of_edge_pkts[rate_counter])
    '''

    out_q.put(Xs)
    out_q.put(Ys)
    out_q.put(Xts)
    out_q.put(Yts)
    out_q.put(list_of_optimal_fone_scores_at_this_exfil_rates)
    out_q.put(trained_models)

def determine_and_assign_exfil_paths(calc_vals, skip_model_part, function_list, goal_train_test_split, goal_attack_NoAttack_split,
                                     ignore_physical_attacks_p, time_each_synthetic_exfil):
    if calc_vals and not skip_model_part:
        print function_list
        exp_infos = []
        for experiment_object in function_list:
            print "calc_vals", calc_vals
            total_experiment_length, exfil_start_time, exfil_end_time, system_startup_time = \
                experiment_object.get_exp_info()
            print "func_exp_info", total_experiment_length, exfil_start_time, exfil_end_time
            exp_infos.append({"total_experiment_length":total_experiment_length, "exfil_start_time":exfil_start_time,
                             "exfil_end_time":exfil_end_time, "startup_time": system_startup_time})

        ## get the exfil_paths that were generated using the mulval component...
        ## this'll require passing a parameter to the single-experiment pipeline and then getting the set of paths
        exps_exfil_paths = []
        exps_initiator_info = []
        total_training_injections_possible, total_testing_injections_possible, _, _ = \
            determine_injection_amnts(exp_infos, goal_train_test_split, goal_attack_NoAttack_split,
                                      ignore_physical_attacks_p,
                                      time_each_synthetic_exfil, float("inf"))
        max_number_of_paths = min(total_training_injections_possible, total_testing_injections_possible)
        orig_max_number_of_paths=  max_number_of_paths
        for experiment_object in function_list:
            print "experiment_object", experiment_object
            synthetic_exfil_paths, initiator_info_for_paths = \
                experiment_object.generate_synthetic_exfil_paths(max_number_of_paths=max_number_of_paths)
            max_number_of_paths = None
            exps_exfil_paths.append(synthetic_exfil_paths)
            exps_initiator_info.append(initiator_info_for_paths)

        print "orig_max_number_of_paths", orig_max_number_of_paths
        #print exps_exfil_paths
        for counter,exp_path in enumerate(exps_exfil_paths[0]):
            print counter,exp_path,len(exp_path)
        #exit(344)
        training_exfil_paths, testing_exfil_paths, end_of_train_portions = assign_exfil_paths_to_experiments(exp_infos, goal_train_test_split,
                                                                                      goal_attack_NoAttack_split,time_each_synthetic_exfil,
                                                                                      exps_exfil_paths, ignore_physical_attacks_p)
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
    time_gran_to_aggregate_mod_score_dfs_training = {}
    time_gran_to_aggregate_mod_score_dfs_testing = {}
    print "list_time_gran_to_mod_zscore_df",list_time_gran_to_mod_zscore_df
    for time_gran_to_mod_zscore_df in list_time_gran_to_mod_zscore_df:
        print "time_gran_to_mod_zscore_df",time_gran_to_mod_zscore_df
        for time_gran, mod_zscore_df in time_gran_to_mod_zscore_df.iteritems():
            if time_gran not in time_gran_to_aggregate_mod_score_dfs.keys():
                time_gran_to_aggregate_mod_score_dfs[time_gran] = mod_zscore_df
                '''
                time_gran_to_aggregate_mod_score_dfs_training[time_gran] = mod_zscore_df[mod_zscore_df['is_exfil'] == 0]
                time_gran_to_aggregate_mod_score_dfs_testing[time_gran] =  mod_zscore_df[mod_zscore_df['is_exfil'] == 1]
                '''
                print "post_initializing_aggregate_dataframe", len(time_gran_to_aggregate_mod_score_dfs[time_gran]), \
                    type(time_gran_to_aggregate_mod_score_dfs[time_gran]), time_gran

            else:
                time_gran_to_aggregate_mod_score_dfs[time_gran] = \
                    time_gran_to_aggregate_mod_score_dfs[time_gran].append(mod_zscore_df, sort=True)
                '''
                time_gran_to_aggregate_mod_score_dfs_training[time_gran] = \
                    time_gran_to_aggregate_mod_score_dfs_training[time_gran].append(mod_zscore_df[mod_zscore_df['is_exfil'] == 0], sort=True)
                time_gran_to_aggregate_mod_score_dfs_testing[time_gran] = \
                    time_gran_to_aggregate_mod_score_dfs_training[time_gran].append(mod_zscore_df[mod_zscore_df['is_exfil'] == 1], sort=True)
                '''
                print "should_be_appending_mod_z_scores", len(time_gran_to_aggregate_mod_score_dfs[time_gran]), \
                    type(time_gran_to_aggregate_mod_score_dfs[time_gran]), time_gran
    return time_gran_to_aggregate_mod_score_dfs#, time_gran_to_aggregate_mod_score_dfs_training, time_gran_to_aggregate_mod_score_dfs_testing

# this function determines which experiments should have which synthetic exfil paths injected into them
def assign_exfil_paths_to_experiments(exp_infos, goal_train_test_split, goal_attack_NoAttack_split,time_each_synthetic_exfil,
                                      exps_exfil_paths, ignore_physical_attacks_p):

    flat_exps_exfil_paths = [tuple(exfil_path) for exp_exfil_paths in exps_exfil_paths for exfil_path in exp_exfil_paths]
    print "flat_exps_exfil_paths",flat_exps_exfil_paths
    possible_exfil_paths = list(set(flat_exps_exfil_paths))

    total_training_injections_possible,total_testing_injections_possible,possible_exfil_path_injections,end_of_train_portions = \
        determine_injection_amnts(exp_infos, goal_train_test_split, goal_attack_NoAttack_split,
                                  ignore_physical_attacks_p,
                                  time_each_synthetic_exfil, possible_exfil_paths)

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
    testing_number_times_inject_all_paths = math.floor(total_testing_injections_possible / float(len(possible_exfil_paths)))
    training_number_times_inject_all_paths = math.floor(total_training_injections_possible / float(len(possible_exfil_paths)))
    if training_number_times_inject_all_paths < 1.0:
        print "can't inject all exfil paths in training set... "
        exit(33)
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
        ## let's NOT do this stochastically... let's just iterate through the dict and assign stuff whenever we can
        ## (NOTE: this WILL NEED TO BE MODIFIED LATER...)
        current_training_exfil_paths = []
        training_times_to_inject_this_exp = possible_exfil_path_injection['training']
        print "(initial)training_times_to_inject_this_exp",training_times_to_inject_this_exp
        while training_times_to_inject_this_exp > 0:
            path = max(exfil_paths_to_train_injection_counts.iteritems(), key=operator.itemgetter(1))[0]
            print "current_max_path", path
            if exfil_paths_to_train_injection_counts[path] > 0:
                current_training_exfil_paths.append(list(path))
                training_times_to_inject_this_exp -= 1
                exfil_paths_to_train_injection_counts[path] -= 1
            else:
                # note: this isn't actually a problem b/c we rounded down when assigning the # of injection counts for each path
                break
                #print "problem w/ exfil assignment! (training)"
                #print path, exfil_paths_to_train_injection_counts[path]
                #print exfil_paths_to_train_injection_counts
                #print training_times_to_inject_this_exp
                #exit(433)
        training_exfil_paths.append(current_training_exfil_paths)

        current_testing_exfil_paths = []
        testing_times_to_inject_this_exp = possible_exfil_path_injection['testing']
        while testing_times_to_inject_this_exp > 0:
            path = max(exfil_paths_to_test_injection_counts.iteritems(), key=operator.itemgetter(1))[0]
            print "current_max_testing_path", path
            if exfil_paths_to_test_injection_counts[path] > 0:
                current_testing_exfil_paths.append(list(path))
                testing_times_to_inject_this_exp -= 1
                exfil_paths_to_test_injection_counts[path] -= 1
            else:
                # note: this isn't actually a problem b/c we rounded down when assigning the # of injection counts for each path
                break
                #print "problem w/ exfil assignment (testing)!"
                #print path, exfil_paths_to_test_injection_counts[path]
                #print exfil_paths_to_test_injection_counts
                #exit(434)
        testing_exfil_paths.append(current_testing_exfil_paths)

    print "training_exfil_paths", training_exfil_paths
    print "testing_exfil_paths", testing_exfil_paths

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

def determine_injection_amnts(exp_infos, goal_train_test_split, goal_attack_NoAttack_split, ignore_physical_attacks_p,
                              time_each_synthetic_exfil, possible_exfil_paths):
    ## now perform the actual assignment portion...
    # first, find the amt of time available for attack injections in each experiments training/testing phase...
    inject_times,end_of_train_portions = determine_injection_times(exp_infos, goal_train_test_split,
                                                                   goal_attack_NoAttack_split, ignore_physical_attacks_p)
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
    print "possible_exfil_path_injections", possible_exfil_path_injections
    #exit(34)
    return total_training_injections_possible,total_testing_injections_possible,possible_exfil_path_injections,end_of_train_portions

# generates a df indicating how long each logical exfil path occurs during each experiment, and returns a handle DF
# for use in the generated report.
def generate_exfil_path_occurence_df(list_of_time_gran_to_mod_zscore_df, experiment_names):
    experiments_to_exfil_path_time_dicts = []
    for time_gran_to_mod_zscore_df in list_of_time_gran_to_mod_zscore_df:
        print time_gran_to_mod_zscore_df.keys()
        min_time_gran = min(time_gran_to_mod_zscore_df.keys())
        print time_gran_to_mod_zscore_df[min_time_gran]
        # I *hope* this solves the list is unhashable problem....
        time_gran_to_mod_zscore_df[min_time_gran]['exfil_path'] = \
            [tuple(i) if type(i) == list  else i for i in time_gran_to_mod_zscore_df[min_time_gran]['exfil_path']]
        print time_gran_to_mod_zscore_df[min_time_gran]['exfil_path']
        print time_gran_to_mod_zscore_df[min_time_gran]['exfil_path']
        print time_gran_to_mod_zscore_df[min_time_gran]['exfil_path'].values
        print time_gran_to_mod_zscore_df[min_time_gran]['exfil_path'].value_counts()
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

            print "time_gran_to_exfil_param_list", time_gran_to_exfil_param_list
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
    print "INITIAL time_gran_to_attack_labels", time_gran_to_attack_labels
    synthetic_exfil_paths = [['a', 'b'], ['b', 'c']]
    time_of_synethic_exfil = 2
    startup_time_before_injection = 4
    time_gran_to_attack_labels, time_gran_to_attack_ranges, time_gran_to_physical_attack_ranges = \
        determine_attacks_to_times(time_gran_to_attack_labels, synthetic_exfil_paths,
                                   time_of_synethic_exfil, startup_time_before_injection)
    print "time_gran_to_attack_labels", time_gran_to_attack_labels
    print "time_gran_to_attack_ranges", time_gran_to_attack_ranges
    print "time_gran_to_physical_attack_ranges", time_gran_to_physical_attack_ranges

