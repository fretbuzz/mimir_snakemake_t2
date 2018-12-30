import gc
import pyximport
import analysis_pipeline.generate_alerts
import analysis_pipeline.generate_graphs
import analysis_pipeline.prepare_graph
from pcap_to_edgelists import create_mappings
import analysis_pipeline.src.analyze_edgefiles
import process_graph_metrics
import generate_alerts
pyximport.install() # to leverage cpython
import simplified_graph_metrics
import process_pcap
import gen_attack_templates
import random
import math
import pandas as pd
import time
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LassoCV, Lasso
import sklearn
import logging
from sklearn.impute import SimpleImputer, MissingIndicator
import numpy as np
import matplotlib.pyplot as plt
import generate_report
import os
import errno

def calculate_raw_graph_metrics(time_interval_lengths, interval_to_filenames, ms_s, basegraph_name, calc_vals, window_size,
                                mapping, is_swarm, make_net_graphs_p, list_of_infra_services,synthetic_exfil_paths,
                                initiator_info_for_paths, time_gran_to_attacks_to_times, fraction_of_edge_weights,
                                fraction_of_edge_pkts, size_of_neighbor_training_window):
    total_calculated_vals = {}
    for time_interval_length in time_interval_lengths:
        print "analyzing edgefiles...", "timer_interval...", time_interval_length

        #newly_calculated_values = simplified_graph_metrics.pipeline_subset_analysis_step(interval_to_filenames[str(time_interval_length)], ms_s,
        #                                                                                 time_interval_length, basegraph_name, calc_vals, window_size,
        #                                                                                 mapping, is_swarm, make_net_graphs_p, list_of_infra_services,
        #                                                                                 synthetic_exfil_paths, initiator_info_for_paths,
        #                                                                                 time_gran_to_attacks_to_times[time_interval_length],
        #                                                                                 fraction_of_edge_weights,
        #                                                                                 fraction_of_edge_pkts)

        if is_swarm:
            svcs = analysis_pipeline.prepare_graph.get_svc_equivalents(is_swarm, mapping)
        else:
            print "this is k8s, so using these sevices", ms_s
            svcs = ms_s

        total_calculated_vals[(time_interval_length, '')] = \
            simplified_graph_metrics.calc_subset_graph_metrics(interval_to_filenames[str(time_interval_length)],
                                                               time_interval_length, basegraph_name + '_subset_',
                                                               calc_vals, window_size, ms_s, mapping, is_swarm, svcs,
                                                               list_of_infra_services, synthetic_exfil_paths,
                                                               initiator_info_for_paths,
                                                               time_gran_to_attacks_to_times[time_interval_length],
                                                               fraction_of_edge_weights, fraction_of_edge_pkts,
                                                               int(size_of_neighbor_training_window/time_interval_length))

        #total_calculated_vals.update(newly_calculated_values)
        gc.collect()
    #exit() ### TODO <---- remove!!!
    return total_calculated_vals

def calc_zscores(alert_file, training_window_size, minimum_training_window,
                 sub_path, time_gran_to_attack_labels, time_gran_to_feature_dataframe, calc_zscore_p):

    #time_gran_to_mod_zscore_df = process_graph_metrics.calculate_mod_zscores_dfs(total_calculated_vals, minimum_training_window,
    #                                                                             training_window_size, time_interval_lengths)

    mod_z_score_df_basefile_name = alert_file + 'mod_z_score_' + sub_path
    z_score_df_basefile_name = alert_file + 'norm_z_score_' + sub_path

    if calc_zscore_p:
        time_gran_to_mod_zscore_df = process_graph_metrics.calc_time_gran_to_mod_zscore_dfs(time_gran_to_feature_dataframe,
                                                                                            training_window_size,
                                                                                            minimum_training_window)

        process_graph_metrics.save_feature_datafames(time_gran_to_mod_zscore_df, mod_z_score_df_basefile_name,
                                                     time_gran_to_attack_labels)

        time_gran_to_zscore_dataframe = process_graph_metrics.calc_time_gran_to_zscore_dfs(time_gran_to_feature_dataframe,
                                                                                           training_window_size,
                                                                                           minimum_training_window)

        process_graph_metrics.save_feature_datafames(time_gran_to_zscore_dataframe, z_score_df_basefile_name,
                                                     time_gran_to_attack_labels)
    else:
        time_gran_to_zscore_dataframe = {}
        time_gran_to_mod_zscore_df = {}
        for interval in time_gran_to_feature_dataframe.keys():
            time_gran_to_zscore_dataframe[interval] = pd.read_csv(z_score_df_basefile_name + str(interval) + '.csv', na_values='?')
            time_gran_to_mod_zscore_df[interval] = pd.read_csv(mod_z_score_df_basefile_name + str(interval) + '.csv', na_values='?')

    return time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe

def generate_rocs(time_gran_to_anom_score_df, alert_file, sub_path):
    for time_gran, df_with_anom_features in time_gran_to_anom_score_df.iteritems():
        cur_alert_function,features_to_use = generate_alerts.determine_alert_function(df_with_anom_features)
        generate_alerts.generate_all_anom_ROCs(df_with_anom_features, time_gran, alert_file, sub_path, cur_alert_function,
                               features_to_use)

# returns whether the range does not already have an attack at that location... so if an attack is found
# then the range is not valid (So you'd wanna return false)
def exfil_time_valid(potential_starting_point, time_slots_attack, attack_labels):
    attack_found = False
    # now check if there's not already an attack selected for that time...
    #print potential_starting_point, potential_starting_point + time_slots_attack
    for i in attack_labels[potential_starting_point:int(potential_starting_point + time_slots_attack)]:
        if i:  # ==1
            attack_found = True
            break
    return not attack_found

##### the goal needs to be some mapping of times to attacks to time (ranges) + updated attack labels
##### so, in effect, there are TWO outputs... and it makes a lot more sense to pick the range then modify
##### the labels
def determine_attacks_to_times(time_gran_to_attack_labels, synthetic_exfil_paths, time_of_synethic_exfil, min_starting):
    time_grans = time_gran_to_attack_labels.keys()
    largest_time_gran = sorted(time_grans)[-1]
    print "LARGEST_TIME_GRAN", largest_time_gran
    time_periods_attack = float(time_of_synethic_exfil) / float(largest_time_gran)
    time_periods_startup = math.ceil(float(min_starting) / float(largest_time_gran))
    time_gran_to_attack_ranges = {} # a list that'll correspond w/ the synthetic exfil paths
    for time_gran in time_gran_to_attack_labels.keys():
        time_gran_to_attack_ranges[time_gran] = []

    for synthetic_exfil_path in synthetic_exfil_paths:
        # randomly choose ranges using highest granularity (then after this we'll choose for the smaller granularities...)
        attack_spot_found = False
        number_free_spots = time_gran_to_attack_labels[largest_time_gran][int(time_periods_startup):].count(0)
        if number_free_spots < time_periods_attack:
            exit(1244) # should break now b/c infinite loop (note: we're not handling the case where it is fragmented...)
        counter = 0
        while not attack_spot_found:
            ## NOTE: not sure if the -1 is necessary...
            # NOTE: this random thing causes all types of problems. Let's just ignore it and do it right after startup??, maybe?
            #potential_starting_point = random.randint(time_periods_startup,
            #                                len(time_gran_to_attack_labels[largest_time_gran]) - time_periods_attack - 1)
            potential_starting_point = int(time_periods_startup + counter)

            print "potential_starting_point", potential_starting_point
            attack_spot_found = exfil_time_valid(potential_starting_point, time_periods_attack,
                                                 time_gran_to_attack_labels[largest_time_gran])
            if attack_spot_found:
                # if the time range is valid, we gotta store it...
                time_gran_to_attack_ranges[largest_time_gran].append((int(potential_starting_point),
                                                                      int(potential_starting_point + time_periods_attack)))
                # and also modify the attack labels
                print "RANGE", potential_starting_point, int(potential_starting_point + time_periods_attack)
                for i in range(potential_starting_point, int(potential_starting_point + time_periods_attack)):
                    #print i, time_gran_to_attack_labels[largest_time_gran]
                    time_gran_to_attack_labels[largest_time_gran][i] = 1
            #print "this starting point failed", potential_starting_point
            counter += 1

    # okay, so now we have the times selected for the largest time granularity... we have to make sure
    # that the other granularities agree...

    print "HIGHEST GRAN SYNTHETIC ATTACKS CHOSEN -- START MAPPING TO LOWER GRAN NOW!"
    for j in range(0, len(synthetic_exfil_paths)):
        for time_gran, attack_labels in time_gran_to_attack_labels.iteritems():
            if time_gran == largest_time_gran:
                continue
            attack_ranges_at_highest_gran = time_gran_to_attack_ranges[largest_time_gran]
            current_attack_range_at_highest_gran = attack_ranges_at_highest_gran[j]
            time_period_conversion_ratio = float(largest_time_gran) / float(time_gran)
            #print "TIME_PERIOD_CONVERSION_RATIO", time_period_conversion_ratio,  float(largest_time_gran), float(time_gran)
            current_start_of_attack = int(current_attack_range_at_highest_gran[0] * time_period_conversion_ratio)
            current_end_of_attack = int(current_attack_range_at_highest_gran[1] * time_period_conversion_ratio)
            time_gran_to_attack_ranges[time_gran].append( (current_start_of_attack, current_end_of_attack) )
            # also, modify the attack_labels
            for z in range(current_start_of_attack, current_end_of_attack):
                #print "z",z
                attack_labels[z] = 1
    return time_gran_to_attack_labels, time_gran_to_attack_ranges

## TODO: this function is an atrocity and should be converted into a snakemake spec so we can use that instead...###
## todo (aim to get it done today...) : change  run_data_analysis_pipeline signature plus the feeder...

# run_data_anaylsis_pipeline : runs the whole analysis_pipeline pipeline (or a part of it)
# (1) creates edgefiles, (2) creates communication graphs from edgefiles, (3) calculates (and stores) graph metrics
# (4) makes graphs of the graph metrics
# Note: see run_analysis_pipeline_recipes for pre-configured sets of parameters (there are rather a lot)
def run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths, ms_s,
                               make_edgefiles_p, basegraph_name, window_size, colors, exfil_start_time, exfil_end_time,
                               wiggle_room, start_time=None, end_time=None, calc_vals=True, graph_p=True,
                               kubernetes_svc_info=None, make_net_graphs_p=False, cilium_config_path=None,
                               rdpcap_p=False, kubernetes_pod_info=None, alert_file=None, ROC_curve_p=False,
                               calc_zscore_p=False, training_window_size=200, minimum_training_window=5,
                               sec_between_exfil_events=1, time_of_synethic_exfil=60,
                               fraction_of_edge_weights=0.1, fraction_of_edge_pkts=0.1,
                               size_of_neighbor_training_window=300):

    print "log file can be found at: " + str(basefile_name) + '_logfile.log'
    logging.basicConfig(filename=basefile_name + '_logfile.log', level=logging.INFO)
    logging.info('run_data_anaylsis_pipeline Started')

    gc.collect()

    print "starting pipeline..."

    sub_path = 'sub_'  # NOTE: make this an empty string if using the full pipeline (and not the subset)
    mapping,list_of_infra_services = create_mappings(is_swarm, container_info_path, kubernetes_svc_info,
                                                     kubernetes_pod_info, cilium_config_path, ms_s)

    experiment_folder_path = basefile_name.split('edgefiles')[0]
    pcap_file = pcap_paths[0].split('/')[-1] # NOTE: assuming only a single pcap file...
    exp_name = basefile_name.split('/')[-1]
    interval_to_filenames = process_pcap.process_pcap(experiment_folder_path, pcap_file, time_interval_lengths,
                                                      exp_name, make_edgefiles_p, mapping)

    if calc_vals or graph_p:
        # TODO: 90% sure that there is a problem with this function...
        largest_interval = int(max(interval_to_filenames.keys()))
        exp_length = len(interval_to_filenames[str(largest_interval)]) * largest_interval
        print "exp_length_ZZZ", exp_length, type(exp_length)
        time_gran_to_attack_labels = process_graph_metrics.generate_time_gran_to_attack_labels(time_interval_lengths,
                                                                                               exfil_start_time, exfil_end_time,
                                                                                                sec_between_exfil_events,
                                                                                               exp_length)

        #print "interval_to_filenames_ZZZ",interval_to_filenames
        for interval, filenames in interval_to_filenames.iteritems():
            print "interval_ZZZ", interval, len(filenames)
        for time_gran, attack_labels in time_gran_to_attack_labels.iteritems():
            print "time_gran_right_after_creation", time_gran, "len of attack labels", len(attack_labels)

        print interval_to_filenames, type(interval_to_filenames), 'stufff', interval_to_filenames.keys()


        # todo: might wanna specify this is in the attack descriptions...
        for ms in ms_s:
            if 'User' in ms:
                sensitive_ms = ms
            if 'my-release' in ms:
                sensitive_ms = ms
        synthetic_exfil_paths, initiator_info_for_paths = gen_attack_templates.generate_synthetic_attack_templates(mapping, ms_s, sensitive_ms)


        # most of the parameters are kinda arbitrary ATM...
        print "INITIAL time_gran_to_attack_labels", time_gran_to_attack_labels
        ## okay, I'll probably wanna write tests for the below function, but it seems to be working pretty well on my
        # informal tests...
        time_gran_to_attack_labels, time_gran_to_attack_ranges = determine_attacks_to_times(time_gran_to_attack_labels,
                                                                                            synthetic_exfil_paths,
                                                                                            time_of_synethic_exfil=time_of_synethic_exfil,
                                                                                            min_starting=training_window_size+size_of_neighbor_training_window)
        print "time_gran_to_attack_labels",time_gran_to_attack_labels
        print "time_gran_to_attack_ranges", time_gran_to_attack_ranges
        #time.sleep(50)

        # OKAY, let's verify that this determine_attacks_to_times function is wokring before moving on to the next one...
        total_calculated_vals = calculate_raw_graph_metrics(time_interval_lengths, interval_to_filenames, ms_s, basegraph_name, calc_vals,
                                                            window_size, mapping, is_swarm, make_net_graphs_p, list_of_infra_services,
                                                            synthetic_exfil_paths, initiator_info_for_paths, time_gran_to_attack_ranges,
                                                            fraction_of_edge_weights, fraction_of_edge_pkts, size_of_neighbor_training_window)

        time_gran_to_feature_dataframe = process_graph_metrics.generate_feature_dfs( total_calculated_vals, time_interval_lengths)

        process_graph_metrics.save_feature_datafames(time_gran_to_feature_dataframe, alert_file + sub_path, time_gran_to_attack_labels)

        analysis_pipeline.generate_graphs.generate_feature_multitime_boxplots(total_calculated_vals, basegraph_name,
                                                                              window_size, colors, time_interval_lengths,
                                                                              exfil_start_time, exfil_end_time, wiggle_room)

    else:
        time_gran_to_feature_dataframe = {}
        time_gran_to_attack_labels = {}
        for interval in interval_to_filenames.keys():
            time_gran_to_feature_dataframe[interval] = pd.read_csv(alert_file + sub_path + str(interval) + '.csv', na_values='?')
            time_gran_to_attack_labels[interval] = time_gran_to_feature_dataframe[interval]['labels']

    print "about to calculate some alerts!"

    time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe = \
        calc_zscores(alert_file, training_window_size, minimum_training_window, sub_path, time_gran_to_attack_labels,
                     time_gran_to_feature_dataframe, calc_zscore_p)

    print "analysis_pipeline about to return!"

    # okay, so can return it here...
    return time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, time_gran_to_feature_dataframe

# this function loops through multiple experiments (or even just a single experiment), accumulates the relevant
# feature dataframes, and then performs LASSO regression to determine a concise graphical model that can detect
# the injected synthetic attacks
def multi_experiment_pipeline(function_list, base_output_name, ROC_curve_p):
    ### Okay, so what is needed here??? We need, like, a list of sets of input (appropriate for run_data_analysis_pipeline),
    ### followed by the LASSO stuff, and finally the ROC stuff... okay, let's do this!!!

    ## step (1) : iterate through individual experiments...
    ##  # 1a. list of inputs [done]
    ##  # 1b. acculate DFs
    list_time_gran_to_mod_zscore_df = []
    list_time_gran_to_zscore_dataframe = []
    list_time_gran_to_feature_dataframe = []
    for func in function_list:
        time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, time_gran_to_feature_dataframe = func()
        list_time_gran_to_mod_zscore_df.append(time_gran_to_mod_zscore_df)
        list_time_gran_to_zscore_dataframe.append(time_gran_to_zscore_dataframe)
        list_time_gran_to_feature_dataframe.append(time_gran_to_feature_dataframe)

    # step (2) :  take the dataframes and feed them into the LASSO component...
    ### 2a. split into training and testing data
    ###### at the moment, we'll make the simplfying assumption to only use the modified z-scores...
    ######## 2a.I. get aggregate dfs for each time granularity...
    time_gran_to_aggregate_mod_score_dfs = {}
    for time_gran_to_mod_zscore_df in list_time_gran_to_mod_zscore_df:
        for time_gran, mod_zscore_df in time_gran_to_mod_zscore_df.iteritems():
            if time_gran not in time_gran_to_aggregate_mod_score_dfs.keys():
                time_gran_to_aggregate_mod_score_dfs[time_gran] = mod_zscore_df
                print "post_initializing_aggregate_dataframe", len(time_gran_to_aggregate_mod_score_dfs[time_gran]), \
                    type(time_gran_to_aggregate_mod_score_dfs[time_gran]), time_gran

            else:
                time_gran_to_aggregate_mod_score_dfs[time_gran] = \
                    time_gran_to_aggregate_mod_score_dfs[time_gran].append(mod_zscore_df, sort=True)
                print "should_be_appending_mod_z_scores", len(time_gran_to_aggregate_mod_score_dfs[time_gran]), \
                    type(time_gran_to_aggregate_mod_score_dfs[time_gran]), time_gran


    #print time_gran_to_aggregate_mod_score_dfs['60']

    ######### 2a.II. do the actual splitting
    # note: labels have the column name 'labels' (noice)
    time_gran_to_model = {}
    #images = 0
    list_of_rocs = []
    list_of_feat_coefs_dfs = []
    time_grans = []
    for time_gran,aggregate_mod_score_dfs in time_gran_to_aggregate_mod_score_dfs.iteritems():
        time_grans.append(time_gran)
        #try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='timemod_z_score')   # might wanna just stop these from being generated...
        #except:
        #    pass
        #try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='labelsmod_z_score') # might wanna just stop these from being generaetd
        print aggregate_mod_score_dfs.columns
        #except:
        #    pass
        X = aggregate_mod_score_dfs.loc[:, aggregate_mod_score_dfs.columns != 'labels']
        y = aggregate_mod_score_dfs.loc[:, aggregate_mod_score_dfs.columns == 'labels']
        print X.shape, "X.shape"
        X_train, X_test, y_train, y_test =  sklearn.model_selection.train_test_split(X, y, test_size = 0.3, random_state = 42)
        print X_train.shape, "X_train.shape"

        ### 2b. feed the lasso to get the high-impact features...
        ## NOTE: THIS IS ALL VERY CONFUSING. I'M GOING TO WORRY ABOUT DOING ANYTHING FANCY LATER ON JUST MAKE IT SIMPLE NOW
        ###### okay, this is the current step ^^ feed the LASSO
        ######## 2b.I. Use LassoCV to find the alpha value
        #cv_outer = KFold(len(X))
        #lasso = LassoCV(cv=3)  # cv=3 makes a KFold inner splitting with 3 folds
        #reg = LassoCV(cv=3, random_state=42).fit(X_train, y_train)
        #alpha = reg.alpha_

        ######## 2b.II. then use Lasso to find the fit and parameters
        #scores = cross_validate(lasso, X, y, return_estimator=True, cv=cv_outer)
        clf = Lasso(alpha=20) ## Okay, this value was just chosen by me somewhat randomly (knew that I wanted it v strong)

        # need to replace the missing values in the data w/ meaningful values...
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        imp = imp.fit(X_train)
        X_train = imp.transform(X_train)
        X_test = imp.transform(X_test)

        #print "X_train", X_train
        #print "y_train", y_train, len(y_train)
        #print "y_test", y_test, len(y_test)
        #print "-- y_train", len(y_train), "y_test", len(y_test), "time_gran", time_gran, "--"
        clf.fit(X_train, y_train)
        score_val = clf.score(X_test, y_test)
        print "score_val", score_val
        test_predictions = clf.predict(X=X_test)
        print "LASSO model", clf.get_params()
        print '----------------------'
        print "Coefficients: "
        #coefficients = pd.DataFrame({"Feature": X.columns, "Coefficients": np.transpose(clf.coef_)})
        #print coefficients
        #clf.coef_, "intercept", clf.intercept_
        coef_dict = {}
        for coef, feat in zip(clf.coef_, list(X.columns.values)):
            coef_dict[feat] = coef
        coef_dict['intercept'] = clf.intercept_[0]
        for coef,feature in coef_dict.iteritems():
            print coef,feature

        #print "COEF_DICT", coef_dict
        coef_feature_df = pd.DataFrame.from_dict(coef_dict, orient='index')
        #print coef_feature_df.columns.values
        #coef_feature_df.index.name = 'Features'
        coef_feature_df.columns = ['Coefficient']

        list_of_feat_coefs_dfs.append(coef_feature_df)
        print '--------------------------'


        '''
        print X_train
        coefficients = clf.coef_
        print len(coefficients), "<- len coefficients,", len(list(X.columns.values))
        for counter,column in enumerate(list(X.columns.values)):
            print counter
            print column, coefficients[counter]
        print "score_val", score_val
        time_gran_to_model[time_gran] = clf
        ##print "time_gran", time_gran, "scores", scores
        '''

        ### step (3)
        ## use the generate sklearn model to create the detection ROC
        if ROC_curve_p:
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=y_test, y_score=test_predictions, pos_label=1)
            x_vals = fpr
            y_vals = tpr
            ROC_path = base_output_name + '_good_roc_'
            title = 'ROC Linear Combination of Features at ' + str(time_gran)
            plot_name = 'sub_roc_lin_comb_features_' + str(time_gran)

            try:
                os.makedirs('./temp_outputs')
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            ax, _, plot_path = generate_alerts.construct_ROC_curve(x_vals, y_vals, title, ROC_path + plot_name, show_p=False)
            list_of_rocs.append(plot_path)

    recipes_used = [recipe.__name__ for recipe in function_list]
    for recipe in function_list:
        print "recipe_in_functon_list", recipe.__name__
    list_of_attacks_found_dfs = []  ## TODO::  <<<<------- THIS IS GOING TO BE NEEDED FOR THE REPORTS TO BE COMPRHENSIBLE!!
    list_of_attacks_found_dfs.append( pd.DataFrame([1,2]) )
    list_of_attacks_found_dfs.append( pd.DataFrame([4,5]) )
    list_of_attacks_found_dfs.append( pd.DataFrame([8,9]) )

    generate_report.generate_report(list_of_rocs, list_of_feat_coefs_dfs, list_of_attacks_found_dfs,
                                    recipes_used, base_output_name, time_grans)

    print "multi_experiment_pipeline is all done! (NO ERROR DURING RUNNING)"
    #print "recall that this was the list of alert percentiles", percentile_thresholds

if __name__ == "__main__":
    time_gran_to_attack_labels = {1: [0, 0, 1, 1, 0, 0, 0, 0, 0, 0], 2: [0, 1, 0, 0, 0]}
    print "INITIAL time_gran_to_attack_labels", time_gran_to_attack_labels
    synthetic_exfil_paths = [['a', 'b'], ['b', 'c']]
    time_of_synethic_exfil = 2
    startup_time_before_injection = 4
    time_gran_to_attack_labels, time_gran_to_attack_ranges = \
        determine_attacks_to_times(time_gran_to_attack_labels, synthetic_exfil_paths,
                                  time_of_synethic_exfil, startup_time_before_injection)
    print "time_gran_to_attack_labels", time_gran_to_attack_labels
    print "time_gran_to_attack_ranges", time_gran_to_attack_ranges