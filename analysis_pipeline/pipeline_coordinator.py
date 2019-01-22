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
import process_roc
import ast
from itertools import groupby
from operator import itemgetter
import operator
import copy
import multiprocessing

def calculate_raw_graph_metrics(time_interval_lengths, interval_to_filenames, ms_s, basegraph_name, calc_vals, window_size,
                                mapping, is_swarm, make_net_graphs_p, list_of_infra_services,synthetic_exfil_paths,
                                initiator_info_for_paths, time_gran_to_attacks_to_times, fraction_of_edge_weights,
                                fraction_of_edge_pkts, size_of_neighbor_training_window):
    total_calculated_vals = {}
    time_gran_to_list_of_concrete_exfil_paths = {}
    time_gran_to_list_of_exfil_amts = {}
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

        total_calculated_vals[(time_interval_length, '')], list_of_concrete_container_exfil_paths, list_of_exfil_amts = \
            simplified_graph_metrics.calc_subset_graph_metrics(interval_to_filenames[str(time_interval_length)],
                                                               time_interval_length, basegraph_name + '_subset_',
                                                               calc_vals, window_size, ms_s, mapping, is_swarm, svcs,
                                                               list_of_infra_services, synthetic_exfil_paths,
                                                               initiator_info_for_paths,
                                                               time_gran_to_attacks_to_times[time_interval_length],
                                                               fraction_of_edge_weights, fraction_of_edge_pkts,
                                                               int(size_of_neighbor_training_window/time_interval_length))
        time_gran_to_list_of_concrete_exfil_paths[time_interval_length] = list_of_concrete_container_exfil_paths
        time_gran_to_list_of_exfil_amts[time_interval_length] = list_of_exfil_amts

        #total_calculated_vals.update(newly_calculated_values)
        gc.collect()
    #exit() ### TODO <---- remove!!!
    return total_calculated_vals, time_gran_to_list_of_concrete_exfil_paths, time_gran_to_list_of_exfil_amts

def calc_zscores(alert_file, training_window_size, minimum_training_window,
                 sub_path, time_gran_to_attack_labels, time_gran_to_feature_dataframe, calc_zscore_p, time_gran_to_synthetic_exfil_paths_series,
                 time_gran_to_list_of_concrete_exfil_paths, time_gran_to_list_of_exfil_amts, end_of_training):

    #time_gran_to_mod_zscore_df = process_graph_metrics.calculate_mod_zscores_dfs(total_calculated_vals, minimum_training_window,
    #                                                                             training_window_size, time_interval_lengths)

    mod_z_score_df_basefile_name = alert_file + 'mod_z_score_' + sub_path
    z_score_df_basefile_name = alert_file + 'norm_z_score_' + sub_path

    if calc_zscore_p:
        time_gran_to_mod_zscore_df = process_graph_metrics.calc_time_gran_to_mod_zscore_dfs(time_gran_to_feature_dataframe,
                                                                                            training_window_size,
                                                                                            minimum_training_window)

        process_graph_metrics.save_feature_datafames(time_gran_to_mod_zscore_df, mod_z_score_df_basefile_name,
                                                     time_gran_to_attack_labels, time_gran_to_synthetic_exfil_paths_series,
                                                     time_gran_to_list_of_concrete_exfil_paths,
                                                     time_gran_to_list_of_exfil_amts, end_of_training)

        time_gran_to_zscore_dataframe = process_graph_metrics.calc_time_gran_to_zscore_dfs(time_gran_to_feature_dataframe,
                                                                                           training_window_size,
                                                                                           minimum_training_window)

        process_graph_metrics.save_feature_datafames(time_gran_to_zscore_dataframe, z_score_df_basefile_name,
                                                     time_gran_to_attack_labels, time_gran_to_synthetic_exfil_paths_series,
                                                     time_gran_to_list_of_concrete_exfil_paths,
                                                     time_gran_to_list_of_exfil_amts, end_of_training)
    else:
        time_gran_to_zscore_dataframe = {}
        time_gran_to_mod_zscore_df = {}
        for interval in time_gran_to_feature_dataframe.keys():
            time_gran_to_zscore_dataframe[interval] = pd.read_csv(z_score_df_basefile_name + str(interval) + '.csv', na_values='?')
            time_gran_to_mod_zscore_df[interval] = pd.read_csv(mod_z_score_df_basefile_name + str(interval) + '.csv', na_values='?')
            try:
                del time_gran_to_zscore_dataframe[interval]['exfil_path']
                del time_gran_to_mod_zscore_df[interval]['exfil_path']
                del time_gran_to_zscore_dataframe[interval]['concrete_exfil_path']
                del time_gran_to_mod_zscore_df[interval]['concrete_exfil_path']
                del time_gran_to_zscore_dataframe[interval]['exfil_weight']
                del time_gran_to_mod_zscore_df[interval]['exfil_weight']
                del time_gran_to_zscore_dataframe[interval]['exfil_pkts']
                del time_gran_to_mod_zscore_df[interval]['exfil_pkts']
                del time_gran_to_zscore_dataframe[interval]['is_test']
                del time_gran_to_mod_zscore_df[interval]['is_test']
            except:
                pass

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

def assign_attacks_to_first_available_spots(time_gran_to_attack_labels, largest_time_gran, time_periods_startup, time_periods_attack,
                                            counter, time_gran_to_attack_ranges, synthetic_exfil_paths, current_exfil_paths):
    for synthetic_exfil_path in synthetic_exfil_paths:
        if synthetic_exfil_path in current_exfil_paths:
            # randomly choose ranges using highest granularity (then after this we'll choose for the smaller granularities...)
            attack_spot_found = False
            number_free_spots = time_gran_to_attack_labels[largest_time_gran][int(time_periods_startup):].count(0)
            if number_free_spots < time_periods_attack:
                exit(1244) # should break now b/c infinite loop (note: we're not handling the case where it is fragmented...)
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
            else:
                ### by making these two points the same, this value will be 'passed over' by the other functions...
                potential_starting_point = int(time_periods_startup + counter)
                time_gran_to_attack_ranges[largest_time_gran].append((potential_starting_point,potential_starting_point))
    return time_gran_to_attack_labels, time_gran_to_attack_ranges

##### the goal needs to be some mapping of times to attacks to time (ranges) + updated attack labels
##### so, in effect, there are TWO outputs... and it makes a lot more sense to pick the range then modify
##### the labels
## NOTE: portion_for_training is the percentage to devote to using for the training period (b/c attacks will be injected
## into both the training period and the testing period)
def determine_attacks_to_times(time_gran_to_attack_labels, synthetic_exfil_paths, time_of_synethic_exfil, min_starting,
                               end_of_train, synthetic_exfil_paths_train, synthetic_exfil_paths_test):
    time_grans = time_gran_to_attack_labels.keys()
    largest_time_gran = sorted(time_grans)[-1]
    print "LARGEST_TIME_GRAN", largest_time_gran
    time_periods_attack = float(time_of_synethic_exfil) / float(largest_time_gran)
    time_periods_startup = math.ceil(float(min_starting) / float(largest_time_gran))
    time_gran_to_attack_ranges = {} # a list that'll correspond w/ the synthetic exfil paths
    for time_gran in time_gran_to_attack_labels.keys():
        time_gran_to_attack_ranges[time_gran] = []

    ## assign injected attacks to times here...
    ### (a) add to time_gran_to_attack_ranges... just put the existing ranges w/ 'injection' as the marker'
    time_gran_to_physical_attack_ranges = {}
    for time_gran in time_gran_to_attack_labels.keys():
        time_gran_to_physical_attack_ranges[time_gran] = determine_physical_attack_ranges(time_gran_to_attack_labels[time_gran])
        print "physical_attack_ranges", time_gran_to_physical_attack_ranges[time_gran]

    # first, let's assign for the training period...
    counter = 0
    time_gran_to_attack_labels, time_gran_to_attack_ranges = assign_attacks_to_first_available_spots(time_gran_to_attack_labels, largest_time_gran, time_periods_startup,
                                            time_periods_attack, counter, time_gran_to_attack_ranges, synthetic_exfil_paths, synthetic_exfil_paths_train)
    # second, let's assign for the testing period...
    print end_of_train, largest_time_gran
    counter = int(math.ceil(end_of_train/largest_time_gran)) #int(math.ceil(len(time_gran_to_attack_labels[largest_time_gran]) * end_of_train - time_periods_startup))
    print "second_counter!!", counter
    time_gran_to_attack_labels, time_gran_to_attack_ranges = assign_attacks_to_first_available_spots(time_gran_to_attack_labels, largest_time_gran, time_periods_startup,
                                            time_periods_attack, counter, time_gran_to_attack_ranges, synthetic_exfil_paths, synthetic_exfil_paths_test)

    # okay, so now we have the times selected for the largest time granularity... we have to make sure
    # that the other granularities agree...

    print "HIGHEST GRAN SYNTHETIC ATTACKS CHOSEN -- START MAPPING TO LOWER GRAN NOW!"
    for j in range(0, len(time_gran_to_attack_ranges[largest_time_gran])):
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
                # print "z",z
                attack_labels[z] = 1
    return time_gran_to_attack_labels, time_gran_to_attack_ranges, time_gran_to_physical_attack_ranges

def determine_physical_attack_ranges(physical_attack_labels):
    ## determine the indexes of contiguous sets of 1's...
    # step 1: find indexes of all the ones (using list comprehension)
    indexes_of_attack_labels = [i for i,j in enumerate(physical_attack_labels) if j == 1]
    print "indexes_of_attack_labels", indexes_of_attack_labels
    # step 2: find contiguous size of contigous numbers
    ### a solution to this is in the docs, so let's just
    #### do it that way: https://docs.python.org/2.6/library/itertools.html#examples
    physical_attack_ranges = []
    for k, g in groupby(enumerate(indexes_of_attack_labels), lambda (i, x): i - x):
        attack_grp =  map(itemgetter(1), g) #groupby, itemgetter
        physical_attack_ranges.append((attack_grp[0], attack_grp[-1]))
    #print "physical_attack_ranges", physical_attack_ranges
    return physical_attack_ranges

def determine_time_gran_to_synthetic_exfil_paths_series(time_gran_to_attack_ranges, synthetic_exfil_paths,
                                                        interval_to_filenames, time_gran_to_physical_attack_ranges,
                                                        injected_exfil_path):
    time_gran_to_synthetic_exfil_paths_series = {}
    for time_gran, attack_ranges in time_gran_to_attack_ranges.iteritems():
        print interval_to_filenames.keys()
        time_steps = len(interval_to_filenames[str(time_gran)])
        current_exfil_path_series = pd.Series([0 for i in range(0,time_steps)])
        print "time_gran_attack_ranges", time_gran, attack_ranges

        # first add the physical attacks
        physical_attack_ranges = time_gran_to_physical_attack_ranges[time_gran]
        for attack_counter, attack_range in enumerate(physical_attack_ranges):
            for i in range(attack_range[0], attack_range[1]):
                current_exfil_path_series[i] = ['physical:'] + injected_exfil_path

        # then add the injected attacks
        for attack_counter, attack_range in enumerate(attack_ranges):
            for i in range(attack_range[0], attack_range[1]):
                current_exfil_path_series[i] = synthetic_exfil_paths[attack_counter % len(synthetic_exfil_paths)]
        #current_exfil_path_series.index *= 10
        time_gran_to_synthetic_exfil_paths_series[time_gran] = current_exfil_path_series
    #print "time_gran_to_synthetic_exfil_paths_series", time_gran_to_synthetic_exfil_paths_series

    #time.sleep(60)
    return time_gran_to_synthetic_exfil_paths_series

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
                               sec_between_exfil_events=1, time_of_synethic_exfil=30,
                               fraction_of_edge_weights=0.1, fraction_of_edge_pkts=0.1,
                               size_of_neighbor_training_window=300,
                               end_of_training=None, injected_exfil_path='None', only_exp_info=False,
                               initiator_info_for_paths=None,
                               synthetic_exfil_paths_train=None, synthetic_exfil_paths_test=None,
                               skip_model_part=False):

    print "log file can be found at: " + str(basefile_name) + '_logfile.log'
    logging.basicConfig(filename=basefile_name + '_logfile.log', level=logging.INFO)
    logging.info('run_data_anaylsis_pipeline Started')

    if 'kube-dns' not in ms_s:
        ms_s.append('kube-dns') # going to put this here so I don't need to re-write all the recipes...

    gc.collect()

    print "starting pipeline..."


    sub_path = 'sub_'  # NOTE: make this an empty string if using the full pipeline (and not the subset)
    mapping,list_of_infra_services = create_mappings(is_swarm, container_info_path, kubernetes_svc_info,
                                                     kubernetes_pod_info, cilium_config_path, ms_s)


    if (synthetic_exfil_paths_train is None or initiator_info_for_paths is None or synthetic_exfil_paths_test is None) and not only_exp_info:
        print "about_to_return_synthetic_exfil_paths", not synthetic_exfil_paths_train, not initiator_info_for_paths,\
                not synthetic_exfil_paths_test
        # todo: might wanna specify this is in the attack descriptions...
        for ms in ms_s:
            if 'user' in ms:
                sensitive_ms = ms
            if 'my-release' in ms:
                sensitive_ms = ms
        synthetic_exfil_paths, initiator_info_for_paths = gen_attack_templates.generate_synthetic_attack_templates(mapping, ms_s, sensitive_ms)
        return synthetic_exfil_paths, initiator_info_for_paths, None, None,None

    system_startup_time = training_window_size+size_of_neighbor_training_window

    experiment_folder_path = basefile_name.split('edgefiles')[0]
    pcap_file = pcap_paths[0].split('/')[-1] # NOTE: assuming only a single pcap file...
    exp_name = basefile_name.split('/')[-1]
    make_edgefiles_p = make_edgefiles_p and only_exp_info
    interval_to_filenames = process_pcap.process_pcap(experiment_folder_path, pcap_file, time_interval_lengths,
                                                      exp_name, make_edgefiles_p, mapping)

    time_grans = [int(i) for i in interval_to_filenames.keys()]
    smallest_time_gran = min(time_grans)
    if only_exp_info:
        print "only_exp_info_section", only_exp_info
        total_experiment_length = len(interval_to_filenames[str(smallest_time_gran)]) * smallest_time_gran
        print "about to return from only_exp_info section",total_experiment_length, exfil_start_time, exfil_end_time,\
            system_startup_time,None
        return total_experiment_length, exfil_start_time, exfil_end_time, system_startup_time,None

    if calc_vals or graph_p:
        # TODO: 90% sure that there is a problem with this function...
        #largest_interval = int(min(interval_to_filenames.keys()))
        exp_length = len(interval_to_filenames[str(smallest_time_gran)]) * smallest_time_gran
        print "exp_length_ZZZ", exp_length, type(exp_length)
        if not skip_model_part:
            time_gran_to_attack_labels = process_graph_metrics.generate_time_gran_to_attack_labels(time_interval_lengths,
                                                                                               exfil_start_time, exfil_end_time,
                                                                                                sec_between_exfil_events,
                                                                                               exp_length)
        else:
            time_gran_to_attack_labels = {}
            for time_gran in time_interval_lengths:
                time_gran_to_attack_labels[time_gran] = [(1,1)]
                #pass

        #print "interval_to_filenames_ZZZ",interval_to_filenames
        for interval, filenames in interval_to_filenames.iteritems():
            print "interval_ZZZ", interval, len(filenames)
        for time_gran, attack_labels in time_gran_to_attack_labels.iteritems():
            print "time_gran_right_after_creation", time_gran, "len of attack labels", len(attack_labels)

        print interval_to_filenames, type(interval_to_filenames), 'stufff', interval_to_filenames.keys()

        # most of the parameters are kinda arbitrary ATM...
        print "INITIAL time_gran_to_attack_labels", time_gran_to_attack_labels
        ## okay, I'll probably wanna write tests for the below function, but it seems to be working pretty well on my
        # informal tests...
        end_of_training = end_of_training
        synthetic_exfil_paths = []
        for path in synthetic_exfil_paths_train + synthetic_exfil_paths_test:
            if path not in synthetic_exfil_paths:
                synthetic_exfil_paths.append(path)

        time_gran_to_attack_labels, time_gran_to_attack_ranges, time_gran_to_physical_attack_ranges = \
            determine_attacks_to_times(time_gran_to_attack_labels, synthetic_exfil_paths, time_of_synethic_exfil=time_of_synethic_exfil,
                                       min_starting=system_startup_time, end_of_train=end_of_training,
                                       synthetic_exfil_paths_train=synthetic_exfil_paths_train,
                                       synthetic_exfil_paths_test=synthetic_exfil_paths_test)
        print "time_gran_to_attack_labels",time_gran_to_attack_labels
        print "time_gran_to_attack_ranges", time_gran_to_attack_ranges
        #time.sleep(50)

        time_gran_to_synthetic_exfil_paths_series = determine_time_gran_to_synthetic_exfil_paths_series(time_gran_to_attack_ranges,
                                                                            synthetic_exfil_paths, interval_to_filenames,
                                                                            time_gran_to_physical_attack_ranges, injected_exfil_path)

        print "time_gran_to_synthetic_exfil_paths_series", time_gran_to_synthetic_exfil_paths_series
        #time.sleep(50)

        #####exit(200) ## TODO ::: <<<---- remove!!
        # OKAY, let's verify that this determine_attacks_to_times function is wokring before moving on to the next one...
        total_calculated_vals, time_gran_to_list_of_concrete_exfil_paths, time_gran_to_list_of_exfil_amts = \
            calculate_raw_graph_metrics(time_interval_lengths, interval_to_filenames, ms_s, basegraph_name, calc_vals,
                                        window_size, mapping, is_swarm, make_net_graphs_p, list_of_infra_services,
                                        synthetic_exfil_paths, initiator_info_for_paths, time_gran_to_attack_ranges,
                                        fraction_of_edge_weights, fraction_of_edge_pkts, size_of_neighbor_training_window)

        time_gran_to_feature_dataframe = process_graph_metrics.generate_feature_dfs( total_calculated_vals, time_interval_lengths)

        process_graph_metrics.save_feature_datafames(time_gran_to_feature_dataframe, alert_file + sub_path,
                                                     time_gran_to_attack_labels,time_gran_to_synthetic_exfil_paths_series,
                                                     time_gran_to_list_of_concrete_exfil_paths, time_gran_to_list_of_exfil_amts,
                                                     end_of_training)

        analysis_pipeline.generate_graphs.generate_feature_multitime_boxplots(total_calculated_vals, basegraph_name,
                                                                              window_size, colors, time_interval_lengths,
                                                                              exfil_start_time, exfil_end_time, wiggle_room)


    else:
        time_gran_to_feature_dataframe = {}
        time_gran_to_attack_labels = {}
        time_gran_to_synthetic_exfil_paths_series = {}
        time_gran_to_list_of_concrete_exfil_paths = {}
        time_gran_to_list_of_exfil_amts = {}
        min_interval = min(time_interval_lengths)
        for interval in time_interval_lengths:
            #if interval in time_interval_lengths:
            print "time_interval_lengths",time_interval_lengths, "interval", interval
            print "feature_df_path", alert_file + sub_path + str(interval) + '.csv'
            time_gran_to_feature_dataframe[interval] = pd.read_csv(alert_file + sub_path + str(interval) + '.csv', na_values='?')
            time_gran_to_attack_labels[interval] = time_gran_to_feature_dataframe[interval]['labels']
            time_gran_to_synthetic_exfil_paths_series[interval] = time_gran_to_feature_dataframe[interval]['exfil_path']
            ##recover time_gran_to_list_of_concrete_exfil_paths, time_gran_to_list_of_exfil_amts
            time_gran_to_list_of_concrete_exfil_paths[interval] = time_gran_to_feature_dataframe[interval]['concrete_exfil_path']
            list_of_exfil_amts = []
            for counter in range(0, len(time_gran_to_feature_dataframe[interval]['exfil_weight'])):
                weight = time_gran_to_feature_dataframe[interval]['exfil_weight'][counter]
                pkts = time_gran_to_feature_dataframe[interval]['exfil_pkts'][counter]
                current_exfil_dict = {'weight':weight, 'frames': pkts}
                list_of_exfil_amts.append( current_exfil_dict )
            time_gran_to_list_of_exfil_amts[interval] = list_of_exfil_amts
            if min_interval:
                print time_gran_to_feature_dataframe[interval]['is_test'], type(time_gran_to_feature_dataframe[interval]['is_test'])
                end_of_training = time_gran_to_feature_dataframe[interval]['is_test'].tolist().index(1) * min_interval

    print "about to calculate some alerts!"

    time_gran_to_feature_dataframe_copy = copy.deepcopy(time_gran_to_feature_dataframe)
    for time_gran, feature_dataframe in time_gran_to_feature_dataframe.iteritems():
        try:
            del feature_dataframe['exfil_path']
            del feature_dataframe['exfil_weight']
            del feature_dataframe['exfil_pkts']
            del feature_dataframe['concrete_exfil_path']
            del feature_dataframe['is_test']
        except:
            pass


    time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe = \
        calc_zscores(alert_file, training_window_size, minimum_training_window, sub_path, time_gran_to_attack_labels,
                     time_gran_to_feature_dataframe, calc_zscore_p, time_gran_to_synthetic_exfil_paths_series,
                     time_gran_to_list_of_concrete_exfil_paths, time_gran_to_list_of_exfil_amts, end_of_training)

    print "analysis_pipeline about to return!"

    # okay, so can return it here...
    #### exit(121)

    #for time_gran, mod_zscore_df in time_gran_to_mod_zscore_df.iteritems():
    #    mod_zscore_df['exfil_paths'] = time_gran_to_synthetic_exfil_paths_series[time_gran]
    '''
    return_dict = {}
    return_dict['time_gran_to_mod_zscore_df'] = time_gran_to_mod_zscore_df
    return_dict['time_gran_to_zscore_dataframe'] = time_gran_to_zscore_dataframe
    return_dict['time_gran_to_feature_dataframe'] = time_gran_to_feature_dataframe
    return_dict['time_gran_to_synthetic_exfil_paths_series'] = time_gran_to_synthetic_exfil_paths_series
    return_dict['end_of_training'] = end_of_training
    return return_dict
    '''

    return time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, time_gran_to_feature_dataframe_copy, \
           time_gran_to_synthetic_exfil_paths_series, end_of_training

# this function determines how much time to is available for injection attacks in each experiment.
# it takes into account when the physical attack starts (b/c need to split into training/testing set
# temporally before the physical attack starts) and the goal percentage of split that we are aiming for.
# I think we're going to aim for the desired split in each experiment, but we WON'T try  to compensate
# not meeting one experiment's goal by modifying how we handle another experiment.
def determine_injection_times(exps_info, goal_train_test_split, goal_attack_NoAttack_split):
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
        testing_time_without_physical_attack = testing_time - physical_attack_time ## <<-- used to use this to calc testing_time_for_attack_injection
        ## but switched it later on b/c you get a (fairly big) label imbalance
        testing_time_for_attack_injection = physical_attack_time * goal_attack_NoAttack_split

        # now let's find the time to inject during training... this'll be a percentage of the time between
        # system startup and the training/testing split point...
        training_time_after_startup = time_split - exp_info["startup_time"]
        training_time_for_attack_injection = training_time_after_startup * goal_attack_NoAttack_split

        exp_injection_info.append({'testing': testing_time_for_attack_injection,
                                   "training": training_time_for_attack_injection})
        #time_splits.append(time_split)
    return exp_injection_info,end_of_train_portions

# this function loops through multiple experiments (or even just a single experiment), accumulates the relevant
# feature dataframes, and then performs LASSO regression to determine a concise graphical model that can detect
# the injected synthetic attacks
def multi_experiment_pipeline(function_list_exp_info, function_list, base_output_name, ROC_curve_p, time_each_synthetic_exfil,
                              goal_train_test_split, goal_attack_NoAttack_split, training_window_size,
                              size_of_neighbor_training_window, calc_vals, skip_model_part):
    ### Okay, so what is needed here??? We need, like, a list of sets of input (appropriate for run_data_analysis_pipeline),
    ### followed by the LASSO stuff, and finally the ROC stuff... okay, let's do this!!!

    # step(0): need to find out the  meta-data for each experiment so we can coordinate the
    # synthetic attack injections between experiments
    if calc_vals and not skip_model_part:
        print function_list_exp_info
        exp_infos = []
        for func_exp_info in function_list_exp_info:
            print "calc_vals", calc_vals
            total_experiment_length, exfil_start_time, exfil_end_time, system_startup_time, _ = \
                func_exp_info(training_window_size=training_window_size, size_of_neighbor_training_window=size_of_neighbor_training_window,calc_vals=calc_vals)
            print "func_exp_info", total_experiment_length, exfil_start_time, exfil_end_time
            exp_infos.append({"total_experiment_length":total_experiment_length, "exfil_start_time":exfil_start_time,
                             "exfil_end_time":exfil_end_time, "startup_time": system_startup_time})

        ## get the exfil_paths that were generated using the mulval component...
        ## this'll require passing a parameter to the single-experiment pipeline and then getting the set of paths
        exps_exfil_paths = []
        exps_initiator_info = []
        for func in function_list:
            synthetic_exfil_paths, initiator_info_for_paths, _, _,_ = func(time_of_synethic_exfil=time_each_synthetic_exfil,
                                                                           calc_vals=calc_vals)
            exps_exfil_paths.append(synthetic_exfil_paths)
            exps_initiator_info.append(initiator_info_for_paths)

        training_exfil_paths, testing_exfil_paths, end_of_train_portions = assign_exfil_paths_to_experiments(exp_infos, goal_train_test_split,
                                                                                      goal_attack_NoAttack_split,time_each_synthetic_exfil,
                                                                                      exps_exfil_paths)
        print "end_of_train_portions", end_of_train_portions
        possible_exps_exfil_paths = []
        for exp_exfil_paths in exps_exfil_paths:
            for exp_exfil_path in exp_exfil_paths:
                if exp_exfil_path not in possible_exps_exfil_paths:
                    possible_exps_exfil_paths.append(exp_exfil_path)
        print "possible_exps_exfil_paths:"
        for possible_exp_exfil_path in possible_exps_exfil_paths:
            print possible_exp_exfil_path
        ###### exit(122) ### TODO::: <--- remove!!!

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

    ## step (1) : iterate through individual experiments...
    ##  # 1a. list of inputs [done]
    ##  # 1b. acculate DFs
    list_time_gran_to_mod_zscore_df = []
    list_time_gran_to_mod_zscore_df_training = []
    list_time_gran_to_mod_zscore_df_testing = []
    list_time_gran_to_zscore_dataframe = []
    list_time_gran_to_feature_dataframe = []

    experiments_to_exfil_path_time_dicts = []
    starts_of_testing = []
    for counter,func in enumerate(function_list):
        #time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, time_gran_to_feature_dataframe, _ = func()
        print "exps_exfil_paths[counter]_to_func",exps_exfil_paths[counter], exps_initiator_info
        ## TODO: need to make this use the multiprocessing library (so I can force the
        ## stupid garbage collector to actually work...)
        '''
        kwards = {'time_of_synethic_exfil':time_each_synthetic_exfil,
                 'initiator_info_for_paths':exps_initiator_info[counter],
                 'training_window_size':training_window_size,
                 'size_of_neighbor_training_window':size_of_neighbor_training_window,
                 'portion_for_training':end_of_train_portions[counter],
                 'synthetic_exfil_paths_train' : training_exfil_paths[counter],
                 'synthetic_exfil_paths_test' : testing_exfil_paths[counter],
                 'calc_vals':calc_vals}

        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        p = multiprocessing.Process(target=func, kwargs=kwards,  args=(return_dict,))
        p.start()
        p.join()
        time_gran_to_mod_zscore_df = return_dict['time_gran_to_mod_zscore_df']
        time_gran_to_zscore_dataframe = return_dict['time_gran_to_zscore_dataframe']
        time_gran_to_feature_dataframe = return_dict['time_gran_to_feature_dataframe']
        '''
        time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, time_gran_to_feature_dataframe, _, start_of_testing = \
            func(time_of_synethic_exfil=time_each_synthetic_exfil,
                 initiator_info_for_paths=exps_initiator_info[counter],
                 training_window_size=training_window_size,
                 size_of_neighbor_training_window=size_of_neighbor_training_window,
                 portion_for_training=end_of_train_portions[counter],
                 synthetic_exfil_paths_train = training_exfil_paths[counter],
                 synthetic_exfil_paths_test = testing_exfil_paths[counter],
                 calc_vals=calc_vals,
                 skip_model_part=skip_model_part)
        print "exps_exfil_pathas[time_gran_to_mod_zscore_df]", time_gran_to_mod_zscore_df
        list_time_gran_to_mod_zscore_df.append(time_gran_to_mod_zscore_df)
        list_time_gran_to_zscore_dataframe.append(time_gran_to_zscore_dataframe)
        list_time_gran_to_feature_dataframe.append(time_gran_to_feature_dataframe)
        list_time_gran_to_mod_zscore_df_training.append(generate_time_gran_sub_dataframes(time_gran_to_mod_zscore_df, 'is_test', 0))
        list_time_gran_to_mod_zscore_df_testing.append(generate_time_gran_sub_dataframes(time_gran_to_mod_zscore_df, 'is_test', 1))
        starts_of_testing.append(start_of_testing)
        gc.collect()

    # step (2) :  take the dataframes and feed them into the LASSO component...
    ### 2a. split into training and testing data
    ###### at the moment, we'll make the simplfying assumption to only use the modified z-scores...
    ######## 2a.I. get aggregate dfs for each time granularity...
    time_gran_to_aggregate_mod_score_dfs = {}
    time_gran_to_feature_dfs = {}
    print "about_to_do_list_time_gran_to_mod_zscore_df"
    time_gran_to_aggregate_mod_score_dfs = aggregate_dfs(list_time_gran_to_mod_zscore_df)
    #time_gran_to_aggregate_mod_score_dfs_training = aggregate_dfs(list_time_gran_to_mod_zscore_df_training)
    #time_gran_to_aggregate_mod_score_dfs_testing = aggregate_dfs(list_time_gran_to_mod_zscore_df_testing)
    print "about_to_do_list_time_gran_to_feature_dataframe"
    time_gran_to_aggreg_feature_dfs = aggregate_dfs(list_time_gran_to_feature_dataframe)

    for time_gran, aggregate_feature_df in time_gran_to_aggreg_feature_dfs.iteritems():
        aggregate_feature_df.to_csv(base_output_name + 'aggregate_feature_df_at_time_gran_of_' + str(time_gran) + '_sec.csv',
                                    na_rep='?')
    for time_gran, aggregate_feature_df in time_gran_to_aggregate_mod_score_dfs.iteritems():
        aggregate_feature_df.to_csv(base_output_name + 'modz_feat_df_at_time_gran_of_' + str(time_gran) + '_sec.csv',
                                    na_rep='?')

    recipes_used = [recipe.__name__ for recipe in function_list]
    names = []
    for counter,recipe in enumerate(recipes_used):
        #print "recipe_in_functon_list", recipe.__name__
        #name = recipe.__name__
        name = '_'.join(recipe.split('_')[1:])
        names.append(name)

    path_occurence_training_df = generate_exfil_path_occurence_df(list_time_gran_to_mod_zscore_df_training, names)
    path_occurence_testing_df = generate_exfil_path_occurence_df(list_time_gran_to_mod_zscore_df_testing, names)

    ##################################
    # todo: okay, I'm going to mess w/ the aggregate mod_score_dfs b/c the vip vals should actually be just raw (instead
    # of the mod-z score of hte vals)
    for time_gran, aggregate_mod_df in time_gran_to_aggregate_mod_score_dfs.iteritems():
        # okay, want to do 2 things:
        # drop old vip terms
        # add non-mod-z-score-terms
        pass
        ##time_gran_to_aggregate_mod_score_dfs[time_gran] = None ## TODO

    ##################################
    '''
    statistically_analyze_graph_features(time_gran_to_aggregate_mod_score_dfs, ROC_curve_p, base_output_name,
                                         names, starts_of_testing, path_occurence_training_df,
                                         path_occurence_testing_df, recipes_used)
    '''
    statistically_analyze_graph_features(time_gran_to_aggreg_feature_dfs, ROC_curve_p, base_output_name,
                                         names, starts_of_testing, path_occurence_training_df,
                                         path_occurence_testing_df, recipes_used, skip_model_part)

def statistically_analyze_graph_features(time_gran_to_aggregate_mod_score_dfs, ROC_curve_p, base_output_name, names,
                                         starts_of_testing, path_occurence_training_df, path_occurence_testing_df,
                                         recipes_used, skip_model_part):
    #print time_gran_to_aggregate_mod_score_dfs['60']
    ######### 2a.II. do the actual splitting
    # note: labels have the column name 'labels' (noice)
    time_gran_to_model = {}
    #images = 0
    time_gran_to_debugging_csv = {} # note: going to be used for (shockingly) debugging purposes....
    percent_attacks = []
    list_percent_attacks_training = []
    list_of_rocs = []
    list_of_feat_coefs_dfs = []
    time_grans = []
    list_of_model_parameters = []
    list_of_attacks_found_dfs = []
    list_of_attacks_found_training_df = []
    list_of_optimal_fone_scores = []
    for time_gran,aggregate_mod_score_dfs in time_gran_to_aggregate_mod_score_dfs.iteritems():
        time_grans.append(time_gran)
        #'''
        try:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='timemod_z_score')   # might wanna just stop these from being generated...
        except:
            pass
        try:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='labelsmod_z_score') # might wanna just stop these from being generaetd
            print aggregate_mod_score_dfs.columns
        except:
            pass
        try:

            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='Unnamed: 0mod_z_score') # might wanna just stop these from being generaetd
        except:
            pass

        #'''
        if not skip_model_part:
            ### TODO TODO TODO TODO TODO TODO
            ### todo: might wanna remove? might wanna keep? not sure...
            ### todo: drop the test physical attacks from the test sets...
            aggregate_mod_score_dfs = \
                aggregate_mod_score_dfs[~((aggregate_mod_score_dfs['labels'] == 1) &
                                          (aggregate_mod_score_dfs['exfil_pkts'] == 0))]
            #####

            aggregate_mod_score_dfs_training = aggregate_mod_score_dfs[aggregate_mod_score_dfs['is_test'] == 0]
            aggregate_mod_score_dfs_testing = aggregate_mod_score_dfs[aggregate_mod_score_dfs['is_test'] == 1]
        else:
            ## note: generally you'd want to split into test and train sets, but if we're not doing logic
            ## part anyway, we just want quick-and-dirty results, so don't bother (note: so for formal purposes,
            ## DO NOT USE WITHOUT LOGIC CHECKING OR SOLVE THE TRAINING-TESTING split problem)
            aggregate_mod_score_dfs_training = aggregate_mod_score_dfs
            aggregate_mod_score_dfs_testing = aggregate_mod_score_dfs

        time_gran_to_debugging_csv[time_gran] = copy.deepcopy(aggregate_mod_score_dfs)


        X_train = aggregate_mod_score_dfs_training.loc[:, aggregate_mod_score_dfs_training.columns != 'labels']
        y_train = aggregate_mod_score_dfs_training.loc[:, aggregate_mod_score_dfs_training.columns == 'labels']
        X_test = aggregate_mod_score_dfs_testing.loc[:, aggregate_mod_score_dfs_training.columns != 'labels']
        y_test = aggregate_mod_score_dfs_testing.loc[:, aggregate_mod_score_dfs_training.columns == 'labels']

        #print "X_train", X_train
        #print "y_train", y_train

        ##X_train, X_test, y_train, y_test =  sklearn.model_selection.train_test_split(X, y, test_size = 1-goal_train_test_split, random_state = 42)
        #print X_train.shape, "X_train.shape"

        exfil_paths = X_test['exfil_path'].replace('0','[]')
        exfil_paths_train = X_train['exfil_path'].replace('0','[]')
        #print "----"
        #print "exfil_path_pre_literal_eval", exfil_paths, type(exfil_paths)
        #exfil_paths = ast.literal_eval(exfil_paths)
        #print "----"

        ## todo: extract and put in a 'safe' spot...
        ##calculated_values['max_ewma_control_chart_scores'] = list_of_max_ewma_control_chart_scores
        try:
            ewma_train = X_train['max_ewma_control_chart_scores']
            ewma_test = X_test['max_ewma_control_chart_scores']
            X_train = X_train.drop(columns='max_ewma_control_chart_scores')
            X_test = X_test.drop(columns='max_ewma_control_chart_scores')
        except:
            ewma_train = [0 for i in range(0,len(X_train))]
            ewma_test = [0 for i in range(0,len(X_test))]

        try:
            ide_train =  X_train['ide_angles']
            ide_test = X_test['ide_angles']
            X_train = X_train.drop(columns='ide_angles')
            X_test = X_test.drop(columns='ide_angles')
        except:
            ide_train = [0 for i in range(0,len(X_train))]
            ide_test = [0 for i in range(0,len(X_test))]


        X_train = X_train.drop(columns='exfil_path')
        X_train = X_train.drop(columns='concrete_exfil_path')
        X_train_exfil_weight = X_train['exfil_weight']
        X_train = X_train.drop(columns='exfil_weight')
        X_train = X_train.drop(columns='exfil_pkts')
        X_train = X_train.drop(columns='is_test')
        X_test = X_test.drop(columns='exfil_path')
        X_test = X_test.drop(columns='concrete_exfil_path')
        X_test_exfil_weight = X_test['exfil_weight']
        X_test = X_test.drop(columns='exfil_weight')
        X_test = X_test.drop(columns='exfil_pkts')
        X_test = X_test.drop(columns='is_test')

        ## TODO: might to put these back in...
        dropped_feature_list = []


        '''
        dropped_feature_list = ['New Class-Class Edges with DNS_mod_z_score', 'New Class-Class Edges with Outside_mod_z_score',
                                'New Class-Class Edges_mod_z_score']
        X_train = X_train.drop(columns='New Class-Class Edges with DNS_mod_z_score')
        X_train = X_train.drop(columns='New Class-Class Edges with Outside_mod_z_score')
        X_train = X_train.drop(columns='New Class-Class Edges_mod_z_score')
        X_test = X_test.drop(columns='New Class-Class Edges with DNS_mod_z_score')
        X_test = X_test.drop(columns='New Class-Class Edges with Outside_mod_z_score')
        X_test = X_test.drop(columns='New Class-Class Edges_mod_z_score')
        '''

        print '-------'
        print type(X_train)
        print X_train.columns.values
        ###exit(344) ### TODO TODO TODO <<<----- remove!!!

        print "columns", X_train.columns
        print "columns", X_test.columns

        print X_train.dtypes
        # need to replace the missing values in the data w/ meaningful values...
        ''' ## imputer is dropping a column... let's do this w/ pandas dataframes instead....
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        imp = imp.fit(X_train)
        X_train = imp.transform(X_train)
        X_test = imp.transform(X_test)
        '''
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())
        print "X_train_median", X_train.median()

        X_train = X_train.dropna(axis=1)
        X_test = X_test.dropna(axis=1)
        print "columns", X_train.columns
        print "columns", X_test.columns

        X_train_dtypes = X_train.dtypes
        X_train_columns = X_train.columns.values
        X_test_columns = X_test.columns.values
        print y_test
        number_attacks_in_test = len(y_test[y_test['labels'] == 1])
        number_non_attacks_in_test = len(y_test[y_test['labels'] == 0])
        percent_attacks.append(float(number_attacks_in_test) / (number_non_attacks_in_test + number_attacks_in_test))

        number_attacks_in_train = len(y_train[y_train['labels'] == 1])
        number_non_attacks_in_train = len(y_train[y_train['labels'] == 0])
        list_percent_attacks_training.append(float(number_attacks_in_train) / (number_non_attacks_in_train + number_attacks_in_train))

        #print "X_train", X_train
        #print "y_train", y_train, len(y_train)
        #print "y_test", y_test, len(y_test)
        #print "-- y_train", len(y_train), "y_test", len(y_test), "time_gran", time_gran, "--"


        ### train the model and generate predictions (2B)
        # note: I think this is where I'd need to modify it to make the anomaly-detection using edge correlation work...

        clf = LassoCV(cv=3, max_iter=8000) ## TODO TODO TODO <<-- instead of having choosing the alpha be magic, let's use cross validation to choose it instead...
        #alpha = 5
        #clf=Lasso(alpha=alpha)
        clf.fit(X_train, y_train)
        score_val = clf.score(X_test, y_test)
        print "score_val", score_val
        test_predictions = clf.predict(X=X_test)
        train_predictions = clf.predict(X=X_train)
        #coefficients = pd.DataFrame({"Feature": X.columns, "Coefficients": np.transpose(clf.coef_)})
        #print coefficients
        #clf.coef_, "intercept", clf.intercept_

        ### get the coefficients used in the model...
        print "Coefficients: "
        print "LASSO model", clf.get_params()
        print '----------------------'
        print len(time_gran_to_debugging_csv[time_gran]["labels"]), len(np.concatenate([train_predictions, test_predictions]))
        time_gran_to_debugging_csv[time_gran].loc[:, "aggreg_anom_score"] = np.concatenate([train_predictions, test_predictions])
        coef_dict = {}
        print "len(clf.coef_)", len(clf.coef_), "len(X_train_columns)", len(X_train_columns), "time_gran", time_gran, \
            "len(X_test_columns)", len(X_test_columns), X_train.shape, X_test.shape
        if len(clf.coef_) != (len(X_train_columns)): # there is no plus one b/c the intercept is stored in clf.intercept_
            print "coef_ is different length than X_train_columns!", X_train_columns
            for  counter,i in enumerate(X_train_dtypes):
                print counter,i, X_train_columns[counter], clf.coef_[counter]
            exit(888)
        for coef, feat in zip(clf.coef_, X_train_columns):
            coef_dict[feat] = coef
        print "intercept...", clf.intercept_
        coef_dict['intercept'] = clf.intercept_
        for coef,feature in coef_dict.iteritems():
            print coef,feature

        #print "COEF_DICT", coef_dict
        coef_feature_df = pd.DataFrame.from_dict(coef_dict, orient='index')
        #print coef_feature_df.columns.values
        #coef_feature_df.index.name = 'Features'
        coef_feature_df.columns = ['Coefficient']

        list_of_feat_coefs_dfs.append(coef_feature_df)
        print '--------------------------'

        model_params = clf.get_params()
        try:
            model_params['alpha_val'] = clf.alpha_
        except:
            model_params['alpha_val'] = alpha
        list_of_model_parameters.append(model_params)

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

            ##  ewma_train = X_train['max_ewma_control_chart_scores']
            ##  ewma_test = X_test['max_ewma_control_chart_scores']
            # now for the ewma part...
            fpr_ewma, tpr_ewma, thresholds_ewma = sklearn.metrics.roc_curve(y_true=y_test, y_score = ewma_test, pos_label=1)
            fpr_ide, tpr_ide, thresholds_ide = sklearn.metrics.roc_curve(y_true=y_test, y_score = ide_test, pos_label=1)


            line_titles = ['ensemble model', 'edge correlation', 'ide_angles']
            list_of_x_vals = [x_vals, fpr_ewma, fpr_ide]
            list_of_y_vals = [y_vals, tpr_ewma, tpr_ide]
            ax, _, plot_path = generate_alerts.construct_ROC_curve(list_of_x_vals, list_of_y_vals, title, ROC_path + plot_name,\
                                                                   line_titles, show_p=False)
            list_of_rocs.append(plot_path)

            ### determination of the optimal operating point goes here (take all the thresh vals and predictions,
            ### find the corresponding f1 scores (using sklearn func), and then return the best.
            optimal_f1_score, optimal_thresh = process_roc.determine_optimal_threshold(y_test, test_predictions, thresholds)
            print "optimal_f1_score", optimal_f1_score, "optimal_thresh",optimal_thresh
            list_of_optimal_fone_scores.append(optimal_f1_score)
            ### get confusion matrix... take predictions from classifer. THreshold
            ### using optimal threshold determined previously. Extract the labels too. This gives two lists, appropriate
            ### for using the confusion_matrix function of sklearn. However, this does NOT handle the different
            ### categories... (for display will probably want to make a df)
            optimal_predictions = [int(i > optimal_thresh) for i in test_predictions]
            print "optimal_predictions", optimal_predictions
            ### determine categorical-level behavior... Split the two lists from the previous step into 2N lists,
            ### where N is the # of categories, and then can just do the confusion matrix function on them...
            ### (and then display the results somehow...)

            categorical_cm_df = determine_categorical_cm_df(y_test, optimal_predictions, exfil_paths, X_test_exfil_weight)
            list_of_attacks_found_dfs.append(categorical_cm_df)

            optimal_train_predictions = [int(i>optimal_thresh) for i in train_predictions]
            categorical_cm_df_training = determine_categorical_cm_df(y_train, optimal_train_predictions, exfil_paths_train,
                                                                     X_train_exfil_weight)
            list_of_attacks_found_training_df.append(categorical_cm_df_training)

            time_gran_to_debugging_csv[time_gran].loc[:, "anom_val_at_opt_pt"] = \
                np.concatenate([optimal_train_predictions, optimal_predictions])

            # I don't want the attributes w/ zero coefficients to show up in the debugging csv b/c it makes it hard to read
            ## TODO
            for feature,coef in coef_dict.iteritems():
                print "coef_check", coef, not coef, feature
                if not coef:
                    print "just_dropped", feature
                    time_gran_to_debugging_csv[time_gran] = time_gran_to_debugging_csv[time_gran].drop([feature],axis=1)
            for dropped_feature in dropped_feature_list:
                time_gran_to_debugging_csv[time_gran] = time_gran_to_debugging_csv[time_gran].drop([dropped_feature], axis=1)

            time_gran_to_debugging_csv[time_gran].to_csv(base_output_name + 'DEBUGGING_modz_feat_df_at_time_gran_of_'+\
                                                         str(time_gran) + '_sec.csv', na_rep='?')

    starts_of_testing_dict = {}
    for counter,name in enumerate(names):
        starts_of_testing_dict[name] = starts_of_testing[counter]

    starts_of_testing_df = pd.DataFrame(starts_of_testing_dict, index=['start_of_testing_phase'])

    print "list_of_rocs", list_of_rocs
    generate_report.generate_report(list_of_rocs, list_of_feat_coefs_dfs, list_of_attacks_found_dfs,
                                    recipes_used, base_output_name, time_grans, list_of_model_parameters,
                                    list_of_optimal_fone_scores, starts_of_testing_df, path_occurence_training_df,
                                    path_occurence_testing_df, percent_attacks, list_of_attacks_found_training_df,
                                    list_percent_attacks_training)

    print "multi_experiment_pipeline is all done! (NO ERROR DURING RUNNING)"
    #print "recall that this was the list of alert percentiles", percentile_thresholds

def determine_categorical_cm_df(y_test, optimal_predictions, exfil_paths, exfil_weights):
    y_test = y_test['labels'].tolist()
    print "new_y_test", y_test
    attack_type_to_predictions, attack_type_to_truth, attack_type_to_weights = \
        process_roc.determine_categorical_labels(y_test, optimal_predictions, exfil_paths, exfil_weights.tolist())
    attack_type_to_confusion_matrix_values = process_roc.determine_cm_vals_for_categories(attack_type_to_predictions,
                                                                                          attack_type_to_truth)
    categorical_cm_df = process_roc.determine_categorical_cm_df(attack_type_to_confusion_matrix_values, attack_type_to_weights)
    ## re-name the row without any attacks in it...
    print "categorical_cm_df.index", categorical_cm_df.index
    categorical_cm_df = categorical_cm_df.rename({(): 'No Attack'}, axis='index')
    return categorical_cm_df

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
                                      exps_exfil_paths):

    flat_exps_exfil_paths = [tuple(exfil_path) for exp_exfil_paths in exps_exfil_paths for exfil_path in exp_exfil_paths]
    print "flat_exps_exfil_paths",flat_exps_exfil_paths
    possible_exfil_paths = list(set(flat_exps_exfil_paths))

    ## now perform the actual assignment portion...
    # first, find the amt of time available for attack injections in each experiments training/testing phase...
    inject_times,end_of_train_portions = determine_injection_times(exp_infos, goal_train_test_split,
                                                                   goal_attack_NoAttack_split)
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

# generates a df indicating how long each logical exfil path occurs during each experiment, and returns a handle DF
# for use in the generated report.
def generate_exfil_path_occurence_df(list_of_time_gran_to_mod_zscore_df, experiment_names):
    experiments_to_exfil_path_time_dicts = []
    for time_gran_to_mod_zscore_df in list_of_time_gran_to_mod_zscore_df:
        min_time_gran = min(time_gran_to_mod_zscore_df.keys())
        logical_exfil_paths_freq = time_gran_to_mod_zscore_df[min_time_gran]['exfil_path'].value_counts().to_dict()
        for path, occur in logical_exfil_paths_freq.iteritems():
            logical_exfil_paths_freq[path] = occur * min_time_gran
        experiments_to_exfil_path_time_dicts.append(logical_exfil_paths_freq)
    path_occurence_df = pd.DataFrame(experiments_to_exfil_path_time_dicts, index=experiment_names)
    return path_occurence_df

def generate_time_gran_sub_dataframes(time_gran_to_df_dataframe, column_name, column_value):
    time_gran_to_sub_dataframe = {}
    for time_gran, dataframe in time_gran_to_df_dataframe.iteritems():
        sub_dataframe = dataframe[dataframe[column_name] == column_value]
        time_gran_to_sub_dataframe[time_gran] = sub_dataframe
    return time_gran_to_sub_dataframe

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