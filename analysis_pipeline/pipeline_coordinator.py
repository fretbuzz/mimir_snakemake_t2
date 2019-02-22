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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LassoCV, Lasso, RidgeCV, Ridge, ElasticNetCV, LogisticRegressionCV
import sklearn
from sklearn import tree
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
from sklearn.ensemble import RandomForestClassifier
import generate_heatmap
import cPickle as pickle
import pyximport
pyximport.install() # am I sure that I want this???

def process_one_set_of_graphs(fraction_of_edge_weights, fraction_of_edge_pkts, time_interval_length, window_size,
                                filenames, svcs, is_swarm, ms_s, mapping,  list_of_infra_services,
                                synthetic_exfil_paths, initiator_info_for_paths, attacks_to_times,
                               collected_metrics_location, current_set_of_graphs_loc, calc_vals, out_q):

    if calc_vals:
        current_set_of_graphs = simplified_graph_metrics.set_of_injected_graphs(fraction_of_edge_weights,
                                         fraction_of_edge_pkts, time_interval_length, window_size,
                                         filenames, svcs, is_swarm, ms_s, mapping, list_of_infra_services,
                                         synthetic_exfil_paths, initiator_info_for_paths, attacks_to_times,
                                          collected_metrics_location, current_set_of_graphs_loc)
        current_set_of_graphs.generate_injected_edgefiles()
        current_set_of_graphs.calcuated_single_step_metrics()
        current_set_of_graphs.calc_serialize_metrics()
        current_set_of_graphs.save()
    else:
        with open(current_set_of_graphs_loc, mode='rb') as f:
            current_set_of_graphs_loc_contents = f.read()
            current_set_of_graphs = pickle.loads(current_set_of_graphs_loc_contents)
        current_set_of_graphs.load_serialized_metrics()
    current_set_of_graphs.put_values_into_outq(out_q)

def calculate_raw_graph_metrics(time_interval_lengths, interval_to_filenames, ms_s, basegraph_name, calc_vals, window_size,
                                mapping, is_swarm, make_net_graphs_p, list_of_infra_services,synthetic_exfil_paths,
                                initiator_info_for_paths, time_gran_to_attacks_to_times, fraction_of_edge_weights,
                                fraction_of_edge_pkts, size_of_neighbor_training_window):
    total_calculated_vals = {}
    time_gran_to_list_of_concrete_exfil_paths = {}
    time_gran_to_list_of_exfil_amts = {}
    time_gran_to_new_neighbors_outside = {}
    time_gran_to_new_neighbors_dns = {}
    time_gran_to_new_neighbors_all = {}
    for time_interval_length in time_interval_lengths:
        print "analyzing edgefiles...", "timer_interval...", time_interval_length

        if is_swarm:
            svcs = analysis_pipeline.prepare_graph.get_svc_equivalents(is_swarm, mapping)
        else:
            print "this is k8s, so using these sevices", ms_s
            svcs = ms_s
        out_q = multiprocessing.Queue()

        '''
        args = (interval_to_filenames[str(time_interval_length)],
               time_interval_length, basegraph_name + '_subset_',
               calc_vals, window_size, ms_s, mapping, is_swarm, svcs,
               list_of_infra_services, synthetic_exfil_paths,
               initiator_info_for_paths,
               time_gran_to_attacks_to_times[time_interval_length],
               fraction_of_edge_weights, fraction_of_edge_pkts,
               int(size_of_neighbor_training_window/time_interval_length),
               out_q)
        p = multiprocessing.Process(
            target=simplified_graph_metrics.calc_subset_graph_metrics,
            args=args)
        p.start()
        total_calculated_vals[(time_interval_length, '')] = out_q.get()
        list_of_concrete_container_exfil_paths = out_q.get()
        list_of_exfil_amts = out_q.get()
        new_neighbors_outside =  out_q.get()
        new_neighbors_dns =  out_q.get()
        new_neighbors_all = out_q.get()
        p.join()
        
        #'''
        collected_metrics_location = basegraph_name + 'collected_metrics_time_gran_' + str(time_interval_length) + '.csv'
        current_set_of_graphs_loc = basegraph_name + 'set_of_graphs' + str(time_interval_length) + '.csv'
        args = [fraction_of_edge_weights, fraction_of_edge_pkts, time_interval_length, window_size,
                interval_to_filenames[str(time_interval_length)], svcs, is_swarm, ms_s, mapping,
                list_of_infra_services, synthetic_exfil_paths,  initiator_info_for_paths,
                time_gran_to_attacks_to_times[time_interval_length], collected_metrics_location, current_set_of_graphs_loc,
                calc_vals, out_q]
        p = multiprocessing.Process(
            target=process_one_set_of_graphs,
            args=args)
        p.start()
        total_calculated_vals[(time_interval_length, '')] = out_q.get()
        list_of_concrete_container_exfil_paths = out_q.get()
        list_of_exfil_amts = out_q.get()
        new_neighbors_outside =  out_q.get()
        new_neighbors_dns =  out_q.get()
        new_neighbors_all = out_q.get()
        p.join()

        print "process returned!"
        time_gran_to_list_of_concrete_exfil_paths[time_interval_length] = list_of_concrete_container_exfil_paths
        time_gran_to_list_of_exfil_amts[time_interval_length] = list_of_exfil_amts
        time_gran_to_new_neighbors_outside[time_interval_length] = None # no longer used
        time_gran_to_new_neighbors_dns[time_interval_length] = None # no longer used
        time_gran_to_new_neighbors_all[time_interval_length] = None # no longer used

        #total_calculated_vals.update(newly_calculated_values)
        gc.collect()
    return total_calculated_vals, time_gran_to_list_of_concrete_exfil_paths, time_gran_to_list_of_exfil_amts,\
        time_gran_to_new_neighbors_outside, time_gran_to_new_neighbors_dns, time_gran_to_new_neighbors_all

def calc_zscores(alert_file, training_window_size, minimum_training_window,
                 sub_path, time_gran_to_attack_labels, time_gran_to_feature_dataframe, calc_zscore_p, time_gran_to_synthetic_exfil_paths_series,
                 time_gran_to_list_of_concrete_exfil_paths, time_gran_to_list_of_exfil_amts, end_of_training,
                 time_gran_to_new_neighbors_outside, time_gran_to_new_neighbors_dns, time_gran_to_new_neighbors_all):

    #time_gran_to_mod_zscore_df = process_graph_metrics.calculate_mod_zscores_dfs(total_calculated_vals, minimum_training_window,
    #                                                                             training_window_size, time_interval_lengths)

    mod_z_score_df_basefile_name = alert_file + 'mod_z_score_' + sub_path
    #z_score_df_basefile_name = alert_file + 'norm_z_score_' + sub_path
    robustScaler_df_basefile_name = alert_file + 'robustScaler_score_' + sub_path

    if calc_zscore_p:
        time_gran_to_mod_zscore_df = process_graph_metrics.calc_time_gran_to_mod_zscore_dfs(time_gran_to_feature_dataframe,
                                                                                            training_window_size,
                                                                                            minimum_training_window)

        #print "end_of_training", end_of_training
        #exit(344)
        process_graph_metrics.save_feature_datafames(time_gran_to_mod_zscore_df, mod_z_score_df_basefile_name,
                                                     time_gran_to_attack_labels, time_gran_to_synthetic_exfil_paths_series,
                                                     time_gran_to_list_of_concrete_exfil_paths,
                                                     time_gran_to_list_of_exfil_amts, end_of_training,
                                                     time_gran_to_new_neighbors_outside, time_gran_to_new_neighbors_dns,
                                                     time_gran_to_new_neighbors_all)

        '''
        time_gran_to_zscore_dataframe = process_graph_metrics.calc_time_gran_to_zscore_dfs(time_gran_to_feature_dataframe,
                                                                                           training_window_size,
                                                                                           minimum_training_window)

        process_graph_metrics.save_feature_datafames(time_gran_to_zscore_dataframe, z_score_df_basefile_name,
                                                     time_gran_to_attack_labels, time_gran_to_synthetic_exfil_paths_series,
                                                     time_gran_to_list_of_concrete_exfil_paths,
                                                     time_gran_to_list_of_exfil_amts, end_of_training,
                                                     time_gran_to_new_neighbors_outside, time_gran_to_new_neighbors_dns,
                                                     time_gran_to_new_neighbors_all)
        '''
        '''
        time_gran_to_RobustScaler_df = process_graph_metrics.calc_time_gran_to_robustScaker_dfs(time_gran_to_feature_dataframe, training_window_size)

        process_graph_metrics.save_feature_datafames(time_gran_to_RobustScaler_df, robustScaler_df_basefile_name,
                                                     time_gran_to_attack_labels, time_gran_to_synthetic_exfil_paths_series,
                                                     time_gran_to_list_of_concrete_exfil_paths,
                                                     time_gran_to_list_of_exfil_amts, end_of_training,
                                                     time_gran_to_new_neighbors_outside, time_gran_to_new_neighbors_dns,
                                                     time_gran_to_new_neighbors_all)
        '''
    else:
        #time_gran_to_zscore_dataframe = {}
        time_gran_to_mod_zscore_df = {}
        #time_gran_to_RobustScaler_df = {}
        for interval in time_gran_to_feature_dataframe.keys():
            #time_gran_to_zscore_dataframe[interval] = pd.read_csv(z_score_df_basefile_name + str(interval) + '.csv', na_values='?')
            time_gran_to_mod_zscore_df[interval] = pd.read_csv(mod_z_score_df_basefile_name + str(interval) + '.csv', na_values='?')
            #time_gran_to_RobustScaler_df[interval] = pd.read_csv(robustScaler_df_basefile_name + str(interval) + '.csv', na_values='?')

            try:
                pass
                '''
                del time_gran_to_zscore_dataframe[interval]['exfil_path']
                del time_gran_to_mod_zscore_df[interval]['exfil_path']

                del time_gran_to_zscore_dataframe[interval]['concrete_exfil_path']
                del time_gran_to_mod_zscore_df[interval]['concrete_exfil_path']

                del time_gran_to_zscore_dataframe[interval]['exfil_weight']
                del time_gran_to_mod_zscore_df[interval]['exfil_weight']


                del time_gran_to_zscore_dataframe[interval]['exfil_pkts']
                del time_gran_to_mod_zscore_df[interval]['exfil_pkts']
                '''
                #del time_gran_to_zscore_dataframe[interval]['is_test']
                #del time_gran_to_mod_zscore_df[interval]['is_test']

                #def time_gran_to_RobustScaler_df[interval]['exfil_path']
                #del time_gran_to_RobustScaler_df[interval]['concrete_exfil_path']
                #del time_gran_to_RobustScaler_df[interval]['exfil_weight']
                #del time_gran_to_RobustScaler_df[interval]['exfil_pkts']
                #del time_gran_to_RobustScaler_df[interval]['is_test']

                ''' # we'll drop these in the other part of the program...
                del time_gran_to_mod_zscore_df[interval]['new_neighbors_dns']
                del time_gran_to_mod_zscore_df[interval]['new_neighbors_all']
                del time_gran_to_mod_zscore_df[interval]['new_neighbors_outside']

                del time_gran_to_zscore_dataframe[interval]['new_neighbors_dns']
                del time_gran_to_zscore_dataframe[interval]['new_neighbors_all']
                del time_gran_to_zscore_dataframe[interval]['new_neighbors_outside']
                '''
            except:
                pass

    return time_gran_to_mod_zscore_df, None, None# time_gran_to_RobustScaler_df # todo<-- put back
    #return time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, None# time_gran_to_RobustScaler_df # todo<-- put back
    #return time_gran_to_RobustScaler_df, time_gran_to_zscore_dataframe, time_gran_to_RobustScaler_df

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
        print synthetic_exfil_path, synthetic_exfil_path in current_exfil_paths
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
                        print time_gran_to_attack_labels[largest_time_gran],i,len(time_gran_to_attack_labels[largest_time_gran])
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
    print "time_of_synethic_exfil",time_of_synethic_exfil
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
    print "second_counter!!", counter, "attacks_to_assign",len(synthetic_exfil_paths_test), time_gran_to_attack_labels[time_gran][counter:],time_gran_to_attack_labels[time_gran][counter:].count(0)
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
class data_anylsis_pipline(object):
    def __init__(self, pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths, ms_s,
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
                               skip_model_part=False, max_number_of_paths=None, netsec_policy=None):
        self.ms_s = ms_s
        print "log file can be found at: " + str(basefile_name) + '_logfile.log'
        logging.basicConfig(filename=basefile_name + '_logfile.log', level=logging.INFO)
        logging.info('run_data_anaylsis_pipeline Started')

        if 'kube-dns' not in ms_s:
            self.ms_s.append('kube-dns')  # going to put this here so I don't need to re-write all the recipes...

        gc.collect()

        print "starting pipeline..."

        # sub_path = 'sub_only_edge_corr_'  # NOTE: make this an empty string if using the full pipeline (and not the subset)
        # sub_path = 'sub_only_ide_'  # NOTE: make this an empty string if using the full pipeline (and not the subset)
        ### TODO put VVV back in...
        self.sub_path = 'sub_'  # NOTE: make this an empty string if using the full pipeline (and not the subset)
        self.mapping, self.list_of_infra_services = create_mappings(is_swarm, container_info_path, kubernetes_svc_info,
                                                          kubernetes_pod_info, cilium_config_path, ms_s)

        self.calc_zscore_p=calc_zscore_p
        self.is_swarm = is_swarm
        self.container_info_path = container_info_path
        self.kubernetes_svc_info = kubernetes_svc_info
        self.kubernetes_pod_info = kubernetes_pod_info
        self.cilium_config_path = cilium_config_path
        self.time_interval_lengths = time_interval_lengths
        self.basegraph_name = basegraph_name
        self.window_size = window_size
        self.colors = colors
        self.exfil_start_time = exfil_start_time
        self.exfil_end_time = exfil_end_time
        self.minimum_training_window = minimum_training_window
        self.experiment_folder_path = basefile_name.split('edgefiles')[0]
        self.pcap_file = pcap_paths[0].split('/')[-1]  # NOTE: assuming only a single pcap file...
        self.exp_name = basefile_name.split('/')[-1]
        self.base_exp_name = self.exp_name
        self.make_edgefiles_p = make_edgefiles_p and only_exp_info
        self.netsec_policy = netsec_policy
        self.make_edgefiles_p=make_edgefiles_p
        self.graph_p = graph_p
        self.sensitive_ms = None
        self.time_of_synethic_exfil = time_of_synethic_exfil
        self.injected_exfil_path = injected_exfil_path
        self.make_net_graphs_p=make_net_graphs_p
        self.alert_file=alert_file
        self.wiggle_room=wiggle_room
        self.sec_between_exfil_events=sec_between_exfil_events
        self.orig_alert_file = self.alert_file
        self.orig_basegraph_name = self.basegraph_name
        self.orig_exp_name = self.exp_name

        self.synthetic_exfil_paths = None
        self.initiator_info_for_paths = None
        self.training_window_size = training_window_size
        self.size_of_neighbor_training_window = size_of_neighbor_training_window
        print training_window_size,size_of_neighbor_training_window
        self.system_startup_time = training_window_size + size_of_neighbor_training_window
        self.calc_vals = calc_vals

        self.time_gran_to_feature_dataframe=None
        self.time_gran_to_attack_labels=None
        self.time_gran_to_synthetic_exfil_paths_series=None
        self.time_gran_to_list_of_concrete_exfil_paths  = None
        self.time_gran_to_list_of_exfil_amts=None
        self.time_gran_to_new_neighbors_outside=None
        self.time_gran_to_new_neighbors_dns=None
        self.time_gran_to_new_neighbors_all=None

        for ms in ms_s:
            if 'user' in ms and 'db' in ms:
                self.sensitive_ms = ms
            if 'my-release' in ms:
                self.sensitive_ms = ms

        self.process_pcaps()

    def generate_synthetic_exfil_paths(self, max_number_of_paths):
        self.netsec_policy = gen_attack_templates.parse_netsec_policy(self.netsec_policy)
        synthetic_exfil_paths, initiator_info_for_paths = \
            gen_attack_templates.generate_synthetic_attack_templates(self.mapping, self.ms_s, self.sensitive_ms,
                                                                     max_number_of_paths, self.netsec_policy)
        self.synthetic_exfil_paths = synthetic_exfil_paths
        self.initiator_info_for_paths = initiator_info_for_paths
        return synthetic_exfil_paths, initiator_info_for_paths

    def process_pcaps(self):
        self.interval_to_filenames = process_pcap.process_pcap(self.experiment_folder_path, self.pcap_file, self.time_interval_lengths,
                                                          self.exp_name, self.make_edgefiles_p, self.mapping)

    def get_exp_info(self):
        time_grans = [int(i) for i in self.interval_to_filenames.keys()]
        smallest_time_gran = min(time_grans)
        self.smallest_time_gran = smallest_time_gran
        self.total_experiment_length = len(self.interval_to_filenames[str(smallest_time_gran)]) * smallest_time_gran
        print "about to return from only_exp_info section", self.total_experiment_length, self.exfil_start_time, self.exfil_end_time, \
            self.system_startup_time, None
        #return total_experiment_length, self.exfil_start_time, self.exfil_end_time, self.system_startup_time
        return self.total_experiment_length, self.exfil_start_time, self.exfil_end_time, self.system_startup_time

    def calculate_values(self,end_of_training, synthetic_exfil_paths_train, synthetic_exfil_paths_test, fraction_of_edge_weights, fraction_of_edge_pkts):
        self.end_of_training = end_of_training
        if self.calc_vals or self.graph_p:
            # TODO: 90% sure that there is a problem with this function...
            # largest_interval = int(min(interval_to_filenames.keys()))
            exp_length = len(self.interval_to_filenames[str(self.smallest_time_gran)]) * self.smallest_time_gran
            print "exp_length_ZZZ", exp_length, type(exp_length)
            # if not skip_model_part:
            time_gran_to_attack_labels = process_graph_metrics.generate_time_gran_to_attack_labels(
                self.time_interval_lengths,
                self.exfil_start_time, self.exfil_end_time,
                self.sec_between_exfil_events,
                exp_length)
            # else:
            # time_gran_to_attack_labels = {}
            # for time_gran in time_interval_lengths:
            #    time_gran_to_attack_labels[time_gran] = [(1,1)]
            # pass

            # print "interval_to_filenames_ZZZ",interval_to_filenames
            for interval, filenames in self.interval_to_filenames.iteritems():
                print "interval_ZZZ", interval, len(filenames)
            for time_gran, attack_labels in time_gran_to_attack_labels.iteritems():
                print "time_gran_right_after_creation", time_gran, "len of attack labels", len(attack_labels)

            print self.interval_to_filenames, type(self.interval_to_filenames), 'stufff', self.interval_to_filenames.keys()

            # most of the parameters are kinda arbitrary ATM...
            print "INITIAL time_gran_to_attack_labels", time_gran_to_attack_labels
            ## okay, I'll probably wanna write tests for the below function, but it seems to be working pretty well on my
            # informal tests...
            end_of_training = end_of_training
            synthetic_exfil_paths = []
            for path in synthetic_exfil_paths_train + synthetic_exfil_paths_test:
                if path not in synthetic_exfil_paths:
                    synthetic_exfil_paths.append(path)

            print "synthetic_exfil_paths_train", synthetic_exfil_paths_train
            print "synthetic_exfil_paths_test", synthetic_exfil_paths_test
            print "synthetic_exfil_paths", synthetic_exfil_paths
            time_gran_to_attack_labels, time_gran_to_attack_ranges, time_gran_to_physical_attack_ranges = \
                determine_attacks_to_times(time_gran_to_attack_labels, synthetic_exfil_paths,
                                           time_of_synethic_exfil=self.time_of_synethic_exfil,
                                           min_starting=self.system_startup_time, end_of_train=end_of_training,
                                           synthetic_exfil_paths_train=synthetic_exfil_paths_train,
                                           synthetic_exfil_paths_test=synthetic_exfil_paths_test)
            print "time_gran_to_attack_labels", time_gran_to_attack_labels
            print "time_gran_to_attack_ranges", time_gran_to_attack_ranges
            # time.sleep(50)

            time_gran_to_synthetic_exfil_paths_series = determine_time_gran_to_synthetic_exfil_paths_series(
                time_gran_to_attack_ranges,
                synthetic_exfil_paths, self.interval_to_filenames,
                time_gran_to_physical_attack_ranges, self.injected_exfil_path)

            print "time_gran_to_synthetic_exfil_paths_series", time_gran_to_synthetic_exfil_paths_series
            # time.sleep(50)

            # exit(200) ## TODO ::: <<<---- remove!!
            ### OKAY, this is where I'd need to add in the component that loops over the various injected exfil weights
            # OKAY, let's verify that this determine_attacks_to_times function is wokring before moving on to the next one...
            total_calculated_vals, time_gran_to_list_of_concrete_exfil_paths, time_gran_to_list_of_exfil_amts, \
            time_gran_to_new_neighbors_outside, time_gran_to_new_neighbors_dns, time_gran_to_new_neighbors_all = \
                calculate_raw_graph_metrics(self.time_interval_lengths, self.interval_to_filenames, self.ms_s, self.basegraph_name,
                                            self.calc_vals,
                                            self.window_size, self.mapping, self.is_swarm, self.make_net_graphs_p,
                                            self.list_of_infra_services,
                                            synthetic_exfil_paths, self.initiator_info_for_paths, time_gran_to_attack_ranges,
                                            fraction_of_edge_weights, fraction_of_edge_pkts,
                                            self.size_of_neighbor_training_window)

            time_gran_to_feature_dataframe = process_graph_metrics.generate_feature_dfs(total_calculated_vals,
                                                                                        self.time_interval_lengths)

            process_graph_metrics.save_feature_datafames(time_gran_to_feature_dataframe, self.alert_file + self.sub_path,
                                                         time_gran_to_attack_labels,
                                                         time_gran_to_synthetic_exfil_paths_series,
                                                         time_gran_to_list_of_concrete_exfil_paths,
                                                         time_gran_to_list_of_exfil_amts,
                                                         int(end_of_training), time_gran_to_new_neighbors_outside,
                                                         time_gran_to_new_neighbors_dns, time_gran_to_new_neighbors_all)

            analysis_pipeline.generate_graphs.generate_feature_multitime_boxplots(total_calculated_vals, self.basegraph_name,
                                                                                  self.window_size, self.colors,
                                                                                  self.time_interval_lengths,
                                                                                  self.exfil_start_time, self.exfil_end_time,
                                                                                  self.wiggle_room)


        else:
            time_gran_to_feature_dataframe = {}
            time_gran_to_attack_labels = {}
            time_gran_to_synthetic_exfil_paths_series = {}
            time_gran_to_list_of_concrete_exfil_paths = {}
            time_gran_to_list_of_exfil_amts = {}
            time_gran_to_new_neighbors_outside, time_gran_to_new_neighbors_dns, time_gran_to_new_neighbors_all = {}, {}, {}
            min_interval = min(self.time_interval_lengths)
            for interval in self.time_interval_lengths:
                # if interval in time_interval_lengths:
                print "time_interval_lengths", self.time_interval_lengths, "interval", interval
                print "feature_df_path", self.alert_file + self.sub_path + str(interval) + '.csv'
                time_gran_to_feature_dataframe[interval] = pd.read_csv(self.alert_file + self.sub_path + str(interval) + '.csv',
                                                                       na_values='?')
                # time_gran_to_feature_dataframe[interval] = time_gran_to_feature_dataframe[interval].apply(lambda x: np.real(x))
                print "dtypes_of_df", time_gran_to_feature_dataframe[interval].dtypes
                time_gran_to_attack_labels[interval] = time_gran_to_feature_dataframe[interval]['labels']
                try:
                    time_gran_to_new_neighbors_outside[interval] = time_gran_to_feature_dataframe[interval][
                        'new_neighbors_outside']
                    time_gran_to_new_neighbors_dns[interval] = time_gran_to_feature_dataframe[interval][
                        'new_neighbors_dns']
                    time_gran_to_new_neighbors_all[interval] = time_gran_to_feature_dataframe[interval][
                        'new_neighbors_all']
                except:
                    time_gran_to_new_neighbors_outside[interval] = [[] for i in
                                                                    range(0, len(time_gran_to_attack_labels[interval]))]
                    time_gran_to_new_neighbors_dns[interval] = [[] for i in
                                                                range(0, len(time_gran_to_attack_labels[interval]))]
                    time_gran_to_new_neighbors_all[interval] = [[] for i in
                                                                range(0, len(time_gran_to_attack_labels[interval]))]

                time_gran_to_synthetic_exfil_paths_series[interval] = time_gran_to_feature_dataframe[interval][
                    'exfil_path']
                ##recover time_gran_to_list_of_concrete_exfil_paths, time_gran_to_list_of_exfil_amts
                time_gran_to_list_of_concrete_exfil_paths[interval] = time_gran_to_feature_dataframe[interval][
                    'concrete_exfil_path']
                list_of_exfil_amts = []
                for counter in range(0, len(time_gran_to_feature_dataframe[interval]['exfil_weight'])):
                    weight = time_gran_to_feature_dataframe[interval]['exfil_weight'][counter]
                    pkts = time_gran_to_feature_dataframe[interval]['exfil_pkts'][counter]
                    current_exfil_dict = {'weight': weight, 'frames': pkts}
                    list_of_exfil_amts.append(current_exfil_dict)
                time_gran_to_list_of_exfil_amts[interval] = list_of_exfil_amts
                if min_interval:
                    print time_gran_to_feature_dataframe[interval]['is_test'], type(
                        time_gran_to_feature_dataframe[interval]['is_test'])
                    self.end_of_training = time_gran_to_feature_dataframe[interval]['is_test'].tolist().index(
                        1) * min_interval

        print "about to calculate some alerts!"

        self.time_gran_to_feature_dataframe_copy = copy.deepcopy(time_gran_to_feature_dataframe)
        for time_gran, feature_dataframe in time_gran_to_feature_dataframe.iteritems():
            try:
                del feature_dataframe['exfil_path']
                del feature_dataframe['exfil_weight']
                del feature_dataframe['exfil_pkts']
                del feature_dataframe['concrete_exfil_path']
                del feature_dataframe['is_test']
            except:
                pass

            try:
                time_gran_to_feature_dataframe[time_gran] = time_gran_to_feature_dataframe[time_gran].drop(
                    columns=[u'new_neighbors_dns'])
            except:
                pass
            try:
                time_gran_to_feature_dataframe[time_gran] = time_gran_to_feature_dataframe[time_gran].drop(
                    columns=[u'new_neighbors_all '])
            except:
                pass
            try:
                time_gran_to_feature_dataframe[time_gran] = time_gran_to_feature_dataframe[time_gran].drop(
                    columns=[u'new_neighbors_outside'])
            except:
                pass
            print "feature_dataframe_columns", time_gran_to_feature_dataframe[time_gran].columns

        self.time_gran_to_feature_dataframe=time_gran_to_feature_dataframe
        self.time_gran_to_attack_labels=time_gran_to_attack_labels
        self.time_gran_to_synthetic_exfil_paths_series=time_gran_to_synthetic_exfil_paths_series
        self.time_gran_to_list_of_concrete_exfil_paths  = time_gran_to_list_of_concrete_exfil_paths
        self.time_gran_to_list_of_exfil_amts=time_gran_to_list_of_exfil_amts
        self.time_gran_to_new_neighbors_outside=time_gran_to_new_neighbors_outside
        self.time_gran_to_new_neighbors_dns=time_gran_to_new_neighbors_dns
        self.time_gran_to_new_neighbors_all=time_gran_to_new_neighbors_all

        return self.calculate_z_scores_and_get_stat_vals()

    def calculate_z_scores_and_get_stat_vals(self):
        time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, time_gran_to_RobustScaler_df = \
            calc_zscores(self.alert_file, self.training_window_size, self.minimum_training_window, self.sub_path,
                         self.time_gran_to_attack_labels,
                         self.time_gran_to_feature_dataframe, self.calc_zscore_p, self.time_gran_to_synthetic_exfil_paths_series,
                         self.time_gran_to_list_of_concrete_exfil_paths, self.time_gran_to_list_of_exfil_amts, self.end_of_training,
                         self.time_gran_to_new_neighbors_outside, self.time_gran_to_new_neighbors_dns,
                         self.time_gran_to_new_neighbors_all)

        print "analysis_pipeline about to return!"

        return time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, self.time_gran_to_feature_dataframe_copy, \
               self.time_gran_to_synthetic_exfil_paths_series, self.end_of_training

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
                              fraction_of_edge_weights=[0.1], fraction_of_edge_pkts=[0.1],
                              calculate_z_scores_p=True):
    ### Okay, so what is needed here??? We need, like, a list of sets of input (appropriate for run_data_analysis_pipeline),
    ### followed by the LASSO stuff, and finally the ROC stuff... okay, let's do this!!!

    # step(0): need to find out the  meta-data for each experiment so we can coordinate the
    # synthetic attack injections between experiments
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

    list_of_optimal_fone_scores_at_exfil_rates = []
    for rate_counter in range(0,len(fraction_of_edge_weights)):
        ## step (1) : iterate through individual experiments...
        ##  # 1a. list of inputs [done]
        ##  # 1b. acculate DFs
        prefix_for_inject_params = 'weight_' + str(fraction_of_edge_weights[rate_counter]) +\
            '_pkt_' + str(fraction_of_edge_pkts[rate_counter]) + '_'
        cur_base_output_name = base_output_name + prefix_for_inject_params
        list_time_gran_to_mod_zscore_df = []
        list_time_gran_to_mod_zscore_df_training = []
        list_time_gran_to_mod_zscore_df_testing = []
        list_time_gran_to_zscore_dataframe = []
        list_time_gran_to_feature_dataframe = []


        ### NOTE: I could modify this portion to loop over several different exfiltration rates...
        ### and then store the resulting dicts in some kinda dict that is indexed by the exfil rate/amt...
        ### my only concern is that it might lead to storing alot of stuff in RAM and also I'm not so sure how
        ### reasonable the rate injector even is ATM...
        experiments_to_exfil_path_time_dicts = []
        starts_of_testing = []

        for counter,experiment_object in enumerate(function_list):
            #time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, time_gran_to_feature_dataframe, _ = func()
            print "exps_exfil_paths[counter]_to_func",exps_exfil_paths[counter], exps_initiator_info

            #''' todo:check if this works and then modify the results accordingly (note: i think this'll require that new vals are calculated...)
            #prefix_for_inject_params
            experiment_object.alert_file = experiment_object.orig_alert_file + prefix_for_inject_params ## TODO
            #experiment_object.basefile_name = experiment_object.basefile_name +  prefix_for_inject_params## TODO
            experiment_object.basegraph_name = experiment_object.orig_basegraph_name + prefix_for_inject_params ## TODO
            experiment_object.exp_name = experiment_object.orig_exp_name + prefix_for_inject_params ## TODO
            experiment_object.calc_zscore_p = calculate_z_scores_p or calc_vals
            #experiment_object.sub_path = None ## TODO: actually might not be needed
            #'''

            time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, time_gran_to_feature_dataframe, _, start_of_testing = \
            experiment_object.calculate_values(end_of_training=end_of_train_portions[counter],
                                               synthetic_exfil_paths_train=training_exfil_paths[counter],
                                               synthetic_exfil_paths_test=testing_exfil_paths[counter],
                                               fraction_of_edge_weights=fraction_of_edge_weights[rate_counter],
                                               fraction_of_edge_pkts=fraction_of_edge_pkts[rate_counter])

            print "exps_exfil_pathas[time_gran_to_mod_zscore_df]", time_gran_to_mod_zscore_df
            print time_gran_to_mod_zscore_df[time_gran_to_mod_zscore_df.keys()[0]].columns.values
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
            aggregate_feature_df.to_csv(cur_base_output_name + 'aggregate_feature_df_at_time_gran_of_' + str(time_gran) + '_sec.csv',
                                        na_rep='?')
        for time_gran, aggregate_feature_df in time_gran_to_aggregate_mod_score_dfs.iteritems():
            aggregate_feature_df.to_csv(cur_base_output_name + 'modz_feat_df_at_time_gran_of_' + str(time_gran) + '_sec.csv',
                                        na_rep='?')

        recipes_used = [recipe.base_exp_name for recipe in function_list]
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
        #time_gran_to_aggreg_feature_dfs
        ## okay, so now us the time to get a little tricky with everything... we gotta generate seperate reports for the different
        ## modls used...

        #'''
        clf = LassoCV(cv=3, max_iter=8000)
        list_of_optimal_fone_scores_at_this_exfil_rates = \
            statistically_analyze_graph_features(time_gran_to_aggregate_mod_score_dfs, ROC_curve_p,
                                             cur_base_output_name + 'lasso_mod_z_',
                                             names, starts_of_testing, path_occurence_training_df,
                                             path_occurence_testing_df, recipes_used, skip_model_part, clf,
                                             ignore_physical_attacks_p, fraction_of_edge_weights[rate_counter],
                                             fraction_of_edge_pkts[rate_counter])
        list_of_optimal_fone_scores_at_exfil_rates.append(list_of_optimal_fone_scores_at_this_exfil_rates)
        '''
        statistically_analyze_graph_features(time_gran_to_aggreg_feature_dfs, ROC_curve_p, base_output_name + 'lasso_raw_',
                                             names, starts_of_testing, path_occurence_training_df,
                                             path_occurence_testing_df, recipes_used, skip_model_part, clf,
                                             ignore_physical_attacks_p)
        #'''
        # lass_feat_sel
        clf = LogisticRegressionCV(penalty="l1", cv=10, max_iter=10000, solver='saga')
        statistically_analyze_graph_features(time_gran_to_aggregate_mod_score_dfs, ROC_curve_p,
                                             cur_base_output_name + 'logistic_l1_mod_z_lass_feat_sel_',
                                             names, starts_of_testing, path_occurence_training_df,
                                             path_occurence_testing_df, recipes_used, skip_model_part, clf,
                                             ignore_physical_attacks_p, fraction_of_edge_weights[rate_counter],
                                             fraction_of_edge_pkts[rate_counter])

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

    # todo: graph f_one versus exfil rates...
    graph_fone_versus_exfil_rate(list_of_optimal_fone_scores_at_exfil_rates, fraction_of_edge_weights,
                                 fraction_of_edge_pkts, time_gran_to_aggregate_mod_score_dfs.keys())

def statistically_analyze_graph_features(time_gran_to_aggregate_mod_score_dfs, ROC_curve_p, base_output_name, names,
                                         starts_of_testing, path_occurence_training_df, path_occurence_testing_df,
                                         recipes_used, skip_model_part, clf, ignore_physical_attacks_p,
                                         fraction_of_edge_weights, fraction_of_edge_pkts):
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
    feature_activation_heatmaps = []
    feature_raw_heatmaps = []
    ideal_thresholds = []
    feature_activation_heatmaps_training, feature_raw_heatmaps_training = [], []
    for time_gran,aggregate_mod_score_dfs in time_gran_to_aggregate_mod_score_dfs.iteritems():
        # drop columns with all identical values b/c they are useless and too many of them makes LASSO wierd
        #aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(aggregate_mod_score_dfs.std()[(aggregate_mod_score_dfs.std() == 0)].index, axis=1)

        '''
        print aggregate_mod_score_dfs.columns
        for column in aggregate_mod_score_dfs.columns:
            if 'coef_of_var_' in column or 'reciprocity' in column or '_density_' in column: # todo
                aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(column, axis=1)
        '''

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

        try:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='Unnamed: 0')
        except:
            pass
        try:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='Communication Between Pods not through VIPs (no abs)_mod_z_score')
        except:
            pass
        try:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='Fraction of Communication Between Pods not through VIPs (no abs)_mod_z_score')
        except:
            pass
        try:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='DNS inside_mod_z_score')
        except:
            pass
        try:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='into_dns_from_outside_mod_z_score')
        except:
            pass
        try:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='DNS outside_mod_z_score')
        except:
            pass
        #'''
        try:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='Angle of DNS edge weight vectors_mod_z_score')
        except:
            pass
        #'''
        #'''
        try:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='Angle of DNS edge weight vectors (w abs)_mod_z_score')
        except:
            pass
        #outside_to_sum_of_max_pod_to_dns_from_each_svc_ratio_mod_z_score
        #'''
        try:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='outside_to_sum_of_max_pod_to_dns_from_each_svc_ratio_mod_z_score')
        except:
            pass
        #sum_of_max_pod_to_dns_from_each_svc_mod_z_score
        try:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='sum_of_max_pod_to_dns_from_each_svc_mod_z_score')
        except:
            pass
        #Communication Not Through VIPs
        try:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='Communication Not Through VIPs')
        except:
            pass
        #'''
        #'Communication Between Pods not through VIPs (no abs)_mod_z_score'
        #'Fraction of Communication Between Pods not through VIPs (no abs)_mod_z_score'

        #'''
        if not skip_model_part:
            ### TODO TODO TODO TODO TODO TODO
            ### todo: might wanna remove? might wanna keep? not sure...
            ### todo: drop the test physical attacks from the test sets...
            #'''
            if ignore_physical_attacks_p:
                aggregate_mod_score_dfs = \
                aggregate_mod_score_dfs[~((aggregate_mod_score_dfs['labels'] == 1) &
                                          ((aggregate_mod_score_dfs['exfil_pkts'] == 0) &
                                           (aggregate_mod_score_dfs['exfil_weight'] == 0)) )]
            #'''
            #####

            aggregate_mod_score_dfs_training = aggregate_mod_score_dfs[aggregate_mod_score_dfs['is_test'] == 0]
            aggregate_mod_score_dfs_testing = aggregate_mod_score_dfs[aggregate_mod_score_dfs['is_test'] == 1]
            time_gran_to_debugging_csv[time_gran] = aggregate_mod_score_dfs.copy(deep=True)
            print "aggregate_mod_score_dfs_training",aggregate_mod_score_dfs_training
            print "aggregate_mod_score_dfs_testing",aggregate_mod_score_dfs_testing
            print aggregate_mod_score_dfs['is_test']
            #exit(344)

        else:
            ## note: generally you'd want to split into test and train sets, but if we're not doing logic
            ## part anyway, we just want quick-and-dirty results, so don't bother (note: so for formal purposes,
            ## DO NOT USE WITHOUT LOGIC CHECKING OR SOLVE THE TRAINING-TESTING split problem)
            aggregate_mod_score_dfs_training, aggregate_mod_score_dfs_testing = train_test_split(aggregate_mod_score_dfs, test_size=0.5)
            #aggregate_mod_score_dfs_training = aggregate_mod_score_dfs
            #aggregate_mod_score_dfs_testing = aggregate_mod_score_dfs
            time_gran_to_debugging_csv[time_gran] = aggregate_mod_score_dfs_training.copy(deep=True).append(aggregate_mod_score_dfs_testing.copy(deep=True))

        #time_gran_to_debugging_csv[time_gran] = copy.deepcopy(aggregate_mod_score_dfs)

        print aggregate_mod_score_dfs_training.index
        aggregate_mod_score_dfs_training = aggregate_mod_score_dfs_training.drop(columns='new_neighbors_outside')
        aggregate_mod_score_dfs_testing = aggregate_mod_score_dfs_testing.drop(columns='new_neighbors_outside')
        aggregate_mod_score_dfs_training = aggregate_mod_score_dfs_training.drop(columns='new_neighbors_dns')
        aggregate_mod_score_dfs_testing = aggregate_mod_score_dfs_testing.drop(columns='new_neighbors_dns')
        try:
            aggregate_mod_score_dfs_training = aggregate_mod_score_dfs_training.drop(columns=u'new_neighbors_all')
            aggregate_mod_score_dfs_testing = aggregate_mod_score_dfs_testing.drop(columns=u'new_neighbors_all')
        except:
            pass
        try:
            aggregate_mod_score_dfs_training = aggregate_mod_score_dfs_training.drop(columns=u'new_neighbors_all ')
            aggregate_mod_score_dfs_testing = aggregate_mod_score_dfs_testing.drop(columns=u'new_neighbors_all ')
        except:
            pass

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
            #if True:
            print X_train.columns
            ide_train =  copy.deepcopy(X_train['ide_angles_'])
            ide_train.fillna(ide_train.mean())
            print "ide_train", ide_train
            #exit(1222)
            copy_of_X_test = X_test.copy(deep=True)
            ide_test = copy.deepcopy(copy_of_X_test['ide_angles_'])
            ide_test = ide_test.fillna(ide_train.mean())
            print "ide_test",ide_test
            X_train = X_train.drop(columns='ide_angles_')
            X_test = X_test.drop(columns='ide_angles_')
            #if np. ide_test.tolist():
            #    ide_train = [0 for i in range(0, len(X_train))]
            #    ide_test = [0 for i in range(0, len(X_test))]
        except:
            try:
                #ide_train = copy.deepcopy(X_train['ide_angles_mod_z_score'])
                ide_train = copy.deepcopy(X_train['ide_angles (w abs)_mod_z_score'])
                X_train = X_train.drop(columns='ide_angles_mod_z_score')
                X_train = X_train.drop(columns='ide_angles (w abs)_mod_z_score')
                ide_train.fillna(ide_train.mean())
                print "ide_train", ide_train
                # exit(1222)
            except:
                ide_train = [0 for i in range(0,len(X_train))]
            try:
                #copy_of_X_test = X_test.copy(deep=True)
                #ide_test = copy.deepcopy(copy_of_X_test['ide_angles_mod_z_score'])
                ide_test = copy.deepcopy(X_test['ide_angles (w abs)_mod_z_score'])
                X_test = X_test.drop(columns='ide_angles_mod_z_score')
                X_test = X_test.drop(columns='ide_angles (w abs)_mod_z_score')
                ide_test = ide_test.fillna(ide_train.mean())
            except:
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

        #'''
        print "X_train_columns", X_train.columns, "---"
        try:
            ## TODO: probably wanna keep the outside_mod_z_score in...
            dropped_feature_list = ['New Class-Class Edges with DNS_mod_z_score',
                                    'New Class-Class Edges_mod_z_score',
                                    'New Class-Class Edges with Outside_mod_z_score',
                                    '1-step-induced-pod density_mod_z_score']
            X_train = X_train.drop(columns='New Class-Class Edges with DNS_mod_z_score')
            X_train = X_train.drop(columns='New Class-Class Edges with Outside_mod_z_score')
            X_train = X_train.drop(columns='New Class-Class Edges_mod_z_score')
            X_train = X_train.drop(columns='1-step-induced-pod density_mod_z_score')
            X_test = X_test.drop(columns='New Class-Class Edges with DNS_mod_z_score')
            X_test = X_test.drop(columns='New Class-Class Edges with Outside_mod_z_score')
            X_test = X_test.drop(columns='New Class-Class Edges_mod_z_score')
            X_test = X_test.drop(columns='1-step-induced-pod density_mod_z_score')

        except:
            dropped_feature_list = ['New Class-Class Edges with DNS_', 'New Class-Class Edges with Outside_',
                                    'New Class-Class Edges_',
                                    '1-step-induced-pod density_']
            print "X_train_columns",X_train.columns, "---"
            try:
                X_train = X_train.drop(columns='New Class-Class Edges with DNS_')
            except:
                pass
            try:
                X_train = X_train.drop(columns='New Class-Class Edges with Outside_')
            except:
                pass
            try:
                X_train = X_train.drop(columns='New Class-Class Edges_')
            except:
                pass
            try:
                X_train = X_train.drop(columns='1-step-induced-pod density_')
            except:
                pass
            try:
                X_test = X_test.drop(columns='New Class-Class Edges with DNS_')
            except:
                pass
            try:
                X_test = X_test.drop(columns='New Class-Class Edges with Outside_')
            except:
                pass
            try:
                X_test = X_test.drop(columns='New Class-Class Edges_')
            except:
                pass
            try:
                X_test = X_test.drop(columns='1-step-induced-pod density_')
            except:
                pass
        #'''

        print '-------'
        print type(X_train)
        print "X_train_columns_values", X_train.columns.values
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

        print X_train
        #exit(233)
        pre_drop_X_train = X_train.copy(deep=True)
        X_train = X_train.dropna(axis=1)
        print X_train
        #exit(233)
        X_test = X_test.dropna(axis=1)

        ## TODO: okay, let's try to use the lasso for feature selection by logistic regresesion for the actual model...
        ''' ## TODO: seperate feature selection step goes here...
        clf_featuree_selection = LassoCV(cv=5)
        # Set a minimum threshold of 0.25
        sfm = sklearn.SelectFromModel(clf_featuree_selection)
        sfm.fit(X, y)
        n_features = sfm.transform(X).shape[1]
        '''
        if 'lass_feat_sel' in base_output_name:
            clf_featuree_selection = LassoCV(cv=5)
            sfm = sklearn.feature_selection.SelectFromModel(clf_featuree_selection)
            sfm.fit(X_train, y_train)
            feature_idx = sfm.get_support()
            selected_columns = X_train.columns[feature_idx]
            X_train = pd.DataFrame(sfm.transform(X_train),index=X_train.index,columns=selected_columns)
            X_test = pd.DataFrame(sfm.transform(X_test),index=X_test.index,columns=selected_columns)
            #X_test = sfm.transform(X_test)


        dropped_columns = list(pre_drop_X_train.columns.difference(X_train.columns))
        print "dropped_columns", dropped_columns
        #exit(233)
        #dropped_columns=[]

        print "columns", X_train.columns
        print "columns", X_test.columns

        X_train_dtypes = X_train.dtypes
        X_train_columns = X_train.columns.values
        X_test_columns = X_test.columns.values
        print y_test
        number_attacks_in_test = len(y_test[y_test['labels'] == 1])
        number_non_attacks_in_test = len(y_test[y_test['labels'] == 0])
        percent_attacks.append(float(number_attacks_in_test) / (number_non_attacks_in_test + number_attacks_in_test))

        print y_train
        number_attacks_in_train = len(y_train[y_train['labels'] == 1])
        number_non_attacks_in_train = len(y_train[y_train['labels'] == 0])
        print number_non_attacks_in_train,number_attacks_in_train
        list_percent_attacks_training.append(float(number_attacks_in_train) / (number_non_attacks_in_train + number_attacks_in_train))

        #print "X_train", X_train
        #print "y_train", y_train, len(y_train)
        #print "y_test", y_test, len(y_test)
        #print "-- y_train", len(y_train), "y_test", len(y_test), "time_gran", time_gran, "--"


        ### train the model and generate predictions (2B)
        # note: I think this is where I'd need to modify it to make the anomaly-detection using edge correlation work...

        #clf = sklearn.tree.DecisionTreeClassifier()
        #clf = RandomForestClassifier(n_estimators=10)
        #clf = clf.fit(X_train, y_train)

        #clf = ElasticNetCV(l1_ratio=1.0)
        #clf = RidgeCV(cv=10) ## TODO TODO TODO <<-- instead of having choosing the alpha be magic, let's use cross validation to choose it instead...
        #alpha = 5 # note: not used unless the line underneath is un-commented...
        #clf=Lasso(alpha=alpha)
        print X_train.dtypes
        print y_train

        clf.fit(X_train, y_train)
        score_val = clf.score(X_test, y_test)
        print "score_val", score_val
        test_predictions = clf.predict(X=X_test)
        train_predictions = clf.predict(X=X_train)
        #coefficients = pd.DataFrame({"Feature": X.columns, "Coefficients": np.transpose(clf.coef_)})
        #print coefficients
        #clf.coef_, "intercept", clf.intercept_

        coef_dict = {}
        #coef_feature_df = pd.DataFrame() # TODO: remove if we go back to LASSO
        #'''
        ### get the coefficients used in the model...
        print "Coefficients: "
        print "LASSO model", clf.get_params()
        print '----------------------'
        print len(time_gran_to_debugging_csv[time_gran]["labels"]), len(np.concatenate([train_predictions, test_predictions]))
        print len(time_gran_to_debugging_csv[time_gran].index)
        if not skip_model_part:
            time_gran_to_debugging_csv[time_gran].loc[:, "aggreg_anom_score"] = np.concatenate([train_predictions, test_predictions])
        else:
            #time_gran_to_debugging_csv[time_gran].loc[:, "aggreg_anom_score"] = test_predictions
            time_gran_to_debugging_csv[time_gran].loc[:, "aggreg_anom_score"] = np.concatenate([train_predictions, test_predictions])

        print "len(clf.coef_)", len(clf.coef_), "len(X_train_columns)", len(X_train_columns), "time_gran", time_gran, \
            "len(X_test_columns)", len(X_test_columns), X_train.shape, X_test.shape

        if 'logistic' in base_output_name:
            model_coefs = clf.coef_[0]
        else:
            model_coefs = clf.coef_

        if len(model_coefs) != (len(X_train_columns)): # there is no plus one b/c the intercept is stored in clf.intercept_
            print "coef_ is different length than X_train_columns!", X_train_columns
            for  counter,i in enumerate(X_train_dtypes):
                print counter,i, X_train_columns[counter]
                print model_coefs#[counter]
                print len(model_coefs)
            exit(888)
        for coef, feat in zip(model_coefs, X_train_columns):
            coef_dict[feat] = coef
        print "COEFS_HERE"
        print "intercept...", clf.intercept_
        coef_dict['intercept'] = clf.intercept_
        for coef,feature in coef_dict.iteritems():
            print coef,feature
        #exit(233) ## TODO REMOVE!!!!

        #print "COEF_DICT", coef_dict

        coef_feature_df = pd.DataFrame.from_dict(coef_dict, orient='index')

        #plt.savefig(local_graph_loc, format='png', dpi=1000)
        #print coef_feature_df.columns.values
        #coef_feature_df.index.name = 'Features'
        coef_feature_df.columns = ['Coefficient']
        #'''


        print '--------------------------'

        model_params = clf.get_params()
        try:
            model_params['alpha_val'] = clf.alpha_
        except:
            try:
                pass
                #model_params['alpha_val'] = alpha
            except:
                pass
                #tree.export_graphviz(clf,out_file = 'tree.dot')
                #pass
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

        current_heatmap_val_path = base_output_name + 'coef_val_heatmap_' + str(time_gran) + '.png'
        local_heatmap_val_path = 'temp_outputs/heatmap_coef_val_at_' +  str(time_gran) + '.png'
        current_heatmap_path = base_output_name + 'coef_act_heatmap_' + str(time_gran) + '.png'
        local_heatmap_path = 'temp_outputs/heatmap_coef_contribs_at_' +  str(time_gran) + '.png'
        coef_impact_df, raw_feature_val_df = generate_heatmap.generate_covariate_heatmap(coef_dict, X_test, exfil_paths)
        generate_heatmap.generate_heatmap(coef_impact_df, local_heatmap_path, current_heatmap_path)
        generate_heatmap.generate_heatmap(raw_feature_val_df, local_heatmap_val_path, current_heatmap_val_path)
        feature_activation_heatmaps.append('../' + local_heatmap_path)
        feature_raw_heatmaps.append('../' + local_heatmap_val_path)
        print coef_impact_df
        #exit(233)

        current_heatmap_raw_val_path_training = base_output_name + 'training_raw_val_heatmap_' + str(time_gran) + '.png'
        local_heatmap_raw_val_path_training = 'temp_outputs/training_heatmap_raw_val_at_' +  str(time_gran) + '.png'
        current_heatmap_path_training = base_output_name + 'training_coef_act_heatmap_' + str(time_gran) + '.png'
        local_heatmap_path_training = 'temp_outputs/training_heatmap_coef_contribs_at_' +  str(time_gran) + '.png'
        coef_impact_df, raw_feature_val_df = generate_heatmap.generate_covariate_heatmap(coef_dict, X_train, exfil_paths_train)
        generate_heatmap.generate_heatmap(coef_impact_df, local_heatmap_path_training, current_heatmap_path_training)
        generate_heatmap.generate_heatmap(raw_feature_val_df, local_heatmap_raw_val_path_training, current_heatmap_raw_val_path_training)
        feature_activation_heatmaps_training.append('../' + local_heatmap_path_training)
        feature_raw_heatmaps_training.append('../' + local_heatmap_raw_val_path_training)

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
            #fpr_ewma, tpr_ewma, thresholds_ewma = sklearn.metrics.roc_curve(y_true=y_test, y_score = ewma_test, pos_label=1)
            print "y_test",y_test
            print "ide_test",ide_test, ide_train
            try:
                fpr_ide, tpr_ide, thresholds_ide = sklearn.metrics.roc_curve(y_true=y_test, y_score = ide_test, pos_label=1)
                line_titles = ['ensemble model', 'ide_angles']
                list_of_x_vals = [x_vals, fpr_ide]
                list_of_y_vals = [y_vals, tpr_ide]
            except:
                #ide_test = [0 for i in range(0, len(X_test))]
                #fpr_ide, tpr_ide, thresholds_ide = sklearn.metrics.roc_curve(y_true=y_test, y_score = ide_test, pos_label=1)
                line_titles = ['ensemble model']
                list_of_x_vals = [x_vals]
                list_of_y_vals = [y_vals]

            ax, _, plot_path = generate_alerts.construct_ROC_curve(list_of_x_vals, list_of_y_vals, title, ROC_path + plot_name,\
                                                                   line_titles, show_p=False)
            list_of_rocs.append(plot_path)

            ### determination of the optimal operating point goes here (take all the thresh vals and predictions,
            ### find the corresponding f1 scores (using sklearn func), and then return the best.
            optimal_f1_score, optimal_thresh = process_roc.determine_optimal_threshold(y_test, test_predictions, thresholds)
            ideal_thresholds.append(optimal_thresh)
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

            if not skip_model_part:
                time_gran_to_debugging_csv[time_gran].loc[:, "anom_val_at_opt_pt"] = \
                    np.concatenate([optimal_train_predictions, optimal_predictions])
            else:
                #time_gran_to_debugging_csv[time_gran].loc[:, "anom_val_at_opt_pt"] = optimal_predictions
                time_gran_to_debugging_csv[time_gran].loc[:, "anom_val_at_opt_pt"] = \
                    np.concatenate([optimal_train_predictions, optimal_predictions])

            # I don't want the attributes w/ zero coefficients to show up in the debugging csv b/c it makes it hard to read
            ## TODO
            for feature,coef in coef_dict.iteritems():
                print "coef_check", coef, not coef, feature
                if not coef:
                    print "just_dropped", feature
                    try:
                        time_gran_to_debugging_csv[time_gran] = time_gran_to_debugging_csv[time_gran].drop([feature],axis=1)
                        coef_feature_df = coef_feature_df.drop(feature, axis=0)
                    except:
                        pass
                for dropped_feature in dropped_feature_list + dropped_columns:
                    try:
                        time_gran_to_debugging_csv[time_gran] = time_gran_to_debugging_csv[time_gran].drop([dropped_feature], axis=1)
                    except:
                        pass

            time_gran_to_debugging_csv[time_gran].to_csv(base_output_name + 'DEBUGGING_modz_feat_df_at_time_gran_of_'+\
                                                         str(time_gran) + '_sec.csv', na_rep='?')
            print "ide_angles", ide_train, ide_test

        list_of_feat_coefs_dfs.append(coef_feature_df)

    starts_of_testing_dict = {}
    for counter,name in enumerate(names):
        starts_of_testing_dict[name] = starts_of_testing[counter]

    starts_of_testing_df = pd.DataFrame(starts_of_testing_dict, index=['start_of_testing_phase'])

    print "list_of_rocs", list_of_rocs
    generate_report.generate_report(list_of_rocs, list_of_feat_coefs_dfs, list_of_attacks_found_dfs,
                                    recipes_used, base_output_name, time_grans, list_of_model_parameters,
                                    list_of_optimal_fone_scores, starts_of_testing_df, path_occurence_training_df,
                                    path_occurence_testing_df, percent_attacks, list_of_attacks_found_training_df,
                                    list_percent_attacks_training, feature_activation_heatmaps, feature_raw_heatmaps,
                                    ideal_thresholds, feature_activation_heatmaps_training, feature_raw_heatmaps_training,
                                    fraction_of_edge_weights, fraction_of_edge_pkts)

    print "multi_experiment_pipeline is all done! (NO ERROR DURING RUNNING)"
    #print "recall that this was the list of alert percentiles", percentile_thresholds
    return list_of_optimal_fone_scores

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

def graph_fone_versus_exfil_rate(optimal_fone_scores, exfil_weights_frac, exfil_pkts_frac, time_grans):
    time_gran_to_fone_list = {}
    time_gran_to_exfil_param_list = {}
    for exfil_counter, optimal_fones in enumerate(optimal_fone_scores):
        for timegran_counter, optimal_score in enumerate(optimal_fones):
            if time_grans[timegran_counter] in time_gran_to_fone_list:
                time_gran_to_fone_list[time_grans[timegran_counter]].append(optimal_score)
            else:
                time_gran_to_fone_list[time_grans[timegran_counter]] = [optimal_score]
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