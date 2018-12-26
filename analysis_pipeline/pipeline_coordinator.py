import gc
import pyximport

import analysis_pipeline.generate_alerts
import analysis_pipeline.generate_graphs
from pcap_to_edgelists import create_edgelists,create_mappings
import process_graph_metrics
import generate_alerts
pyximport.install() # to leverage cpython
import simplified_graph_metrics
import process_pcap
import gen_attack_templates
import random
import math
import time

def calculate_raw_graph_metrics(time_interval_lengths, interval_to_filenames, ms_s, basegraph_name, calc_vals, window_size,
                                mapping, is_swarm, make_net_graphs_p, list_of_infra_services,synthetic_exfil_paths,
                                initiator_info_for_paths, time_gran_to_attacks_to_times, fraction_of_edge_weights,
                                fraction_of_edge_pkts):
    total_calculated_vals = {}
    for time_interval_length in time_interval_lengths:
        print "analyzing edgefiles..."
        ### TODO: change back to analyze_edgefiles.pipeline_analysis_step if you want to use the whole pipeline
        newly_calculated_values = simplified_graph_metrics.pipeline_subset_analysis_step(interval_to_filenames[str(time_interval_length)], ms_s,
                                                                                         time_interval_length, basegraph_name, calc_vals, window_size,
                                                                                         mapping, is_swarm, make_net_graphs_p, list_of_infra_services,
                                                                                         synthetic_exfil_paths, initiator_info_for_paths,
                                                                                         time_gran_to_attacks_to_times[time_interval_length],
                                                                                         fraction_of_edge_weights,
                                                                                         fraction_of_edge_pkts)
        total_calculated_vals.update(newly_calculated_values)
        gc.collect()
    exit() ### TODO <---- remove!!!
    return total_calculated_vals

def calc_zscores(total_calculated_vals, time_interval_lengths, alert_file, training_window_size, minimum_training_window,
                 sub_path, time_gran_to_attack_labels, time_gran_to_feature_dataframe):

    time_gran_to_mod_zscore_df = process_graph_metrics.calculate_mod_zscores_dfs(total_calculated_vals, minimum_training_window,
                                                                                 training_window_size, time_interval_lengths)

    process_graph_metrics.save_feature_datafames(time_gran_to_mod_zscore_df, alert_file + 'mod_z_score_' + sub_path,
                                                 time_gran_to_attack_labels)

    time_gran_to_zscore_dataframe = process_graph_metrics.calc_time_gran_to_zscore_dfs(time_gran_to_feature_dataframe,
                                                                                       training_window_size,
                                                                                       minimum_training_window)

    process_graph_metrics.save_feature_datafames(time_gran_to_zscore_dataframe, alert_file + 'norm_z_score_' + sub_path,
                                                 time_gran_to_attack_labels)


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
        while not attack_spot_found:
            potential_starting_point = random.randint(time_periods_startup,
                                            len(time_gran_to_attack_labels[largest_time_gran]) - time_periods_attack)
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
                               fraction_of_edge_weights=0.1, fraction_of_edge_pkts=0.1):
    gc.collect()
    print "starting pipeline..."

    mapping,list_of_infra_services = create_mappings(is_swarm, container_info_path, kubernetes_svc_info,
                                                     kubernetes_pod_info, cilium_config_path, ms_s)
    #interval_to_filenames = create_edgelists(pcap_paths, start_time, make_edgefiles_p, time_interval_lengths, rdpcap_p,
    #                                         basefile_name, mapping)

    experiment_folder_path = basefile_name.split('edgefiles')[0]
    pcap_file = pcap_paths[0].split('/')[-1] # NOTE: assuming only a single pcap file...
    exp_name = basefile_name.split('/')[-1]
    interval_to_filenames = process_pcap.process_pcap(experiment_folder_path, pcap_file, time_interval_lengths,
                                                      exp_name, make_edgefiles_p, mapping)

    # TODO: 90% sure that there is a problem with this function...
    time_gran_to_attack_labels = process_graph_metrics.generate_time_gran_to_attack_labels(time_interval_lengths,
                                                                                           exfil_start_time, exfil_end_time,
                                                                                            sec_between_exfil_events)

    print interval_to_filenames, type(interval_to_filenames), 'stufff', interval_to_filenames.keys()

    # todo: might wanna specify this is in the attack descriptions...
    for ms in ms_s:
        if 'User' in ms:
            sensitive_ms = ms
        if 'my-release' in ms:
            sensitive_ms = ms
    synthetic_exfil_paths, initiator_info_for_paths = gen_attack_templates.generate_synthetic_attack_templates(mapping, ms_s, sensitive_ms)
    #######
    ### TODO: this is where I'd prbobably want to create the synthetic data... the plan would probably be to make copies
    ### of the given sequence, and then inject attacks into it, and I could use a loop over the code below to make it work...
    ########
    ### next steps: build synthetic attacks (leveraging existing work on mulval) and fix time_gran to attack labels
    ### and THEN (waay after): going to want to probably do a big rewrite of the graph metrics calculation stuff...
    ########
    ########################

    # most of the parameters are kinda arbitrary ATM...
    print "INITIAL time_gran_to_attack_labels", time_gran_to_attack_labels
    ## okay, I'll probably wanna write tests for the below function, but it seems to be working pretty well on my
    # informal tests...
    time_gran_to_attack_labels, time_gran_to_attack_ranges = determine_attacks_to_times(time_gran_to_attack_labels,
                                                                                        synthetic_exfil_paths,
                                                                                        time_of_synethic_exfil=time_of_synethic_exfil,
                                                                                        min_starting=training_window_size)
    print "time_gran_to_attack_labels",time_gran_to_attack_labels
    print "time_gran_to_attack_ranges", time_gran_to_attack_ranges
    #time.sleep(50)

    # OKAY, let's verify that this determine_attacks_to_times function is wokring before moving on to the next one...
    total_calculated_vals = calculate_raw_graph_metrics(time_interval_lengths, interval_to_filenames, ms_s, basegraph_name, calc_vals,
                                                        window_size, mapping, is_swarm, make_net_graphs_p, list_of_infra_services,
                                                        synthetic_exfil_paths, initiator_info_for_paths, time_gran_to_attack_ranges,
                                                        fraction_of_edge_weights, fraction_of_edge_pkts)

    exit() ### <<<----- TODO REMOVE
    sub_path = 'sub_'  # NOTE: make this an empty string if using the full pipeline (and not the subset)

    time_gran_to_feature_dataframe = process_graph_metrics.generate_feature_dfs( total_calculated_vals, time_interval_lengths)

    process_graph_metrics.save_feature_datafames(time_gran_to_feature_dataframe, alert_file + sub_path, time_gran_to_attack_labels)

    if graph_p:
        analysis_pipeline.generate_graphs.generate_feature_multitime_boxplots(total_calculated_vals, basegraph_name, window_size, colors, time_interval_lengths,
                                                                              exfil_start_time, exfil_end_time, wiggle_room)

    print "about to calculate some alerts!"

    if calc_zscore_p:
        time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe = \
            calc_zscores(total_calculated_vals, time_interval_lengths, alert_file, training_window_size,
                         minimum_training_window, sub_path, time_gran_to_attack_labels, time_gran_to_feature_dataframe)

    if calc_zscore_p and ROC_curve_p:
        generate_rocs(time_gran_to_mod_zscore_df, alert_file, sub_path)

    print "and analysis_pipeline pipeline is all done!"
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