import csv
import gc
import json
import pyximport
import yaml
from scapy.all import *

import analysis_pipeline.generate_alerts
import analysis_pipeline.generate_graphs
from pcap_to_edgelists import create_edgelists
import process_graph_metrics
import generate_alerts

pyximport.install() # am I sure that I want this???
import simplified_graph_metrics
import next_gen_metrics

def calculate_raw_graph_metrics(time_interval_lengths, interval_to_filenames, ms_s, basegraph_name, calc_vals, window_size,
                                mapping, is_swarm, make_net_graphs_p, list_of_infra_services):
    total_calculated_vals = {}
    for time_interval_length in time_interval_lengths:
        print "analyzing edgefiles..."
        ### TODO: change back to analyze_edgefiles.pipeline_analysis_step if you want to use the whole pipeline
        newly_calculated_values = simplified_graph_metrics.pipeline_subset_analysis_step(interval_to_filenames[str(time_interval_length)], ms_s,
                                                                                         time_interval_length, basegraph_name, calc_vals, window_size,
                                                                                         mapping, is_swarm, make_net_graphs_p, list_of_infra_services)
        total_calculated_vals.update(newly_calculated_values)
        gc.collect()
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


## TODO: this function is an atrocity and should be converted into a snakemake spec so we can use that instead...###
# run_data_anaylsis_pipeline : runs the whole analysis_pipeline pipeline (or a part of it)
# (1) creates edgefiles, (2) creates communication graphs from edgefiles, (3) calculates (and stores) graph metrics
# (4) makes graphs of the graph metrics
# Note: see run_analysis_pipeline_recipes for pre-configured sets of parameters (there are rather a lot)
def run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles_p, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time = None, end_time = None, calc_vals=True,
                               graph_p = True, kubernetes_svc_info=None, make_net_graphs_p=False, cilium_config_path=None,
                               rdpcap_p=False, kubernetes_pod_info=None, calc_alerts_p=False,
                               percentile_thresholds=None, anomaly_window = None, anom_num_outlier_vals_in_window = None,
                               alert_file = None, ROC_curve_p=False, calc_zscore_p=False, calc_packet_vals_p = False,
                               training_window_size=200, minimum_training_window=5, sec_between_exfil_events=1):
                                # <--- training window size is going to be forty somewhat arbitrarily

    gc.collect()

    interval_to_filenames, mapping, list_of_infra_services = create_edgelists(pcap_paths, is_swarm, ms_s, kubernetes_pod_info,
                                                                              cilium_config_path,  start_time, make_edgefiles_p,
                                                                              time_interval_lengths, rdpcap_p, basefile_name)

    total_calculated_vals = calculate_raw_graph_metrics(time_interval_lengths, interval_to_filenames, ms_s, basegraph_name, calc_vals,
                                                        window_size, mapping, is_swarm, make_net_graphs_p, list_of_infra_services)

    # TODO: 90% sure that there is a problem with this function...
    time_gran_to_attack_labels = process_graph_metrics.generate_time_gran_to_attack_labels( time_interval_lengths,
                                                                                            exfil_start_time, exfil_end_time, sec_between_exfil_events)
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