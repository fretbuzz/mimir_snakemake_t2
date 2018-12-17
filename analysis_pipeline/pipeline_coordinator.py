import csv
import gc
import json
import pyximport
import yaml
from scapy.all import *
import alert_triggers
import analysis_pipeline.generate_graphs
from analysis_pipeline import analyze_edgefiles
from pcap_to_edgelists import create_edgelists

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

def z_scores_and_alerts(total_calculated_vals, percentile_thresholds, anomaly_window, anom_num_outlier_vals_in_window,
                        time_interval_lengths, alert_file, calc_alerts_p, training_window_size, csv_path, minimum_training_window,
                        exfil_start_time, exfil_end_time,sec_between_exfil_events, sub_path):

    # okay, let's modify this a little more... let's get the time_gran_to_feature_dataframe
    # and then calc the z(mod) scores
    # then let's calculate alerts
    # ATM the whole thing is nonsense...

    params_to_alerts, time_gran_to_feature_dataframe, time_gran_to_mod_zscore_df = alert_triggers.compute_alerts(
        total_calculated_vals, percentile_thresholds, anomaly_window, anom_num_outlier_vals_in_window,
        time_interval_lengths, alert_file, calc_alerts_p, training_window_size, csv_path, minimum_training_window)

    time_gran_to_attack_labels = alert_triggers.generate_time_gran_to_attack_labels(time_gran_to_feature_dataframe,
                                                                                    exfil_start_time, exfil_end_time,
                                                                                    sec_between_exfil_events)

    alert_triggers.save_feature_datafames(time_gran_to_feature_dataframe, alert_file + sub_path,
                                          time_gran_to_attack_labels)

    time_gran_feature_dataframe_to_time_gran_z_score_dataframe = alert_triggers.time_gran_feature_dataframe_to_time_gran_z_score_dataframe(
        time_gran_to_feature_dataframe, training_window_size, minimum_training_window)

    alert_triggers.save_feature_datafames(time_gran_feature_dataframe_to_time_gran_z_score_dataframe,
                                          alert_file + 'norm_z_score_' + sub_path, time_gran_to_attack_labels)

    alert_triggers.save_feature_datafames(time_gran_to_mod_zscore_df, alert_file + 'mod_z_score_' + sub_path,
                                          time_gran_to_attack_labels)

    alert_triggers.save_alerts_to_csv(params_to_alerts, alert_file + sub_path, time_gran_to_attack_labels)

    ''' # TODO: probably want to include this section again at some point...
    params_to_tpr_fpr = alert_triggers.calc_all_fp_and_tps(params_to_alerts, exfil_start_time, exfil_end_time,
                                                           wiggle_room, time_gran_to_attack_labels)

    #print "params_to_tpr_fpr", params_to_tpr_fpr
    #time.sleep(300)
    params_to_method_to_tpr_fpr = alert_triggers.organize_tpr_fpr_results(params_to_tpr_fpr, percentile_thresholds)
    alert_triggers.store_organized_tpr_fpr_results(params_to_method_to_tpr_fpr, alert_file + sub_path,
                                                   percentile_thresholds)
    '''
    return time_gran_to_mod_zscore_df, time_gran_feature_dataframe_to_time_gran_z_score_dataframe, params_to_alerts

def generate_rocs(time_gran_to_mod_zscore_df, alert_file, sub_path):
    ''' # TODO: if re-enable the above block, will want to re-enable this one too
     params_to_method_to_tpr_fpr = alert_triggers.read_organized_tpr_fpr_file(alert_file + sub_path + 'tpr_fpr.csv')
     alert_triggers.make_roc_graphs(params_to_method_to_tpr_fpr, alert_file + sub_path + '_ROC')
     '''
    # note: this new function is much better, so I might wanna use it instead...
    # TODO: make the csv/datagram to send to the next function
    for time_gran, mod_zscore_df in time_gran_to_mod_zscore_df.iteritems():
        df_with_anom_features = mod_zscore_df  # pd.concat([mod_zscore_df, time_gran_to_feature_dataframe[time_gran]])
        # print "df_with_anom_features",df_with_anom_features
        next_gen_metrics.next_gen_ROCS(df_with_anom_features, time_gran, alert_file, sub_path)
    '''
    for time_gran, mod_zscore_df in time_gran_to_mod_zscore_df.iteritems():
        df_with_anom_features = mod_zscore_df # TODO construct w/ the values that I want it to have...
        features_to_use = ['New Class-Class Edges50_5__mod_z_score',
                           'Communication Between Pods not through VIPs (no abs)50_5__mod_z_score',
                           'DNS outside-to-inside ratio50_5__mod_z_score'] # TODO: fill out (will need to figure out what the others will be first tho...)
        weights = {'New Class-Class Edges50_5__mod_z_score': 1,
                   'Communication Between Pods not through VIPs (no abs)50_5__mod_z_score': 1,
                   'DNS outside-to-inside ratio50_5__mod_z_score': 4} # TODO: fill out (will need to figure out what the others will be first tho...)
        ROC_path = alert_file + sub_path + '_good_roc_'
        cur_alert_function = partial(next_gen_metrics.alert_fuction, weights, features_to_use)
        title = 'ROC Linear Combination of Features at ' + str(time_gran)
        plot_name = 'sub_roc_lin_comb_features_' + str(time_gran)
        alert_triggers.create_ROC_of_anom_score(df_with_anom_features, time_gran, ROC_path, cur_alert_function, title, plot_name)

        for feature in features_to_use:
            title = 'ROC ' + feature + ' at ' + str(time_gran)
            plot_name = 'sub_roc_' + feature + '_' + str(time_gran)
            cur_alert_function = partial(next_gen_metrics.alert_fuction, weights, [feature])
            alert_triggers.create_ROC_of_anom_score(df_with_anom_features, time_gran, ROC_path, cur_alert_function,
                                                    title, plot_name)
    '''

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
                               alert_file = None, ROC_curve_p=False, calc_tpr_fpr_p=False, calc_packet_vals_p = False,
                               training_window_size=200, minimum_training_window=5, sec_between_exfil_events=1):
                                # <--- training window size is going to be forty somewhat arbitrarily

    gc.collect()

    interval_to_filenames, mapping, list_of_infra_services = create_edgelists(pcap_paths, is_swarm, ms_s, kubernetes_pod_info,
                                                                              cilium_config_path,  start_time, make_edgefiles_p,
                                                                              time_interval_lengths, rdpcap_p, basefile_name)

    total_calculated_vals = calculate_raw_graph_metrics(time_interval_lengths, interval_to_filenames, ms_s, basegraph_name, calc_vals,
                                                        window_size, mapping, is_swarm, make_net_graphs_p, list_of_infra_services)

    if graph_p:
        analysis_pipeline.generate_graphs.generate_feature_multitime_boxplots(total_calculated_vals, basegraph_name, window_size, colors, time_interval_lengths,
                                                                              exfil_start_time, exfil_end_time, wiggle_room)


    print "about to calculate some alerts!"
    sub_path = 'sub_'  # NOTE: make this an empty string if using the full pipeline (and not the subset)


    if calc_tpr_fpr_p:
        csv_path = alert_file + 'features_'
        time_gran_to_mod_zscore_df, time_gran_feature_dataframe_to_time_gran_z_score_dataframe, params_to_alerts = \
            z_scores_and_alerts(total_calculated_vals, percentile_thresholds, anomaly_window, anom_num_outlier_vals_in_window,
                        time_interval_lengths, alert_file, calc_alerts_p, training_window_size, csv_path, minimum_training_window,
                        exfil_start_time, exfil_end_time,sec_between_exfil_events, sub_path)

    if ROC_curve_p:
        generate_rocs(time_gran_to_mod_zscore_df, alert_file, sub_path)

    print "and analysis_pipeline pipeline is all done!"
    #print "recall that this was the list of alert percentiles", percentile_thresholds