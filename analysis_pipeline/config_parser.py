import json
import pyximport
pyximport.install()
import sys
import matplotlib
matplotlib.use('Agg',warn=False, force=True)
from pipeline_coordinator import multi_experiment_pipeline
from single_experiment_pipeline import data_anylsis_pipline
import argparse
import os
import time

'''
This file is essentially just sets of parameters for the run_data_analysis_pipeline function in pipeline_coordinator.py
There are a lot of parameters, and some of them are rather long, so I decided to make a function to store them in
'''

def parse_experimental_data_json(config_file, experimental_folder, experiment_name, make_edgefiles,
                                 time_interval_lengths, pcap_file_path, pod_creation_log_path,
                                 netsec_policy=None, time_of_synethic_exfil=None):
    with open(config_file) as f:
        config_file = json.load(f)
        basefile_name = experimental_folder + experiment_name + '/edgefiles/' + experiment_name + '_'
        basegraph_name = experimental_folder + experiment_name + '/graphs/' + experiment_name + '_'
        alert_file = experimental_folder + experiment_name + '/alerts/' + experiment_name + '_'
        base_experiment_dir =  experimental_folder + experiment_name + '/'

        sec_between_exfil_pkts = config_file["exfiltration_info"]['sec_between_exfil_pkts']
        pcap_paths = [ pcap_file_path + config_file['pcap_file_name'] ]
        pod_creation_log = [ pod_creation_log_path + config_file['pod_creation_log_name']]
        sensitive_ms = config_file["exfiltration_info"]['sensitive_ms']

        exfil_StartEnd_times = config_file["exfiltration_info"]['exfil_StartEnd_times']
        physical_exfil_paths = config_file["exfiltration_info"]['exfil_paths']

        pipeline_object = data_anylsis_pipline(pcap_paths=pcap_paths, basefile_name=basefile_name,
                                               time_interval_lengths=time_interval_lengths,
                                               make_edgefiles_p=make_edgefiles,
                                               basegraph_name=basegraph_name,
                                               alert_file=alert_file,
                                               sec_between_exfil_pkts=sec_between_exfil_pkts,
                                               cluster_creation_log=pod_creation_log,
                                               netsec_policy=netsec_policy, sensitive_ms=sensitive_ms,
                                               exfil_StartEnd_times=exfil_StartEnd_times,
                                               physical_exfil_paths=physical_exfil_paths,
                                               base_experiment_dir=base_experiment_dir,
                                               time_of_synethic_exfil=time_of_synethic_exfil)
    return pipeline_object

def parse_experimental_config(experimental_config_file):
    with open(experimental_config_file) as f:
        config_file = json.load(f)

        skip_model_part = config_file['skip_model_part']
        ignore_physical_attacks_p = config_file['ignore_physical_attacks_p']

        time_of_synethic_exfil = config_file['time_of_synethic_exfil']
        goal_train_test_split_training = config_file['goal_train_test_split_training']
        goal_attack_NoAttack_split_training = config_file['goal_attack_NoAttack_split_training']
        goal_attack_NoAttack_split_testing = config_file['goal_attack_NoAttack_split_testing']

        time_interval_lengths = config_file['time_interval_lengths']
        ide_window_size = config_file['ide_window_size']

        avg_exfil_per_min = config_file['avg_exfil_per_min']
        exfil_per_min_variance = config_file['exfil_per_min_variance']
        avg_pkt_size = config_file['avg_pkt_size']
        pkt_size_variance = config_file['pkt_size_variance']

        BytesPerMegabyte = 1000000
        avg_exfil_per_min = [BytesPerMegabyte * i for i in avg_exfil_per_min]
        exfil_per_min_variance = [BytesPerMegabyte * i for i in exfil_per_min_variance]

        calc_vals = config_file['calc_vals']
        calculate_z_scores = config_file['calculate_z_scores']
        drop_pairwise_features = config_file['drop_pairwise_features']
        perform_cilium_component = config_file['perform_cilium_component']

        cur_experiment_name = config_file['cur_experiment_name']
        base_output_location = config_file['base_output_location']
        drop_infra_from_graph = config_file['drop_infra_from_graph']
        if drop_infra_from_graph:
            cur_experiment_name += 'dropInfra'
        base_output_location += cur_experiment_name

        skip_graph_injection = config_file['skip_graph_injection']
        get_endresult_from_memory = config_file['get_endresult_from_memory']

        make_edgefiles = config_file['make_edgefiles']
        experimental_folder = config_file['experimental_folder']
        pcap_file_path = config_file['pcap_file_path']
        pod_creation_log_path = config_file["pod_creation_log_path"]
        exp_config_file = config_file['exp_config_file']
        netsec_policy = config_file['netsec_policy']

        experiment_classes = [parse_experimental_data_json(exp_config_file, experimental_folder, cur_experiment_name,
                                                           make_edgefiles, time_interval_lengths, pcap_file_path,
                                                           pod_creation_log_path, netsec_policy, time_of_synethic_exfil)]

        # this is a work-around until I finish refactoring the actual system...
        # this part exists b/c I never want to interfere with actually working
        include_ide = config_file['include_ide']
        calc_ide = config_file['calc_ide']
        only_ide = config_file['only_ide']

        #print "REMOVE THE WAITING!!!"
        #time.sleep(1500)

        multi_experiment_object = \
            multi_experiment_pipeline(experiment_classes, base_output_location, True, time_of_synethic_exfil,
                                  goal_train_test_split_training, goal_attack_NoAttack_split_training, None,
                                  None, calc_vals, skip_model_part, ignore_physical_attacks_p,
                                  calculate_z_scores_p=calculate_z_scores,
                                  avg_exfil_per_min=avg_exfil_per_min, exfil_per_min_variance=exfil_per_min_variance,
                                  avg_pkt_size=avg_pkt_size, pkt_size_variance=pkt_size_variance,
                                  skip_graph_injection=skip_graph_injection,
                                  get_endresult_from_memory=get_endresult_from_memory,
                                  goal_attack_NoAttack_split_testing=goal_attack_NoAttack_split_testing,
                                  calc_ide=calc_ide, include_ide=include_ide, only_ide=only_ide,
                                  drop_pairwise_features=drop_pairwise_features,
                                  ide_window_size=ide_window_size, drop_infra_from_graph=drop_infra_from_graph,
                                  perform_cilium_component=perform_cilium_component)

        min_rate_statspipelines = multi_experiment_object.run_pipelines()

    return min_rate_statspipelines

if __name__=="__main__":
    print "RUNNING"
    print sys.argv


    parser = argparse.ArgumentParser(description='This is the central CL interface for the MIMIR system')
    parser.add_argument('--training_config_json', dest='training_config_json', default=None,
                        help='this is the configuration file used to train/retrieve the model')
    parser.add_argument('--eeval_config_json', dest='config_json', default=None,
                        help='this is the configuration file used to generate actual alerts')
    args = parser.parse_args()

    ## TODO: need to hook this into the system in a coherent fashion...
    ## TODO: change all of the import statments so that they are relevant to the analysis_pipeline direectory...


    if len(sys.argv) == 1:
        print "running_preset..."
        #autoscaling_sockshop_recipe()
        #nonauto_sockshop_recipe()
        #new_wordpress_autoscaling_recipe()
        #new_wordpress_recipe()

        #print os.getcwd()
        parse_experimental_config('./analysis_json/sockshop_one_v2.json')
        #parse_experimental_config('./analysis_json/wordpress_one_v2_nonauto.json')
        #parse_experimental_config('./analysis_json/sockshop_one_v2_nonauto.json')
    elif len(sys.argv) == 2:
        experimental_config_file = sys.argv[1]
        parse_experimental_config(experimental_config_file)
    else:
        print "too many args!"