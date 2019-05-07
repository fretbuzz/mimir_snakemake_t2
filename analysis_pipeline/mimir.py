import json
import pyximport
pyximport.install()
import sys
import matplotlib
matplotlib.use('Agg',warn=False, force=True)
from pipeline_coordinator import multi_experiment_pipeline
from single_experiment_pipeline import data_anylsis_pipline
import argparse
import os,errno
import time
from tabulate import tabulate

'''
This file runs the MIMIR anomaly detection system by parsing configuration files.
'''

def parse_experimental_data_json(config_file, experimental_folder, experiment_name, make_edgefiles,
                                 time_interval_lengths, pcap_file_path, pod_creation_log_path,
                                 netsec_policy=None, time_of_synethic_exfil=None, no_processing_at_all=False):
    with open(config_file) as f:
        config_file = json.load(f)
        basefile_name = experimental_folder + experiment_name + '/edgefiles/' + experiment_name + '_'
        basegraph_name = experimental_folder + experiment_name + '/graphs/' + experiment_name + '_'
        alert_file = experimental_folder + experiment_name + '/alerts/' + experiment_name + '_'
        base_experiment_dir =  experimental_folder + experiment_name + '/'

        try:
            sec_between_exfil_pkts = config_file["exfiltration_info"]['sec_between_exfil_pkts']
        except:
            sec_between_exfil_pkts = 1.0

        pcap_paths = [ pcap_file_path + config_file['pcap_file_name'] ]
        pod_creation_log = [ pod_creation_log_path + config_file['pod_creation_log_name']]
        sensitive_ms = config_file["exfiltration_info"]['sensitive_ms']

        try:
            physical_exfil_p = config_file["exfiltration_info"]["physical_exfil_performed"]
        except:
            physical_exfil_p = False

        try:
            if physical_exfil_p:
                exfil_StartEnd_times = config_file["exfiltration_info"]['exfil_StartEnd_times']
            else:
                exfil_StartEnd_times = [[]]
        except:
            exfil_StartEnd_times = [[]]

        try:
            if physical_exfil_p:
                physical_exfil_paths = config_file["exfiltration_info"]['exfil_paths']
            else:
                physical_exfil_paths = [[]]
        except:
            physical_exfil_paths = [[]]

        pipeline_object = data_anylsis_pipline(pcap_paths=pcap_paths, basefile_name=basefile_name,
                                               time_interval_lengths=time_interval_lengths,
                                               make_edgefiles_p=make_edgefiles, basegraph_name=basegraph_name,
                                               alert_file=alert_file, sec_between_exfil_pkts=sec_between_exfil_pkts,
                                               time_of_synethic_exfil=time_of_synethic_exfil,
                                               netsec_policy=netsec_policy, cluster_creation_log=pod_creation_log,
                                               sensitive_ms=sensitive_ms, exfil_StartEnd_times=exfil_StartEnd_times,
                                               physical_exfil_paths=physical_exfil_paths,
                                               base_experiment_dir=base_experiment_dir,
                                               no_processing_at_all=no_processing_at_all)
    return pipeline_object

def parse_experimental_config(experimental_config_file, live=False, is_eval=False):
    with open(experimental_config_file) as f:
        config_file = json.load(f)

        if 'skip_model_part' in config_file:
            skip_model_part = config_file['skip_model_part']
        else:
            skip_model_part = False

        if 'time_of_synethic_exfil' in config_file:
            time_of_synethic_exfil = config_file['time_of_synethic_exfil']
        else:
            time_of_synethic_exfil = 60

        if 'goal_train_test_split_training' in config_file:
            goal_train_test_split_training = config_file['goal_train_test_split_training']
        else:
            if is_eval:
                goal_train_test_split_training = 0.0 # if eval, everything is test
            else:
                goal_train_test_split_training = 1.0 # if train, everything is train

        if "goal_attack_NoAttack_split_training" in config_file:
            goal_attack_NoAttack_split_training = config_file['goal_attack_NoAttack_split_training']
        else:
            goal_attack_NoAttack_split_training = 0.50

        if "goal_attack_NoAttack_split_testing" in config_file:
            goal_attack_NoAttack_split_testing = config_file['goal_attack_NoAttack_split_testing']
        else:
            goal_attack_NoAttack_split_testing = 0.2

        if 'time_interval_lengths' in config_file:
            time_interval_lengths = config_file['time_interval_lengths']
        else:
            time_interval_lengths = [10, 60]

        if 'ide_window_size' in config_file:
            ide_window_size = config_file['ide_window_size']
        else:
            ide_window_size = 12

        if 'avg_exfil_per_min' in config_file:
            avg_exfil_per_min = config_file['avg_exfil_per_min']
        else:
            avg_exfil_per_min = [10.0, 1.0, 0.1, 0.05, 0.01]

        if 'exfil_per_min_variance' in config_file:
            exfil_per_min_variance = config_file['exfil_per_min_variance']
        else:
            exfil_per_min_variance = [0.3, 0.15, 0.025, 0.0125, 0.0025]

        if 'avg_pkt_size' in config_file:
            avg_pkt_size = config_file['avg_pkt_size']
        else:
            # note: this literally has no effect on the system, so don't worry about it
            avg_pkt_size = [500.0 for i in range(0,len(avg_exfil_per_min))]

        if 'pkt_size_variance' in config_file:
            pkt_size_variance = config_file['pkt_size_variance']
        else:
            # note: this literally has no effect on the system, so don't worry about it
            pkt_size_variance = [100 for i in range(0,len(avg_exfil_per_min))]

        BytesPerMegabyte = 1000000
        avg_exfil_per_min = [BytesPerMegabyte * i for i in avg_exfil_per_min]
        exfil_per_min_variance = [BytesPerMegabyte * i for i in exfil_per_min_variance]

        calc_vals = config_file['calc_vals']
        calculate_z_scores = config_file['calculate_z_scores']

        if 'drop_pairwise_features' in config_file:
            drop_pairwise_features = config_file['drop_pairwise_features']
        else:
            drop_pairwise_features = False

        if 'perform_cilium_component' in config_file:
            perform_cilium_component = config_file['perform_cilium_component']
        else:
            perform_cilium_component = True

        cur_experiment_name = config_file['cur_experiment_name']

        exp_config_file = config_file['exp_config_file']
        if 'experimental_folder' in config_file:
            experimental_folder = config_file['experimental_folder']
        else:
            experimental_folder = "/".join(exp_config_file.split('/')[:-1]) + "/"

        if 'base_output_location' in config_file:
            base_output_location = config_file['base_output_location']
        else:
            base_output_location = experimental_folder + 'results/'

        try:
            os.makedirs(base_output_location)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        if 'drop_infra_from_graph' in config_file:
            drop_infra_from_graph = config_file['drop_infra_from_graph']
        else:
            drop_infra_from_graph = True

        if drop_infra_from_graph:
            cur_experiment_name += 'dropInfra'
        base_output_location += cur_experiment_name

        skip_graph_injection = config_file['skip_graph_injection']

        get_endresult_from_memory = config_file['get_endresult_from_memory']

        make_edgefiles = config_file['make_edgefiles']

        if 'pcap_file_path' in config_file:
            pcap_file_path = config_file['pcap_file_path']
        else:
            pcap_file_path = experimental_folder

        if 'pod_creation_log_path' in config_file:
            pod_creation_log_path = config_file["pod_creation_log_path"]
        else:
            pod_creation_log_path = experimental_folder

        if 'netsec_policy' in config_file:
            netsec_policy = config_file['netsec_policy']
        else:
            netsec_policy = "None"

        if 'auto_open_pdfs' in config_file:
            auto_open_pdfs = config_file['auto_open_pdfs']
        else:
            auto_open_pdfs = True

        experiment_classes = [parse_experimental_data_json(exp_config_file, experimental_folder, cur_experiment_name,
                                                           make_edgefiles, time_interval_lengths, pcap_file_path,
                                                           pod_creation_log_path, netsec_policy, time_of_synethic_exfil,
                                                           no_processing_at_all=get_endresult_from_memory)]

        # this is a work-around until I finish refactoring the actual system...
        # this part exists b/c I never want to interfere with actually working
        if 'include_ide' in config_file:
            include_ide = config_file['include_ide']
        else:
            include_ide = False

        if 'calc_ide' in config_file:
            calc_ide = config_file['calc_ide']
        else:
            calc_ide = False

        if 'only_ide' in config_file:
            only_ide = config_file['only_ide']
        else:
            only_ide = False

        #print "REMOVE THE WAITING!!!"
        #time.sleep(1500)

        if 'skip_heatmap_p' in config_file:
            skip_heatmap_p = config_file['skip_heatmap_p']
        else:
            skip_heatmap_p = False

        no_labeled_data = False

    if live:
        skip_model_part = True
        no_labeled_data = True
        avg_exfil_per_min = [0.0]

    multi_experiment_object = \
        multi_experiment_pipeline(experiment_classes, base_output_location, True, time_of_synethic_exfil,
                                  goal_train_test_split_training, goal_attack_NoAttack_split_training, None, None,
                                  calc_vals, skip_model_part, calculate_z_scores_p=calculate_z_scores,
                                  avg_exfil_per_min=avg_exfil_per_min, exfil_per_min_variance=exfil_per_min_variance,
                                  avg_pkt_size=avg_pkt_size, pkt_size_variance=pkt_size_variance,
                                  skip_graph_injection=skip_graph_injection,
                                  get_endresult_from_memory=get_endresult_from_memory,
                                  goal_attack_NoAttack_split_testing=goal_attack_NoAttack_split_testing,
                                  calc_ide=calc_ide, include_ide=include_ide, only_ide=only_ide,
                                  perform_cilium_component=perform_cilium_component,
                                  drop_pairwise_features=drop_pairwise_features,
                                  drop_infra_from_graph=drop_infra_from_graph, ide_window_size=ide_window_size,
                                  auto_open_pdfs=auto_open_pdfs, skip_heatmap_p=skip_heatmap_p,
                                  no_labeled_data=no_labeled_data)


    return multi_experiment_object

def run_analysis(training_config, eval_config=None, live=False):
    training_experimente_object = parse_experimental_config(training_config, is_eval=False)
    min_rate_training_statspipelines, training_results = training_experimente_object.run_pipelines()

    print "training_results", training_results
    eval_results = None
    #time.sleep(35)

    if eval_config:
        eval_experimente_object = parse_experimental_config(eval_config, live=live, is_eval=True)
        _, eval_results = eval_experimente_object.run_pipelines(pretrained_model_object=min_rate_training_statspipelines)

        print "----------------------------"
        print "eval_results:"

        if eval_results and live:
            lowest_timegran = min(eval_results["ensemble"].keys())
            predicted_vals =  eval_results["ensemble"][lowest_timegran] # predicted alert vals at each time granularity
            data = [ (lowest_timegran * counter, val) for counter,val in enumerate(predicted_vals)]
            print(tabulate(data, headers=['time', 'alert_value']))
        elif eval_results:
            print eval_results

    return eval_results

if __name__=="__main__":
    print "RUNNING"
    print sys.argv


    parser = argparse.ArgumentParser(description='This is the central CL interface for the MIMIR system')
    parser.add_argument('--training_config_json', dest='training_config_json', default=None,
                        help='this is the configuration file used to train/retrieve the model')
    parser.add_argument('--eval_config_json', dest='config_json', default=None,
                        help='this the data that the trained model is applied to')
    parser.add_argument('--live', dest='live', default=False, action='store_true',
                        help='the eval set doesn\'t have attack labels')
    args = parser.parse_args()

    if not args.training_config_json:
        print "running_preset..."
        #run_analysis('./analysis_json/hipsterStore_mk1.json')
        #run_analysis('./analysis_json/sockshop_one_auto_mk11long.json')
        #run_analysis('./analysis_json/sockshop_mk13.json', eval_config = './analysis_json/sockshop_exfil_test.json', live=True)
        #run_analysis('./analysis_json/sockshop_mk13.json', eval_config = './new_analysis_json/sockshop_mk20.json')
        #run_analysis('./analysis_json/sockshop_mk13.json', eval_config = './new_analysis_json/sockshop_mk22.json')
        ###run_analysis('./analysis_json/sockshop_mk13.json')
        #time.sleep(7200)
        #run_analysis('./analysis_json/sockshop_mk13.json', eval_config = './new_analysis_json/sockshop_mk23.json')
        #run_analysis('./analysis_json/sockshop_mk13.json', eval_config = './new_analysis_json/sockshop_mk24.json')

        #run_analysis('./analysis_json/sockshop_mk13.json', eval_config = './new_analysis_json/sockshop_auto_mk27.json')

        run_analysis('./analysis_json/wordpress_one_3_auto_mk5.json', eval_config='./analysis_json/wordpress_one_v2_na_eval.json')
        #run_analysis('./analysis_json/sockshop_exfil_test.json')
        #run_analysis('analysis_json/wordpress_model.json', eval_config='analysis_json/wordpress_example.json')
        #run_analysis('./analysis_json/sockshop_one_auto_mk12long.json', eval_config='./analysis_json/sockshop_example.json')
        #run_analysis('./analysis_json/sockshop_one_auto_mk12long.json')

        #run_analysis('./analysis_json/sockshop_one_auto_mk11long.json', eval_config='./analysis_json/sockshop_example.json')
        ##
        #run_analysis('./analysis_json/sockshop_one_v2_nonauto.json', eval_config='./analysis_json/sockshop_example.json')

        #run_analysis('./analysis_json/sockshop_example.json')
        #run_analysis('./analysis_json/sockshop_one_v2_mk7.json')
        #run_analysis('./analysis_json/sockshop_one_v2.json')
        #run_analysis('./analysis_json/sockshop_one_v2_minimal.json')
        #run_analysis('./analysis_json/wordpress_one_v2_nonauto.json')
        #run_analysis('./analysis_json/sockshop_one_v2_nonauto.json')
        #run_analysis('./analysis_json/sockshop_short.json')
    else:
        #parse_experimental_config(args.training_config_json, eval_config)
        run_analysis(args.training_config_json, eval_config=args.config_json, live=args.live)
