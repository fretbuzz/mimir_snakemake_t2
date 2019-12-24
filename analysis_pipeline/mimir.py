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
from dlp_pipeline import end_to_end_microservice

'''
This file runs the MIMIR anomaly detection system by parsing configuration files.
'''

def parse_experimental_data_json(config_file, experimental_folder, experiment_name, make_edgefiles,
                                 time_interval_lengths, pcap_file_path, pod_creation_log_path,
                                 netsec_policy=None, time_of_synethic_exfil=None, skip_to_calc_zscore=False,
                                 no_processing_at_all=False):
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
                                               no_processing_at_all=no_processing_at_all,
                                               skip_to_calc_zscore=skip_to_calc_zscore)
    return pipeline_object

def parse_experimental_config(experimental_config_file, return_new_model_function, live=False, is_eval=False,
                              add_dropInfo_to_name=True, skip_to_calc_zscore=False, exp_data_dir=None):

    print "experimental_config_file", type(experimental_config_file), experimental_config_file
    with open(experimental_config_file, 'r') as f:
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

        if 'perform_cilium_component' in config_file or 'perform_svcpair_sec_component' in config_file:
            if 'perform_cilium_component' in config_file:
                perform_svcpair_sec_component = config_file['perform_cilium_component']
            else:
                perform_svcpair_sec_component = config_file['perform_svcpair_sec_component']
        else:
            perform_svcpair_sec_component = False

        cur_experiment_name = config_file['cur_experiment_name']

        exp_config_file = config_file['exp_config_file']
        if exp_data_dir is not None:
            # if the experiment directory differs from the one listed in the config file, then exp_data_dir is not none and we need to adjust things here...
            if exp_data_dir[-1] != '/':
                exp_data_dir = exp_data_dir + '/'
            exp_config_file = exp_data_dir + '/'.join(exp_config_file.split('/')[-2:])

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
            if add_dropInfo_to_name: # exists for compatibility reasons
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

        if 'calc_ide' in config_file:
            calc_ide = config_file['calc_ide']
        else:
            calc_ide = False

        if 'ide_window_size' in config_file:
            ide_window_size = config_file['ide_window_size']
        else:
            ide_window_size = 10

        # in this case, we want to only retrain the model (WITHOUT recalculating the features-- just pull the features from the csv...)
        if skip_to_calc_zscore:
            print "skip_to_calc_zscore was called!!"
            make_edgefiles = False
            skip_graph_injection = True
            calc_vals = False
            get_endresult_from_memory = False
            calc_ide = False
            calculate_z_scores = True

        experiment_classes = [parse_experimental_data_json(exp_config_file, experimental_folder, cur_experiment_name,
                                                           make_edgefiles, time_interval_lengths, pcap_file_path,
                                                           pod_creation_log_path, netsec_policy, time_of_synethic_exfil,
                                                           skip_to_calc_zscore, no_processing_at_all=get_endresult_from_memory)]


        if 'skip_heatmap_p' in config_file:
            skip_heatmap_p = config_file['skip_heatmap_p']
        else:
            skip_heatmap_p = True

        no_labeled_data = False

    if live:
        skip_model_part = True
        no_labeled_data = True
        avg_exfil_per_min = [0.0]

    #if calc_ide and (not calc_vals):
    #    print "if calculating ide scores, must also calculate values for the graphs"
    #    print "i.e. if calc_ide true, then calc_vals must be true"
    #    exit(233)

    multi_experiment_object = \
        multi_experiment_pipeline(experiment_classes, base_output_location, True, time_of_synethic_exfil,
                                  goal_train_test_split_training, goal_attack_NoAttack_split_training, None, None,
                                  calc_vals, skip_model_part, return_new_model_function, calculate_z_scores_p=calculate_z_scores,
                                  avg_exfil_per_min=avg_exfil_per_min, exfil_per_min_variance=exfil_per_min_variance,
                                  avg_pkt_size=avg_pkt_size, pkt_size_variance=pkt_size_variance,
                                  skip_graph_injection=skip_graph_injection,
                                  get_endresult_from_memory=get_endresult_from_memory,
                                  goal_attack_NoAttack_split_testing=goal_attack_NoAttack_split_testing,
                                  calc_ide=calc_ide,
                                  perform_svcpair_sec_component=perform_svcpair_sec_component,
                                  drop_pairwise_features=drop_pairwise_features,
                                  drop_infra_from_graph=drop_infra_from_graph, ide_window_size=ide_window_size,
                                  auto_open_pdfs=auto_open_pdfs, skip_heatmap_p=skip_heatmap_p,
                                  no_labeled_data=no_labeled_data)


    return multi_experiment_object

def run_analysis(return_new_model_function, training_config, eval_config=None, live=False, no_tsl=True,
                 decanter_configs=None, skip_to_calc_zscore=False, exp_data_dir=None,
                 per_svc_exfil_model_p=False, load_old_pipelines=False):

    training_experimente_object = parse_experimental_config(training_config, return_new_model_function, is_eval=False,
                                                            skip_to_calc_zscore=skip_to_calc_zscore, exp_data_dir=exp_data_dir)

    min_rate_training_statspipelines, training_results, svcpair_model = training_experimente_object.run_pipelines(no_tsl=no_tsl, per_svc_exfil_model_p=per_svc_exfil_model_p,
                                                                                                                  load_old_pipelines=load_old_pipelines)

    print "min_rate_training_statspipelines",min_rate_training_statspipelines
    print "training_results", training_results
    eval_results = None
    #time.sleep(35)
    ##exit(233)

    if eval_config:
        eval_experimente_object = parse_experimental_config(eval_config, None, live=live, is_eval=True, skip_to_calc_zscore=skip_to_calc_zscore,
                                                            exp_data_dir=exp_data_dir)
        _, eval_results,_ = eval_experimente_object.run_pipelines(pretrained_model_object=min_rate_training_statspipelines,
                                                                  no_tsl=no_tsl, svcpair_model=svcpair_model,
                                                                  per_svc_exfil_model_p=per_svc_exfil_model_p,
                                                                  load_old_pipelines=load_old_pipelines)

        print "----------------------------"
        print "eval_results:"

        if eval_results and live:
            print "eval_results.keys()", eval_results.keys()
            lowest_timegran = min(eval_results["ensemble"].keys())
            predicted_vals =  eval_results["ensemble"][lowest_timegran] # predicted alert vals at each time granularity
            data = [ (lowest_timegran * counter, val) for counter,val in enumerate(predicted_vals)]
            print(tabulate(data, headers=['time', 'alert_value']))
        elif eval_results:
            print eval_results

        if decanter_configs:
            print "eval_results_b4_dec", eval_results
            eval_results = run_decanter_component(decanter_configs, training_config, eval_config, eval_results)
            # TODO: update the decanter configs appropriately....

    return eval_results

def run_decanter_component(decanter_configs, training_config, eval_config, eval_results):
    print "decanter_configs", decanter_configs

    print "params_in_call_to_end_to_end_microservices", (training_config, eval_config,
        decanter_configs['train_gen_bro_log'], decanter_configs['test_gen_bro_log'], decanter_configs['gen_fingerprints_p']),\
        decanter_configs['fraction_of_training_pcap_to_use']

    if 'fraction_of_training_pcap_to_use' in decanter_configs:
        print "found fraction_of_training_pcap_to_use in decanter_configs"
        fraction_of_training_pcap_to_use = float(decanter_configs['fraction_of_training_pcap_to_use'])
    else:
        print "did NOT find fraction_of_training_pcap_to_use in decanter_configs"
        fraction_of_training_pcap_to_use = 1.0

    decanter_results = end_to_end_microservice(training_config, eval_config,
                                               decanter_configs['train_gen_bro_log'],
                                               decanter_configs['test_gen_bro_log'],
                                               decanter_configs['gen_fingerprints_p'],
                                               fraction_of_training_pcap_to_use=fraction_of_training_pcap_to_use)

    ###return

    print "decanter_results", decanter_results

    # TODO: this is being incorporated incorrectly!! We need to examine the old schema before just
    # throwing new data in there...
    # applies to ALL exfil rates
    # eval_results['decanter'] = decanter_results

    for exfil_rate, perf_at_exfil_rate in eval_results.iteritems():
        for time_gran, perf_at_timegran in perf_at_exfil_rate.iteritems():
            print "ggpg", type(perf_at_timegran.values()[0]), perf_at_timegran.values()[0]
            try:
                perf_at_timegran['decanter'] = decanter_results[time_gran]
            except:
                print time_gran, "is not a timegran that the decanter component handles currently"

    print "eval_results_after_dec", eval_results
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
    parser.add_argument('--retrain_model', dest='retrain_model', default=False, action='store_true',
                        help='retrains the model and applies it to the testing data-- note that it does NOT recalculate the features, it just uses the feature CSV')
    parser.add_argument('--exp_data_dir', dest='exp_data_dir', default=None,
                        help='if the experiment directory differs from the one listed in the config file, you can specify it here (useful for running locally)')
    parser.add_argument('--return_new_model_func', dest='ret_new_mod_func', default=False,
                        help='when training, returns the model from statistical_analysis_perSvc.py for use on the evaluation data')

    parser.add_argument('--load_old_pipelines', dest='load_old_pipelines', default=False, action='store_true',
                        help='[for dev purposes] loads the old pipelines (from statistical_analysis.py), so that the new one can be tested more easily')




    args = parser.parse_args()

    if not args.training_config_json:
        print "running_preset..."
        #run_analysis('./analysis_json/hipsterStore_mk1.json')
        #run_analysis('./analysis_json/sockshop_one_auto_mk11long.json')
        #run_analysis('./analysis_json/training_config_example.json', eval_config = './analysis_json/sockshop_exfil_test.json', live=True)
        #run_analysis('./analysis_json/training_config_example.json', eval_config = './new_analysis_json/sockshop_mk20.json')
        #run_analysis('./analysis_json/training_config_example.json', eval_config = './new_analysis_json/sockshop_mk22.json')
        #run_analysis('./analysis_json/training_config_example.json', eval_config = './new_analysis_json/sockshop_mk23.json')
        run_analysis(args.ret_new_mod_func, './new_analysis_json/training_config_example.json',
                     eval_config = './new_analysis_json/sockshop_mk24.json')
        #run_analysis('./analysis_json/training_config_example.json', eval_config = './new_analysis_json/sockshop_auto_mk27.json')

        #run_analysis('./analysis_json/wordpress_one_3_auto_mk5.json', eval_config='./analysis_json/wordpress_one_v2_na_eval.json')
        #run_analysis('./analysis_json/sockshop_exfil_test.json')
        #run_analysis('analysis_json/wordpress_model.json', eval_config='analysis_json/wordpress_example.json')
        #run_analysis('./analysis_json/sockshop_one_auto_mk12long.json', eval_config='./analysis_json/sockshop_example.json')
        #run_analysis('./analysis_json/sockshop_one_auto_mk12long.json')
        #run_analysis('./analysis_json/sockshop_one_auto_mk11long.json', eval_config='./analysis_json/sockshop_example.json')

    else:
        #parse_experimental_config(args.training_config_json, eval_config)
        print "training_config_json", args.training_config_json
        print "eval_config", args.config_json
        print "retrain_model", args.retrain_model
        run_analysis(args.ret_new_mod_func, args.training_config_json, eval_config=args.config_json, live=args.live,
                     skip_to_calc_zscore = args.retrain_model, exp_data_dir = args.exp_data_dir,
                     load_old_pipelines = args.load_old_pipelines)
