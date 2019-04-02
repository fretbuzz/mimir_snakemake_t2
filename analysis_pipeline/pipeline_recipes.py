import json
import pyximport
pyximport.install()
import sys
import matplotlib
matplotlib.use('Agg',warn=False, force=True)
from analysis_pipeline.pipeline_coordinator import multi_experiment_pipeline
from analysis_pipeline.single_experiment_pipeline import data_anylsis_pipline
import argparse
import os

'''
This file is essentially just sets of parameters for the run_data_analysis_pipeline function in pipeline_coordinator.py
There are a lot of parameters, and some of them are rather long, so I decided to make a function to store them in
'''

microservices_sockshop = ['carts-db', 'cart', 'catalogue-db', 'catalogue', 'front-end', 'orders-db', 'orders',
                         'payment', 'queue-master', 'rabbitmq', 'shipping', 'user-db', 'user']
minikube_infrastructure = ['etcd', 'kube-addon-manager', 'kube-apiserver', 'kube-controller-manager',
                           'kube-dns', 'kube-proxy', 'kube-scheduler', 'kubernetes-dashboard', 'metrics-server',
                           'storage-provisioner']
microservices_wordpress = ['mariadb-master', 'mariadb-slave', 'wordpress']

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

        rate_to_time_gran_to_xs, rate_to_time_gran_to_ys, rate_to_timegran_list_of_methods_to_attacks_found_training_df, \
        rate_to_timegran_to_methods_to_attacks_found_dfs = \
            multi_experiment_pipeline(experiment_classes, base_output_location, True, time_of_synethic_exfil,
                                  goal_train_test_split_training, goal_attack_NoAttack_split_training, None,
                                  None, calc_vals, skip_model_part, ignore_physical_attacks_p,
                                  calculate_z_scores_p=calculate_z_scores,
                                  avg_exfil_per_min=avg_exfil_per_min, exfil_per_min_variance=exfil_per_min_variance,
                                  avg_pkt_size=avg_pkt_size, pkt_size_variance=pkt_size_variance,
                                  skip_graph_injection=skip_graph_injection,
                                  get_endresult_from_memory=get_endresult_from_memory,
                                  goal_attack_NoAttack_split_testing=goal_attack_NoAttack_split_testing,
                                  calc_ide=False, include_ide=False, only_ide=only_ide,
                                  drop_pairwise_features=drop_pairwise_features,
                                  ide_window_size=ide_window_size, drop_infra_from_graph=drop_infra_from_graph,
                                  perform_cilium_component=perform_cilium_component)

        if calc_ide:
            calc_vals = False
            skip_graph_injection = True
            rate_to_time_gran_to_xs, rate_to_time_gran_to_ys, rate_to_timegran_list_of_methods_to_attacks_found_training_df, \
            rate_to_timegran_to_methods_to_attacks_found_dfs = \
                multi_experiment_pipeline(experiment_classes, base_output_location, True, time_of_synethic_exfil,
                                          goal_train_test_split_training, goal_attack_NoAttack_split_training, None,
                                          None, calc_vals, skip_model_part, ignore_physical_attacks_p,
                                          calculate_z_scores_p=True,
                                          avg_exfil_per_min=avg_exfil_per_min,
                                          exfil_per_min_variance=exfil_per_min_variance,
                                          avg_pkt_size=avg_pkt_size, pkt_size_variance=pkt_size_variance,
                                          skip_graph_injection=skip_graph_injection,
                                          get_endresult_from_memory=get_endresult_from_memory,
                                          goal_attack_NoAttack_split_testing=goal_attack_NoAttack_split_testing,
                                          calc_ide=calc_ide, include_ide=True, only_ide=True,
                                          drop_pairwise_features=drop_pairwise_features,
                                          ide_window_size=ide_window_size, drop_infra_from_graph=drop_infra_from_graph,
                                          perform_cilium_component=perform_cilium_component)

    return rate_to_time_gran_to_xs, rate_to_time_gran_to_ys, rate_to_timegran_list_of_methods_to_attacks_found_training_df, \
            rate_to_timegran_to_methods_to_attacks_found_dfs

def wordpress_thirteen_t2(time_of_synethic_exfil=None, time_interval_lengths=None):

    basefile_name = '/Volumes/exM2/experimental_data/wordpress_info/wordpress_thirteen_t2/edgefiles/wordpress_thirteen_t2_'
    basegraph_name = '/Volumes/exM2/experimental_data/wordpress_info/wordpress_thirteen_t2/graphs/wordpress_thirteen_t2_'
    alert_file = '/Volumes/exM2/experimental_data/wordpress_info/wordpress_thirteen_t2/alerts/wordpress_thirteen_t2_'

    old_mulval_info = {}
    old_mulval_info["container_info_path"] = "/Volumes/exM2/experimental_data/wordpress_info/wordpress_thirteen_t2/wordpress_thirteen_t2_docker_0_network_configs.txt"
    old_mulval_info["cilium_config_path"] = None # does NOT use cilium on reps 2-4
    old_mulval_info["kubernetes_svc_info"] = '/Volumes/exM2/experimental_data/wordpress_info/wordpress_thirteen_t2/wordpress_thirteen_t2_svc_config_0.txt'
    old_mulval_info["kubernetes_pod_info"] = '/Volumes/exM2/experimental_data/wordpress_info/wordpress_thirteen_t2/wordpress_thirteen_t2_pod_config_0.txt'
    old_mulval_info["ms_s"] = ["my-release-pxc", "wwwppp-wordpress"]

    pcap_paths = [
        "/Volumes/exM2/experimental_data/wordpress_info/wordpress_thirteen_t2/wordpress_thirteen_t2_default_bridge_0any.pcap"]
    pod_creation_log = None

    sensitive_ms = "my-release-pxc"
    exfil_start_time = 6090
    exfil_end_time = 6090
    sec_between_exfil_events = 15
    physical_exfil_path = []

    make_edgefiles = False ## already done!
    wordpress_thirteen_t2_object = data_anylsis_pipline(pcap_paths=pcap_paths,
                                                        basefile_name=basefile_name,
                                                        time_interval_lengths=time_interval_lengths,
                                                        make_edgefiles_p=make_edgefiles,
                                                        basegraph_name=basegraph_name,
                                                        exfil_start_time=exfil_start_time,
                                                        exfil_end_time=exfil_end_time,
                                                        alert_file=alert_file,
                                                        sec_between_exfil_pkts=sec_between_exfil_events,
                                                        injected_exfil_path = physical_exfil_path,
                                                        time_of_synethic_exfil=time_of_synethic_exfil,
                                                        cluster_creation_log=pod_creation_log,
                                                        netsec_policy=None, sensitive_ms=sensitive_ms,
                                                        old_mulval_info=old_mulval_info)

    return wordpress_thirteen_t2_object

def sockshop_thirteen_NOautoscale_mark1(time_of_synethic_exfil=None, time_interval_lengths=None):

    basefile_name = '/Volumes/exM2/experimental_data/sockshop_info/sockshop_thirteen_NOautoscale_mark1/edgefiles/sockshop_thirteen_NOautoscale_mark1_'
    basegraph_name = '/Volumes/exM2/experimental_data/sockshop_info/sockshop_thirteen_NOautoscale_mark1/graphs/sockshop_thirteen_NOautoscale_mark1_'
    alert_file = '/Volumes/exM2/experimental_data/sockshop_info/sockshop_thirteen_NOautoscale_mark1/alerts/sockshop_thirteen_NOautoscale_mark1_'

    pod_creation_log = None

    old_mulval_info = {}
    old_mulval_info["container_info_path"] = "/Volumes/exM2/experimental_data/sockshop_info/sockshop_thirteen_NOautoscale_mark1/sockshop_thirteen_NOautoscale_mark1_docker_0_network_configs.txt"
    old_mulval_info["cilium_config_path"] = None # does NOT use cilium on reps 2-4
    old_mulval_info["kubernetes_svc_info"] = '/Volumes/exM2/experimental_data/sockshop_info/sockshop_thirteen_NOautoscale_mark1/sockshop_thirteen_NOautoscale_mark1_svc_config_0.txt'
    old_mulval_info["kubernetes_pod_info"] = '/Volumes/exM2/experimental_data/sockshop_info/sockshop_thirteen_NOautoscale_mark1/sockshop_thirteen_NOautoscale_mark1_pod_config_0.txt'
    old_mulval_info["ms_s"] = microservices_sockshop


    pcap_paths = [ "/Volumes/exM2/experimental_data/sockshop_info/sockshop_thirteen_NOautoscale_mark1/sockshop_thirteen_NOautoscale_mark1_default_bridge_0any.pcap"]
    netsec_policy = '/Volumes/exM2/experimental_data/sockshop_info/sockshop_nine_better_exfil_netsec_seg.txt'
    sensitive_ms = 'user-db'
    exfil_start_time = 8000
    exfil_end_time = 8000
    sec_between_exfil_events = 15
    physical_exfil_path = []


    make_edgefiles = False ## already done!
    sockshop_thirteen_NOautoscale_mark1_object = data_anylsis_pipline(pcap_paths=pcap_paths,
                                                                      basefile_name=basefile_name,
                                                                      time_interval_lengths=time_interval_lengths,
                                                                      make_edgefiles_p=make_edgefiles,
                                                                      basegraph_name=basegraph_name,
                                                                      exfil_start_time=exfil_start_time,
                                                                      exfil_end_time=exfil_end_time,
                                                                      alert_file=alert_file,
                                                                      sec_between_exfil_pkts=sec_between_exfil_events,
                                                                      injected_exfil_path = physical_exfil_path,
                                                                      time_of_synethic_exfil=time_of_synethic_exfil,
                                                                      cluster_creation_log=pod_creation_log,
                                                                      netsec_policy=netsec_policy,
                                                                      sensitive_ms=sensitive_ms,old_mulval_info=old_mulval_info)

    return sockshop_thirteen_NOautoscale_mark1_object


def wordpress_fourteen_mark7(time_of_synethic_exfil=None, time_interval_lengths=None):

    basefile_name = '/Volumes/exM2/experimental_data/wordpress_info/wordpress_fourteen_mark7_final/edgefiles/wordpress_fourteen_mark7_final_'
    basegraph_name = '/Volumes/exM2/experimental_data/wordpress_info/wordpress_fourteen_mark7_final/graphs/wordpress_fourteen_mark7_final_'
    alert_file = '/Volumes/exM2/experimental_data/wordpress_info/wordpress_fourteen_mark7_final/alerts/wordpress_fourteen_mark7_final_'

    old_mulval_info = {}
    old_mulval_info["container_info_path"] = "/Volumes/exM2/experimental_data/wordpress_info/wordpress_fourteen_mark7_final/wordpress_fourteen_mark7_final_docker_0_network_configs.txt"
    old_mulval_info["cilium_config_path"] = None
    old_mulval_info["kubernetes_svc_info"] = '/Volumes/exM2/experimental_data/wordpress_info/wordpress_fourteen_mark7_final/wordpress_fourteen_mark7_final_svc_config_0.txt'
    old_mulval_info["kubernetes_pod_info"] = '/Volumes/exM2/experimental_data/wordpress_info/wordpress_fourteen_mark7_final/wordpress_fourteen_mark7_final_pod_config_0.txt'
    old_mulval_info["ms_s"] = ["my-release-pxc", "wwwppp-wordpress"]

    # eventaully these should be the only files that actually exist.
    pod_creation_log = "/Volumes/exM2/experimental_data/wordpress_info/wordpress_fourteen_mark7_final/wordpress_fourteen_mark7_final_pod_creation_log.txt"
    pcap_paths = [ "/Volumes/exM2/experimental_data/wordpress_info/wordpress_fourteen_mark7_final/wordpress_fourteen_mark7_final_default_bridge_0any.pcap"]

    sensitive_ms = "my-release-pxc"
    exfil_start_time = 8000
    exfil_end_time = 8000
    sec_between_exfil_events = 15
    physical_exfil_path = []

    make_edgefiles = False ## already done!
    wordpress_fourteen_mark7_object = data_anylsis_pipline(pcap_paths=pcap_paths,
                                                           basefile_name=basefile_name,
                                                           time_interval_lengths=time_interval_lengths,
                                                           make_edgefiles_p=make_edgefiles,
                                                           basegraph_name=basegraph_name,
                                                           exfil_start_time=exfil_start_time,
                                                           exfil_end_time=exfil_end_time,
                                                           alert_file=alert_file, old_mulval_info=old_mulval_info,
                                                           sec_between_exfil_pkts=sec_between_exfil_events,
                                                           injected_exfil_path = physical_exfil_path,
                                                           time_of_synethic_exfil=time_of_synethic_exfil,
                                                           cluster_creation_log=pod_creation_log,
                                                           netsec_policy=None,
                                                           sensitive_ms=sensitive_ms)

    return wordpress_fourteen_mark7_object


def sockshop_thirteen_autoscale_mark4(time_of_synethic_exfil=None, time_interval_lengths=None):

    experiment_folder = '/Volumes/exM2/experimental_data/sockshop_info/sockshop_thirteen_autoscale_mark4/'

    basefile_name = experiment_folder + 'edgefiles/sockshop_thirteen_autoscale_mark4_'
    basegraph_name = experiment_folder + 'graphs/sockshop_thirteen_autoscale_mark4_'
    alert_file = experiment_folder + 'alerts/sockshop_thirteen_autoscale_mark4_'

    pod_creation_log = experiment_folder + "sockshop_thirteen_autoscale_mark4_pod_creation_log.txt"
    pcap_paths = [ experiment_folder + "sockshop_thirteen_autoscale_mark4_default_bridge_0any.pcap"]

    old_mulval_info = {}
    old_mulval_info["container_info_path"] = experiment_folder + "sockshop_thirteen_autoscale_mark4_docker_0_network_configs.txt"
    old_mulval_info["cilium_config_path"] = None
    old_mulval_info["kubernetes_svc_info"] = experiment_folder + 'sockshop_thirteen_autoscale_mark4_svc_config_0.txt'
    old_mulval_info["kubernetes_pod_info"] = experiment_folder + 'sockshop_thirteen_autoscale_mark4_pod_config_0.txt'
    old_mulval_info["ms_s"] = microservices_sockshop

    netsec_policy = '/Volumes/exM2/experimental_data/sockshop_info/sockshop_nine_better_exfil_netsec_seg.txt'
    sensitive_ms = 'user-db'
    exfil_start_time = 8000
    exfil_end_time = 8000
    physical_exfil_path = []
    sec_between_exfil_events = 15

    make_edgefiles = False ## already done!

    sockshop_thirteen_autoscale_mark4_object = data_anylsis_pipline(pcap_paths=pcap_paths,
                                                                    basefile_name=basefile_name,
                                                                    time_interval_lengths=time_interval_lengths,
                                                                    make_edgefiles_p=make_edgefiles,
                                                                    basegraph_name=basegraph_name,
                                                                    exfil_start_time=exfil_start_time,
                                                                    exfil_end_time=exfil_end_time,
                                                                    alert_file=alert_file,
                                                                    sec_between_exfil_pkts=sec_between_exfil_events,
                                                                    injected_exfil_path = physical_exfil_path,
                                                                    time_of_synethic_exfil=time_of_synethic_exfil,
                                                                    cluster_creation_log=pod_creation_log,
                                                                    netsec_policy=netsec_policy,
                                                                    sensitive_ms=sensitive_ms, old_mulval_info=old_mulval_info)

    return sockshop_thirteen_autoscale_mark4_object

def nonauto_sockshop_recipe():
    skip_model_part = False
    ignore_physical_attacks_p = True

    time_of_synethic_exfil = 30 # sec
    goal_train_test_split_training = 0.5
    goal_attack_NoAttack_split_training = 0.6
    goal_attack_NoAttack_split_testing = 0.2

    time_interval_lengths = [30, 10]#, 10] #[30, 10, 1] #[30, 10, 1] #[30, 10, 1]#,

    #####
    # IN MEGABYTES / MINUTE
    avg_exfil_per_min = [10.0, 2.0, 1.0, 0.25, 0.1] # [10.0, 2.0, 1.0, 0.25, 0.1] # [10.0, 2.0, 1.0, 0.25, 0.1]
    exfil_per_min_variance = [0.3, 0.2, 0.15, 0.08, 0.05] # [0.3. 0.2, 0.15, 0.08, 0.05] #[0.3, 0.2, 0.15, 0.08, 0.05]
    avg_pkt_size = [500.0, 500.0, 500.00, 500.00, 500.0]
    pkt_size_variance = [100, 100, 100, 100, 100]

    BytesPerMegabyte = 1000000
    avg_exfil_per_min = [BytesPerMegabyte * i for i in avg_exfil_per_min]
    exfil_per_min_variance = [BytesPerMegabyte * i for i in exfil_per_min_variance]
    ######

    calc_vals = False
    calculate_z_scores = True
    include_ide = False # include ide vals? this'll involve either calculating them (below) or grabbing them from the file location
    calc_ide = False
    only_ide = False ## ONLY calculate the ide values... this'll be useful if I wanna first calc all the other values and THEN ide...
    ide_window_size = 10
    drop_pairwise_features = False # drops pairwise features (i.e. serviceX_to_serviceY_reciprocity)

    ####
    cur_experiment_name = "mark1_"
    base_output_location = '/Volumes/exM2/experimental_data/sockshop_summary_new_nonauto13/nonauto13_'# + 'lasso_roc'
    base_output_location += cur_experiment_name
    if drop_pairwise_features:
        base_output_location += 'dropPairWise_'

    #####

    skip_graph_injection = False
    get_endresult_from_memory = False # in this case, you'd skip literally the whole pipeline and just get the
                                      # trained model + the results (from that model) out of memory
                                      # I anticpate that this'll mostly be useful for working on generating
                                      # the final results report + the graphs + other stuff kinda...

    experiment_classes = [sockshop_thirteen_NOautoscale_mark1(time_of_synethic_exfil=time_of_synethic_exfil,
                                                              time_interval_lengths=time_interval_lengths)]

    return multi_experiment_pipeline(experiment_classes, base_output_location, True, time_of_synethic_exfil,
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
                              ide_window_size=ide_window_size)


def autoscaling_sockshop_recipe():
    skip_model_part = False
    ignore_physical_attacks_p = True

    time_of_synethic_exfil = 30 # sec
    goal_train_test_split_training = 0.5
    goal_attack_NoAttack_split_training = 0.6
    goal_attack_NoAttack_split_testing = 0.2

    time_interval_lengths = [30, 10]#, 10] #[30, 10, 1] #[30, 10, 1] #[30, 10, 1]#,

    #####
    # IN MEGABYTES / MINUTE
    avg_exfil_per_min = [10.0, 2.0, 1.0, 0.25, 0.1] #, 0.1] #[10.0, 2.0, 1.0, 0.25, 0.1] # [10.0, 2.0, 1.0, 0.25, 0.1] # [10.0, 2.0, 1.0, 0.25, 0.1]
    exfil_per_min_variance = [0.3, 0.2, 0.15, 0.08, 0.05] #, 0.05] #[0.3, 0.2, 0.15, 0.08, 0.05] # [0.3. 0.2, 0.15, 0.08, 0.05] #[0.3, 0.2, 0.15, 0.08, 0.05]
    avg_pkt_size = [500.0, 500.0, 500.00, 500.00, 500.0]
    pkt_size_variance = [100, 100, 100, 100, 100]

    BytesPerMegabyte = 1000000
    avg_exfil_per_min = [BytesPerMegabyte * i for i in avg_exfil_per_min]
    exfil_per_min_variance = [BytesPerMegabyte * i for i in exfil_per_min_variance]

    # max_number_of_paths = 20 ## not sure if I want to do this still...
    ######

    calc_vals = True
    calculate_z_scores = True
    include_ide = False # include ide vals? this'll involve either calculating them (below) or grabbing them from the file location
    calc_ide = False
    only_ide = False ## ONLY calculate the ide values... this'll be useful if I wanna first calc all the other values and THEN ide...
    ide_window_size = 10 # size of the sliding window over which ide operates
    drop_pairwise_features = False # drops pairwise features (i.e. serviceX_to_serviceY_reciprocity)
    drop_infra_from_graph = False ## TODO: doesn't do anything yet.

    ####
    cur_experiment_name = "mark4_26_adjAT_"
    base_output_location = '/Volumes/exM2/experimental_data/sockshop_summary_new/new_'# + 'lasso_roc'
    base_output_location += cur_experiment_name
    if drop_pairwise_features:
        base_output_location += 'dropPairWise_'

    #####

    skip_graph_injection = False
    get_endresult_from_memory = False # in this case, you'd skip literally the whole pipeline and just get the
                                      # trained model + the results (from that model) out of memory
                                      # I anticpate that this'll mostly be useful for working on generating
                                      # the final results report + the graphs + other stuff kinda...

    # let's actually remove rest of these too...
    experiment_classes = [sockshop_thirteen_autoscale_mark4(time_of_synethic_exfil=time_of_synethic_exfil,
                                                time_interval_lengths=time_interval_lengths)]

    return multi_experiment_pipeline(experiment_classes, base_output_location, True, time_of_synethic_exfil,
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
                              drop_infra_from_graph=drop_infra_from_graph,
                              ide_window_size=ide_window_size)


def new_wordpress_autoscaling_recipe():
    skip_model_part = False
    ignore_physical_attacks_p = True

    time_of_synethic_exfil = 30 # sec
    goal_train_test_split_training = 0.5
    goal_attack_NoAttack_split_training = 0.6
    goal_attack_NoAttack_split_testing = 0.2

    #time_interval_lengths = [10, 30]#, 10] #[30, 10, 1] #[30, 10, 1] #[30, 10, 1]#,
    time_interval_lengths = [10, 30]#, 10, 100]#, 10] #[30, 10, 1] #[30, 10, 1] #[30, 10, 1]#,

    # ide_window_size is for ide_angles and the other things that use the angle between the principal eigenvectors
    ide_window_size = 10

    #####
    # IN MEGABYTES / MINUTE
    avg_exfil_per_min = [10.0, 1.0, 0.25, 0.1]#, 0.1 ] #[10.0, 2.0,
    exfil_per_min_variance = [0.3, 0.15, 0.08, 0.05] #, 0.05] # 0.3, 0.2,
    avg_pkt_size = [500.0, 500.00, 500.00, 500.00]#, 500.0] # 500.0, 500.0,
    pkt_size_variance = [100, 100, 100, 100]#, 100] # 100, 100,

    BytesPerMegabyte = 1000000
    avg_exfil_per_min = [BytesPerMegabyte * i for i in avg_exfil_per_min]
    exfil_per_min_variance = [BytesPerMegabyte * i for i in exfil_per_min_variance]
    ######

    calc_vals = True
    calculate_z_scores = True
    include_ide = False # include ide vals? this'll involve either calculating them (below) or grabbing them from the file location
    calc_ide = False
    only_ide = False ## ONLY calculate the ide values... this'll be useful if I wanna first calc all the other values and THEN ide...
    drop_pairwise_features = False # drops pairwise features (i.e. serviceX_to_serviceY_reciprocity)

    ####
    cur_experiment_name = "autoscaling_mark7_orderMagTimeGran_"
    base_output_location = '/Volumes/exM2/experimental_data/wordpress_summary_new/new_'# + 'lasso_roc'
    base_output_location += cur_experiment_name
    if drop_pairwise_features:
        base_output_location += 'dropPairWise_'
    #####

    skip_graph_injection = False
    get_endresult_from_memory = False # in this case, you'd skip literally the whole pipeline and just get the
                                      # trained model + the results (from that model) out of memory
                                      # I anticpate that this'll mostly be useful for working on generating
                                      # the final results report + the graphs + other stuff kinda...

    experiment_classes = [wordpress_fourteen_mark7(time_of_synethic_exfil=time_of_synethic_exfil,
                                                time_interval_lengths=time_interval_lengths)]

    return multi_experiment_pipeline(experiment_classes, base_output_location, True, time_of_synethic_exfil,
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
                              ide_window_size=ide_window_size)


def new_wordpress_recipe():
    skip_model_part = False
    ignore_physical_attacks_p = True

    time_of_synethic_exfil = 30 # sec
    goal_train_test_split_training = 0.5
    goal_attack_NoAttack_split_training = 0.6
    goal_attack_NoAttack_split_testing = 0.2

    time_interval_lengths = [10, 30] #[30, 10] #[30, 10, 1] #[30, 10, 1] #[30, 10, 1]#,

    ###

    #####
    # IN MEGABYTES / MINUTE
    avg_exfil_per_min =  [10.0, 1.0, 0.25, 0.1]#, 10.0 2.0, 1.0, 0.25, 0.1] #[100000000.0] # [100.0
    exfil_per_min_variance = [0.3, 0.15, 0.08, 0.05]# 0.3, 0.2, 0.15, 0.08, 0.05] #[100.0] # 1.0,
    avg_pkt_size = [500.0, 500.0, 500.0, 500.0] #500.0 , 500.0, 500.00, 500.00, 500.0] #[1000.0] # 500.0,
    pkt_size_variance = [100, 100, 100, 100] #100, 100, 100, 100, 100] #[100] #100,

    BytesPerMegabyte = 1000000
    avg_exfil_per_min = [BytesPerMegabyte * i for i in avg_exfil_per_min]
    exfil_per_min_variance = [BytesPerMegabyte * i for i in exfil_per_min_variance]
    ######

    calc_vals = True
    calculate_z_scores = True
    calc_ide = False
    include_ide = True
    only_ide = False ## ONLY calculate the ide values... this'll be useful if I wanna first calc all the other values and THEN ide...
    ide_window_size = 12
    drop_pairwise_features = False # drops pairwise features (i.e. serviceX_to_serviceY_reciprocity)

    ###############

    ####
    cur_experiment_name = "v2_testingNewPipeline"  # can modify if you want, probably with:  new_wordpress_recipe.__name__
    base_output_location = '/Volumes/exM2/experimental_data/wordpress_summary_new/new_'# + 'lasso_roc'
    base_output_location += cur_experiment_name
    if drop_pairwise_features:
        base_output_location += 'dropPairWise_'
    #####

    skip_graph_injection = False
    get_endresult_from_memory = False # in this case, you'd skip literally the whole pipeline and just get the
                                      # trained model + the results (from that model) out of memory
                                      # I anticpate that this'll mostly be useful for working on generating
                                      # the final results report + the graphs + other stuff kinda...

    experiment_classes = [wordpress_thirteen_t2(time_of_synethic_exfil=time_of_synethic_exfil,
                                                time_interval_lengths=time_interval_lengths)]

    return multi_experiment_pipeline(experiment_classes, base_output_location, True, time_of_synethic_exfil,
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
                              ide_window_size=ide_window_size)

if __name__=="__main__":
    print "RUNNING"
    print sys.argv

    if len(sys.argv) == 1:
        print "running_preset..."
        #autoscaling_sockshop_recipe()
        #nonauto_sockshop_recipe()
        #new_wordpress_autoscaling_recipe()
        #new_wordpress_recipe()

        #print os.getcwd()
        #parse_experimental_config('./analysis_json/sockshop_one_v2.json')
        parse_experimental_config('./analysis_json/wordpress_one_v2_nonauto.json')
    elif len(sys.argv) == 2:
        experimental_config_file = sys.argv[1]
        parse_experimental_config(experimental_config_file)
    else:
        print "too many args!"