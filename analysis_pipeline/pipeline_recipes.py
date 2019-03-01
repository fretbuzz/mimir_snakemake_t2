import json
import time

import pyximport
pyximport.install() # am I sure that I want this???

import sys
import gc
import functools

import matplotlib
matplotlib.use('Agg',warn=False, force=True)

from analysis_pipeline.pipeline_coordinator import data_anylsis_pipline, multi_experiment_pipeline

'''
This file is essentially just sets of parameters for the run_data_analysis_pipeline function in pipeline_coordinator.py
There are a lot of parameters, and some of them are rather long, so I decided to make a function to store them in
'''

# these lists are only need for processing the k8s pod info
#microservices_sockshop = ['carts-db', 'carts', 'catalogue-db', 'catalogue', 'front-end', 'orders-db', 'orders',
#                         'payment', 'queue-master', 'rabbitmq', 'session-db', 'shipping', 'user-db', 'user',
#                          'load-test']
microservices_sockshop = ['carts-db', 'carts', 'catalogue-db', 'catalogue', 'front-end', 'orders-db', 'orders',
                         'payment', 'queue-master', 'rabbitmq', 'shipping', 'user-db', 'user']
minikube_infrastructure = ['etcd', 'kube-addon-manager', 'kube-apiserver', 'kube-controller-manager',
                           'kube-dns', 'kube-proxy', 'kube-scheduler', 'kubernetes-dashboard', 'metrics-server',
                           'storage-provisioner']
microservices_wordpress = ['mariadb-master', 'mariadb-slave', 'wordpress']

# here are some 'recipes'
# comment out the ones you are not using
def run_analysis_pipeline_recipes():
    # atsea store recipe
    '''
    pcap_paths = ['/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/seastore_redux_back-tier_1.pcap',
                   '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/seastore_redux_front-tier_1.pcap']
    is_swarm = 1
    basefile_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/edgefiles/seastore_swarm'
    basegraph_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/graphs/seastore_swarm'
    container_info_path = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_redux_docker_container_configs.txt'
    time_interval_lengths = [10, 1, 0.1] # seconds # note: 100 used to be here too
    ms_s = ['appserver', 'reverse_proxy', 'database']
    make_edgefiles = False
    start_time = 1529180898.56
    end_time = 1529181277.03
    exfil_start_time = 40
    exfil_end_time = 70
    calc_vals = False
    window_size = 6
    graph_p = True # should I make graphs?
    colors = ['b', 'r']
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p)
    '''

    # sockshop recipe
    '''
    # note: still gotta do calc_vals again...
    pcap_paths = ["/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/sockshop_swarm_fixed_br0_0.pcap"]
    is_swarm = 1
    basefile_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/edgefiles/sockshop_swarm_pipeline_br0'
    basegraph_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/graphs/sockshop_swarm'
    container_info_path = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/sockshop_swarm_fixed_containers_config.txt'
    time_interval_lengths = [10, 1] # seconds
    ms_s = microservices_sockshop
    make_edgefiles = True
    calc_vals = True
    graph_p = True # should I make graphs?
    start_time = 1529527610.6
    end_time = 1529527979.54
    window_size = 6
    colors = ['b', 'r']
    # here are some example colors:
    # b: blue ;  g: green  ;  r: red   ;   c: cyan    ; m: magenta
    exfil_start_time = None
    exfil_end_time = None
    wiggle_room = ??
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time, 
                               calc_vals = calc_vals, graph_p = graph_p)
    #'''
    # wordpress recipe [rep1 = any/all network interfaces]
    '''
    pcap_paths = ["/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_exp_one_rep1_default_bridge_0any.pcap"]
    #['/Users/jseverin/Documents/Microservices/munnin/experimental_data/wordpress_info/wordpress_exp_one_rep1_default_bridge_0docker0.pcap',
    #              '/Users/jseverin/Documents/Microservices/munnin/experimental_data/wordpress_info/wordpress_exp_one_rep1_default_bridge_0eth0.pcap',
    #              '/Users/jseverin/Documents/Microservices/munnin/experimental_data/wordpress_info/wordpress_exp_one_rep1_default_bridge_0eth1.pcap']
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/edgefiles/wordpress_exp_one_rep1'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/graphs/wordpress_exp_one_rep1'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_exp_one_rep1_docker_0_network_configs.txt'
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_exp_one_rep1_svc_config_0.txt'
    time_interval_lengths = [50, 30, 10, 1] #, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = ["k8s_POD_dbcmmz-mariadb-slave",  "k8s_POD_dbcmmz-mariadb-master", "k8s_POD_awwwppp-wordpress"]
    make_edgefiles = False
    start_time = None
    end_time = None
    exfil_start_time = None
    exfil_end_time = None
    calc_vals = True
    window_size = 6
    graph_p = True # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2 # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p, kubernetes_svc_info = kubernetes_svc_info)
    #'''

    '''
    # wordpress exp 1 (k8s, no network plugins)
    pcap_paths = ['/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_one_default_bridge_0docker0.pcap',
                  '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_one_default_bridge_0eth0.pcap',
                  '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_one_default_bridge_0eth1.pcap']
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/edgefiles/wordpress_exp_one'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/graphs/wordpress_exp_one'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_one_docker_0_network_configs.txt'
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_exp_one_rep1_svc_config_0.txt'
    time_interval_lengths = [50, 30, 10, 1] #, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = ["k8s_POD_dbcmmz-mariadb-slave",  "k8s_POD_dbcmmz-mariadb-master", "k8s_POD_awwwppp-wordpress"]
    make_edgefiles = True
    start_time = None
    end_time = None
    exfil_start_time = None
    exfil_end_time = None
    calc_vals = True
    window_size = 6
    graph_p = True # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2 # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p, kubernetes_svc_info = kubernetes_svc_info)

    #'''

    # sockshop recipe (new exp1)
    '''
    pcap_paths = ["/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_exp_six_rep1_sockshop_default_0.pcap",
                  "/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_exp_six_rep1_ingress_sbox_0.pcap",
                  "/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_exp_six_rep1_ingress_0.pcap",
                  "/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_exp_six_rep1_bridge_0.pcap"]
    is_swarm = 1
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/edgefiles/sockshop_exp_six_rep1_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/graphs/sockshop_exp_six_rep1_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_exp_six_rep1_docker_0_network_configs.txt'
    time_interval_lengths = [50, 30, 10, 1] #, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = microservices_sockshop
    make_edgefiles = False
    start_time = 1533994537.35
    end_time = 1533995436.18
    exfil_start_time = 270
    exfil_end_time = 330
    calc_vals = True
    window_size = 6
    graph_p = True # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2 # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p)
    #'''

    # sockshop exp 3 (no exfil, on k8s)
    '''
    pcap_paths = ["/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_three_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/edgefiles/sockshop_exp_three_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/graphs/sockshop_exp_three_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_three_docker_0_network_configs.txt'
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_three_svc_config_0.txt'
    time_interval_lengths = [50]#, 30, 10, 1] #, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = microservices_sockshop
    make_edgefiles = True
    start_time = None
    end_time = None
    exfil_start_time = None
    exfil_end_time = None
    calc_vals = True
    window_size = 6
    graph_p = True # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2 # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p, kubernetes_svc_info=kubernetes_svc_info)
    #'''

    # sockshop exp 1 (rep 0)
    ''' # note: still gotta do calc_vals again...
    pcap_paths = ["/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/sockshop_one_sockshop_default_0.pcap"]
    is_swarm = 1
    basefile_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/edgefiles/sockshop_one_pipeline_br0'
    basegraph_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/graphs/sockshop_one'
    container_info_path = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/sockshop_one_docker_0_container_configs.txt'
    time_interval_lengths = [10, 1]# , .1] # seconds (note eventually the 0.1 gran should be done and can re-enable)
    ms_s = microservices_sockshop
    make_edgefiles = False
    calc_vals = True
    graph_p = True # should I make graphs?
    start_time = None
    end_time = None
    window_size = 6
    colors = ['b', 'r']
    exfil_start_time = 270
    exfil_end_time = 310
    wiggle_room = ??
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p)
    #'''
    # sockshop exp 2 (rep 0)
    ''' # note: still gotta do calc_vals again...
    pcap_paths = ["/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/sockshop_two_sockshop_default_0.pcap"]
    is_swarm = 1
    basefile_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/edgefiles/sockshop_two_pipeline_br0'
    basegraph_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/graphs/sockshop_two'
    container_info_path = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/sockshop_two_docker_0_container_configs.txt'
    time_interval_lengths = [50, 10, 1]# , .1] # seconds (note eventually the 0.1 gran should be done and can re-enable)
    ms_s = microservices_sockshop
    make_edgefiles = True
    calc_vals = False
    graph_p = False # should I make graphs?
    start_time = None
    end_time = None
    window_size = 6
    colors = ['b', 'r']
    exfil_start_time = 270
    exfil_end_time = 330
    wiggle_room = ???
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p)
    #'''
    '''
    # sockshop exp 3 (rep 0) [[old, probably do not want]]
    pcap_paths = ["/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/sockshop_three_sockshop_default_0.pcap"]
    is_swarm = 1
    basefile_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/edgefiles/sockshop_three_0'
    basegraph_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/graphs/sockshop_three_0'
    container_info_path = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/sockshop_three_docker_0_container_configs.txt'
    time_interval_lengths = [50, 10, 1]#, .1] # seconds
    ms_s = microservices_sockshop
    make_edgefiles = False
    calc_vals = False
    graph_p = True # should I make graphs?
    start_time = None
    end_time = None
    window_size = 6
    colors = ['b', 'r']
    exfil_start_time = 300
    exfil_end_time = 360
    wiggle_room = ??
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p)

    #'''

    # sockshop exp 8 (no exfil, on k8s, using cilium network plguin)
    '''
    pcap_paths = ["/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_eight_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/edgefiles/sockshop_exp_eight_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/graphs/sockshop_exp_eight_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_eight_docker_0_network_configs.txt'
    cilium_config_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_eight_0_cilium_network_configs.txt'
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_eight_svc_config_0.txt'
    time_interval_lengths = [50, 30, 10, 1] #, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = microservices_sockshop
    make_edgefiles = True
    start_time = None
    end_time = None
    exfil_start_time = None
    exfil_end_time = None
    calc_vals = True
    window_size = 6
    graph_p = True # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2 # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p, kubernetes_svc_info=kubernetes_svc_info, 
                               cilium_config_path=cilium_config_path)
    #'''

    # sockshop exp 8 (no exfil, on k8s, using cilium network plguin) [rep2 = I stopped those wierd load-balancer containers]
    '''
    pcap_paths = ["/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_eight_rep2_noloadtest_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/edgefiles/sockshop_exp_eight_rep2'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/graphs/sockshop_exp_eight_rep2'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_eight_rep2_noloadtest_docker_0_network_configs.txt'
    cilium_config_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_eight_rep2_noloadtest_0_cilium_network_configs.txt'
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_eight_rep2_noloadtest_svc_config_0.txt'
    time_interval_lengths = [50, 30, 10, 1] #, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = microservices_sockshop
    make_edgefiles = True
    start_time = None
    end_time = None
    exfil_start_time = None
    exfil_end_time = None
    calc_vals = True
    window_size = 6
    graph_p = True # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2 # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p, kubernetes_svc_info=kubernetes_svc_info,
                               cilium_config_path=cilium_config_path, rdpcap_p=False)
    #'''

    # atsea exp 2 (v2)
    '''
    pcap_paths = ['/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_two_v2__atsea_back-tier_0.pcap',
                   '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_two_v2__atsea_front-tier_0.pcap',
                  '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_two_v2__ingress_0.pcap']
    is_swarm = 1
    basefile_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/edgefiles/atsea_store_exp_two_v2_'
    basegraph_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/graphs/atsea_store_exp_two_v2_'
    container_info_path = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_two_v2__docker_0_network_configs.txt'
    time_interval_lengths = [50, 10]#50, , 1] #, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = ['appserver_VIP', 'reverse_proxy_VIP', 'database_VIP', 'appserver', 'reverse_proxy', 'database', 'back-tier', 'front-tier']
    make_edgefiles = True
    start_time = 1533310837.05
    end_time = 1533311351.12
    exfil_start_time = 270
    exfil_end_time = 330
    calc_vals = True
    make_net_graphs_p = True
    window_size = 6
    graph_p = True # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p, make_net_graphs_p=make_net_graphs_p)
    #'''

    # atsea exp 2 (v7) [good]
    '''
    pcap_paths = ['/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_exp_two_v7__atsea_back-tier_0.pcap',
                   '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_exp_two_v7__atsea_front-tier_0.pcap']#,
                  # note: If I go back to doing the normal thing, then I should re-enable the stuff below
                  #'/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_exp_two_v7__ingress_0.pcap',
                  #'/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_exp_two_v7__bridge_0.pcap',
                  #'/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_exp_two_v7__ingress_sbox_0.pcap']
    is_swarm = True
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/edgefiles/atsea_store_exp_two_v7_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/graphs/atsea_store_exp_two_v7_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_exp_two_v7__docker_0_network_configs.txt'
    time_interval_lengths = [.000001]#[30, 10, 1] #, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = ['appserver_VIP', 'reverse_proxy_VIP', 'database_VIP', 'appserver', 'reverse_proxy', 'database', 'back-tier', 'front-tier', 'visualizer']
    make_edgefiles = True
    start_time = 1533377817.89
    end_time = 1533378712.2
    exfil_start_time = 270
    exfil_end_time = 330
    make_net_graphs_p = False # do you want to make network
    calc_vals = False
    window_size = 6
    graph_p = False # should I make graphs?
    colors = ['b', 'r']
    rdpcap_p = True
    wiggle_room = 2 # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p, make_net_graphs_p=make_net_graphs_p, rdpcap_p=rdpcap_p)
    #'''
    # atsea exp 3 (v2) [good]
    '''
    pcap_paths = ['/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_exp_three_v2__atsea_back-tier_0.pcap',
                  '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_exp_three_v2__atsea_front-tier_0.pcap',
                  '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_exp_three_v2__ingress_0.pcap',
                  '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_exp_three_v2__bridge_0.pcap',
                  '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_exp_three_v2__ingress_sbox_0.pcap']
    is_swarm = 1
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/edgefiles/atsea_store_exp_three_v2_pcapreader_' #pcapreader_
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/graphs/atsea_store_exp_three_v2_pcapreader' # pcapreader
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_exp_three_v2__docker_0_network_configs.txt'
    time_interval_lengths = [50, 30, 10, 1]#50, , 1] #, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = ['appserver_VIP', 'reverse_proxy_VIP', 'database_VIP', 'appserver', 'reverse_proxy', 'database', 'back-tier', 'front-tier']
    make_edgefiles = False
    start_time = 1533381724.66 #None
    end_time = 1533382619.64 #None
    exfil_start_time = 300
    exfil_end_time = 360
    calc_vals = False
    window_size = 6
    graph_p = True # should I make graphs?
    make_net_graphs_p = True
    colors = ['b', 'r']
    wiggle_room = 2
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p, make_net_graphs_p=make_net_graphs_p)
    #'''

    '''
    # atsea exp 3 (rep 0)
    pcap_paths = ['/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_three_atsea_front-tier_0.pcap']
    # ^^ only a single value in list b/c connected database to front-tier network (so backtier didn't do anything)
    is_swarm = 1
    basefile_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/edgefiles/atsea_store_three'
    basegraph_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/graphs/atsea_store_three'
    container_info_path = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_three_docker_0_container_configs.txt'
    time_interval_lengths = [50] #, 10, 1, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = ['appserver', 'reverse_proxy', 'database']
    make_edgefiles = True
    start_time = None
    end_time = None
    exfil_start_time = 300
    exfil_end_time = 360
    calc_vals = True
    window_size = 6
    graph_p = True # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = ??
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p)
    #'''

    '''
    # Wordpress exp 4 (wordpress w/ HA cluster on cilium that is not configured)
    pcap_paths = ["/mydata/mimir/wordpress_four_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/mydata/mimir/temp_expName/edgefiles/wordpress_four_'
    basegraph_name = '/mydata/mimir/temp_expName/graphs/wordpress_four_'
    container_info_path = '/mydata/mimir/wordpress_four_docker_0_network_configs.txt'
    cilium_config_path = '/mydata/mimir/wordpress_four_0_cilium_network_configs.txt'
    kubernetes_svc_info = '/mydata/mimir/wordpress_four_svc_config_0.txt'
    time_interval_lengths = [50, 30, 10, 1] #, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s =  ["my-release-pxc", "wwwppp-wordpress"] 
    make_edgefiles = False
    start_time = None
    end_time = None
    exfil_start_time = 100
    exfil_end_time = 150
    calc_vals = True
    window_size = 6
    graph_p = True # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2 # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p, kubernetes_svc_info=kubernetes_svc_info, cilium_config_path=cilium_config_path,
    			       rdpcap_p=True)
    #'''
    '''
    # Wordpress exp 5 (wordpress w/ HA cluster on cilium that is configured per the yaml files in the experimental_configuration folder)
    pcap_paths = ["/mydata/mimir/wordpress_six_take_2_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/mydata/mimir/temp_expName/edgefiles/wordpress_six_take_2_dns_'
    basegraph_name = '/mydata/mimir/temp_expName/graphs/wordpress_six_take_2_dns_'
    container_info_path = '/mydata/mimir/wordpress_six_take_2_docker_0_network_configs.txt'
    cilium_config_path = '/mydata/mimir/wordpress_six_take_2_0_cilium_network_configs.txt'
    kubernetes_svc_info = '/mydata/mimir/wordpress_six_take_2_svc_config_0.txt'
    time_interval_lengths = [50, 30, 10, 1] #, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s =  ["my-release-pxc", "wwwppp-wordpress"]
    make_edgefiles = True
    start_time = None
    end_time = None
    exfil_start_time = 250
    exfil_end_time = 300
    calc_vals = True
    window_size = 6
    graph_p = True # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2 # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p, kubernetes_svc_info=kubernetes_svc_info, cilium_config_path=cilium_config_path,
                               rdpcap_p=True)
    #'''

    '''
    # Wordpress, DNSCAT2 test
    pcap_paths = ["/mydata/mimir/dnscat_test_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/mydata/mimir/temp_expName/edgefiles/dnscat_test_'
    basegraph_name = '/mydata/mimir/temp_expName/graphs/dnscat_test_'
    container_info_path = '/mydata/mimir/dnscat_test_docker_0_network_configs.txt'
    cilium_config_path = '/mydata/mimir/dnscat_test_0_cilium_network_configs.txt'
    kubernetes_svc_info = '/mydata/mimir/dnscat_test_svc_config_0.txt'
    time_interval_lengths = [50, 30, 10, 1] #, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s =  ["my-release-pxc", "wwwppp-wordpress"] 
    make_edgefiles = True
    start_time = None
    end_time = None
    exfil_start_time = None
    exfil_end_time = None
    calc_vals = True
    window_size = 6
    graph_p = True # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2 # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p, kubernetes_svc_info=kubernetes_svc_info, cilium_config_path=cilium_config_path,
                               rdpcap_p=True)
    #'''

    # atsea exp6 -- send exfil data through VIP (note that only a portion of the data sent from the DB pod actually
    # made it to the outside of them deployment
    '''
    pcap_paths = ['/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_six_atsea_back-tier_0.pcap',
                  '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_six_atsea_front-tier_0.pcap',
                  '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_six_ingress_0.pcap',
                  '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_six_ingress_sbox_0.pcap']
    is_swarm = 1
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/edgefiles/atsea_store_six_' #pcapreader_
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/graphs/atsea_store_exp_six_' # pcapreader
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_six_docker_0_network_configs.txt'
    time_interval_lengths = [50, 30, 10, 1]#50, , 1] #, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = ['appserver_VIP', 'reverse_proxy_VIP', 'database_VIP', 'appserver', 'reverse_proxy', 'database', 'back-tier', 'front-tier']
    make_edgefiles = True
    start_time = None
    end_time = None
    exfil_start_time = 270
    exfil_end_time = 330
    calc_vals = True
    window_size = 6
    graph_p = True # should I make graphs?
    make_net_graphs_p = True
    colors = ['b', 'r']
    wiggle_room = 2
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p, make_net_graphs_p=make_net_graphs_p, rdpcap_p=False)
    #'''

    ''' # NOTE: I'm missing some pcaps that I'd need to do this analysis_pipeline... will probably want to re-run it
        # at some point
    # atsea exp7 - attempts to send the data through the endpoint (load-balancer). Note that this does not actualyl
    # work as no data is able to reach the outside
    pcap_paths = [
        '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_seven_atsea_back-tier_0.pcap',
        '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_six_atsea_front-tier_0.pcap',
        '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_six_ingress_0.pcap',
        '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_six_ingress_sbox_0.pcap']
    is_swarm = 1
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/edgefiles/atsea_store_six_'  # pcapreader_
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/graphs/atsea_store_exp_six_'  # pcapreader
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_six_docker_0_network_configs.txt'
    time_interval_lengths = [50, 30, 10,
                             1]  # 50, , 1] #, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = ['appserver_VIP', 'reverse_proxy_VIP', 'database_VIP', 'appserver', 'reverse_proxy', 'database', 'back-tier',
            'front-tier']
    make_edgefiles = True
    start_time = None
    end_time = None
    exfil_start_time = 270
    exfil_end_time = 330
    calc_vals = True
    window_size = 6
    graph_p = True  # should I make graphs?
    make_net_graphs_p = True
    colors = ['b', 'r']
    wiggle_room = 2
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals=calc_vals, graph_p=graph_p, make_net_graphs_p=make_net_graphs_p,
                               rdpcap_p=False)
    '''
    '''
    # wordpress- exp3: w/ exfil
    pcap_paths = ["/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_three_default_bridge_0any.pcap"]
    #['/Users/jseverin/Documents/Microservices/munnin/experimental_data/wordpress_info/wordpress_exp_one_rep1_default_bridge_0docker0.pcap',
    #              '/Users/jseverin/Documents/Microservices/munnin/experimental_data/wordpress_info/wordpress_exp_one_rep1_default_bridge_0eth0.pcap',
    #              '/Users/jseverin/Documents/Microservices/munnin/experimental_data/wordpress_info/wordpress_exp_one_rep1_default_bridge_0eth1.pcap']
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/edgefiles/wordpress_three'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/graphs/wordpress_three'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_three_docker_0_network_configs.txt'
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_three_svc_config_0.txt'
    time_interval_lengths = [50, 30, 10, 1] #, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = ["k8s_POD_dbcmmz-mariadb-slave",  "k8s_POD_dbcmmz-mariadb-master", "k8s_POD_awwwppp-wordpress"]
    make_edgefiles = True
    start_time = None
    end_time = None
    exfil_start_time = None
    exfil_end_time = None
    calc_vals = True
    window_size = 6
    graph_p = True # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2 # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p, kubernetes_svc_info = kubernetes_svc_info)
    #'''

    '''
    # Wordpress exp 4 (wordpress w/ HA cluster on cilium that is not configured)
    pcap_paths = ["/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_four_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/edgefiles/wordpress_four_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/graphs/wordpress_four_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_four_docker_0_network_configs.txt'
    cilium_config_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_four_0_cilium_network_configs.txt'
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_four_svc_config_0.txt'
    time_interval_lengths = [50, 30, 10, 1] #, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s =  ["my-release-pxc", "wwwppp-wordpress"]
    make_edgefiles = False
    start_time = None
    end_time = None
    exfil_start_time = 100
    exfil_end_time = 150
    calc_vals = True
    window_size = 6
    graph_p = True # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2 # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p, kubernetes_svc_info=kubernetes_svc_info, cilium_config_path=cilium_config_path)
    #'''
    '''
    # Wordpress exp 5 (wordpress w/ HA cluster on cilium that is configured per the yaml files in the experimental_configuration folder)
    pcap_paths = ["/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_five_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/edgefiles/wordpress_five_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/graphs/wordpress_five_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_five_docker_0_network_configs.txt'
    cilium_config_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_five_0_cilium_network_configs.txt'
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_five_svc_config_0.txt'
    time_interval_lengths = [50, 30, 10, 1] #, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s =  ["my-release-pxc", "wwwppp-wordpress"]
    make_edgefiles = False
    start_time = None
    end_time = None
    exfil_start_time = 100
    exfil_end_time = 150
    calc_vals = True
    window_size = 6
    graph_p = True # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2 # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p, kubernetes_svc_info=kubernetes_svc_info, cilium_config_path=cilium_config_path)
    #'''

    '''
    # Wordpress exp 6 (wordpress w/ HA cluster on cilium w/o security config, dnscat exfil from single db w/ 10 sec delay)
    pcap_paths = [
        "/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_full_scaled_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/edgefiles/wordpress_six_full_scaled_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/graphs/wordpress_six_full_scaled_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_full_scaled_docker_0_network_configs.txt'
    cilium_config_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_full_scaled_0_cilium_network_configs.txt'
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_full_scaled_svc_config_0.txt'
    kubernetes_pod_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_full_scaled_pod_config_0.txt'
    time_interval_lengths = [50, 30, 10]#,
                             #1]  # , 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = ["my-release-pxc", "wwwppp-wordpress"]
    make_edgefiles = False
    start_time = None
    end_time = None
    exfil_start_time = 600
    exfil_end_time = 650
    calc_vals = False
    window_size = 6
    graph_p = False  # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2  # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    percentile_thresholds = [50, 75, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    anomaly_window = 4
    anom_num_outlier_vals_in_window = 2
    alert_file = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/alerts/wordpress_six_'
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals=calc_vals, graph_p=graph_p, kubernetes_svc_info=kubernetes_svc_info,
                               cilium_config_path=cilium_config_path, rdpcap_p=False, calc_alerts_p=True,
                               percentile_thresholds=percentile_thresholds, anomaly_window = anomaly_window,
                               anom_num_outlier_vals_in_window = anom_num_outlier_vals_in_window,
                               alert_file = alert_file, ROC_curve_p=True, calc_tpr_fpr_p=True,
                               kubernetes_pod_info=kubernetes_pod_info)
    '''

    '''
    pcap_paths = [
        "/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/edgefiles/wordpress_six_syn_inter_packets_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/graphs/wordpress_six_syn_inter_packets_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_docker_0_network_configs.txt'
    cilium_config_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_0_cilium_network_configs.txt'
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_svc_config_0.txt'
    time_interval_lengths = [50, 30, 10,
                             1]  # , 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = ["my-release-pxc", "wwwppp-wordpress"]
    make_edgefiles = False
    start_time = None
    end_time = None
    exfil_start_time = 600
    exfil_end_time = 650
    calc_vals = False
    window_size = 6
    graph_p = True  # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2  # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals=calc_vals, graph_p=graph_p, kubernetes_svc_info=kubernetes_svc_info,
                               cilium_config_path=cilium_config_path, rdpcap_p=False)
    #'''

    #func_calls_here
    #process_wordpress6_rep4() # this is good (DNSCAT, from DB, 15 sec delay between packets)
    #time.sleep(27600)
    #process_wordpress7_rep3() # this is good.

    #process_wordpress8() # 20 min of scaled-up wordpress (wordpress_eight) (DET exfil on normal path)
    gc.collect()
    #process_sockshop9() # sockshop experiment 9 (an hour of scaled-up activity w/ exfil along normal path)
    gc.collect()
    #process_sockshop11() # sockshop experiment 11 (an hour of scaled-up activity w/ exfil NOT on normal path)
    gc.collect()
    #process_sockshop12() # sockshop experiment 12 (an hour of scaled-up activity w/ exfil going straight out)
    print "wow, we ran all that stuff"

    # other stuff
    #process_wordpress6_rep3()
    #process_wordpress7_rep2()
    process_wordpress8_rep_2()
    gc.collect()
    process_wordpress8_rep_3()

    # should NOTE use (for now at least)
    ### NO ## process_wordpress7() # Wordpress exp 7 (wordpress w/ HA cluster on cilium w/o security config, dnscat exfil from single WP w/ 15 sec delay)
    ### NO ##   process_wordpress6() # 20 min of moderate-scale wordpress w/ dnscat exfil from DB w/ 15 sec pause between packets

# 20 min of moderate-scale wordpress w/ dnscat exfil from DB w/ 15 sec pause between packets
def process_wordpress6():
    #time.sleep(9000)
    #''' # w/ next-gen metrics
    pcap_paths = [
        "/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_rep_2_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/edgefiles/wordpress_six_rep_2_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/graphs/wordpress_six_rep_2_'
    alert_file = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/alerts/wordpress_six_rep_2_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_rep_2_docker_0_network_configs.txt'
    cilium_config_path = None # does NOT sue cilium on reps 2-4
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_rep_2_svc_config_0.txt'
    kubernetes_pod_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_rep_2_pod_config_0.txt'
    time_interval_lengths = [60, 30, 10]#,
    #1]  # , 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = ["my-release-pxc", "wwwppp-wordpress"]
    make_edgefiles = False
    start_time = None
    end_time = None
    exfil_start_time = 600
    exfil_end_time = 650
    calc_vals = True
    window_size = 6
    graph_p = False  # should I make graphs?
    colors = ['b', 'r']
    sec_between_exfil_events = 15
    wiggle_room = 2  # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    percentile_thresholds = [30, 40, 50, 60, 70, 75, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    anomaly_window = [1, 4]
    anom_num_outlier_vals_in_window = [1, 2] # note: these vals correspond to anoamly_window (so the first vals get matched, etc.)
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths, ms_s,
                               make_edgefiles, basegraph_name, window_size, colors, exfil_start_time, exfil_end_time,
                               wiggle_room, start_time=start_time, end_time=end_time, calc_vals=calc_vals,
                               graph_p=graph_p, kubernetes_svc_info=kubernetes_svc_info,
                               cilium_config_path=cilium_config_path, rdpcap_p=False,
                               kubernetes_pod_info=kubernetes_pod_info, alert_file=alert_file, ROC_curve_p=True,
                               calc_zscore_p=True, sec_between_exfil_events=sec_between_exfil_events)
    #'''

def process_wordpress6_rep2():
    #'''
    # Wordpress exp 6 rep3 (wordpress w/ HA cluster on cilium w/o security config, dnscat exfil from single db w/ 15 sec delay)
    pcap_paths = [
        "/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_rep_2_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/edgefiles/wordpress_six_rep_2_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/graphs/wordpress_six_rep_2_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_rep_2_docker_0_network_configs.txt'
    cilium_config_path = None # does NOT sue cilium on reps 2-4
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_rep_2_svc_config_0.txt'
    kubernetes_pod_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_rep_2_pod_config_0.txt'
    time_interval_lengths = [60, 30, 10]#,
                             #1]  # , 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = ["my-release-pxc", "wwwppp-wordpress"]
    make_edgefiles = True
    start_time = False
    end_time = None
    exfil_start_time = 600
    exfil_end_time = 650
    calc_vals = True
    window_size = 6
    graph_p = False  # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2  # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    percentile_thresholds = [50, 75, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    anomaly_window = [1, 4]
    sec_between_exfil_events = 15
    anom_num_outlier_vals_in_window = [1, 2] # note: these vals correspond to anoamly_window (so the first vals get matched, etc.)
    alert_file = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/alerts/wordpress_six_rep_2_'
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths, ms_s,
                               make_edgefiles, basegraph_name, window_size, colors, exfil_start_time, exfil_end_time,
                               wiggle_room, start_time=start_time, end_time=end_time, calc_vals=calc_vals,
                               graph_p=graph_p, kubernetes_svc_info=kubernetes_svc_info,
                               cilium_config_path=cilium_config_path, rdpcap_p=False,
                               kubernetes_pod_info=kubernetes_pod_info, alert_file=alert_file, ROC_curve_p=True,
                               calc_zscore_p=True, sec_between_exfil_events=sec_between_exfil_events)
    #'''

def process_wordpress6_rep3(time_of_synethic_exfil=None, only_exp_info=False, initiator_info_for_paths=None,
                            portion_for_training=None, training_window_size=None, size_of_neighbor_training_window=None,
                            synthetic_exfil_paths_train=None,
                            synthetic_exfil_paths_test=None, calc_vals=False,
                            skip_model_part=False,max_number_of_paths=None):
    #'''
    #calc_vals=False
    # Wordpress exp 6 rep3 (wordpress w/ HA cluster on cilium w/o security config, dnscat exfil from single db w/ 15 sec delay)
    pcap_paths = [
        "/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_rep_3_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/edgefiles/wordpress_six_rep_3_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/graphs/wordpress_six_rep_3_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_rep_3_docker_0_network_configs.txt'
    cilium_config_path = None # does NOT use cilium on reps 2-4
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_rep_3_svc_config_0.txt'
    kubernetes_pod_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_rep_3_pod_config_0.txt'
    time_interval_lengths = [30, 10]#,
                             #1]  # , 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = ["my-release-pxc", "wwwppp-wordpress"]
    make_edgefiles = False ## already done!
    start_time = False
    end_time = None
    exfil_start_time = 600
    exfil_end_time = 650
    #calc_vals = False #
    window_size = 6
    graph_p = False  # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2  # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    percentile_thresholds = [50, 75, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    anomaly_window = [1, 4]
    anom_num_outlier_vals_in_window = [1, 2] # note: these vals correspond to anoamly_window (so the first vals get matched, etc.)
    alert_file = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/alerts/wordpress_six_rep_3_'
    sec_between_exfil_events = 15
    physical_exfil_path = ['my_release_pxc_pod', 'kube_dns_vip', 'kube_dns_pod', 'internet']
    #training_window_size
    process_wordpress6_rep3_object = data_anylsis_pipline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths, ms_s,
                                   make_edgefiles, basegraph_name, window_size, colors, exfil_start_time, exfil_end_time,
                                   wiggle_room, start_time=start_time, end_time=end_time, calc_vals=calc_vals,
                                   graph_p=graph_p, kubernetes_svc_info=kubernetes_svc_info,
                                   cilium_config_path=cilium_config_path, rdpcap_p=False,
                                   kubernetes_pod_info=kubernetes_pod_info, alert_file=alert_file, ROC_curve_p=True,
                                   calc_zscore_p=True, sec_between_exfil_events=sec_between_exfil_events,
                                   injected_exfil_path = physical_exfil_path, only_exp_info=only_exp_info,
                                   time_of_synethic_exfil=time_of_synethic_exfil,
                                   initiator_info_for_paths=initiator_info_for_paths,
                                   end_of_training=portion_for_training,
                                   training_window_size=training_window_size, size_of_neighbor_training_window=size_of_neighbor_training_window,
                                   synthetic_exfil_paths_train=synthetic_exfil_paths_train, synthetic_exfil_paths_test=synthetic_exfil_paths_test,
                                   skip_model_part=skip_model_part,
                                   max_number_of_paths=max_number_of_paths)

    return process_wordpress6_rep3_object

def process_wordpress6_rep4(time_of_synethic_exfil=None, only_exp_info=False, initiator_info_for_paths=None,
                            portion_for_training=None, training_window_size=None, size_of_neighbor_training_window=None,
                            synthetic_exfil_paths_train=None,
                            synthetic_exfil_paths_test=None, calc_vals=False, skip_model_part=False,
                            max_number_of_paths=None):
     #'''
     #calc_vals = False
     # Wordpress exp 6 rep4 (wordpress w/ HA cluster on cilium w/o security config, dnscat exfil from single db w/ 15 sec delay)
     pcap_paths = [
         "/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_rep_4_default_bridge_0any.pcap"]
     is_swarm = 0
     basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/edgefiles/wordpress_six_rep_4_'
     basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/graphs/wordpress_six_rep_4_'
     container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_rep_4_docker_0_network_configs.txt'
     cilium_config_path = None  # does NOT sue cilium on reps 2-4
     kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_rep_4_svc_config_0.txt'
     kubernetes_pod_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_six_rep_4_pod_config_0.txt'
     time_interval_lengths = [30, 10]  # ,
     # 1]  # , 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
     ms_s = ["my-release-pxc", "wwwppp-wordpress"]
     make_edgefiles = False
     start_time = None
     end_time = None
     exfil_start_time = 600
     exfil_end_time = 650
     #calc_vals = False
     window_size = 6
     graph_p = False  # should I make graphs?
     colors = ['b', 'r']
     wiggle_room = 2  # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
     percentile_thresholds = [30, 40, 50, 60, 70, 75, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
     anomaly_window = [1, 4]
     anom_num_outlier_vals_in_window = [1, 2] # note: these vals correspond to anoamly_window (so the first vals get matched, etc.)
     alert_file = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/alerts/wordpress_six_rep_4_'
     sec_between_exfil_events = 15
     physical_exfil_path = ['my_release_pxc_pod', 'kube_dns_vip', 'kube_dns_pod', 'internet']

     process_wordpress6_rep4_object = data_anylsis_pipline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths, ms_s,
                                        make_edgefiles, basegraph_name, window_size, colors, exfil_start_time, exfil_end_time,
                                        wiggle_room, start_time=start_time, end_time=end_time, calc_vals=calc_vals,
                                        graph_p=graph_p, kubernetes_svc_info=kubernetes_svc_info,
                                        cilium_config_path=cilium_config_path, rdpcap_p=False,
                                        kubernetes_pod_info=kubernetes_pod_info, alert_file=alert_file, ROC_curve_p=True,
                                        calc_zscore_p=True, sec_between_exfil_events=sec_between_exfil_events,
                                        injected_exfil_path=physical_exfil_path, only_exp_info=only_exp_info,
                                        time_of_synethic_exfil=time_of_synethic_exfil,
                                        initiator_info_for_paths = initiator_info_for_paths,
                                        end_of_training=portion_for_training,
                                        training_window_size=training_window_size,
                                        size_of_neighbor_training_window=size_of_neighbor_training_window,
                                        synthetic_exfil_paths_train=synthetic_exfil_paths_train,
                                        synthetic_exfil_paths_test=synthetic_exfil_paths_test,
                                        skip_model_part=skip_model_part,
                                        max_number_of_paths=max_number_of_paths)
     return process_wordpress6_rep4_object

def process_wordpress7_rep3(time_of_synethic_exfil=None, only_exp_info=False, initiator_info_for_paths=None,
                            portion_for_training=None,training_window_size=None, size_of_neighbor_training_window=None,
                            synthetic_exfil_paths_train=None,
                            synthetic_exfil_paths_test=None, calc_vals=False, skip_model_part=False,
                            max_number_of_paths=None):
    #''' # here
    #calc_vals=False

    # Wordpress exp 7 rep 3(wordpress w/ HA cluster on cilium w/o security config, dnscat exfil from single WP w/ 15 sec delay)
    pcap_paths = ["/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_seven_rep_3_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/edgefiles/wordpress_seven_rep_3_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/graphs/wordpress_seven_rep_3_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_seven_rep_3_docker_0_network_configs.txt'
    cilium_config_path = None
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_seven_rep_3_svc_config_0.txt'
    kubernetes_pod_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_seven_rep_3_pod_config_0.txt'
    time_interval_lengths = [30, 10]#,
    # 1]  # , 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = ["my-release-pxc", "wwwppp-wordpress"]
    make_edgefiles = False
    start_time = None
    end_time = None
    exfil_start_time = 600
    exfil_end_time = 650
    #calc_vals = False
    window_size = 6
    graph_p = False  # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2  # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    percentile_thresholds = [30, 40, 50, 60, 70, 75, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    anomaly_window = [1, 4]
    anom_num_outlier_vals_in_window = [1, 2]
    sec_between_exfil_events = 15
    physical_exfil_path = ['wwwppp_wordpress_pod', 'kube_dns_vip', 'kube_dns_pod', 'internet']
    alert_file = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/alerts/wordpress_seven_rep_3_'
    process_wordpress7_rep3_object = \
        data_anylsis_pipline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths, ms_s,
                                   make_edgefiles, basegraph_name, window_size, colors, exfil_start_time, exfil_end_time,
                                   wiggle_room, start_time=start_time, end_time=end_time, calc_vals=calc_vals,
                                   graph_p=graph_p, kubernetes_svc_info=kubernetes_svc_info,
                                   cilium_config_path=cilium_config_path, rdpcap_p=False,
                                   kubernetes_pod_info=kubernetes_pod_info, alert_file=alert_file, ROC_curve_p=True,
                                   calc_zscore_p=True, sec_between_exfil_events=sec_between_exfil_events,
                                   injected_exfil_path=physical_exfil_path, only_exp_info=only_exp_info,
                                   time_of_synethic_exfil=time_of_synethic_exfil,
                                   initiator_info_for_paths = initiator_info_for_paths,
                                   end_of_training=portion_for_training,
                                   training_window_size=training_window_size,
                                   size_of_neighbor_training_window=size_of_neighbor_training_window,
                                   synthetic_exfil_paths_train=synthetic_exfil_paths_train,
                                   synthetic_exfil_paths_test=synthetic_exfil_paths_test,
                                   skip_model_part=skip_model_part,
                                   max_number_of_paths=max_number_of_paths)
    #return time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, time_gran_to_feature_dataframe, fourth_return_val, fifth_val
    # '''
    return process_wordpress7_rep3_object

# Wordpress exp 7 (wordpress w/ HA cluster on cilium w/o security config, dnscat exfil from single WP w/ 15 sec delay)
def process_wordpress7():
    #time.sleep(18000)
    #''' next-gen metrics
    # Wordpress exp 7 (wordpress w/ HA cluster on cilium w/o security config, dnscat exfil from single WP w/ 15 sec delay)
    pcap_paths = [
        "/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_seven_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/edgefiles/wordpress_seven_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/graphs/wordpress_seven_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_seven_docker_0_network_configs.txt'
    cilium_config_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_seven_0_cilium_network_configs.txt'
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_seven_svc_config_0.txt'
    kubernetes_pod_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_seven_pod_config_0.txt'
    time_interval_lengths = [60, 30, 10]#,
                             #1]  # , 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = ["my-release-pxc", "wwwppp-wordpress"]
    make_edgefiles = False
    start_time = None
    end_time = None
    exfil_start_time = 600
    exfil_end_time = 650
    calc_vals = True
    window_size = 6
    graph_p = False  # should I make graphs?
    colors = ['b', 'r', 'y']
    wiggle_room = 2  # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    percentile_thresholds = [50, 75, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    anomaly_window = [1, 4]
    anom_num_outlier_vals_in_window = [1, 2] # note: these vals correspond to anoamly_window (so the first vals get matched, etc.)
    alert_file = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/alerts/wordpress_seven_'
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths, ms_s,
                               make_edgefiles, basegraph_name, window_size, colors, exfil_start_time, exfil_end_time,
                               wiggle_room, start_time=start_time, end_time=end_time, calc_vals=calc_vals,
                               graph_p=graph_p, kubernetes_svc_info=kubernetes_svc_info,
                               cilium_config_path=cilium_config_path, rdpcap_p=False,
                               kubernetes_pod_info=kubernetes_pod_info, alert_file=alert_file, ROC_curve_p=True,
                               calc_zscore_p=True)
    #'''

def process_wordpress7_rep2(time_of_synethic_exfil=None, only_exp_info=False, initiator_info_for_paths=None,
                            portion_for_training=None, training_window_size=None, size_of_neighbor_training_window=None,
                            synthetic_exfil_paths_train=None,
                            synthetic_exfil_paths_test=None, calc_vals=False, skip_model_part=False,
                            max_number_of_paths=None):
    #'''
    #calc_vals=False

    # Wordpress exp 7 rep 2(wordpress w/ HA cluster on cilium w/o security config, dnscat exfil from single WP w/ 15 sec delay)
    pcap_paths = [
        "/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_seven_rep_2_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/edgefiles/wordpress_seven_rep_2_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/graphs/wordpress_seven_rep_2_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_seven_rep_2_docker_0_network_configs.txt'
    cilium_config_path = None
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_seven_rep_2_svc_config_0.txt'
    kubernetes_pod_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_seven_rep_2_pod_config_0.txt'
    time_interval_lengths = [30, 10]#,
                             #1]  # , 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = ["my-release-pxc", "wwwppp-wordpress"]
    make_edgefiles = False
    start_time = None
    end_time = None
    exfil_start_time = 600
    exfil_end_time = 650
    #calc_vals = False
    window_size = 6
    graph_p = False  # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2  # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    percentile_thresholds = [30, 40, 50, 60, 70, 75, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    anomaly_window = [1, 4]
    sec_between_exfil_events = 15
    anom_num_outlier_vals_in_window = [1, 2]
    physical_exfil_path = ['wwwppp_wordpress_pod', 'kube_dns_vip', 'kube_dns_pod', 'internet']
    alert_file = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/alerts/wordpress_seven_rep_2_'
    process_wordpress7_rep2_object = \
        data_anylsis_pipline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths, ms_s,
                                   make_edgefiles, basegraph_name, window_size, colors, exfil_start_time, exfil_end_time,
                                   wiggle_room, start_time=start_time, end_time=end_time, calc_vals=calc_vals,
                                   graph_p=graph_p, kubernetes_svc_info=kubernetes_svc_info,
                                   cilium_config_path=cilium_config_path, rdpcap_p=False,
                                   kubernetes_pod_info=kubernetes_pod_info, alert_file=alert_file, ROC_curve_p=True,
                                   calc_zscore_p=True, sec_between_exfil_events=sec_between_exfil_events,
                                   injected_exfil_path=physical_exfil_path, only_exp_info=only_exp_info,
                                   time_of_synethic_exfil=time_of_synethic_exfil,
                                   initiator_info_for_paths=initiator_info_for_paths,
                                   end_of_training=portion_for_training,
                                   training_window_size=training_window_size,
                                   size_of_neighbor_training_window=size_of_neighbor_training_window,
                                   synthetic_exfil_paths_train=synthetic_exfil_paths_train,
                                   synthetic_exfil_paths_test=synthetic_exfil_paths_test,
                                   skip_model_part=skip_model_part,
                                   max_number_of_paths=max_number_of_paths)
    #return time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, time_gran_to_feature_dataframe, fourth_return_val, fifth_val
    #'''
    return process_wordpress7_rep2_object

# 20 min of scaled-up wordpress (wordpress_eight) (dnscat exfil)
def process_wordpress8():
    #time.sleep(27000)
    #'''
    # 20 min of scaled-up wordpress (wordpress_eight)
    pcap_paths = [
        "/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_eight_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/edgefiles/wordpress_eight_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/graphs/wordpress_eight_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_eight_docker_0_network_configs.txt'
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_eight_svc_config_0.txt'
    kubernetes_pod_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_eight_pod_config_0.txt'
    time_interval_lengths = [60, 30, 10]#,
                             #1]  # , 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = ["my-release-pxc", "wwwppp-wordpress"]
    make_edgefiles = False
    start_time = None
    end_time = None
    exfil_start_time = 600
    exfil_end_time = 900
    calc_vals = True
    window_size = 6
    graph_p = False  # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2  # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    percentile_thresholds = [30, 40, 50, 60, 70, 75, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    anomaly_window = [1, 4]
    anom_num_outlier_vals_in_window = [1, 2] # note: these vals correspond to anoamly_window (so the first vals get matched, etc.)
    sec_between_exfil_events = 1
    physical_exfil_path = '(my_release_pxc_pod, wwwppp_wordpress_pod, internet)'
    alert_file = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/alerts/wordpress_eight_'
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths, ms_s,
                               make_edgefiles, basegraph_name, window_size, colors, exfil_start_time, exfil_end_time,
                               wiggle_room, start_time=start_time, end_time=end_time, calc_vals=calc_vals,
                               graph_p=graph_p, kubernetes_svc_info=kubernetes_svc_info, rdpcap_p=False,
                               kubernetes_pod_info=kubernetes_pod_info, alert_file=alert_file, ROC_curve_p=True,
                               calc_zscore_p=True, sec_between_exfil_events=sec_between_exfil_events,
                               injected_exfil_path=physical_exfil_path)
    #'''

def process_wordpress8_rep_2(time_of_synethic_exfil=None, only_exp_info=False, initiator_info_for_paths=None,
                             portion_for_training=None,training_window_size=None, size_of_neighbor_training_window=None,
                             synthetic_exfil_paths_train=None,
                             synthetic_exfil_paths_test=None, calc_vals=False, skip_model_part=False,
                             max_number_of_paths=None):
    #time.sleep(27000)
    #'''
    #calc_vals=False

    # 20 min of scaled-up wordpress (wordpress_eight)
    pcap_paths = [
        "/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_eight_rep_2_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/edgefiles/wordpress_eight_rep_2_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/graphs/wordpress_eight_rep_2_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_eight_rep_2_docker_0_network_configs.txt'
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_eight_rep_2_svc_config_0.txt'
    kubernetes_pod_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_eight_rep_2_pod_config_0.txt'
    time_interval_lengths = [30, 10]#,
                             #1]  # , 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = ["my-release-pxc", "wwwppp-wordpress"]
    make_edgefiles = False
    start_time = None
    end_time = None
    exfil_start_time = 600
    exfil_end_time = 900
    #calc_vals = False
    window_size = 6
    graph_p = False  # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2  # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    percentile_thresholds = [30, 40, 50, 60, 70, 75, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    anomaly_window = [1, 4]
    anom_num_outlier_vals_in_window = [1, 2] # note: these vals correspond to anoamly_window (so the first vals get matched, etc.)
    sec_between_exfil_events = 1
    alert_file = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/alerts/wordpress_eight_rep_2_'
    physical_exfil_path = ['my_release_pxc_pod', 'wwwppp_wordpress_pod', 'internet']
    process_wordpress8_rep_2_object = \
        data_anylsis_pipline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths, ms_s,
                                   make_edgefiles, basegraph_name, window_size, colors, exfil_start_time, exfil_end_time,
                                   wiggle_room, start_time=start_time, end_time=end_time, calc_vals=calc_vals,
                                   graph_p=graph_p, kubernetes_svc_info=kubernetes_svc_info, rdpcap_p=False,
                                   kubernetes_pod_info=kubernetes_pod_info, alert_file=alert_file, ROC_curve_p=True,
                                   calc_zscore_p=True, sec_between_exfil_events=sec_between_exfil_events,
                                   injected_exfil_path = physical_exfil_path, only_exp_info=only_exp_info,
                                   time_of_synethic_exfil=time_of_synethic_exfil,
                                   initiator_info_for_paths = initiator_info_for_paths,
                                   end_of_training=portion_for_training,
                                   training_window_size=training_window_size,
                                   size_of_neighbor_training_window=size_of_neighbor_training_window,
                                   synthetic_exfil_paths_train=synthetic_exfil_paths_train,
                                   synthetic_exfil_paths_test=synthetic_exfil_paths_test,
                                   skip_model_part=skip_model_part,
                                   max_number_of_paths=max_number_of_paths)
    #return time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, time_gran_to_feature_dataframe, fourth_return_val, fifth_val
    #'''
    return process_wordpress8_rep_2_object

def process_wordpress8_rep_3(time_of_synethic_exfil=None,only_exp_info=False, initiator_info_for_paths=None,
                             portion_for_training=None,training_window_size=None, size_of_neighbor_training_window=None,
                             synthetic_exfil_paths_train=None,
                             synthetic_exfil_paths_test=None, calc_vals=False, skip_model_part=False,
                             max_number_of_paths=None):
    #time.sleep(27000)
    #'''
    # 20 min of scaled-up wordpress (wordpress_eight)
    pcap_paths = [
        "/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_eight_rep_3_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/edgefiles/wordpress_eight_rep_3_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/graphs/wordpress_eight_rep_3_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_eight_rep_3_docker_0_network_configs.txt'
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_eight_rep_3_svc_config_0.txt'
    kubernetes_pod_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_eight_rep_3_pod_config_0.txt'
    time_interval_lengths = [30, 10]#,
                             #1]  # , 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = ["my-release-pxc", "wwwppp-wordpress"]
    make_edgefiles = False
    start_time = None
    end_time = None
    exfil_start_time = 600
    exfil_end_time = 900
    #calc_vals = False
    window_size = 6
    graph_p = False  # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2  # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    percentile_thresholds = [30, 40, 50, 60, 70, 75, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    anomaly_window = [1, 4]
    anom_num_outlier_vals_in_window = [1, 2] # note: these vals correspond to anoamly_window (so the first vals get matched, etc.)
    sec_between_exfil_events = 1
    physical_exfil_path = ['my_release_pxc_pod', 'wwwppp_wordpress_pod', 'internet']
    alert_file = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/alerts/wordpress_eight_rep_3_'
    process_wordpress8_rep_3_object = \
        data_anylsis_pipline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths, ms_s,
                                   make_edgefiles, basegraph_name, window_size, colors, exfil_start_time, exfil_end_time,
                                   wiggle_room, start_time=start_time, end_time=end_time, calc_vals=calc_vals,
                                   graph_p=graph_p, kubernetes_svc_info=kubernetes_svc_info, rdpcap_p=False,
                                   kubernetes_pod_info=kubernetes_pod_info, alert_file=alert_file, ROC_curve_p=True,
                                   calc_zscore_p=True, sec_between_exfil_events=sec_between_exfil_events,
                                   injected_exfil_path=physical_exfil_path, only_exp_info=only_exp_info,
                                   time_of_synethic_exfil=time_of_synethic_exfil,
                                   initiator_info_for_paths = initiator_info_for_paths,
                                   end_of_training=portion_for_training,
                                   training_window_size=training_window_size,
                                   size_of_neighbor_training_window=size_of_neighbor_training_window,
                                   synthetic_exfil_paths_train=synthetic_exfil_paths_train,
                                   synthetic_exfil_paths_test=synthetic_exfil_paths_test,
                                   skip_model_part=skip_model_part,
                                   max_number_of_paths=max_number_of_paths)
    #return time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, time_gran_to_feature_dataframe, fourth_return_val, fifth_val
    #'''
    return process_wordpress8_rep_3_object

## sockshop experiment 9 (an hour of scaled-up activity w/ exfil along normal path)
def process_sockshop9(time_of_synethic_exfil=None,only_exp_info=False, initiator_info_for_paths=None,
                             portion_for_training=None,training_window_size=None, size_of_neighbor_training_window=None,
                             synthetic_exfil_paths_train=None,
                             synthetic_exfil_paths_test=None, calc_vals=False,skip_model_part=False,
                            max_number_of_paths=None):
    ## TODO: REMOVE
    calc_vals=False

    ## sockshop experiment 9 (an hour of scaled-up activity w/ exfil along normal path)
    pcap_paths = ["/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_nine_better_exfil_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/edgefiles/sockshop_nine_better_exfil_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/graphs/sockshop_nine_better_exfil_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_nine_better_exfil_docker_0_network_configs.txt'
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_nine_better_exfil_svc_config_0.txt'
    kubernetes_pod_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_nine_better_exfil_pod_config_0.txt'
    netsec_policy = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_nine_better_exfil_netsec_seg.txt'
    time_interval_lengths = [30, 10]#,
    ms_s = microservices_sockshop
    make_edgefiles = False
    start_time = None
    end_time = None
    exfil_start_time = 2100
    exfil_end_time = 3000
    window_size = 6
    graph_p = False # should I make graphs?
    colors = ['b', 'r']
    percentile_thresholds = [30, 40, 50, 60, 70, 75, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    anomaly_window = [1, 4]
    anom_num_outlier_vals_in_window = [1, 2] # note: these vals correspond to anoamly_window (so the first vals get matched, etc.)
    alert_file = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/alerts/sockshop_nine_better_exfil_'
    wiggle_room = 2 # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    sec_between_exfil_events = 1
    physical_exfil_path = ['user-db', 'user', 'front-end', 'internet']
    #physical_exfil_path = ['user-db', 'user', 'internet']
    process_sockshop9_object = \
        data_anylsis_pipline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths, ms_s,
                               make_edgefiles, basegraph_name, window_size, colors, exfil_start_time, exfil_end_time,
                               wiggle_room, start_time=start_time, end_time=end_time, calc_vals=calc_vals,
                               graph_p=graph_p, kubernetes_svc_info=kubernetes_svc_info, rdpcap_p=False,
                               kubernetes_pod_info=kubernetes_pod_info, alert_file=alert_file, ROC_curve_p=True,
                               calc_zscore_p=True, sec_between_exfil_events=sec_between_exfil_events,
                               injected_exfil_path=physical_exfil_path, only_exp_info=only_exp_info,
                               time_of_synethic_exfil=time_of_synethic_exfil,
                               initiator_info_for_paths=initiator_info_for_paths,
                               end_of_training=portion_for_training,
                               training_window_size=training_window_size,
                               size_of_neighbor_training_window=size_of_neighbor_training_window,
                               synthetic_exfil_paths_train=synthetic_exfil_paths_train,
                               synthetic_exfil_paths_test=synthetic_exfil_paths_test,
                                   skip_model_part=skip_model_part,
                                   max_number_of_paths=max_number_of_paths,
                                   netsec_policy=netsec_policy)
    #'''
    #return time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, time_gran_to_feature_dataframe, fourth_return_val, fifth_val
    return process_sockshop9_object

## sockshop experiment 9_rep2 (an hour of scaled-up activity w/ exfil along normal path)
def process_sockshop9_rep2(time_of_synethic_exfil=None,only_exp_info=False, initiator_info_for_paths=None,
                             portion_for_training=None,training_window_size=None, size_of_neighbor_training_window=None,
                             synthetic_exfil_paths_train=None,
                             synthetic_exfil_paths_test=None, calc_vals=False,skip_model_part=False):
    ## TODO: REMOVE
    # sockshop_nine_rep2_default_bridge_0any.pcap
    #calc_vals=False

    ## sockshop experiment 9 (an hour of scaled-up activity w/ exfil along normal path)
    pcap_paths = ["/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_nine_rep2_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/edgefiles/sockshop_nine_rep2_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/graphs/sockshop_nine_rep2_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_nine_rep2_docker_0_network_configs.txt'
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_nine_rep2_svc_config_0.txt'
    kubernetes_pod_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_nine_rep2_pod_config_0.txt'
    time_interval_lengths = [30, 10]#,
    ms_s = microservices_sockshop
    make_edgefiles = True
    start_time = None
    end_time = None
    exfil_start_time = 2100
    exfil_end_time = 3000
    window_size = 6
    graph_p = False # should I make graphs?
    colors = ['b', 'r']
    percentile_thresholds = [30, 40, 50, 60, 70, 75, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    anomaly_window = [1, 4]
    anom_num_outlier_vals_in_window = [1, 2] # note: these vals correspond to anoamly_window (so the first vals get matched, etc.)
    alert_file = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/alerts/sockshop_nine_rep2_'
    wiggle_room = 2 # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    sec_between_exfil_events = 1
    physical_exfil_path = ['user-db', 'user', 'front-end', 'internet']
    #physical_exfil_path = ['user-db', 'user', 'internet']
    time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, time_gran_to_feature_dataframe, fourth_return_val, fifth_val = \
        run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths, ms_s,
                               make_edgefiles, basegraph_name, window_size, colors, exfil_start_time, exfil_end_time,
                               wiggle_room, start_time=start_time, end_time=end_time, calc_vals=calc_vals,
                               graph_p=graph_p, kubernetes_svc_info=kubernetes_svc_info, rdpcap_p=False,
                               kubernetes_pod_info=kubernetes_pod_info, alert_file=alert_file, ROC_curve_p=True,
                               calc_zscore_p=True, sec_between_exfil_events=sec_between_exfil_events,
                               injected_exfil_path=physical_exfil_path, only_exp_info=only_exp_info,
                               time_of_synethic_exfil=time_of_synethic_exfil,
                               initiator_info_for_paths=initiator_info_for_paths,
                               end_of_training=portion_for_training,
                               training_window_size=training_window_size,
                               size_of_neighbor_training_window=size_of_neighbor_training_window,
                               synthetic_exfil_paths_train=synthetic_exfil_paths_train,
                               synthetic_exfil_paths_test=synthetic_exfil_paths_test,
                                   skip_model_part=skip_model_part)
    #'''
    return time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, time_gran_to_feature_dataframe, fourth_return_val, fifth_val

## sockshop experiment 11 (an hour of scaled-up activity w/ exfil NOT on normal path)
def process_sockshop11(time_of_synethic_exfil=None,only_exp_info=False, initiator_info_for_paths=None,
                             portion_for_training=None,training_window_size=None, size_of_neighbor_training_window=None,
                             synthetic_exfil_paths_train=None,
                             synthetic_exfil_paths_test=None, calc_vals=False, skip_model_part=False,
                            max_number_of_paths=None):
    #time.sleep(45000)
    #calc_vals=False

    #''' # next-gen
    ## sockshop experiment 11 (an hour of scaled-up activity w/ exfil NOT on normal path)
    pcap_paths = ["/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_eleven_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/edgefiles/sockshop_eleven_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/graphs/sockshop_eleven_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_eleven_docker_0_network_configs.txt'
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_eleven_svc_config_0.txt'
    kubernetes_pod_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_eleven_pod_config_0.txt'
    netsec_policy = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_eleven_exfil_netsec_seg.txt'
    time_interval_lengths = [30, 10]#,
    ms_s = microservices_sockshop
    make_edgefiles = False
    start_time = None
    end_time = None
    exfil_start_time = 2100
    exfil_end_time = 3000
    window_size = 6
    graph_p = False # should I make graphs?
    colors = ['b', 'r']
    percentile_thresholds = [30, 40, 50, 60, 70, 75, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    anomaly_window = [1, 4]
    sec_between_exfil_events = 1
    physical_exfil_path = ['user-db', 'front-end', 'internet']
    anom_num_outlier_vals_in_window = [1, 2] # note: these vals correspond to anoamly_window (so the first vals get matched, etc.)
    alert_file = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/alerts/sockshop_eleven_'
    wiggle_room = 2 # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    process_sockshop11_object = \
        data_anylsis_pipline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths, ms_s,
                               make_edgefiles, basegraph_name, window_size, colors, exfil_start_time, exfil_end_time,
                               wiggle_room, start_time=start_time, end_time=end_time, calc_vals=calc_vals,
                               graph_p=graph_p, kubernetes_svc_info=kubernetes_svc_info, rdpcap_p=False,
                               kubernetes_pod_info=kubernetes_pod_info, alert_file=alert_file, ROC_curve_p=True,
                               calc_zscore_p=True, sec_between_exfil_events=sec_between_exfil_events,
                               injected_exfil_path=physical_exfil_path, only_exp_info=only_exp_info,
                               time_of_synethic_exfil=time_of_synethic_exfil,
                               initiator_info_for_paths=initiator_info_for_paths,
                               end_of_training=portion_for_training,
                               training_window_size=training_window_size,
                               size_of_neighbor_training_window=size_of_neighbor_training_window,
                               synthetic_exfil_paths_train=synthetic_exfil_paths_train,
                               synthetic_exfil_paths_test=synthetic_exfil_paths_test,
                                   skip_model_part=skip_model_part,
                                   max_number_of_paths=max_number_of_paths,
                                   netsec_policy=netsec_policy)
    #'''
    #return time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, time_gran_to_feature_dataframe, fourth_return_val, fifth_val
    return process_sockshop11_object

## sockshop experiment 11_rep2 (an hour of scaled-up activity w/ exfil NOT on normal path)
def process_sockshop11_rep2(time_of_synethic_exfil=None,only_exp_info=False, initiator_info_for_paths=None,
                             portion_for_training=None,training_window_size=None, size_of_neighbor_training_window=None,
                             synthetic_exfil_paths_train=None,
                             synthetic_exfil_paths_test=None, calc_vals=False, skip_model_part=False):
    #time.sleep(45000)
    #calc_vals=False

    #''' # next-gen
    ## sockshop experiment 11 (an hour of scaled-up activity w/ exfil NOT on normal path)
    pcap_paths = ["/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_eleven_rep2_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/edgefiles/sockshop_eleven_rep2_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/graphs/sockshop_eleven_rep2_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_eleven_rep2_docker_0_network_configs.txt'
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_eleven_rep2_svc_config_0.txt'
    kubernetes_pod_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_eleven_rep2_pod_config_0.txt'
    time_interval_lengths = [30, 10]#,
    ms_s = microservices_sockshop
    make_edgefiles = True
    start_time = None
    end_time = None
    exfil_start_time = 2100
    exfil_end_time = 3000
    window_size = 6
    graph_p = False # should I make graphs?
    colors = ['b', 'r']
    percentile_thresholds = [30, 40, 50, 60, 70, 75, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    anomaly_window = [1, 4]
    sec_between_exfil_events = 1
    physical_exfil_path = ['user-db', 'front-end', 'internet']
    anom_num_outlier_vals_in_window = [1, 2] # note: these vals correspond to anoamly_window (so the first vals get matched, etc.)
    alert_file = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/alerts/sockshop_eleven_rep2_'
    wiggle_room = 2 # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, time_gran_to_feature_dataframe, fourth_return_val, fifth_val = \
        run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths, ms_s,
                               make_edgefiles, basegraph_name, window_size, colors, exfil_start_time, exfil_end_time,
                               wiggle_room, start_time=start_time, end_time=end_time, calc_vals=calc_vals,
                               graph_p=graph_p, kubernetes_svc_info=kubernetes_svc_info, rdpcap_p=False,
                               kubernetes_pod_info=kubernetes_pod_info, alert_file=alert_file, ROC_curve_p=True,
                               calc_zscore_p=True, sec_between_exfil_events=sec_between_exfil_events,
                               injected_exfil_path=physical_exfil_path, only_exp_info=only_exp_info,
                               time_of_synethic_exfil=time_of_synethic_exfil,
                               initiator_info_for_paths=initiator_info_for_paths,
                               end_of_training=portion_for_training,
                               training_window_size=training_window_size,
                               size_of_neighbor_training_window=size_of_neighbor_training_window,
                               synthetic_exfil_paths_train=synthetic_exfil_paths_train,
                               synthetic_exfil_paths_test=synthetic_exfil_paths_test,
                                   skip_model_part=skip_model_part)
    #'''
    return time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, time_gran_to_feature_dataframe, fourth_return_val, fifth_val


## sockshop experiment 12 (an hour of scaled-up activity w/ exfil going straight out)
def process_sockshop12(time_of_synethic_exfil=None,only_exp_info=False, initiator_info_for_paths=None,
                             portion_for_training=None,training_window_size=None, size_of_neighbor_training_window=None,
                             synthetic_exfil_paths_train=None,
                             synthetic_exfil_paths_test=None, calc_vals=False,
                            skip_model_part=False, max_number_of_paths=None):
    #time.sleep(54000)
    #''' # next-gen
    ## sockshop experiment 12 (an hour of scaled-up activity w/ exfil going straight out)
    pcap_paths = ["/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_twelve_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/edgefiles/sockshop_twelve_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/graphs/sockshop_twelve_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_twelve_docker_0_network_configs.txt'
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_twelve_svc_config_0.txt'
    netsec_policy = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_twelve_exfil_netsec_seg.txt'
    time_interval_lengths = [30, 10]#,
    ms_s = microservices_sockshop
    make_edgefiles = False
    start_time = None
    end_time = None
    exfil_start_time = 2100
    exfil_end_time = 3000
    window_size = 6
    graph_p = False # should I make graphs?
    colors = ['b', 'r']
    percentile_thresholds = [30, 40, 50, 60, 70, 75, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    anomaly_window = [1, 4]
    anom_num_outlier_vals_in_window = [1, 2] # note: these vals correspond to anoamly_window (so the first vals get matched, etc.)
    alert_file = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/alerts/sockshop_twelve_'
    kubernetes_pod_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_twelve_pod_config_0.txt'
    wiggle_room = 2 # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    sec_between_exfil_events = 1
    physical_exfil_path = ['user-db', 'internet']
    process_sockshop12 = \
        data_anylsis_pipline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths, ms_s,
                               make_edgefiles, basegraph_name, window_size, colors, exfil_start_time, exfil_end_time,
                               wiggle_room, start_time=start_time, end_time=end_time, calc_vals=calc_vals,
                               graph_p=graph_p, kubernetes_svc_info=kubernetes_svc_info, rdpcap_p=False,
                               kubernetes_pod_info=kubernetes_pod_info, alert_file=alert_file, ROC_curve_p=True,
                               calc_zscore_p=True, sec_between_exfil_events=sec_between_exfil_events,
                               injected_exfil_path=physical_exfil_path, only_exp_info=only_exp_info,
                               time_of_synethic_exfil=time_of_synethic_exfil,
                               initiator_info_for_paths=initiator_info_for_paths,
                               end_of_training=portion_for_training,
                               training_window_size=training_window_size,
                               size_of_neighbor_training_window=size_of_neighbor_training_window,
                               synthetic_exfil_paths_train=synthetic_exfil_paths_train,
                               synthetic_exfil_paths_test=synthetic_exfil_paths_test,
                                   skip_model_part=skip_model_part,
                                   max_number_of_paths=max_number_of_paths,
                                   netsec_policy=netsec_policy)
    #
    # '''
    #return time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, time_gran_to_feature_dataframe, fourth_return_val, fifth_val
    return process_sockshop12

## TODO
## sockshop experiment 12_rep2 (an hour of scaled-up activity w/ exfil going straight out)
def process_sockshop12_rep2(time_of_synethic_exfil=None,only_exp_info=False, initiator_info_for_paths=None,
                             portion_for_training=None,training_window_size=None, size_of_neighbor_training_window=None,
                             synthetic_exfil_paths_train=None,
                             synthetic_exfil_paths_test=None, calc_vals=False,
                            skip_model_part=False):
    #time.sleep(54000)
    #''' # next-gen
    ## sockshop experiment 12 (an hour of scaled-up activity w/ exfil going straight out)
    pcap_paths = ["/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_twelve_rep2_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/edgefiles/sockshop_twelve_rep2_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/graphs/sockshop_twelve_rep2_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_twelve_rep2_docker_0_network_configs.txt'
    kubernetes_svc_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_twelve_rep2_svc_config_0.txt'
    time_interval_lengths = [30, 10]#,
    ms_s = microservices_sockshop
    make_edgefiles = True
    start_time = None
    end_time = None
    exfil_start_time = 2100
    exfil_end_time = 3000
    window_size = 6
    graph_p = False # should I make graphs?
    colors = ['b', 'r']
    percentile_thresholds = [30, 40, 50, 60, 70, 75, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    anomaly_window = [1, 4]
    anom_num_outlier_vals_in_window = [1, 2] # note: these vals correspond to anoamly_window (so the first vals get matched, etc.)
    alert_file = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/alerts/sockshop_twelve_rep2_'
    kubernetes_pod_info = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/sockshop_twelve_rep2_pod_config_0.txt'
    wiggle_room = 2 # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    sec_between_exfil_events = 1
    physical_exfil_path = ['user-db', 'internet']
    time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, time_gran_to_feature_dataframe, fourth_return_val, fifth_val = \
        run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths, ms_s,
                               make_edgefiles, basegraph_name, window_size, colors, exfil_start_time, exfil_end_time,
                               wiggle_room, start_time=start_time, end_time=end_time, calc_vals=calc_vals,
                               graph_p=graph_p, kubernetes_svc_info=kubernetes_svc_info, rdpcap_p=False,
                               kubernetes_pod_info=kubernetes_pod_info, alert_file=alert_file, ROC_curve_p=True,
                               calc_zscore_p=True, sec_between_exfil_events=sec_between_exfil_events,
                               injected_exfil_path=physical_exfil_path, only_exp_info=only_exp_info,
                               time_of_synethic_exfil=time_of_synethic_exfil,
                               initiator_info_for_paths=initiator_info_for_paths,
                               end_of_training=portion_for_training,
                               training_window_size=training_window_size,
                               size_of_neighbor_training_window=size_of_neighbor_training_window,
                               synthetic_exfil_paths_train=synthetic_exfil_paths_train,
                               synthetic_exfil_paths_test=synthetic_exfil_paths_test,
                                   skip_model_part=skip_model_part)
    #'''
    return time_gran_to_mod_zscore_df, time_gran_to_zscore_dataframe, time_gran_to_feature_dataframe, fourth_return_val, fifth_val


def run_analysis_pipeline_recipes_json(json_file, path_to_experimental_data):
    with open(path_to_experimental_data + json_file) as f:
        data = json.load(f)
        pcap_paths = [path_to_experimental_data + i for i in data["pcap_paths"]]
        is_swarm = int(data["is_swarm"])
        basefile_name = path_to_experimental_data + data["basefile_name"]
        basegraph_name = path_to_experimental_data + data["basegraph_name"]
        container_info_path =  path_to_experimental_data + data["container_info_path"]
        cilium_config_path = (path_to_experimental_data + data["cilium_config_path"])[0] if (data["cilium_config_path"] and data["cilium_config_path"] != "None") else None
        kubernetes_svc_info = path_to_experimental_data + data["kubernetes_svc_info"]
        kubernetes_pod_info = path_to_experimental_data + data["kubernetes_pod_info"]
        time_interval_lengths = data["time_interval_lengths"]
        ms_s = data["ms_s"]
        make_edgefiles =  data["make_edgefiles"]
        start_time = data["start_time"]
        end_time = data["end_time"]
        exfil_start_time = data["exfil_start_time"]
        exfil_end_time = data["exfil_end_time"]
        calc_vals = data["calc_vals"]
        window_size = data["window_size"]
        graph_p = data["graph_p"]
        colors = data["colors"]
        wiggle_room = data["wiggle_room"] # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
        #percentile_thresholds = data["percentile_thresholds"]
        #anomaly_window = data["anomaly_window"]
        #anom_num_outlier_vals_in_window = data["anom_num_outlier_vals_in_window"]
        alert_file = path_to_experimental_data + data["alert_file"]
        ROC_curve_p =  data["ROC_curve_p"]
        calc_tpr_fpr_p =  data["calc_tpr_fpr_p"]
        sec_between_exfil_events = data['sec_between_exfil_events']

        run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                                   ms_s, make_edgefiles, basegraph_name, window_size, colors, exfil_start_time,
                                   exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                                   calc_vals=calc_vals, graph_p=graph_p, kubernetes_svc_info=kubernetes_svc_info,
                                   cilium_config_path=cilium_config_path, rdpcap_p=False,
                                   kubernetes_pod_info=kubernetes_pod_info, alert_file=alert_file,
                                   ROC_curve_p=ROC_curve_p, calc_zscore_p=calc_tpr_fpr_p,
                                   sec_between_exfil_events=sec_between_exfil_events)

def wordpress_thirteen_t1(time_of_synethic_exfil=None, only_exp_info=False, initiator_info_for_paths=None,
                            portion_for_training=None, training_window_size=None, size_of_neighbor_training_window=None,
                            synthetic_exfil_paths_train=None,
                            synthetic_exfil_paths_test=None, calc_vals=False,
                            skip_model_part=False,max_number_of_paths=None):
    #'''
    #calc_vals=False
    # Wordpress exp 6 rep3 (wordpress w/ HA cluster on cilium w/o security config, dnscat exfil from single db w/ 15 sec delay)
    pcap_paths = [
        "/Volumes/exM2/experimental_data/wordpress_info/wordpress_thirteen_t1/wordpress_thirteen_t1_default_bridge_0any.pcap"]
    is_swarm = 0
    basefile_name = '/Volumes/exM2/experimental_data/wordpress_info/wordpress_thirteen_t1/edgefiles/wordpress_thirteen_t1_'
    basegraph_name = '/Volumes/exM2/experimental_data/wordpress_info/wordpress_thirteen_t1/graphs/wordpress_thirteen_t1_'
    container_info_path = "/Volumes/exM2/experimental_data/wordpress_info/wordpress_thirteen_t1/wordpress_thirteen_t1_docker_0_network_configs.txt"
    cilium_config_path = None # does NOT use cilium on reps 2-4
    kubernetes_svc_info = '/Volumes/exM2/experimental_data/wordpress_info/wordpress_thirteen_t1/wordpress_thirteen_t1_svc_config_0.txt'
    kubernetes_pod_info = '/Volumes/exM2/experimental_data/wordpress_info/wordpress_thirteen_t1/wordpress_thirteen_t1_pod_config_0.txt'
    time_interval_lengths = [30, 10, 1]#,
                             #1]  # , 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = ["my-release-pxc", "wwwppp-wordpress"]
    start_time = False
    end_time = None
    exfil_start_time = 3570
    exfil_end_time = 3570
    #calc_vals = False #
    window_size = 6
    graph_p = False  # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2  # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    #percentile_thresholds = [50, 75, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    #anomaly_window = [1, 4]
    #anom_num_outlier_vals_in_window = [1, 2] # note: these vals correspond to anoamly_window (so the first vals get matched, etc.)
    alert_file = '/Volumes/exM2/experimental_data/wordpress_info/wordpress_thirteen_t1/alerts/wordpress_thirteen_t1_'
    sec_between_exfil_events = 15
    physical_exfil_path = []
    #training_window_size


    make_edgefiles = False ## already done!
    wordpress_thirteen_t1_object = data_anylsis_pipline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths, ms_s,
                                   make_edgefiles, basegraph_name, window_size, colors, exfil_start_time, exfil_end_time,
                                   wiggle_room, start_time=start_time, end_time=end_time, calc_vals=calc_vals,
                                   graph_p=graph_p, kubernetes_svc_info=kubernetes_svc_info,
                                   cilium_config_path=cilium_config_path, rdpcap_p=False,
                                   kubernetes_pod_info=kubernetes_pod_info, alert_file=alert_file, ROC_curve_p=True,
                                   calc_zscore_p=True, sec_between_exfil_events=sec_between_exfil_events,
                                   injected_exfil_path = physical_exfil_path, only_exp_info=only_exp_info,
                                   time_of_synethic_exfil=time_of_synethic_exfil,
                                   initiator_info_for_paths=initiator_info_for_paths,
                                   end_of_training=portion_for_training,
                                   training_window_size=training_window_size, size_of_neighbor_training_window=size_of_neighbor_training_window,
                                   synthetic_exfil_paths_train=synthetic_exfil_paths_train, synthetic_exfil_paths_test=synthetic_exfil_paths_test,
                                   skip_model_part=skip_model_part,
                                   max_number_of_paths=max_number_of_paths)

    return wordpress_thirteen_t1_object



def new_wordpress_recipe():
    skip_model_part = False
    ignore_physical_attacks_p = True
    #fraction_of_edge_weights,fraction_of_edge_pkts = [0.001, 0.1],[0.001, 0.1]
    #fraction_of_edge_weights,fraction_of_edge_pkts = [0.1],[0.1]

    # want to put back eventaully...
    fraction_of_edge_weights,fraction_of_edge_pkts = [0.2, 0.1, 0.01, 0.001],[0.2, 0.1, 0.01, 0.001]
    #fraction_of_edge_weights,fraction_of_edge_pkts = [0.2],[0.2]

    # going w/ this now b/c the 0.2's have already been calculated...
    #fraction_of_edge_weights, fraction_of_edge_pkts = [0.001], [0.001]

    time_of_synethic_exfil = 30 # sec
    goal_train_test_split = 0.6
    goal_attack_NoAttack_split = 0.6
    training_window_size = 200
    size_of_neighbor_training_window = 0


    calc_vals = True ## TODO: probably want to turn this off (eventually...)
    calculate_z_scores = True


    experiment_classes = [wordpress_thirteen_t1(training_window_size=training_window_size,
                                                  size_of_neighbor_training_window=size_of_neighbor_training_window,
                                                  calc_vals=calc_vals,
                                                  time_of_synethic_exfil=time_of_synethic_exfil)]

    ## NOTE: process_wordpress8 could be here too, but I'm for the moment I'm keeping each kind of injected
    ## attack w/ two different experiments in which it occurss...

    #function_list = [i for i in raw_function_list]
    #function_list_exp_info = [functools.partial(i,time_of_synethic_exfil=time_of_synethic_exfil, only_exp_info=True) for i in function_list]
    base_output_location = '/Volumes/exM2/experimental_data/wordpress_summary/new_'# + 'lasso_roc'
    multi_experiment_pipeline(experiment_classes, base_output_location, True, time_of_synethic_exfil,
                              goal_train_test_split, goal_attack_NoAttack_split, training_window_size,
                              size_of_neighbor_training_window, calc_vals, skip_model_part, ignore_physical_attacks_p,
                              fraction_of_edge_weights=fraction_of_edge_weights,
                              fraction_of_edge_pkts=fraction_of_edge_pkts,
                              calculate_z_scores_p=calculate_z_scores)


# this function feeds a set of wordpress experiments into the multi_experiment_pipeline() function found in the
# pipeline_coordinator
def multi_experiment_wordpress_recipe():
    skip_model_part = False
    ignore_physical_attacks_p = True
    #fraction_of_edge_weights,fraction_of_edge_pkts = [0.001, 0.1],[0.001, 0.1]
    #fraction_of_edge_weights,fraction_of_edge_pkts = [0.1],[0.1]
    fraction_of_edge_weights,fraction_of_edge_pkts = [0.001],[0.001]
    time_of_synethic_exfil = 30 # sec
    goal_train_test_split = 0.6
    goal_attack_NoAttack_split = 0.6
    training_window_size = 200
    size_of_neighbor_training_window = 0
    calc_vals = True ## TODO: probably want to turn this off
    calculate_z_scores = False
    experiment_classes = [process_wordpress6_rep3(training_window_size=training_window_size,
                                                  size_of_neighbor_training_window=size_of_neighbor_training_window,
                                                  calc_vals=calc_vals,
                                                  time_of_synethic_exfil=time_of_synethic_exfil),
                          process_wordpress6_rep4(training_window_size=training_window_size,
                                                  size_of_neighbor_training_window=size_of_neighbor_training_window,
                                                  calc_vals=calc_vals,
                                                  time_of_synethic_exfil=time_of_synethic_exfil),
                          process_wordpress7_rep2(training_window_size=training_window_size,
                                                  size_of_neighbor_training_window=size_of_neighbor_training_window,
                                                  calc_vals=calc_vals,
                                                  time_of_synethic_exfil=time_of_synethic_exfil),
                          process_wordpress7_rep3(training_window_size=training_window_size,
                                                  size_of_neighbor_training_window=size_of_neighbor_training_window,
                                                  calc_vals=calc_vals,
                                                  time_of_synethic_exfil=time_of_synethic_exfil),
                          process_wordpress8_rep_2(training_window_size=training_window_size,
                                                  size_of_neighbor_training_window=size_of_neighbor_training_window,
                                                  calc_vals=calc_vals,
                                                   time_of_synethic_exfil=time_of_synethic_exfil),
                          process_wordpress8_rep_3(training_window_size=training_window_size,
                                                  size_of_neighbor_training_window=size_of_neighbor_training_window,
                                                  calc_vals=calc_vals,
                                                   time_of_synethic_exfil=time_of_synethic_exfil)]

    ## NOTE: process_wordpress8 could be here too, but I'm for the moment I'm keeping each kind of injected
    ## attack w/ two different experiments in which it occurss...

    #function_list = [i for i in raw_function_list]
    #function_list_exp_info = [functools.partial(i,time_of_synethic_exfil=time_of_synethic_exfil, only_exp_info=True) for i in function_list]
    base_output_location = '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_summary/'# + 'lasso_roc'
    multi_experiment_pipeline(experiment_classes, base_output_location, True, time_of_synethic_exfil,
                              goal_train_test_split, goal_attack_NoAttack_split, training_window_size,
                              size_of_neighbor_training_window, calc_vals, skip_model_part, ignore_physical_attacks_p,
                              fraction_of_edge_weights=fraction_of_edge_weights,
                              fraction_of_edge_pkts=fraction_of_edge_pkts,
                              calculate_z_scores_p=calculate_z_scores)

## TODO TODO TODO TODO
# this function feeds a set of wordpress experiments into the multi_experiment_pipeline() function found in the
# pipeline_coordinator
def multi_experiment_sockshop_recipe():
    time_of_synethic_exfil = 30 # sec
    goal_train_test_split = 0.5
    goal_attack_NoAttack_split = 0.5
    training_window_size = 200
    size_of_neighbor_training_window = 0
    #calc_vals = False#True
    #fraction_of_edge_weights,fraction_of_edge_pkts = [0.001],[0.001]
    skip_model_part = False
    ignore_physical_attacks_p = True

    calc_vals = False #True  ## TODO: PROBABLY WANT TO TURN THIS OFF AT SOME POINT
    calculate_z_scores = False

    ### TODO: can add the other reps of 9,11,12 when I get some time...
    #function_list = [process_sockshop9, process_sockshop9_rep2, process_sockshop11, process_sockshop11_rep2,
    #                 process_sockshop12, process_sockshop12_rep2] ## TODO TODO TODO
    #experiment_classes = [process_sockshop9(), process_sockshop11(), process_sockshop12()] ## TODO TODO TOD

    experiment_classes = [process_sockshop9(training_window_size=training_window_size,
                                                  size_of_neighbor_training_window=size_of_neighbor_training_window,
                                                  calc_vals=calc_vals,
                                                  time_of_synethic_exfil=time_of_synethic_exfil),
                          process_sockshop11(training_window_size=training_window_size,
                                                  size_of_neighbor_training_window=size_of_neighbor_training_window,
                                                  calc_vals=calc_vals,
                                                  time_of_synethic_exfil=time_of_synethic_exfil),
                          process_sockshop12(training_window_size=training_window_size,
                                                  size_of_neighbor_training_window=size_of_neighbor_training_window,
                                                  calc_vals=calc_vals,
                                                  time_of_synethic_exfil=time_of_synethic_exfil)]


    ## NOTE: process_wordpress8 could be here too, but I'm for the moment I'm keeping each kind of injected
    ## attack w/ two different experiments in which it occurss...

    #function_list = [i for i in raw_function_list]

    base_output_location = '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_summary/'# + 'lasso_roc'
    fraction_of_edge_weights = [0.1]
    fraction_of_edge_pkts = [0.1]
    multi_experiment_pipeline(experiment_classes, base_output_location, True, time_of_synethic_exfil,
                              goal_train_test_split, goal_attack_NoAttack_split, training_window_size,
                              size_of_neighbor_training_window, calc_vals, skip_model_part, ignore_physical_attacks_p,
                              fraction_of_edge_weights=fraction_of_edge_weights,
                              fraction_of_edge_pkts=fraction_of_edge_pkts,
                              calculate_z_scores_p=calculate_z_scores)

if __name__=="__main__":
    print "RUNNING"
    print sys.argv

    if len(sys.argv) == 1:
        new_wordpress_recipe()
        #multi_experiment_wordpress_recipe()
        #time.sleep(14400)
        #multi_experiment_sockshop_recipe()

        # TODO: might wanna re-enable the function below...
        #run_analysis_pipeline_recipes()

        #run_analysis_pipeline_recipes_json('wordpress_eleven_dns_1sec_analysis.json',
        #                               '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_eleven_dns_1sec/')

        #run_analysis_pipeline_recipes_json('wordpress_eleven_analysis.json',
        #                               '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_eleven/')

    elif len(sys.argv) == 2:
        if sys.argv[2] == "process_wordpress6_rep4":
            print "matched arg"
            process_wordpress6_rep4()  # this is good (DNSCAT, from DB, 15 sec delay between packets)
        elif sys.argv[2] == "process_wordpress7_rep3":
            process_wordpress7_rep3()
        elif sys.argv[2] == "process_wordpress8":
            process_wordpress8()
        ## more stuff to run at the first available opportuntiy... ##
        elif sys.argv[2] == "process_wordpress6_rep3":
            process_wordpress6_rep3()
        elif sys.argv[2] == "process_wordpress7_rep2":
            process_wordpress7_rep2()
        elif sys.argv[2] == "process_wordpress6_rep2":
            process_wordpress6_rep2()
        elif sys.argv[2] == "process_wordpress8_rep_2":
            process_wordpress8_rep_2()
        elif sys.argv[2] == "process_wordpress8_rep_3":
            process_wordpress8_rep_3()
        else:
            print "argument not recognized"
    else:
        print "too many args!"
    #local_machine_experimental_data_location = '/Volumes/Seagate Backup Plus Drive/experimental_data/'
    #run_analysis_pipeline_recipes_json('/Users/jseverin/Documents/Microservices/munnin/analysis_json/wordpress_six_rep2.json',
    #                                   local_machine_experimental_data_location)

    #run_analysis_pipeline_recipes_json( 'wp_testing_new_file_setup_analysis.json',
    #                        '/Users/jseverin/Documents/Microservices/munnin/experimental_data/wp_testing_new_file_setup/')