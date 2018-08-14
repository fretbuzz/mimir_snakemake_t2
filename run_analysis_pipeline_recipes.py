from pcap_parser import run_data_anaylsis_pipeline

'''
This file is essentially just sets of parameters for the run_data_analysis_pipeline function in pcap_parser.py
There are a lot of parameters, and some of them are rather long, so I decided to make a function to store them in
'''

# here are some 'recipes'
# comment out the ones you are not using
def run_analysis_pipeline_recipes():
    # these lists are only need for processing the k8s pod info
    microservices_sockshop = ['carts-db', 'carts', 'catalogue-db', 'catalogue', 'front-end', 'orders-db', 'orders',
                              'payment', 'queue-master', 'rabbitmq', 'session-db', 'shipping', 'user-db', 'user',
                              'load-test']
    minikube_infrastructure = ['etcd', 'kube-addon-manager', 'kube-apiserver', 'kube-controller-manager',
                               'kube-dns', 'kube-proxy', 'kube-scheduler', 'kubernetes-dashboard', 'metrics-server',
                               'storage-provisioner']
    microservices_wordpress = ['mariadb-master', 'mariadb-slave', 'wordpress']

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
    # wordpress recipe
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
    # sockshop exp 3 (rep 0)
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
    window_size = 6
    graph_p = True # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p)
    '''

    # atsea exp 2 (v7) [good]
    #''''
    pcap_paths = ['/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_exp_two_v7__atsea_back-tier_0.pcap',
                   '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_exp_two_v7__atsea_front-tier_0.pcap',
                  '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_exp_two_v7__ingress_0.pcap',
                  '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_exp_two_v7__bridge_0.pcap',
                  '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_exp_two_v7__ingress_sbox_0.pcap']
    is_swarm = True
    basefile_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/edgefiles/atsea_store_exp_two_v7_'
    basegraph_name = '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/graphs/atsea_store_exp_two_v7_'
    container_info_path = '/Volumes/Seagate Backup Plus Drive/experimental_data/atsea_info/atsea_store_exp_two_v7__docker_0_network_configs.txt'
    time_interval_lengths = [30, 10, 1] #, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = ['appserver_VIP', 'reverse_proxy_VIP', 'database_VIP', 'appserver', 'reverse_proxy', 'database', 'back-tier', 'front-tier', 'visualizer']
    make_edgefiles = False
    start_time = 1533377817.89
    end_time = 1533378712.2
    exfil_start_time = 270
    exfil_end_time = 330
    make_net_graphs_p = True # do you want to make network
    calc_vals = True
    window_size = 6
    graph_p = True # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2 # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p, make_net_graphs_p=make_net_graphs_p)
    #'''
    # atsea exp 3 (v2) [good]
    '''
    pcap_paths = ['/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_three_v2__atsea_back-tier_0.pcap',
                   '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_three_v2__atsea_front-tier_0.pcap',
                  '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_three_v2__ingress_0.pcap',
                  '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_three_v2__bridge_0.pcap',
                  '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_three_v2__ingress_sbox_0.pcap']
    is_swarm = 1
    basefile_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/edgefiles/atsea_store_exp_three_v2_'
    basegraph_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/graphs/atsea_store_exp_three_v2_'
    container_info_path = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_three_v2__docker_0_network_configs.txt'
    time_interval_lengths = [50, 30, 10, 1]#50, , 1] #, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    ms_s = ['appserver_VIP', 'reverse_proxy_VIP', 'database_VIP', 'appserver', 'reverse_proxy', 'database', 'back-tier', 'front-tier']
    make_edgefiles = False
    start_time = 1533381724.66 #None
    end_time = 1533382619.64 #None
    exfil_start_time = 300
    exfil_end_time = 360
    calc_vals = True
    window_size = 6
    graph_p = True # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p)
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
    '''


if __name__=="__main__":
    print "RUNNING"
    run_analysis_pipeline_recipes()