'''
To see usage: python run_experiment.py --help

it'll save the resulting pcaps in the local folder, since it is necessary to transfer them off of
cloudlab anyway, it makes little sense to use a complicated directory structure to store data at
this stage (though it makes more sense later, when I'm actually analyzing the pcaps)
'''

import argparse
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


def main(experiment_name, config_file, prepare_app_p, port, ip, localhostip, install_det_depen_p):
    # step (1) read in the config file
    with open(config_file + '.json') as f:
        config_params = json.load(f)
    if experiment_name == 'None':
        experiment_name = config_params["experiment_name"]
    orchestrator = config_params["orchestrator"]
    class_to_installer = config_params["exfiltration_info"]["exfiltration_path_class_which_installer"]

    min_exfil_bytes_in_packet = int(config_params["exfiltration_info"]["min_exfil_data_per_packet_bytes"])
    max_exfil_bytes_in_packet = int(config_params["exfiltration_info"]["max_exfil_data_per_packet_bytes"])
    avg_exfil_rate_KB_per_sec = float(config_params["exfiltration_info"]["avg_exfiltration_rate_KB_per_sec"])
    # okay, now need to calculate the time between packetes (and throw an error if necessary)
    avg_exfil_bytes_in_packet = (float(min_exfil_bytes_in_packet) + float(max_exfil_bytes_in_packet)) / 2.0
    BYTES_PER_MB = 1024
    avg_number_of_packets_per_second = (avg_exfil_rate_KB_per_sec * BYTES_PER_MB) / avg_exfil_bytes_in_packet
    average_seconds_between_packets = 1.0 / avg_number_of_packets_per_second
    maxsleep = average_seconds_between_packets * 2 # take random value between 0 and the max, so 2*average gives the right
    # will need to calculate the MAX_SLEEP_TIME after I load the webapp (and hence
    # the corresponding database)

    # step (2) setup the application, if necessary (e.g. fill up the DB, etc.)
    # note: it is assumed that the application is already deployed

    print prepare_app_p, config_params["application_name"], config_params["setup"], ip, port

    if prepare_app_p:
        prepare_app(config_params["application_name"], config_params["setup"], ip, port)

    class_to_net = config_params["class_to_networks"]
    print "class_to_net", class_to_net
    # determine the network namespaces
    # this will require mapping the name of the network to the network id, which
    # is then present (in truncated form) in the network namespace
    full_network_ids = get_network_ids(orchestrator, config_params["networks_to_tcpdump_on"])
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
    exfil_path = config_params["exfiltration_info"]["exfiltration_path_class"]
    for proxy_class in exfil_path[:-1]:
        print "current proxy class", proxy_class
        possible_proxies[proxy_class], class_to_networks[proxy_class] = get_class_instances(orchestrator, proxy_class, class_to_net[proxy_class])
        print "new possible proxies", possible_proxies[proxy_class]
        print "new class_to_network mapping", class_to_networks[proxy_class]
        num_proxies_of_this_class = int(config_params["exfiltration_info"]
                                        ["exfiltration_path_how_many_instances_for_each_class"][index])
        selected_proxies[proxy_class] = random.sample(possible_proxies[proxy_class], num_proxies_of_this_class)
        print "new selected proxies", selected_proxies[proxy_class]
        index += 1
    '''
    print "selected proxies", selected_proxies
    # get_class_instances cannot get the ingress network, so we must compensate manually
    if orchestrator == 'docker_swarm':
        print "let's handle the ingress network edgecase"
        classes_connected_to_ingess = config_params["ingess_class"]
        ingress_network = None
        client = docker.from_env()
        for a_network in client.networks.list():
            if a_network.name == 'ingress':
                ingress_network = a_network
                break
        print "the ingress class is", ingress_network, ingress_network.name
        for ingress_class in classes_connected_to_ingess:
            print "this class is connected to the ingress network", ingress_class
            class_to_networks[ingress_class].append(ingress_network)
    '''
    # determine which container instances should be the originator point
    originator_class = config_params["exfiltration_info"]["exfiltration_path_class"][-1]
    possible_originators = {}
    print "originator class", originator_class
    possible_originators[originator_class], class_to_networks[originator_class] = get_class_instances(orchestrator, originator_class, class_to_net[originator_class])
    num_originators = int(config_params["exfiltration_info"]
                                    ["exfiltration_path_how_many_instances_for_each_class"][-1])
    print "num originators", num_originators
    selected_originators = {}
    selected_originators[originator_class] = random.sample(possible_originators[originator_class], num_originators)
    print "selected originators", selected_originators, selected_originators[originator_class]

    # map all of the names of the proxy container instances to their corresponding IP's
    # a dict of dicts (instance -> networks -> ip)
    print "about to map the proxy instances to their networks to their IPs..."
    proxy_instance_to_networks_to_ip = map_container_instances_to_ips(orchestrator, possible_proxies, class_to_networks)
    proxy_instance_to_networks_to_ip.update( map_container_instances_to_ips(orchestrator, possible_originators, class_to_networks) )
    print "proxy_instance_to_networks_to_ip", proxy_instance_to_networks_to_ip
    for container, network_to_ip in proxy_instance_to_networks_to_ip.iteritems():
        print container.name, [i.name for i in network_to_ip.keys()]

    selected_containers = selected_proxies.copy()
    selected_containers.update(selected_originators)


    print "#####"

    for name_of_class, network in class_to_networks.iteritems():
        print name_of_class, [i.name for i in network]

    exfil_protocol = config_params["exfiltration_info"]["exfil_protocol"]
    #'''
    # need to install the pre-reqs for each of the containers (proxies + orgiinator)
    # note: assuming endpoint (i.e. local) pre-reqs are already installed
    if install_det_depen_p:
        for class_name, container_instances in possible_proxies.iteritems():
            for container in container_instances:
                install_det_dependencies(orchestrator, container, class_to_installer[class_name])

        print "possible_originators", possible_originators
        for class_name, container_instances in possible_originators.iteritems():
            for container in container_instances:
                install_det_dependencies(orchestrator, container, class_to_installer[class_name])

    # start thes proxy DET instances (note: will need to dynamically configure some
    # of the configuration file)
    # note: this is only going to work for a single src and a single dst ip, ATM
    for class_name, container_instances in selected_proxies.iteritems():
        # okay, gotta determine srcs and dst
        # okay, so fine location in exfil_path
        # look backward to find src class, then index into selected_proxies, and then index into
        # instances_to_network_to_ips (will need to match up networks)

        dsts,srcs=find_dst_and_srcs_ips_for_det(exfil_path, class_name, selected_containers, localhostip,
                                                proxy_instance_to_networks_to_ip, class_to_networks)

        for container in container_instances:
            # todo: does this work?
            for dst in dsts:
                print "config stuff", container.name, srcs, dst, proxy_instance_to_networks_to_ip[ container ]
                start_det_proxy_mode(orchestrator, container, srcs, dst, exfil_protocol,
                                        maxsleep, max_exfil_bytes_in_packet, min_exfil_bytes_in_packet)

    # start the endpoint (assuming the pre-reqs are installed prior to the running of this script)
    # todo: modify this for the k8s scaneario
    # todo: modify for 1-n, n-1, n-n
    print "ex", exfil_path[0], proxy_instance_to_networks_to_ip[ selected_proxies[exfil_path[0]][0] ]
    srcs = []
    for proxy_instance in selected_proxies[exfil_path[0]]:
        for network, ip in proxy_instance_to_networks_to_ip[ proxy_instance ].iteritems():
            print "finding proxy for local", network.name, ip
            if network.name == 'ingress':
                srcs.append(ip)
                break
    if srcs == []:
        print "cannot find the the hop-point immediately before the local DET instance"
        exit(1)
    #srcs = [ proxy_instance_to_networks_to_ip[ selected_proxies[exfil_path[0]][0] ]['ingress'] ]
    #'''
    print "srcs for local", srcs
    #'''
    start_det_server_local(exfil_protocol, srcs, maxsleep, max_exfil_bytes_in_packet,
                           min_exfil_bytes_in_packet)
    #'''
    # now setup the originator (i.e. the client that originates the exfiltrated data)
    next_instance_ip, _ = find_dst_and_srcs_ips_for_det(exfil_path, originator_class,
                                                                         selected_containers, localhostip,
                                                                         proxy_instance_to_networks_to_ip,
                                                                         class_to_networks)

    print "next ip for the originator to send to", next_instance_ip
    directory_to_exfil = config_params["exfiltration_info"]["folder_to_exfil"]
    regex_to_exfil = config_params["exfiltration_info"]["regex_of_file_to_exfil"]
    files_to_exfil = []
    for class_name, container_instances in selected_originators.iteritems():
        for container in container_instances:
            file_to_exfil = setup_config_file_det_client(next_instance_ip, container, directory_to_exfil, regex_to_exfil,
                                                         maxsleep, min_exfil_bytes_in_packet, max_exfil_bytes_in_packet)
            files_to_exfil.append(file_to_exfil)

    print "files_to_exfil", files_to_exfil
    experiment_length = config_params["experiment"]["experiment_length_sec"]
    for i in range(0, int(config_params["experiment"]["number_of_trials"])):
        exfil_info_file_name = experiment_name + '_docker' + '_' + str(i) + '_exfil_info.txt'

        # step (3b) get docker configs for docker containers (assuming # is constant for the whole experiment)
        container_id_file = experiment_name + '_docker' + '_' + str(i) + '_containers.txt'
        container_config_file = experiment_name + '_docker' '_' + str(i) +  '_container_configs.txt'
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

        out = subprocess.check_output(['bash', './src/docker_container_configs.sh', container_id_file, container_config_file])
        print out

        # step (5) start load generator (okay, this I can do!)
        start_time = time.time()
        max_client_count = int( config_params["experiment"]["number_background_locusts"])
        thread.start_new_thread(generate_background_traffic, ((int(experiment_length)+2.4), max_client_count,
                    config_params["experiment"]["traffic_type"], config_params["experiment"]["background_locust_spawn_rate"],
                                                              config_params["application_name"], ip, port))

        # step (4) setup testing infrastructure (i.e. tcpdump)
        for network_id, network_namespace in network_ids_to_namespaces.iteritems():
            current_network =  client.networks.get(network_id)
            print "about to tcpdump on:", current_network.name
            filename = config_params["experiment_name"] + '_' + current_network.name + '_' + str(i) + '.pcap'
            thread.start_new_thread(start_tcpdump, (orchestrator, network_namespace, str(int(experiment_length)), filename))


        # step (6) start data exfiltration at the relevant time
        ## this will probably be a fairly simple modification of part of step 3
        # for now, assume just a single exfiltration time
        exfil_start_time = int(config_params["exfiltration_info"]["exfil_start_time"])
        exfil_end_time = int(config_params["exfiltration_info"]["exfil_end_time"])

        print "need to wait this long before starting the det client...", start_time + exfil_start_time - time.time()
        print "current time", time.time(), "start time", start_time, "exfil_start_time", exfil_start_time, "exfil_end_time", exfil_end_time
        time.sleep(start_time + exfil_start_time - time.time())
        #file_to_exfil = config_params["exfiltration_info"]["folder_to_exfil"]
        file_to_exfil = files_to_exfil[0]
        for class_name, container_instances in selected_originators.iteritems():
            for container in container_instances:
                thread.start_new_thread(start_det_client, (file_to_exfil, exfil_protocol, container))

        print "need to wait this long before stopping the det client...", start_time + exfil_end_time - time.time()
        time.sleep(start_time + exfil_end_time - time.time())
        for class_name, container_instances in selected_originators.iteritems():
            for container in container_instances:
                stop_det_client(container)

        # step (7) wait, all the tasks are being taken care of elsewhere
        time.sleep(start_time + int(experiment_length) + 7 - time.time())
        # I don't wanna return while the other threads are still doing stuff b/c I'll get confused


    # stopping the proxies can be done the same way (useful if e.g., switching
    # protocols between experiments, etc.)
    for class_name, container_instances in selected_proxies.iteritems():
        for container in container_instances:
            stop_det_client(container)


def prepare_app(app_name, config_params, ip, port):
    if app_name == "sockshop":
        print config_params["number_background_locusts"], config_params["background_locust_spawn_rate"], config_params["number_customer_records"]
        print type(config_params["number_background_locusts"]), type(config_params["background_locust_spawn_rate"]), type(config_params["number_customer_records"])
        request_url = "--host=http://" + ip + ":"+ str(port)
        print request_url
        prepare_cmds = ["locust", "-f", "./sockshop_config/pop_db.py", request_url, "--no-web", "-c",
             config_params["number_background_locusts"], "-r", config_params["background_locust_spawn_rate"],
             "-n", config_params["number_customer_records"]]
        print prepare_cmds
        out = subprocess.check_output(prepare_cmds)
        print out
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
        print out
    else:
        # TODO TODO TODO other applications will require other setup procedures (if they can be automated) #
        # note: some cannot be automated (i.e. wordpress)
        pass


# Func: generate_background_traffic
#   Uses locustio to run a dynamic number of background clients based on 24 time steps
# Args:
#   time: total time for test. Will be subdivided into 24 smaller chunks to represent 1 hour each
#   max_clients: Arg provided by user in parameters.py. Represents maximum number of simultaneous clients
def generate_background_traffic(run_time, max_clients, traffic_type, spawn_rate, app_name, ip, port):
    #minikube = get_IP()#subprocess.check_output(["minikube", "ip"]).rstrip()
    devnull = open(os.devnull, 'wb')  # disposing of stdout manualy

    client_ratio = []

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

    #24 = hours in a day, we're working with 1 hour granularity
    timestep = run_time / 24.0
    for i in xrange(24):

        client_count = str(int(round(normalizer*client_ratio[i]*max_clients)))
        proc = 0

        try:
            if app_name == "sockshop":
                proc = subprocess.Popen(["locust", "-f", "./sockshop_config/background_traffic.py",
                                         "--host=http://"+ip+ ":" +str(port), "--no-web", "-c",
                                        client_count, "-r", spawn_rate],
                                        stdout=devnull, stderr=devnull, preexec_fn=os.setsid)
                #print proc.stdout
            # for use w/ seastore:
            if app_name == "atsea_store":
                proc = subprocess.Popen(["locust", "-f", "./load_generators/seashop_background.py", "--host=https://"+ip+ ":" +str(port),
                                     "--no-web", "-c", client_count, "-r", spawn_rate],
                              stdout=devnull, stderr=devnull, preexec_fn=os.setsid)
            #proc = subprocess.Popen(["locust", "-f", "./load_generators/wordpress_background.py", "--host=https://192.168.99.103:31758",
            #                        "--no-web", "-c", client_count, "-r", spawn_rate],
            #                        stdout=devnull, stderr=devnull, preexec_fn=os.setsid)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

        print("Time: " + str(i) + ". Now running with " + client_count + " simultaneous clients")

        #Run some number of background clients for 1/24th of the total test time
        time.sleep(timestep)
        # this stops the background traffic process
        if proc:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM) # should kill it


def get_IP(orchestrator):
    if orchestrator == "kubernetes":
        ip = subprocess.check_output(["kubectl", "config", "view"])
        for thing in ip.split("\n"):
            if "https" in thing:
                return thing.split(":")[2].split("//")[1]
    if orchestrator == "docker_swarm":
        # TODO TODO TODO TODO
        return "0"
    return "-1"


# note this may need to be implemented as a seperate thread
# in which case it'll also need experimental time + will not need
# to reset the bash situation
def start_tcpdump(orchestrator, network_namespace, tcpdump_time, filename):
    if orchestrator == "kubernetes":
        pass
    elif orchestrator == "docker_swarm":
        # ssh root@MachineB 'bash -s' < local_script.sh
        #args = ["docker-machine", "ssh", "default",  "-s", "./src/start_tcpdump.sh"]
        #, network_namespace, tcpdump_time,
                #orchestrator, filename
        #out = subprocess.check_output(args)
        #print out
        #args = ['docker-machine', 'ssh', 'default', '-t', "sudo ls /var/run/docker/netns"]

        start_netshoot = "docker run -it --rm -v /var/run/docker/netns:/var/run/docker/netns -v /home/docker:/outside --privileged=true nicolaka/netshoot"
        print network_namespace, tcpdump_time
        switch_namespace =  'nsenter --net=/var/run/docker/netns/' + network_namespace + ' ' 'sh'
        start_tcpdum = "tcpdump -G " + tcpdump_time + ' -W 1 -i ' + "br0" + ' -w /outside/' + filename
        cmd_to_send = start_netshoot + ';' + switch_namespace + ';' + start_tcpdum
        print "cmd_to_send", cmd_to_send
        print "start_netshoot", start_netshoot
        print "switch_namespace", switch_namespace
        print "start_tcpdum", start_tcpdum

        args = ['docker-machine', 'ssh', 'default', '-t', cmd_to_send]

        child = pexpect.spawn('docker-machine ssh default')
        child.expect('##')
        print child.before, child.after
        print "###################"
        child.sendline(start_netshoot)
        child.expect('Netshoot')
        print child.before, child.after
        child.sendline(switch_namespace)
        child.expect('#')
        print child.before, child.after
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

        # tcpdump file is safely on minikube but we might wanna move it all the way to localhost
        args2 = ["docker-machine", "scp",
                 "docker@default:/home/docker/" + filename, filename]
        out = subprocess.check_output(args2)
        print out


# returns a list of container names that correspond to the
# selected class
def get_class_instances(orchestrator, class_name, networks):
    print "finding class instances for: ", class_name
    if orchestrator == "kubernetes":
        # TODo
        pass
    elif orchestrator == "docker_swarm":
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
        '''
        for network in client.networks.list(greedy=True):
            try:
                #print 'note', network, network.containers
                for container in network.containers:
                    print "containers", network.containers
                    if class_name +'.' in container.name:
                        print class_name, container.name
                        container_instances.append(container)
                        container_networks_attached.append(network)
            except Exception as e:
                print network.name, "has hidden containers...", "error", e
        #'''
        for network in client.networks.list():
            for connected_nets in networks:
                if connected_nets in network.name:
                    container_networks_attached.append(network)

        return container_instances, list(set(container_networks_attached))
    else:
        pass


def get_network_ids(orchestrator, list_of_network_names):
    if orchestrator == "kubernetes":
        #TODO
        pass
    elif orchestrator == "docker_swarm":
        network_ids = []
        client = docker.from_env()
        for network_name in list_of_network_names:
            for network in client.networks.list(greedy=True):
                print network, network.name, network_name
                if network_name == network.name:
                    network_ids.append(network.id)

        print "just finished getting network id's...", network_ids
        return network_ids
    else:
        # TODO
        pass


def map_container_instances_to_ips(orchestrator, class_to_instances, class_to_networks):
    instance_to_networks_to_ip = {}
    print "class_to_instance_names", class_to_instances.keys()
    print class_to_instances
    for class_name, containers in class_to_instances.iteritems():
        print 'class_to_networks[class_name]', class_to_networks[class_name], class_name,  class_to_networks
        for container in containers:
            instance_to_networks_to_ip[ container ] = {}
            container_atrribs =  container.attrs
            for connected_network in class_to_networks[class_name]:
                instance_to_networks_to_ip[container][connected_network] = []
                try:
                    ip_on_this_network = container_atrribs["NetworkSettings"]["Networks"][connected_network.name]["IPAMConfig"][
                        "IPv4Address"]
                    instance_to_networks_to_ip[container][connected_network] = ip_on_this_network
                except:
                    pass
    print "instance_to_networks_to_ip", instance_to_networks_to_ip
    return instance_to_networks_to_ip


def install_det_dependencies(orchestrator, container, installer):
    if orchestrator == 'kubernetes':
        ## todo
        pass
    elif orchestrator == "docker_swarm":
        # okay, so want to read in the relevant bash script
        # make a list of lists, where each list is a line
        # and then send each to the container
        # (I shall pretest it, so I'm just going to ignore error for the moment...)

        upload_config_command = ["docker", "cp", "./src/modify_resolve_conf.sh", container.id+ ":/modify_resolv.sh"]
        out = subprocess.check_output(upload_config_command)
        print "upload_config_command", upload_config_command, out

        out = container.exec_run(['sh', '//modify_resolv.sh'], stream=True, user="root")
        print out

        if installer == 'apk':
            filename = './install_scripts/apk_det_dependencies.sh'
        elif installer =='apt':
            filename = './install_scripts/apt_det_dependencies.sh'
        elif installer == 'tce-load':
            filename = './install_scripts/tce_load_det_dependencies.sh'
        else:
            print "unrecognized installer, cannot install DET dependencies.."
            filename = ''
            pass

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
    network_ids_to_namespaces = {}
    if orchestrator == 'kubernetes':
        ## todo
        pass
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
        #print "network_ids_to_namespaces", network_ids_to_namespaces
        return network_ids_to_namespaces
    else:
        pass


# note: det must be a single ip, in string form, ATM
def start_det_proxy_mode(orchestrator, container, srcs, dst, protocol, maxsleep, maxbytesread, minbytesread):
    network_ids_to_namespaces = {}
    if orchestrator == 'kubernetes':
        ## todo
        pass
    elif orchestrator == "docker_swarm":
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
        cp_command = ['cp', './src/det_config_template.json', './current_det_config.json']
        out = subprocess.check_output(cp_command)
        print "cp command result", out

        # TODO: modify the switches below for n-n / 1-n
        targetip_switch = "s/TARGETIP/\"" + dst + "\"/"
        print srcs[0], srcs
        src_string = ""
        for src in srcs[:-1]:
            src_string += "\\\"" + src +  "\\\"" + ','
        str_string += "\\\"" + src[-1] +  "\\\""
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
            container.exec_run(start_det_command, user="root", workdir='/DET',stdout=False)
            #print "response from DET proxy start command:"
            #print out
        except:
            print "start det proxy command is hanging, going to hope it is okay and just keep going"
        #for output in out.output:
        #    print output
        #print "\n"

    else:
        pass


def start_det_server_local(protocol, srcs, maxsleep, maxbytesread, minbytesread):
    # okay, need to modify this so that it can work (can use the working version above as a template)
    #'''
    cp_command = ['sudo', 'cp', "./src/det_config_local_template.json", "/DET/det_config_local_configured.json"]
    out = subprocess.check_output(cp_command)
    print "cp command result", out

    # todo: switch this for 1-n and n-n
    #proxiesip_switch = "s/PROXIESIP/" + "[\\\"" + srcs[0] + "\\\"]" + "/"
    src_string = ""
    for src in srcs[:-1]:
        src_string += "\\\"" + src +  "\\\"" + ','
    str_string += "\\\"" + src[-1] +  "\\\""
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
    cmd = subprocess.Popen(cmds, cwd='/DET/', preexec_fn=os.setsid)
    print cmd # okay, I guess I'll just analyze the output manually... (Since this fancy thing doe
    '''
    parsing_thread = thread.start_new_thread(parse_local_det_output, (cmd, exfil_info_file_name))

    # now wait for a certain amount of time and then kill both of those
    # threads, so that we can start on the next experiment
    print "about to start waiting for", exp_time, "seconds!"
    time.sleep(exp_time)
    # honestly, I don't really need to kill it, b/c it'll stop getting input and then it'll
    # just die with the process (i think)...
    parsing_thread.kill()
    if cmd:
        #os.killpg(os.getpgid(cmd.pid), signal.SIGTERM)
        os.system("sudo kill %s" % (cmd.pid,))
    print cmd
    '''

def parse_local_det_output(subprocess_output, exfil_info_file_name):
    print "this is the output parsing function!!"
    cmd = subprocess_output
    time_of_first_arrival = None
    total_bytes = 0
    for line in cmd.stdout:
        print "before recieved", line
        if "Received" in line:
            print "after recieved", line
            matchObj = re.search(r'(.*)Received(.*)bytes (.*)', line)
            bytes_recieved = int(matchObj.group(2))
            total_bytes += bytes_recieved
            print "bytes recieved...", bytes_recieved
            print "total bytes...", total_bytes
            if not time_of_first_arrival:
                time_of_first_arrival = time.time()
            current_time = time.time()
            with open(exfil_info_file_name, 'w') as f:
                print >> f, time_of_first_arrival, '\n', total_bytes, '\n', current_time
                # then just keep printing these values to a file.
                # we can just kill the thread at the end of the program and it'll be fine
                # note: will also need to write current time to file, so that the endpoint
                # in the time can be calculated

def setup_config_file_det_client(dst, container, directory_to_exfil, regex_to_exfil, maxsleep, minbytesread, maxbytesread):
    # note: don't want to actually start the client yet, however
    out = subprocess.check_output(['pwd'])
    print out

    cp_command = ['cp', './src/det_config_client_template.json', './det_config_client.json']
    out = subprocess.check_output(cp_command)
    print "cp command result", out

    print 'dst', dst
    # todo: modify this for n-to-1 (you know what I mean)
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

    # i also want to move ./src/loop.py here (so that I can call it easily later on)
    upload_loop_command = ["docker", "cp", "./src/loop.py", container.id + ":/DET/loop.py"]
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

    # todo: modify this function to handle 1-n and n-1 situations
    # at originator -> no srcs (or rather, it is the src for itself):
    print current_class_name, current_loc_in_exfil_path+1, len(exfil_path)
    if current_loc_in_exfil_path+1 == len(exfil_path):
        srcs = None
        pass
    else: # then it has srcs other than itself
        prev_class_in_path = exfil_path[current_loc_in_exfil_path + 1]
        print selected_containers
        # todo: this is one of the places to modify [[ this whole sectio will need to be modified so that
        # todo: the srcs list has all of the previous ip's append to it ]]
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
    else: # then it'll hop through another microservice
        # then can just pass to another proxy in the exfiltration path

        # todo: this is another one of the places to modify, per the todos above,
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

if __name__=="__main__":
    print "RUNNING"

    #file = open('./sockshop_config/pop_db.py', "r")
    #for line in file:
    #    print line,

    parser = argparse.ArgumentParser(description='Creates microservice-architecture application pcaps')

    parser.add_argument('--exp_name',dest="exp_name", default='None')
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


    #  localhost communicates w/ vm over vboxnet0 ifconfig interface, apparently, so use the
    # address there as the response address, in this case it seems to default to the below
    # value, but that might change at somepoints
    parser.add_argument('--localhostip',dest="localhostip", default="192.168.99.1")

    args = parser.parse_args()
    #print args.restart_minikube, args.setup_sockshop, args.run_experiment, args.analyze, args.output_dict, args.tcpdump, args.on_cloudlab, args.app, args.istio_p, args.hpa
    print args.exp_name, args.config_file, args.prepare_app_p, args.port_number, args.vm_ip, args.localhostip, args.install_det_depen_p

    with open(args.config_file + '.json') as f:
        config_params = json.load(f)
    orchestrator = config_params["orchestrator"]

    if args.vm_ip == 'None':
        ip = get_IP(orchestrator)
    else:
        ip = args.vm_ip

    # need to setup some environmental variables so that the docker python api will interact with
    # the docker daemon on the docker machine
    docker_host_url = "tcp://" + ip + ":" + args.docker_daemon_port
    print "docker_host_url", docker_host_url
    path_to_docker_machine_tls_certs = "/users/jsev/.docker/machine/machines/default"
    print "path_to_docker_machine_tls_certs", path_to_docker_machine_tls_certs
    os.environ['DOCKER_HOST'] = docker_host_url
    os.environ['DOCKER_TLS_VERIFY'] = "1"
    os.environ['DOCKER_CERT_PATH'] = path_to_docker_machine_tls_certs
    client =docker.from_env()

    main(args.exp_name, args.config_file, args.prepare_app_p, int(args.port_number), ip, args.localhostip, args.install_det_depen_p)
