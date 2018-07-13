'''
To see usage: python run_experiment.py --help

it'll save the resulting pcaps in the local folder, since it is necessary to transfer them off of
cloudlab anyway, it makes little sense to use a complicated directory structure to store data at
this stage (though it makes more sense later, when I'm actually analyzing the pcaps)
'''

import argparse
import copy
import os
import pickle
import signal
import subprocess
import thread
import time
import pexpect
import requests
import json
import docker
import random

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

def main(experiment_name, config_file, prepare_app_p, port, vm_ip, localhostip):
    # step (1) read in the config file
    with open(config_file + 'json') as f:
        config_params = json.load(f)
    if experiment_name == 'None':
        experiment_name = config_params["experiment_name"]
    orchestrator = config_params["orchestrator"]
    class_to_installer = config_params["exfiltration_info"]["exfiltration_path_class_which_installer"]

    # step (2) setup the application, if necessary (e.g. fill up the DB, etc.)
    # note: it is assumed that the application is already deployed
    if vm_ip == 'None':
        ip = get_IP(orchestrator)
    else:
        ip = vm_ip

    if prepare_app_p:
        prepare_app(config_params["application_name"], config_params["setup"], ip, port)

    # determine the network namespaces
    # this will require mapping the name of the network to the network id, which
    # is then present (in truncated form) in the network namespace
    full_network_ids = get_network_ids(orchestrator, config_params["networks_to_tcpdump_on"])
    network_ids_to_namespaces = map_network_ids_to_namespaces(orchestrator, full_network_ids)
    # okay, so I have the full network id's now, but these aren't the id's of the network namespace,
    # so I need to do two things: (1) get list of network namespaces, (2) parse the list to get the mapping
    network_namespaces = network_ids_to_namespaces.values()

    # step (3) prepare system for data exfiltration (i.g. get DET working on the relevant containers)
    # note: I may want to re-select the specific instances during each trial
    possible_proxies = {}
    selected_proxies = {}
    class_to_networks = {}
    # the furthest will be the originator, the others will be proxies (endpoint will be local)
    index = 0
    exfil_path = config_params["exfiltration_info"]["exfiltration_path_class"]
    for proxy_class in exfil_path[:-1]:
        possible_proxies[proxy_class], class_to_networks[proxy_class] = get_class_instances(orchestrator, proxy_class)
        num_proxies_of_this_class = int(config_params["exfiltration_info"]
                                        ["exfiltration_path_how_many_instances_for_each_class"][index])
        selected_proxies[proxy_class] = random.sample(possible_proxies[proxy_class], num_proxies_of_this_class)
        index += 1

    # determine which container instances should be the originator point
    originator_class = config_params["exfiltration_info"]["exfiltration_path_class"][-1]
    possible_originators = {}
    possible_originators[originator_class] = get_class_instances(orchestrator, originator_class)
    num_originators = int(config_params["exfiltration_info"]
                                    ["exfiltration_path_how_many_instances_for_each_class"][-1])
    selected_originators = {}
    selected_originators[originator_class] = random.sample(possible_originators[originator_class], num_originators)

    # map all of the names of the proxy container instances to their corresponding IP's
    # a dict of dicts (instance -> networks -> ip)
    proxy_instance_to_networks_to_ip = map_container_instances_to_ips(orchestrator, possible_proxies, class_to_networks)
    orginator_instance_to_networks_to_ip = map_container_instances_to_ips(orchestrator, possible_originators, class_to_networks)

    # need to install the pre-reqs for each of the containers (proxies + orgiinator)
    # note: assuming endpoint (i.e. local) pre-reqs are already installed
    for class_name, container_instances in possible_proxies:
        for container in container_instances:
            install_det_dependencies(orchestrator, container, class_to_installer[class_name])

    for class_name, container_instances in possible_originators:
        for container in container_instances:
            install_det_dependencies(orchestrator, container, class_to_installer[class_name])

    # start thes proxy DET instances (note: will need to dynamically configure some
    # of the configuration file)
    # note: this is only going to work for a single src and a single dst ip, ATM
    exfil_protocol = config_params["exfiltration_info"]["exfil_protocol"]
    for class_name, container_instances in selected_proxies:
        # okay, gotta determine srcs and dst
        # okay, so fine location in exfil_path
        # look backward to find src class, then index into selected_proxies, and then index into
        # instances_to_network_to_ips (will need to match up networks)
        loc_in_exfil_path = exfil_path.index(class_name)
        prev_class_in_path = exfil_path[loc_in_exfil_path + 1]
        prev_instance = selected_proxies[prev_class_in_path]
        current_class_networks = proxy_instance_to_networks_to_ip[class_name].keys()
        prev_class_networks = proxy_instance_to_networks_to_ip[prev_instance].keys()
        prev_and_current_class_network = list(set(current_class_networks + prev_class_networks))[0] # should be precisly one
        prev_instance_ip = [proxy_instance_to_networks_to_ip[prev_instance][prev_and_current_class_network]]

        # look forward to find dst class, do the same indexing thing; but if it is at the front, then it'd want to
        # send the data to the local instance
        if loc_in_exfil_path == 0:
            # then gotta pass it to the endpoint (i.e. the local instance running it)
            next_class_in_path = 'local'
            # todo check if this works for k8s also (so far only checked for swarm)
            next_instance_ip = localhostip

        else:
            # then can just pass to another proxy in the exfiltration path
            next_class_in_path = exfil_path[loc_in_exfil_path - 1]
            next_instance = selected_proxies[next_class_in_path]
            next_class_networks = proxy_instance_to_networks_to_ip[next_instance].keys()
            next_and_current_class_network = list(set(current_class_networks + next_class_networks))[
                0]  # should be precisly one
            next_instance_ip = proxy_instance_to_networks_to_ip[next_class_networks][next_and_current_class_network]

        for container in container_instances:
            start_det_proxy_mode(orchestrator, container, prev_instance_ip, next_instance_ip, exfil_protocol)

    # TODO start the endpoint (assuming the pre-reqs are installed prior to the running of this script)
    # after lunch, start with this. then do the one below, then ready for testing (maybe sketch an
    # architecture diagram first tho?)


    experiment_length = config_params["experiment"]["experiment_length_sec"]
    for i in range(0, int(config_params["experiment"]["number_of_trials"])):
        # step (4) setup testing infrastructure (i.e. tcpdump)
        for network_namespace in network_namespaces:
            filename = config_params["experiment"]["experiment_name"] + '_' + network_namespace + '_' + str(i) + '.pcap'
            thread.start_new_thread(start_tcpdump, (orchestrator, network_namespaces, experiment_length + 5, filename))

        # step (5) start load generator (okay, this I can do!)
        max_client_count = int( config_params["experiment"]["number_background_locusts"])
        thread.start_new_thread(generate_background_traffic, (experiment_length, max_client_count,
                    config_params["experiment"]["traffic_type"], config_params["experiment"]["background_locust_spawn_rate"]))

        # step (6) start data exfiltration at the relevant time
        ## TODO
        ## this will probably be a fairly simple modification of part of step 3

        # step (7) wait, all the tasks are being taken care of elsewhere
        time.sleep(experiment_length + 5)

def prepare_app(app_name, config_params, ip, port):
    if app_name == "sockshop":
        out = subprocess.check_output(
            ["locust", "-f", "./sockshop_config/pop_db.py", "--host=http://" + ip + ":"+ port, "--no-web", "-c",
             config_params["number_background_locusts"], "-r", config_params["background_locust_spawn_rate"],
             "-n", config_params["number_customer_records"]])
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

        try:
            if app_name == "sockshop":
                proc = subprocess.Popen(["locust", "-f", "./sockshop_config/background_traffic.py",
                                         "--host=http://"+ip+ ":" +port, "--no-web", "-c",
                                        client_count, "-r", spawn_rate],
                                        stdout=devnull, stderr=devnull, preexec_fn=os.setsid)
            # for use w/ seastore:
            #proc = subprocess.Popen(["locust", "-f", "./load_generators/seashop_background.py", "--host=https://192.168.99.107",
            #                         "--no-web", "-c", client_count, "-r", spawn_rate],
            #                  stdout=devnull, stderr=devnull, preexec_fn=os.setsid)
            #proc = subprocess.Popen(["locust", "-f", "./load_generators/wordpress_background.py", "--host=https://192.168.99.103:31758",
            #                        "--no-web", "-c", client_count, "-r", spawn_rate],
            #                        stdout=devnull, stderr=devnull, preexec_fn=os.setsid)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

        print("Time: " + str(i) + ". Now running with " + client_count + " simultaneous clients")

        #Run some number of background clients for 1/24th of the total test time
        time.sleep(timestep)
        # this stops the background traffic process 
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
        args = ["docker-machine", "ssh -s", "<", "./src/start_tcpdump.sh", network_namespace, tcpdump_time,
                orchestrator, filename]
        subprocess.Popen(args)

        time.sleep(tcpdump_time)
        # tcpdump file is safely on minikube but we might wanna move it all the way to localhost
        args2 = ["docker-machine", "scp", "-o", "StrictHostKeyChecking=no",
                 "/home/docker/" + filename, filename]
        subprocess.Popen(args2)

# returns a list of container names that correspond to the
# selected class
def get_class_instances(orchestrator, class_name):
    if orchestrator == "kubernetes":
        # TODo
        pass
    elif orchestrator == "docker_swarm":
        client = docker.from_env()
        container_instances = []
        container_networks_attached = []
        for network in client.networks.list(greedy=True):
            for container in network.containers:
                if class_name in container.name:
                    container_instances.append(container)
                    container_networks_attached.append(network)
        return container_instances, list(set(container_networks_attached))
    else
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
                if network_name == network.name:
                    network_ids.append(network.id)

        return network_ids
    else:
        # TODO
        pass

def map_container_instances_to_ips(orchestrator, class_to_instances, class_to_networks):
    instance_to_networks_to_ip = {}
    for class_name, containers in class_to_instances.iteritems():
        for container in containers:
            instance_to_networks_to_ip[ container ] = {}
            container_atrribs =  container.attrs
            for connected_network in class_to_networks[class_name]:
                instance_to_networks_to_ip[container][connected_network] = []
                try:
                    ip_on_this_network = container_atrribs["NetworkSettings"]["Networks"][connected_network]["IPAMConfig"][
                        "IPv4Address"]
                    instance_to_networks_to_ip[container][connected_network] = ip_on_this_network
                except:
                    pass

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
        if installer == 'apk':
            filename = './install_scripts/apk_det_dependencies.sh'
        elif installer =='apt':
            filename = './install_scripts/apt_det_dependencies.sh.sh'
        elif installer == 'tce-load':
            filename = './install_scripts/tce_load_det_dependencies.sh'
        else:
            print "unrecognized installer, cannot install DET dependencies.."
            pass

        with open(filename, 'r') as fp:
            read_lines = fp.readlines()
            read_lines = [line.rstrip('\n') for line in read_lines]

        for command_string in read_lines:
            command_list = command_string.split(' ')
            print "command string", command_string
            print "command list", command_list
            container.exec_run(command_list)

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
        args = ["docker-machine", "ssh -s", "<", "./src/get_network_namespaces.sh"]
        out = subprocess.check_output(args)
        network_namespaces = out.split(' ')

        # (2)
        for network_namespace in network_namespaces:
            if '1-' in network_namespace: # only looking at overlay networks
                for network_id in full_network_ids:
                    if network_namespace[2:] in network_id:
                        network_ids_to_namespaces[network_id] = network_namespace
                        break
        return network_ids_to_namespaces
    else:
        pass

# note: det must be a single ip, in string form, ATM
def start_det_proxy_mode(orchestrator, container, srcs, dst, protocol):
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

        sed_command = ["sed", "-e",  "\'s/TARGETIP/" + dst + "/\'", "-e", "\'s/PROXIESIP/" + srcs +"/\'",
                       "./src/det_config_template.json", ">", "./current_det_config.json"]
        subprocess.Popen(sed_command)

        upload_config_command = ["docker", "cp", container.id+ ":/config.json", "./current_det_config.json"]
        subprocess.Popen(upload_config_command)

        start_det_command = ["python", "/det.py", "-c", "./config.json", "-p", protocol, "-Z"]
        subprocess.Popen(start_det_command)

    else:
        pass

def start_det_server_local(protocol):
    # TODO pretty much this whole function
    cmds = ["python", "det.py", "-L" ,"-c", "./src/det_config_local_configured.json", "-p", protocol]
    pass


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Creates microservice-architecture application pcaps')

    parser.add_argument('--exp_name',dest="None", default='repOne')
    parser.add_argument('--config_file',dest="config_file", default='configFile')
    parser.add_argument('--prepare_app_p', dest='prepare_app_p', action='store_true',
                        default=False,
                        help='sets up the application (i.e. loads db, etc.)')
    parser.add_argument('--port',dest="port_number", default='30001')
    parser.add_argument('--ip',dest="vm_ip", default='None')

    #  localhost communicates w/ vm over vboxnet0 ifconfig interface, apparently, so use the
    # address there as the response address, in this case it seems to default to the below
    # value, but that might change at somepoints
    parser.add_argument('--localhostip',dest="localhostip", default="192.168.99.1")


    args = parser.parse_args()
    #print args.restart_minikube, args.setup_sockshop, args.run_experiment, args.analyze, args.output_dict, args.tcpdump, args.on_cloudlab, args.app, args.istio_p, args.hpa
    print args.exp_name, args.config_file, args.prepare_app_p, args.port_number, args.vm_ip, args.localhostip

    main(args.exp_name, args.config_file, args.prepare_app_p, int(args.port_number), args.vm_ip, args.localhostip)
