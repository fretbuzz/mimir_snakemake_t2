'''
To see usage: python run_experiment.py --help

it'll save the resulting pcaps in the local folder, since it is necessary to transfer them off of
cloudlab anyway, it makes little sense to use a complicated directory structure to store data at
this stage (though it makes more sense later, when I'm actually analyzing the pcaps)
'''

# okay, so the plan for saturday is:
#   TODO: monday: I'm up to about line 155 in making this work. So keep going from there.
#                 I think I can get those whole thing working on Monday (I'd say maybe like 4-5 hours)
#                 Then I can get working on those graph below.
#                 Plus, need to get the configs (below) + write instructions for recovering the pcaps
#                       from the cloudlab machine (should be fairly easy but is still important)
#                 todo: automate getting the relevant docker configs (for the pcap processing function)
#    also, I can't forget about making those graphs. i'm going to want to (1) fix/finish
#       those aggregate boxplots that I wanted (2) implement those other metrics (3) try running
#       on sockshop. This may seem like a lot, but (1) should only take like 1/2 an hour and (3) ideally
#       doesn't take much time either. Now (2) might need to be post-poned....
#   okay, so if I work hard, I think I can stop at... 4? that's like 7 hours or so, so it should
#   be plenty...

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


def main(experiment_name, config_file, prepare_app_p, port, ip, localhostip):
    # step (1) read in the config file
    with open(config_file + '.json') as f:
        config_params = json.load(f)
    if experiment_name == 'None':
        experiment_name = config_params["experiment_name"]
    orchestrator = config_params["orchestrator"]
    class_to_installer = config_params["exfiltration_info"]["exfiltration_path_class_which_installer"]

    # step (2) setup the application, if necessary (e.g. fill up the DB, etc.)
    # note: it is assumed that the application is already deployed

    print prepare_app_p, config_params["application_name"], config_params["setup"], ip, port

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
    print "full_network_ids", full_network_ids
    print "network_ids_to_namespaces", network_ids_to_namespaces
    print "network_namespaces", network_ids_to_namespaces

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
        possible_proxies[proxy_class], class_to_networks[proxy_class] = get_class_instances(orchestrator, proxy_class)
        print "new possible proxies", possible_proxies[proxy_class]
        print "new class_to_network mapping", class_to_networks[proxy_class]
        num_proxies_of_this_class = int(config_params["exfiltration_info"]
                                        ["exfiltration_path_how_many_instances_for_each_class"][index])
        selected_proxies[proxy_class] = random.sample(possible_proxies[proxy_class], num_proxies_of_this_class)
        print "new selected proxies", selected_proxies[proxy_class]
        index += 1

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

    # determine which container instances should be the originator point
    originator_class = config_params["exfiltration_info"]["exfiltration_path_class"][-1]
    possible_originators = {}
    print "originator class", originator_class
    possible_originators[originator_class], class_to_networks[originator_class] = get_class_instances(orchestrator, originator_class)
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

    print "#####"

    for name_of_class, network in class_to_networks.iteritems():
        print name_of_class, [i.name for i in network]

    # need to install the pre-reqs for each of the containers (proxies + orgiinator)
    # note: assuming endpoint (i.e. local) pre-reqs are already installed
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
    # TODO:  I should probably make a function to that takes which pair of containers you
    # want an ip connection between and it returns the ip, since I have pretty much the same
    # code three times below...
    exfil_protocol = config_params["exfiltration_info"]["exfil_protocol"]
    for class_name, container_instances in selected_proxies.iteritems():
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

    # start the endpoint (assuming the pre-reqs are installed prior to the running of this script)
    srcs = [ proxy_instance_to_networks_to_ip[ selected_proxies[exfil_path[0]] ]['ingress'] ]
    start_det_server_local(exfil_protocol, srcs)

    # now setup the originator (i.e. the client that originates the exfiltrated data)
    next_class_in_path = exfil_path[-2]
    next_instance = selected_proxies[next_class_in_path]
    current_class_networks = proxy_instance_to_networks_to_ip[selected_originators[originator_class][0]].keys()
    next_class_networks = proxy_instance_to_networks_to_ip[next_instance].keys()
    next_and_current_class_network = list(set(current_class_networks + next_class_networks))[
        0]  # should be precisly one
    next_instance_ip = proxy_instance_to_networks_to_ip[next_instance][next_and_current_class_network]
    for class_name, container_instances in selected_originators:
        for container in container_instances:
            setup_config_file_det_client(next_instance_ip, container)

    experiment_length = config_params["experiment"]["experiment_length_sec"]
    for i in range(0, int(config_params["experiment"]["number_of_trials"])):
        # step (4) setup testing infrastructure (i.e. tcpdump)
        for network_namespace in network_namespaces:
            filename = config_params["experiment"]["experiment_name"] + '_' + network_namespace + '_' + str(i) + '.pcap'
            thread.start_new_thread(start_tcpdump, (orchestrator, network_namespaces, experiment_length + 5, filename))

        # step (5) start load generator (okay, this I can do!)
        start_time = time.time()
        max_client_count = int( config_params["experiment"]["number_background_locusts"])
        thread.start_new_thread(generate_background_traffic, (experiment_length, max_client_count,
                    config_params["experiment"]["traffic_type"], config_params["experiment"]["background_locust_spawn_rate"]))

        # step (6) start data exfiltration at the relevant time
        ## this will probably be a fairly simple modification of part of step 3
        # for now, assume just a single exfiltration time
        exfil_start_time = int(config_params["exfiltration_info"]["exfil_start_time"])
        exfil_end_time = int(config_params["exfiltration_info"]["exfil_end_time"])

        time.sleep(start_time + exfil_start_time - time.time())
        file_to_exfil = config_params["exfiltration_info"]["file_to_exfil"]
        for class_name, container_instances in selected_originators:
            for container in container_instances:
                start_det_client(file_to_exfil, exfil_protocol, container)

        time.sleep(start_time + exfil_end_time - time.time())
        for class_name, container_instances in selected_originators:
            for container in container_instances:
                stop_det_client(container)

        # step (7) wait, all the tasks are being taken care of elsewhere
        time.sleep(start_time + experiment_length + 5 - time.time())


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
            try:
                #print 'note', network, network.containers
                for container in network.containers:
                    if class_name +'.' in container.name:
                        print class_name, container.name
                        container_instances.append(container)
                        container_networks_attached.append(network)
            except:
                print network.name, "has hidden containers..."

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
        #args = ["docker-machine", "ssh", "<", "./src/get_network_namespaces.sh"]
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

        container.exec_run(start_det_command)

    else:
        pass


def start_det_server_local(protocol, srcs):
    sed_command = ["sed", "\'s/PROXIESIP/" + srcs + "/\'",
                   "./src/det_config_local_template.json", ">", "./src/det_config_local_configured.json"]
    subprocess.Popen(sed_command)

    # note: don't have to move anything b/c the file is already local

    cmds = ["python", "det.py", "-L" ,"-c", "./src/det_config_local_configured.json", "-p", protocol]
    subprocess.Popen(cmds)


def setup_config_file_det_client(dst, container):
    sed_command = ["sed", "\'s/TARGETIP/" + dst + "/\'",
                   "./src/det_config_client_template.sh", ">", "./src/det_config_client.sh.json"]
    subprocess.Popen(sed_command)

    upload_config_command = ["docker", "cp", container.id + ":/config.json", "./src/det_config_client.sh.json"]
    subprocess.Popen(upload_config_command)


def start_det_client(file, protocol, container):
    cmds = ["python", "/det.py", "-c", "./config.json", "-p", protocol, "-f", file]
    container.exec_run(cmds)


def stop_det_client(container):
    ## let's just kill all python processes, that'll be easier than trying to record PIDs, or anything else
    cmds = ["pkill", "-9", "python"]
    container.exec_run(cmds)

def find_dst_and_srcs_ips_for_det(exfil_path, current_class_name):
    current_loc_in_exfil_path = exfil_path.index(current_class_name)

    # at originator -> no srcs (or rather, it is the src for itself):
    if current_loc_in_exfil_path == len(exfil_path):
        srcs = None
        pass
    else: # then it has srcs other than itself
        prev_class_in_path = exfil_path[loc_in_exfil_path + 1]
        pass

    # at last microservice hop -> next dest is local host
    if current_loc_in_exfil_path == 0:
        pass
    else: # then it'll hop through another microservice
        pass

    prev_instance = selected_proxies[prev_class_in_path]
    current_class_networks = proxy_instance_to_networks_to_ip[class_name].keys()
    prev_class_networks = proxy_instance_to_networks_to_ip[prev_instance].keys()
    prev_and_current_class_network = list(set(current_class_networks + prev_class_networks))[
        0]  # should be precisly one
    prev_instance_ip = [proxy_instance_to_networks_to_ip[prev_instance][prev_and_current_class_network]]


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
    parser.add_argument('--port',dest="port_number", default='80')
    parser.add_argument('--ip',dest="vm_ip", default='None')
    parser.add_argument('--docker_daemon_port',dest="docker_daemon_port", default='2376')


    #  localhost communicates w/ vm over vboxnet0 ifconfig interface, apparently, so use the
    # address there as the response address, in this case it seems to default to the below
    # value, but that might change at somepoints
    parser.add_argument('--localhostip',dest="localhostip", default="192.168.99.1")

    args = parser.parse_args()
    #print args.restart_minikube, args.setup_sockshop, args.run_experiment, args.analyze, args.output_dict, args.tcpdump, args.on_cloudlab, args.app, args.istio_p, args.hpa
    print args.exp_name, args.config_file, args.prepare_app_p, args.port_number, args.vm_ip, args.localhostip

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

    main(args.exp_name, args.config_file, args.prepare_app_p, int(args.port_number), ip, args.localhostip)
