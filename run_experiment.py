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
import meta_parameters
import json

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

#def main(restart_kube, setup_sock, multiple_experiments, only_data_analysis):
def main(experiment_name, config_file, prepare_app_p, port, vm_ip):
    # step (1) read in the config file
    with open(config_file + 'json') as f:
        config_params = json.load(f)
    if experiment_name == 'None':
        experiment_name = config_params["experiment_name"]
    orchestrator = config_params["orchestrator"]

    # step (2) setup the application, if necessary (e.g. fill up the DB, etc.)
    # note: it is assumed that the application is already deployed
    if vm_ip == 'None':
        ip = get_IP(orchestrator)
    else:
        ip = vm_ip

    if prepare_app_p:
        prepare_app(config_params["application_name"], config_params["setup"], ip, port)

    # TODO: determine the network namespaces
    network_namespaces = []

    # step (3) prepare system for data exfiltration (i.g. get DET working on the relevant containers)
    ### TODO [probably via bash script that I ssh onto the vm]

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

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Creates microservice-architecture application pcaps')
    '''
    parser.add_argument('--start_minikube', dest='restart_minikube',
                        action='store_true',
                        default=False,
                        help='should minikube (and therefore Kubernetes) be (re)started')
    parser.add_argument('--setup_sockshop', dest='setup_sockshop', action='store_true',
                        default=False,
                        help='does sockshop need to be re(started)?')
    parser.add_argument('--analyze', dest='analyze', action='store_true',
                        default=False,
                        help='do you want to do data analysis??')
    parser.add_argument('--tcpdump', dest='tcpdump', action='store_true',
                        default=False,
                        help='do you want to store record logs using tcpdump?')
    parser.add_argument('--output_dict',dest="output_dict", default='all_results')
    parser.add_argument('--on_cloudlab', dest='on_cloudlab', action='store_true',
                        default=False,
                        help='are we starting minikube on cloudlab? (have dependencies + can make larger)')
    parser.add_argument('--with_istio', dest='istio_p', action='store_true',
                    default=False,
                    help='should we do the stuff involving istio?')
    parser.add_argument('--hpa', dest='hpa', action='store_true',
                    default=False,
                    help='setup horizontap pod autoscalers?')        
    parser.add_argument('--run_experiment', dest='run_experiment', action='store_true',
                        default=False,
                        help='should an actual experiment be run??')

    parser.add_argument("--app", type=str, default="sockshop", dest='app', help='what app do you want to run?')
    '''

    parser.add_argument('--exp_name',dest="None", default='repOne')
    parser.add_argument('--config_file',dest="config_file", default='configFile')
    parser.add_argument('--prepare_app_p', dest='prepare_app_p', action='store_true',
                        default=False,
                        help='sets up the application (i.e. loads db, etc.)')
    parser.add_argument('--port',dest="port_number", default='30001')
    parser.add_argument('--ip',dest="vm_ip", default='None')

    args = parser.parse_args()
    #print args.restart_minikube, args.setup_sockshop, args.run_experiment, args.analyze, args.output_dict, args.tcpdump, args.on_cloudlab, args.app, args.istio_p, args.hpa
    print args.exp_name, args.config_file, args.prepare_app_p, args.port_number, args.vm_ip

    main(args.exp_name, args.config_file, args.prepare_app_p, int(args.port_number), args.vm_ip)
