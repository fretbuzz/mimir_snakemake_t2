## the purpose of this file is to move the results of the testbed
## to the local machine, where it will optionally start processing them

import argparse
import pwnlib.tubes.ssh
from pwn import *
import time
import json

################################

cloudlab_private_key = '/Users/jseverin/Dropbox/cloudlab.pem'
possible_apps = ['drupal', 'sockshop', 'gitlab', 'eShop', 'wordpress', 'hipster']
experiment_sentinal_file = '/mydata/mimir_v2/experiment_coordinator/experiment_done.txt'
sentinal_file = '/mydata/all_done.txt'

###############################

def retrieve_results(s):
    print('hello')

    # check every five minutes until the sentinal file is present
    while True:
        # this is a special 'done' file used to indicate that
        # the experiment is finished.
        if 'experimenet_done' in s.download_data(experiment_sentinal_file):
            break
        time.sleep(200)

    s.download_dir(remote=remote_dir, local=local_dir)

def get_ip_and_port(app_name, sh):
    if app_name == 'sockshop':
        sh.sendline('minikube service front-end  --url --namespace="sock-shop"')
        namespace = 'sock-shop'
    elif app_name == 'wordpress':
        # step 1: get the appropriate ip / port (like above -- need for next step)
        sh.sendline('minikube service wwwppp-wordpress  --url')
    elif app_name == 'hipster':
        sh.sendline('minikube service frontend-external --url')
        pass
    else:
        pass

    line_rec = 'start'
    last_line = ''
    while line_rec != '':
        last_line = line_rec
        line_rec = sh.recvline(timeout=100)
        print("recieved line", line_rec)
    print("--end minikube_front-end port ---")

    # kubernetes_setup_functions.wait_until_pods_done(namespace)
    print "last_line", last_line
    minikube_ip, front_facing_port = last_line.split(' ')[-1].split('/')[-1].rstrip().split(':')
    print "minikube_ip", minikube_ip, "front_facing_port", front_facing_port
    return minikube_ip, front_facing_port

def sendline_and_wait_responses(sh, cmd_str, timeout=5):
    sh.sendline(cmd_str)
    line_rec = 'start'
    while line_rec != '':
        line_rec = sh.recvline(timeout=timeout)
        if 'Please enter your response' in line_rec:
            sh.sendline('n')
        print("recieved line", line_rec)

def run_experiment(app_name, config_file_name, exp_name, skip_setup_p, use_cilium, physical_attacks_p, skip_app_setup, dont_pull_p):
    s = None
    while s == None:
        try:
            s = pwnlib.tubes.ssh.ssh(host=cloudlab_server_ip,
                keyfile=cloudlab_private_key,
                user='jsev')
        except:
            time.sleep(60)

    # Create an initial process
    sh = s.run('sh')

    sendline_and_wait_responses(sh, 'pwd', timeout=5)
    print("--end pwd ---")

    sendline_and_wait_responses(sh, 'sudo newgrp docker', timeout=5)
    sendline_and_wait_responses(sh, 'export MINIKUBE_HOME=/mydata/', timeout=5)
    print("-- newgrp and export DONEE---")

    if not skip_setup_p and not skip_app_setup:
        sendline_and_wait_responses(sh, 'minikube stop', timeout=5)
        print("--end minikube-stop ---")

        sendline_and_wait_responses(sh, 'minikube delete', timeout=5)
        print("--end minikube delete ---")

        clone_mimir_str = "cd /mydata/; git clone https://github.com/fretbuzz/mimir_v2"

        sendline_and_wait_responses(sh, clone_mimir_str, timeout=5)

        if not dont_pull_p:
            update_mimir_str = "cd ./mimir_v2/; git pull"
            sendline_and_wait_responses(sh, update_mimir_str, timeout=5)

        sh.sendline("cd ./experiment_coordinator/")

        ## is this causing the problem??
        #sh.sendline("bash /mydata/mimir_v2/experiment_coordinator/dnscat_component/install_dnscat_server.sh")
        line_rec = "start"
        while line_rec != '':
            line_rec = sh.recvline(timeout=5)
            if 'Please enter your response' in line_rec:
                sh.sendline('n')
            print("recieved line", line_rec)

        sh.sendline('cd /mydata/mimir_v2/experiment_coordinator/exp_support_scripts/')
        time.sleep(300)
        sh.sendline('bash /mydata/mimir_v2/experiment_coordinator/exp_support_scripts/run_experiment.sh ' +
                    app_name + ' ' + str(use_cilium) )

        line_rec = 'start'
        last_line = ''
        while line_rec != '':
            last_line = line_rec
            line_rec = sh.recvline(timeout=240)
            print("recieved line", line_rec)
        print("did run_experiment work???")

        sentinal_file_setup = '/mydata/done_with_setup.txt'
        while True:
            # this is a special 'done' file used to indicate that
            # the experiment is finished.
            print "line_recieved: ", s.download_data(sentinal_file_setup)
            if 'done_with_that' in s.download_data(sentinal_file_setup):
                break
            time.sleep(20)

        sh.sendline("bash /mydata/mimir_snakemake_t2/experiment_coordinator/")

        if app_name == "wordpress":
            sh.sendline("bash /mydata/mimir_snakemake_t2/experiment_coordinator/install_scripts/install_selenium_dependencies.sh")
            line_rec = 'start'
            last_line = ''
            while line_rec != '':
                last_line = line_rec
                line_rec = sh.recvline(timeout=240)
                print("recieved line", line_rec)
            print("did selenium setup work??")

        time.sleep(170)
        sh.sendline('rm ' + experiment_sentinal_file)
        print "removing experimente sential file", sh.recvline(timeout=5)
        sh.sendline('minikube ssh')
        print "minikube sshing", sh.recvline(timeout=5)
        sh.sendline('docker pull nicolaka/netshoot')
        print "docker pulling", sh.recvline(timeout=5)
        sh.sendline('exit')
        print "minikube exiting", sh.recvline(timeout=5)
        time.sleep(170)

    else:
        pass

    print "----after minikube setup---"
    if not dont_pull_p:
        update_mimir_str = "cd /mydata/mimir_v2/; git pull"
        sh.sendline(update_mimir_str)

    # pwd_line = ''
    line_rec = 'something something'
    while line_rec != '':
        last_line = line_rec
        line_rec = sh.recvline(timeout=100)
        print("recieved line", line_rec)

    if app_name == 'hipster':
        #print "hipsterStore (microservice from google) doesn't have an actual run_experiment component defined"
        dwnload_skaffold_str = "curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/latest/skaffold-linux-amd64"
        chg_perm_skaffold_str = "chmod +x skaffold"
        install_skaffold_str = "sudo mv skaffold /usr/local/bin"
        sendline_and_wait_responses(sh, dwnload_skaffold_str, timeout=5)
        sendline_and_wait_responses(sh, chg_perm_skaffold_str, timeout=5)
        sendline_and_wait_responses(sh, install_skaffold_str, timeout=5)
    elif app_name == 'wordpress':
        sh.sendline("cd /mydata/mimir_v2/experiment_coordinator")
        deploy_str = 'python /mydata/mimir_v2/experiment_coordinator/wordpress_setup/scale_wordpress.py' + ' ' + str(7)
        sh.sendline(deploy_str)
        # pwd_line = ''
        line_rec = 'something something'
        while line_rec != '':
            last_line = line_rec
            line_rec = sh.recvline(timeout=100)
            print("recieved line", line_rec)

        time.sleep(120)

        ip, port = get_ip_and_port(app_name, sh)
        sh.sendline("exit")  # need to be a normal user when using selenium
        sh.sendline("cd /mydata/mimir_v2/experiment_coordinator/experimental_configs")

        setup_str = 'python /mydata/mimir_v2/experiment_coordinator/wordpress_setup/setup_wordpress.py ' + str(ip) + ' ' + \
            str(port) + ' \"hi\"'
        try:
            sh.sendline(setup_str)
        except:
            print "I legit do not know if sh tubes can through exceptions... let's find out..."
        # pwd_line = ''
        line_rec = 'something something'
        while line_rec != '':
            last_line = line_rec
            line_rec = sh.recvline(timeout=240)
            print("recieved line", line_rec)

        sh.sendline('sudo newgrp docker')
        sh.sendline('export MINIKUBE_HOME=/mydata/')
        sh.sendline('cd /mydata/mimir_v2/experiment_coordinator/')
        line_rec = 'something something'
        while line_rec != '':
            last_line = line_rec
            line_rec = sh.recvline(timeout=100)
            print("recieved line", line_rec)

    time.sleep(60)
    start_actual_experiment = 'python /mydata/mimir_v2/experiment_coordinator/run_experiment.py --exp_name ' + \
                              exp_name + ' --config_file ' + config_file_name
    if not physical_attacks_p:
        start_actual_experiment += ' --no_exfil'
    print "skip_app_setup",skip_app_setup
    if not skip_app_setup:
        start_actual_experiment += ' --prepare_app_p'


    create_experiment_sential_file = '; echo experimenet_done >> ' + experiment_sentinal_file
    start_actual_experiment += create_experiment_sential_file

    print "start_actual_experiment: ", start_actual_experiment
    sh.sendline('cd /mydata/mimir_v2/experiment_coordinator/')
    timeout = exp_length / 12.0
    sh.sendline(start_actual_experiment)
    #sh.stream()
    #sh.process([start_actual_experiment], cwd='/mydata/mimir_v2/experiment_coordinator/',executable='python').stream()
    line_rec = 'start'
    last_line = ''
    while line_rec != '':
        last_line = line_rec
        line_rec = sh.recvline(timeout=timeout)
        print("recieved line", line_rec)
    while line_rec != '':
        last_line = line_rec
        line_rec = sh.recvline(timeout=timeout)
        print("recieved line", line_rec)

    return s

if __name__ == "__main__":

    #################################
    #app_name = possible_apps[4] # wordpress
    app_name = possible_apps[1] # sockshop
    #app_name = possible_apps[5] # hipsterStore (google's example microservice)
    sock_config_file_name = '/mydata/mimir_v2/experiment_coordinator/experimental_configs/sockshop_exp_one.json'
    wp_config_file_name = '/mydata/mimir_v2/experiment_coordinator/experimental_configs/wordpress_exp_one.json'
    config_file_name = sock_config_file_name #wp_config_file_name
    use_cilium = False # note: if actually running an experiment, will probably want "False"
    physical_attacks_p = False

    #local_dir = '/Volumes/exM2/experimental_data/wordpress_info'  # '/Users/jseverin/Documents'
    #local_dir = '/Volumes/exM2/experimental_data/wordpress_info'
    local_dir = '/Volumes/exM2/experimental_data/sockshop_info'
    #exp_name = 'wordpress_fourteen_mark7_final'
    #exp_name = 'sockshop_thirteen_NOautoscale_mark1' #mark3 is good too
    #exp_name = 'sockshop_autoscaling_tests'
    exp_name = 'sockshop_exp_one_v2_noauto'
    #exp_name = 'wordpress_exp_one_v2_noauto'
    #mimir_1 = 'c220g5-111314.wisc.cloudlab.us'  #'c240g5-110119.wisc.cloudlab.us'
    mimir_1 = 'c240g5-110123.wisc.cloudlab.us'
    mimir_2 = 'c220g5-111211.wisc.cloudlab.us' #
    cloudlab_server_ip = mimir_2  # note: remove the username@ from the beggining
    exp_length = 10800  # 10800 #7200 # in seconds
    #################################

    parser = argparse.ArgumentParser(description='Handles e2e setup, running, and extract of microservice traffic experiments')

    parser.add_argument('--straight_to_experiment', dest='skip_setup_p', action='store_true',
                        default=False,
                        help='only do the running the experiment-- no minikube setup')

    parser.add_argument('--skip_app_setup', dest='skip_app_setup_p', action='store_true',
                        default=False,
                        help='only do the running the experiment-- no minikube setup, application deployment, or \
                        application loading')

    parser.add_argument('--config_json',dest="config_json", default='None')


    parser.add_argument('--dont_pull', dest='dont_pull_p', action='store_true',
                        default=False,
                        help='do NOT pull from the github repo (default is to pull)')

    args = parser.parse_args()

    if args.config_json != 'None':
        with open(args.config_json) as f:
            config_params = json.load(f)

            app_name = config_params["app_name"]
            exp_name = config_params["exp_name"]
            config_file_name = config_params["config_file_name"]
            local_dir = config_params["local_dir"]
            cloudlab_server_ip = config_params["cloudlab_server_ip"]
            exp_length = config_params["exp_length"]
            physical_attacks_p = config_params["physical_attacks_p"]

    remote_dir = '/mydata/mimir_v2/experiment_coordinator/experimental_data/' + exp_name  # TODO
    s = run_experiment(app_name, config_file_name, exp_name, args.skip_setup_p,
                       use_cilium, physical_attacks_p, args.skip_app_setup_p, args.dont_pull_p)
    retrieve_results(s)
