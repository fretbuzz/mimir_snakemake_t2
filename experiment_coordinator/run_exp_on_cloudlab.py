## the purpose of this file is to move the results of the testbed
## to the local machine, where it will optionally start processing them

import argparse
import pwnlib.tubes.ssh
from pwn import *
import time
import json

def retrieve_results(s, experiment_sentinal_file, remote_dir, local_dir):
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

def run_experiment(app_name, local_path_to_exp_config, exp_name, skip_setup_p, use_cilium, physical_attacks_p, skip_app_setup,
                   pull_from_github, exp_length, user, cloudlab_private_key, cloudlab_server_ip,
                   experiment_sentinal_file):
    exp_length = max(10800, exp_length) # min is 10800 b/c it pauses that long during setup...

    s = None
    while s == None:
        try:
            s = pwnlib.tubes.ssh.ssh(host=cloudlab_server_ip,
                keyfile=cloudlab_private_key,
                user=user)
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

        # useful when cycling wordpress deployments...
        sendline_and_wait_responses(sh, 'helm del --purge my-release', timeout=5)
        sendline_and_wait_responses(sh, 'helm del --purge wwwppp', timeout=5)

        clone_mimir_str = "cd /mydata/; git clone https://github.com/fretbuzz/mimir_v2"
        sendline_and_wait_responses(sh, clone_mimir_str, timeout=5)

        sendline_and_wait_responses(sh, "cd ./mimir_v2/", timeout=5)

        if pull_from_github:
            update_mimir_str = "git pull"
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
        sh.sendline("docker image  prune -a")
        line_rec = 'start'
        while line_rec != '':
            line_rec = sh.recvline(timeout=15)
            if 'Please enter your response' in line_rec:
                sh.sendline('n')
            elif "Are you sure you want to continue" in line_rec:
                sh.sendline('y')
            print("recieved line", line_rec)

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
    sendline_and_wait_responses(sh, "cd /mydata/mimir_v2/", timeout=5)
    if pull_from_github:
        update_mimir_str = "git pull"
        sh.sendline(update_mimir_str)
    else:
        pass

    ## move the config_files w/o relying on the push-to-github workaround
    ## two parts:
    # part 1: where is the config file locally?
    # part 2: put it into the correct remote location...
    with open(local_path_to_exp_config, 'r') as f:
        test = f.read()
    print "test",test

    remote_path_to_experimental_config = local_path_to_exp_config.split('/')[-1]
    print "remote_path_to_experimental_config",remote_path_to_experimental_config
    s.put(file_or_directory=local_path_to_exp_config, remote=remote_path_to_experimental_config)

    # pwd_line = ''
    line_rec = 'something something'
    while line_rec != '':
        last_line = line_rec
        line_rec = sh.recvline(timeout=100)
        print("recieved line", line_rec)

    if not skip_app_setup:
        if app_name == 'hipsterStore':
            print "hipsterStore_SPECIFIC"
            #print "hipsterStore (microservice from google) doesn't have an actual run_experiment component defined"
            dwnload_skaffold_str = "curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/latest/skaffold-linux-amd64"
            chg_perm_skaffold_str = "chmod +x skaffold"
            install_skaffold_str = "sudo mv skaffold /usr/local/bin"
            sendline_and_wait_responses(sh, dwnload_skaffold_str, timeout=5)
            sendline_and_wait_responses(sh, chg_perm_skaffold_str, timeout=5)
            sendline_and_wait_responses(sh, install_skaffold_str, timeout=5)


            sendline_and_wait_responses(sh, "cd /mydata/mimir_v2/experiment_coordinator", timeout=5)
            clone_repo = "git clone https://github.com/GoogleCloudPlatform/microservices-demo.git"
            sendline_and_wait_responses(sh, clone_repo, timeout=10)
            sendline_and_wait_responses(sh, "cd ./microservices-demo", timeout=5)
            sendline_and_wait_responses(sh, "skaffold run", timeout=120)
            sendline_and_wait_responses(sh, "cd ..", timeout=10)

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

            ## NOTE: to go multiple times, might need to use sudo chown -R jsev:dna-PG0 geckodriver.log
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
                line_rec = sh.recvline(timeout=360) # it takes 300 sec to timeout at the end, so let's be on th
                print("recieved line", line_rec)

            sendline_and_wait_responses(sh, 'sudo newgrp docker', timeout=10)
            sh.sendline('export MINIKUBE_HOME=/mydata/')
            sendline_and_wait_responses(sh, 'cd /mydata/mimir_v2/experiment_coordinator/', timeout=100)

    time.sleep(60)
    start_actual_experiment = 'python /mydata/mimir_v2/experiment_coordinator/run_experiment.py --exp_name ' + \
                              exp_name + ' --config_file ~/' + remote_path_to_experimental_config
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

    parser = argparse.ArgumentParser(description='Handles e2e setup, running, and extract of microservice traffic experiments')

    parser.add_argument('--straight_to_experiment', dest='skip_setup_p', action='store_true',
                        default=False,
                        help='only do the running the experiment-- no minikube setup')

    parser.add_argument('--skip_app_setup', dest='skip_app_setup_p', action='store_true',
                        default=False,
                        help='only do the running the experiment-- no minikube setup, application deployment, or \
                        application loading')

    parser.add_argument('--config_json',dest="config_json", default='None')

    parser.add_argument('--pull_github', dest='pull_github', action='store_true',
                        default=False,
                        help='do NOT pull from the github repo (default is to pull)')

    args = parser.parse_args()

    if args.config_json == 'None':
        print "please give a config file"
        exit(344)

    with open(args.config_json) as f:
        config_params = json.load(f)

        app_name = config_params["app_name"]
        exp_name = config_params["exp_name"]
        local_path_to_exp_config = config_params["local_path_to_exp_config"]
        local_dir = config_params["local_dir"]
        cloudlab_server_ip = config_params["cloudlab_server_ip"]
        exp_length = config_params["exp_length"]
        physical_attacks_p = config_params["physical_attacks_p"]
        cloudlab_private_key = config_params["cloudlab_private_key"]
        user = config_params["user"]

    use_cilium = False

    remote_dir = '/mydata/mimir_v2/experiment_coordinator/experimental_data/' + exp_name
    experiment_sentinal_file = '/mydata/mimir_v2/experiment_coordinator/experiment_done.txt'
    sentinal_file = '/mydata/all_done.txt'

    s = run_experiment(app_name, local_path_to_exp_config, exp_name, args.skip_setup_p,
                       use_cilium, physical_attacks_p, args.skip_app_setup_p, args.pull_github,
                       exp_length, user, cloudlab_private_key, cloudlab_server_ip,
                       experiment_sentinal_file)

    retrieve_results(s, experiment_sentinal_file, remote_dir, local_dir)
