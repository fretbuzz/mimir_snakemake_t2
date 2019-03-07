## the purpose of this file is to move the results of the testbed
## to the local machine, where it will optionally start processing them
# TODO: will need to fill in these TODOs and then test (maybe integrate
# with auto-running of the analysis capabilities, but will require me to re-work some of the testbed)

import argparse
import pwnlib.tubes.ssh
from pwn import *
import time

cloudlab_private_key = '/Users/jseverin/Dropbox/cloudlab.pem'
local_dir = '/Volumes/exM2/experimental_data/wordpress_info' #'/Users/jseverin/Documents'  # TODO
sentinal_file = '/mydata/all_done.txt'
exp_name = 'wordpress_thirteen_t5_autoscaling'  # TODO
mimir_1 = 'c240g5-110231.wisc.cloudlab.us' #'c240g5-110207.wisc.cloudlab.us' #'c240g5-110217.wisc.cloudlab.us' #'c240g5-110119.wisc.cloudlab.us' #'c240g5-110119.wisc.cloudlab.us'
#mimir_2 = 'c240g5-110105.wisc.cloudlab.us'
cloudlab_server_ip = mimir_1 #note: remove the username@ from the beggining
exp_length = 10800 #7200 # in seconds
remote_dir = '/mydata/mimir_snakemake_t2/experiment_coordinator/experimental_data/' + exp_name  # TODO
possible_apps = ['drupal', 'sockshop', 'gitlab', 'eShop', 'wordpress']
experiment_sentinal_file = '/mydata/mimir_snakemake_t2/experiment_coordinator/experiment_done.txt'

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
    else:
        pass  # TODO

    line_rec = 'start'
    last_line = ''
    while line_rec != '':
        last_line = line_rec
        line_rec = sh.recvline(timeout=100)
        print("recieved line", line_rec)
    print("--end minikube_front-end port ---")

    # kubernetes_setup_functions.wait_until_pods_done(namespace)
    minikube_ip, front_facing_port = last_line.split(' ')[-1].split('/')[-1].rstrip().split(':')
    print "minikube_ip", minikube_ip, "front_facing_port", front_facing_port
    return minikube_ip, front_facing_port

def run_experiment(app_name, config_file_name, exp_name, skip_setup_p, autoscale_p):
    #start_minikube_p = False
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
    # Send the process arguments
    sh.sendline('ls -la')
    # Receive output from the executed command
    line_rec = 'start'
    while line_rec != '':
        line_rec = sh.recvline(timeout=5)
        print("recieved line", line_rec)
    print("--end ls -la ---")

    sh.sendline('pwd')
    # Receive output from the executed command
    line_rec = 'start'
    while line_rec != '':
        line_rec = sh.recvline(timeout=5)
        print("recieved line", line_rec)
    print("--end pwd ---")

    sh.sendline('sudo newgrp docker')
    sh.sendline('export MINIKUBE_HOME=/mydata/')

    if not skip_setup_p:
        sh.sendline('minikube stop')
        line_rec = 'start'
        while line_rec != '':
            line_rec = sh.recvline(timeout=5)
            if 'Please enter your response' in line_rec:
                sh.sendline('n')
            print("recieved line", line_rec)
        print("--end minikube-stop ---")

        sh.sendline('minikube delete')
        line_rec = 'start'
        while line_rec != '':
            line_rec = sh.recvline(timeout=5)
            if 'Please enter your response' in line_rec:
                sh.sendline('n')
            print("recieved line", line_rec)



        print("--end minikube delete ---")

        clone_mimir_str = "git clone https://github.com/fretbuzz/mimir_snakemake_t2"
        sh.sendline(clone_mimir_str)

        while line_rec != '':
            line_rec = sh.recvline(timeout=5)
            if 'Please enter your response' in line_rec:
                sh.sendline('n')
            print("recieved line", line_rec)

        '''
        sh.sendline('minikube ip')
        # Receive output from the executed command
        line_rec = 'start'
        while line_rec != '':
            line_rec = sh.recvline(timeout=5)
            print("recieved line", line_rec)
        print("--end minikube_ip ---")
        '''
        #print last_line.split(' ')
        #print last_line.split(' ')[-1]
        #print last_line.split(' ')[-1].split('/')
        #print last_line.split(' ')[-1].split('/')[-1]
        #print last_line.split(' ')[-1].split('/')[-1].rstrip().split(':')

        #minikube_ip, front_facing_port = None, None
        #print "minikube_ip", minikube_ip, "front_facing_port",front_facing_port

        #sh.sendline('bash /local/repository/run_experiment.sh ' + app_name)
        if app_name == 'wordpress':
            cpu_threshold = 80
        else:
            cpu_threshold = None # b/c doesn't matter

        if autoscale_p:
            sh.sendline('bash /mydata/mimir_snakemake_t2/experiment_coordinator/former_profile/run_experiment.sh ' +
                        app_name + ' ' + str(autoscale_p) + ' ' + str(cpu_threshold))
        else:
            sh.sendline('bash /mydata/mimir_snakemake_t2/experiment_coordinator/former_profile/run_experiment.sh ' + \
                        app_name)

        line_rec = 'start'
        last_line = ''
        while line_rec != '':
            last_line = line_rec
            line_rec = sh.recvline(timeout=40)
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

        '''
        def run_experiment(app_name, config_file_name, exp_name):
        ## delete this part...
        s = pwnlib.tubes.ssh.ssh(host=cloudlab_server_ip,
            keyfile=cloudlab_private_key,
            user='jsev')
        sh = s.run('sh') # Create an initial process
        sh.sendline('sudo newgrp docker')
        sh.sendline('export MINIKUBE_HOME=/mydata/')
        ## delete this part...
        #'''

        minikube_ip, front_facing_port = get_ip_and_port(app_name, sh)

        if app_name == 'wordpress':
            # step 2: setup wordpress (must be done now rather than later in run_experiment like sockshop)
            # todo: it's not local you know... it doesn't make any sense for it be called locally...
            #wp_api_pwd = setup_wordpress.main(minikube_ip, front_facing_port, 'hi')
            sh.sendline("exit") # need to be a normal user when using selenium
            sh.sendline("cd /mydata/mimir_snakemake_t2/experiment_coordinator/experimental_configs")

            prepare_wp_str = "python /mydata/mimir_snakemake_t2/experiment_coordinator/experimental_configs/setup_wordpress.py " + \
                    minikube_ip + " " + front_facing_port + " " + "hi"
            print "prepare_wp_str",prepare_wp_str
            sh.sendline(prepare_wp_str)

            #pwd_line = ''
            line_rec = 'something something'
            while line_rec != '':
                last_line = line_rec
                line_rec = sh.recvline(timeout=360)
                print("recieved line", line_rec)
                #if 'pwd' in line_rec:
                #    pwd_line = line_rec

            # need to get back to the other group now
            sh.sendline('sudo newgrp docker')
            sh.sendline('export MINIKUBE_HOME=/mydata/')
            sh.sendline('cd /mydata/mimir_snakemake_t2/experiment_coordinator/')
            #wp_api_pwd = pwd_line.split("pwd")[1].lstrip()
            #print "wp_api_pwd",wp_api_pwd
            # this is kinda silly... just put the pwd in a file and have the background function read it...
            #change_cwd = 'cd /mydata/mimir_snakemake_t2/experiment_coordinator/load_generators'
            #change_pwd_cmd = "sed -e 's/app_pass.*$/app_pass = \"" + wp_api_pwd +  "\"/g' wordpress_background.py"
            #sh.sendline(change_cwd)
            #sh.sendline(change_pwd_cmd)
            #exit(2)


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
        minikube_ip, front_facing_port = get_ip_and_port(app_name, sh)

    # pwd_line = ''
    line_rec = 'something something'
    while line_rec != '':
        last_line = line_rec
        line_rec = sh.recvline(timeout=100)
        print("recieved line", line_rec)

    start_actual_experiment = 'python /mydata/mimir_snakemake_t2/experiment_coordinator/run_experiment.py --exp_name ' +\
                              exp_name  + ' --config_file ' + config_file_name + ' --prepare_app_p --port ' + \
                              front_facing_port + ' --ip ' + minikube_ip + ' --no_exfil'

    #start_actual_experiment = 'python /mydata/mimir_snakemake_t2/experiment_coordinator/run_experiment.py --exp_name ' +\
    #                          exp_name  + ' --config_file ' + config_file_name + ' --port ' + \
    #                          front_facing_port + ' --ip ' + minikube_ip + ' --no_exfil'

    create_experiment_sential_file = '; echo experimenet_done >> ' + experiment_sentinal_file
    start_actual_experiment += create_experiment_sential_file

    print "start_actual_experiment: ", start_actual_experiment
    sh.sendline('cd /mydata/mimir_snakemake_t2/experiment_coordinator/')
    sh.sendline(start_actual_experiment)
    timeout = exp_length / 12.0
    #sh.stream()
    #sh.process([start_actual_experiment], cwd='/mydata/mimir_snakemake_t2/experiment_coordinator/',executable='python').stream()
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
    app_name = possible_apps[4]
    sock_config_file_name = '/mydata/mimir_snakemake_t2/experiment_coordinator/experimental_configs/sockshop_thirteen'
    wp_config_file_name = '/mydata/mimir_snakemake_t2/experiment_coordinator/experimental_configs/wordpress_thirteen'


    parser = argparse.ArgumentParser(description='Handles e2e setup, running, and extract of microservice traffic experiments')

    parser.add_argument('--straight_to_experiment', dest='skip_setup_p', action='store_true',
                        default=False,
                        help='only do the running the experiment-- no minikube setup, application deployment, or \
                        application loading')
    parser.add_argument('--autoscale_p', dest='autoscale_p', action='store_true',
                       default=False,
                       help='enable autoscaling')

    args = parser.parse_args()

    # NOTE: remember: dont put the .json in the filename!! ^^^
    s = run_experiment(app_name, wp_config_file_name, exp_name, args.skip_setup_p, args.autoscale_p)
    retrieve_results(s)
