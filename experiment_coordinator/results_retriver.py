## the purpose of this file is to move the results of the testbed
## to the local machine, where it will optionally start processing them
# TODO: will need to fill in these TODOs and then test (maybe integrate
# with auto-running of the analysis capabilities, but will require me to re-work some of the testbed)

import pwnlib.tubes.ssh
from pwn import *

import experimental_configs.setup_wordpress as setup_wordpress

cloudlab_private_key = '/Users/jseverin/Dropbox/cloudlab.pem'
local_dir = '/Users/jseverin/Documents'  # TODO
sentinal_file = '/mydata/all_done.txt'
exp_name = 'wordpress_auto_test'  # TODO
mimir_1 = 'c240g5-110105.wisc.cloudlab.us' #'c240g5-110119.wisc.cloudlab.us' #'c240g5-110119.wisc.cloudlab.us'
#mimir_2 = 'c240g5-110105.wisc.cloudlab.us'
cloudlab_server_ip = mimir_1 #note: remove the username@ from the beggining
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

def run_experiment(app_name, config_file_name, exp_name):
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


    #if start_minikube_p:
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

    sh.sendline('bash /local/repository/run_experiment.sh ' + app_name)


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


    if app_name == 'sockshop':
        sh.sendline('minikube service front-end  --url --namespace="sock-shop"')
        namespace = 'sock-shop'
    elif app_name == 'wordpress':
        # step 1: get the appropriate ip / port (like above -- need for next step)
        sh.sendline('minikube service wwwppp-wordpress  --url')
    else:
        pass #TODO


    line_rec = 'start'
    last_line = ''
    while line_rec != '':
        last_line = line_rec
        line_rec = sh.recvline(timeout=100)
        print("recieved line", line_rec)
    print("--end minikube_front-end port ---")

    #kubernetes_setup_functions.wait_until_pods_done(namespace)
    minikube_ip, front_facing_port = last_line.split(' ')[-1].split('/')[-1].rstrip().split(':')
    print "minikube_ip", minikube_ip, "front_facing_port",front_facing_port

    if app_name == 'wordpress':
        # step 2: setup wordpress (must be done now rather than later in run_experiment like sockshop)
        # todo: it's not local you know... it doesn't make any sense for it be called locally...
        #wp_api_pwd = setup_wordpress.main(minikube_ip, front_facing_port, 'hi')
        sh.sendline("python /mydata/mimir_snakemake_t2/experiment_coordinator/experimental_configs/setup_wordpress.py " + \
                minikube_ip + " " + front_facing_port + " " + "hi")

        #pwd_line = ''
        while line_rec != '':
            last_line = line_rec
            line_rec = sh.recvline(timeout=100)
            print("recieved line", line_rec)
            #if 'pwd' in line_rec:
            #    pwd_line = line_rec

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
    #sh.stream()
    #sh.process([start_actual_experiment], cwd='/mydata/mimir_snakemake_t2/experiment_coordinator/',executable='python').stream()
    line_rec = 'start'
    last_line = ''
    while line_rec != '':
        last_line = line_rec
        line_rec = sh.recvline(timeout=200)
        print("recieved line", line_rec)
    while line_rec != '':
        last_line = line_rec
        line_rec = sh.recvline(timeout=200)
        print("recieved line", line_rec)

    '''
    okay, we have a bit of downtime. We should start planning our next move... which is to do what??
    just activate the bash script (which should already be there...)
    
    okay, what do we actually need to do?? well, we need to focus-down bigtime, my homie. This means that the 
    path forward must go through setting up sockshop automatically as soon as possible. then we can also add in
    retrieval + wordpress components (probably plug and play) after that.
    
    getting sockshop setup is:
        1. finishing/fixing run_experiment.sh
            well just hitting run_experiment.sh sockshop seems to work just dandy... however, it doesn't actally run the
            experiment... we need to
                (a) get the params
                    (i) which params?? : 
                        EXP_NAME = None # TODO -- will pass.
                        CONFIG_FILE = None # TODO -- must put into the experimental_confis directory (beforehand)
                        PORT_NUMBER = None # TODO -- need to determine -- extracted
                        VM_IP = None # TODO -- need to determine
                (b) call the func
        2. having this function start/stop things appropriately...
        
    '''
    '''
    # todo: problem is that stream never stops. need to put a sentinal file to identify when that portion is done.
    # also, there's a timeout problem with the thing that starts tcpdump. There was also some problems w/ the function
    # to retrieve the pcap from minikube, but I think it won't be a problem as long as the initial function works finee.
    # and of course I haven't tested the end part at all... the part where I bring the results directory back to the local
    # machine (but prelim analysis indicates that the directories are not appearing where'd you'd expect them...)
    # and then just transform the pcap with the stuff to a recipe and hopefully it'll more-or-less run... and I can get
    # results again... since I don't have them at the moment...
    '''

    return s

if __name__ == "__main__":
    app_name = possible_apps[4]
    sock_config_file_name = '/mydata/mimir_snakemake_t2/experiment_coordinator/experimental_configs/sockshop_thirteen'
    wp_config_file_name = '/mydata/mimir_snakemake_t2/experiment_coordinator/experimental_configs/wordpress_twelve.json'
    # NOTE: remember: dont put the .json in the filename!! ^^^
    s = run_experiment(app_name, wp_config_file_name, exp_name)
    retrieve_results(s)
