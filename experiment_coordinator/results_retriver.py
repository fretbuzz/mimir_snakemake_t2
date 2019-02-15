## the purpose of this file is to move the results of the testbed
## to the local machine, where it will optionally start processing them
# TODO: will need to fill in these TODOs and then test (maybe integrate
# with auto-running of the analysis capabilities, but will require me to re-work some of the testbed)
from pwn import *
import pwnlib.tubes.ssh
import time

cloudlab_private_key = '/Users/jseverin/Dropbox/cloudlab.pem'
local_dir = '/Users/jseverin/Documents'  # TODO
sentinal_file = '/mydata/all_done.txt'
cloudlab_server_ip = 'c240g5-110119.wisc.cloudlab.us' #note: remove the username@ from the beggining
remote_dir = '/mydata/mimir_snakemake_t2/results'  # TODO
possible_apps = ['drupal', 'sockshop', 'gitlab', 'eShop', 'wordpress']

def retrieve_results(s):
    print('hello')

    # check every five minutes until the sentinal file is present
    while True:
        # this is a special 'done' file used to indicate that
        # the experiment is finished.
        if s.download_data(sentinal_file) == 'done':
            break
        time.sleep(300)

    s.download_dir(remote=remote_dir, local=local_dir)

def run_experiment(app_name, config_file_name, exp_name):
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
    sh.sendline('minikube ip')
    # Receive output from the executed command
    line_rec = 'start'
    while line_rec != '':
        line_rec = sh.recvline(timeout=5)
        print("recieved line", line_rec)
    print("--end minikube_ip ---")

    #print last_line.split(' ')
    #print last_line.split(' ')[-1]
    #print last_line.split(' ')[-1].split('/')
    #print last_line.split(' ')[-1].split('/')[-1]
    #print last_line.split(' ')[-1].split('/')[-1].rstrip().split(':')

    minikube_ip, front_facing_port = None, None
    print "minikube_ip", minikube_ip, "front_facing_port",front_facing_port

    sh.sendline('bash /local/repository/run_experiment.sh ' + app_name)


    line_rec = 'start'
    last_line = ''
    while line_rec != '':
        last_line = line_rec
        line_rec = sh.recvline(timeout=5)
        print("recieved line", line_rec)
    print("did run_experiment work???")


    sentinal_file_setup = '/mydata/done_with_setup.txt'
    while True:
        # this is a special 'done' file used to indicate that
        # the experiment is finished.
        print s.download_data(sentinal_file_setup)
        if s.download_data(sentinal_file_setup) == 'done':
            break
        time.sleep(20)


    if app_name == 'sockshop':
        sh.sendline('minikube service front-end  --url --namespace="sock-shop"')
    else:
        pass #TODO

    line_rec = 'start'
    last_line = ''
    while line_rec != '':
        last_line = line_rec
        line_rec = sh.recvline(timeout=5)
        print("recieved line", line_rec)
    print("--end minikube_front-end port ---")


    minikube_ip, front_facing_port = last_line.split(' ')[-1].split('/')[-1].rstrip().split(':')
    print "minikube_ip", minikube_ip, "front_facing_port",front_facing_port

    start_actual_experiment = 'python /mydata/mimir_snakemake_t2/experiment_coordinator/run_experiment.py --exp_name ' +\
                              exp_name  + ' --config_file ' + config_file_name + ' --prepare_app_p --port ' + \
                              front_facing_port + ' -ip ' + minikube_ip + ' --no_exfil'

    sh.sendline(start_actual_experiment)
    line_rec = 'start'
    last_line = ''
    while line_rec != '':
        last_line = line_rec
        line_rec = sh.recvline(timeout=5)
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

    return s

if __name__ == "__main__":
    app_name = possible_apps[1]
    config_file_name = '/mydata/mimir_snakemake_t2/experiment_coordinator/experimental_configs/sockshop_thirteen.json' # TODO
    exp_name = 'completey_crazy_test' # TODO
    s = run_experiment(app_name, config_file_name, exp_name)
    #retrieve_results(s)