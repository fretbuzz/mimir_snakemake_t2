'''
The purpose of this file is that it is a lot of work to manually coordinate the processing of all
the individual experiment instance data. This function will be able to run the experiments and the
analysis component, with a much more concise configuration format. It'll let me handle all the eval
work I am going to do without exhausting me from handling too much stuff by hand.
'''

import json
import multiprocessing
import pwnlib.tubes.ssh
from pwn import *
import time
import copy
import ast
from experiment_coordinator.run_exp_on_cloudlab import run_experiment, retrieve_results

# this is a wrapper around run_exp_on_cloudlab.py
def run_new_experiment(template, template_changes, cloudlab_ip, flags, user, private_key, exp_name, local_dir,
                       remote_experimental_data_dir, experiment_sentinal_file, remote_experimental_config_folder):
    ## NOTE: eventually I'll use the flags param to skip duplicate setup... but not for now...

    # (0) establish connection to cloudlab server
    s = None
    while s == None:
        try:
            s = pwnlib.tubes.ssh.ssh(host=cloudlab_ip,
                keyfile=private_key,
                user=user)
        except:
            time.sleep(60)

    ## (2) modify template appropriately and upload to cloudlab
    # modify template appropriately....
    for template_change_name, template_change_value in template_changes.iteritems():
        #print "template_change_name", template_change_name
        try:
            template_change_name = ast.literal_eval(template_change_name)
        except:
            pass

        ## indexing into nested dictionary is difficult... however... we can just hardcode "two deep"
        ## interactions because we know that that must be the only kind that occurs in this instance...
        #print "template",template
        if type(template_change_name) == tuple:
            template[template_change_name[0]][template_change_name[1]] = template_change_value
        else:
            template[template_change_name] = template_change_value

    # upload to cloudlab...
    filename = exp_name + '_exp.json'
    modified_template_filename = './e2e_eval_configs/' + filename
    with open(modified_template_filename, 'w') as f:
        f.write(json.dumps(template))

    remote = remote_experimental_config_folder + filename ## destination on remote
    s.put(file_or_directory=modified_template_filename, remote=remote)

    ## (3) call the relevant components of run_exp_on_cloudlab.py
    ### what the heck does this mean? I guess it means that the params are all okay and all
    ### the data is still in the same place....
    app_name = template['application_name']
    config_file_name = remote
    try:
        physical_attacks_p = template["exfiltration_info"]["physical_attacks"]
    except:
        physical_attacks_p = False

    exp_length = template["experiment_length_sec"]

    remote_dir = remote_experimental_data_dir + exp_name

    s = run_experiment(app_name, config_file_name, exp_name, skip_setup_p=False, use_cilium=False,
                   physical_attacks_p=physical_attacks_p, skip_app_setup=False,dont_pull_p=False,
                   exp_length = exp_length, user = user, cloudlab_private_key=private_key,
                   cloudlab_server_ip = cloudlab_ip, experiment_sentinal_file = experiment_sentinal_file)

    retrieve_results(s, experiment_sentinal_file, remote_dir, local_dir)


def perform_eval_work(cloudlab_exps_file, cloudlab_exps_dir):
    remote_experimental_data_dir = '/mydata/mimir_v2/experiment_coordinator/experimental_data/'
    experiment_sentinal_file = '/mydata/mimir_v2/experiment_coordinator/experiment_done.txt'
    #remote_experimental_config_folder = "/mydata/mimir_v2/experiment_coordinator/experimental_configs/"
    remote_experimental_config_folder = "/users/jsev/"
    cloudlab_exps_file = cloudlab_exps_dir + cloudlab_exps_file

    ## complication: (a) and (b) must be done simulataneously
    ## (a) run stuff on cloudlab
    #### current step:
    ####    (1) take the cloudlab_exps.json and assign experiments to cloudlab instances
    ####    (2) modify the template accordingly and ssh it into the cloudlab instance
    ####    (3) wait for it to run (simple call to run_exp_on_cloudlab.py)
    with open(cloudlab_exps_file, 'r') as f:
        cloudlab_config = json.loads(f.read())

    minikube_ips = cloudlab_config["minikube_ips_to_lifetime"].keys()
    experiments_running = True

    experiment_name_to_status = {}
    for exp_name in cloudlab_config["name_to_diff_params"].keys():
        experiment_name_to_status[exp_name] = 0 # 0 indicates that this experiment has NOT been run yet.

    cloudlab_instances_to_status = {}
    for minikube_ip in minikube_ips:
        cloudlab_instances_to_status[minikube_ip] = 0 # 0 indicates NOT running an experiment
    number_cloudlab_instances = len(cloudlab_instances_to_status.keys())
    experiment_name_to_assigned_ip = {}

    while experiments_running:
        jobs = []
        for ip,status in cloudlab_instances_to_status.iteritems():
            if status == 0:
                with open(cloudlab_exps_dir + cloudlab_config["template"], 'r') as f:
                    template = json.loads(f.read())
                # now choose the next experiment
                exp_name = None
                template_changes = None
                for exp_name in cloudlab_config["name_to_diff_params"].keys():
                    if experiment_name_to_status[exp_name] == 0:
                        # this is the experiment that we shall run now.
                        template_changes = cloudlab_config["name_to_diff_params"][exp_name]
                        experiment_name_to_assigned_ip[exp_name] = ip
                        break
                ## now that we have the next experiment selected, we must run it
                template_copy = copy.deepcopy(template)
                user = cloudlab_config["user"]
                private_key = cloudlab_config["private_key"]
                local_dir = cloudlab_config["local_dir"] + exp_name + '/'
                cur_args = [template_copy, template_changes, ip, None, user, private_key, exp_name, local_dir,
                            remote_experimental_data_dir, experiment_sentinal_file, remote_experimental_config_folder]
                service = multiprocessing.Process(name=exp_name, target=run_new_experiment, args=cur_args)
                service.start()
                jobs.append(service)
                # and now change the statuses of everything appropriately...
                experiment_name_to_status[exp_name] = 1
                cloudlab_instances_to_status[ip] = 1
        # okay, if we made it out here then all the cloudlab instances must have assigned work.
        # we must wait for a job to complete now...
        while(len(jobs) == number_cloudlab_instances):
            jobs_to_remove = []
            for job in jobs:
                if not job.is_alive():
                    jobs_to_remove.append(job)
                    ## we need to modify the cloudlab_instances_to_status appropriately
                    ## and note that we have a new experiment ready for processing by the latter part
                    ## of the system...
                    finished_exp_name = job.name
                    freed_ip = experiment_name_to_assigned_ip[finished_exp_name]
                    cloudlab_instances_to_status[freed_ip] = 0
            if jobs_to_remove == []:
                time.sleep(60) # wait a min and check again
            else:
                for job_to_remove in jobs_to_remove:
                    jobs.remove(job_to_remove)

    ## TODO: okay, let's take stock of the current situation... the current code is the basic outline
    ## for assign experiments to cloudlab instances and looping through it... this is NOT a large
    ## advantage over just straight-up using the command line... howevever, the killer features are next:
    #### (1) finish run_new_experiment
        ## (1) modify template appropriately
        ## (2) upload to cloudlab
        ## (3) make sure that run_exp_on_cloudlab.py still knows what's up.
    #### (2) start LOCAL experiments... perhaps I can abstract the existing logic into a function???
        ## cause it'll be essentially equivalent for the multiprocessing part... however
            ## (1) ALSO need to autogenerate analysis json
            ## (2) ALSO need to collect the multi-process output
        ## (Note: probably no point in modifying that code that I wrote b/c I don't think it really does
        ## the same thing at atll...)

    ## (b) create analysis json + run stuff locally... this'll also require the same kinda multiprocess logic as above...

    ## create graphs

    pass

if __name__=="__main__":
    cloudlab_exps_file = 'cloudlab_exps.json'
    cloudlab_exps_dir = './e2e_eval_configs/'
    perform_eval_work(cloudlab_exps_file, cloudlab_exps_dir)