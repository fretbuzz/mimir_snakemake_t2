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
from analysis_pipeline.mimir import run_analysis
from analysis_pipeline.multi_experiment_looper import run_looper

# this is a wrapper around run_exp_on_cloudlab.py
def run_new_experiment(template, template_changes, cloudlab_ip, flags, user, private_key, exp_name, local_dir,
                       remote_experimental_data_dir, experiment_sentinal_file, remote_experimental_config_folder,
                       e2e_eval_configs_dir):
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
    modified_template_filename = e2e_eval_configs_dir + filename
    with open(modified_template_filename, 'w') as f:
        json.dump(template, f, indent=2)
        #f.write(json.dumps(template))

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

    skip_app_setup = False
    if flags:
        if 'skip_app_setup' in flags and flags['skip_app_setup']:
            skip_app_setup = True

    s = run_experiment(app_name, config_file_name, exp_name, skip_setup_p=False, use_cilium=False,
                   physical_attacks_p=physical_attacks_p, skip_app_setup=skip_app_setup, dont_pull_p=False,
                   exp_length = exp_length, user = user, cloudlab_private_key=private_key,
                   cloudlab_server_ip = cloudlab_ip, experiment_sentinal_file = experiment_sentinal_file)

    retrieve_results(s, experiment_sentinal_file, remote_dir, local_dir)

def create_analysis_json_from_template(template, exp_name, exp_config_file):
    template["cur_experiment_name"] = exp_name
    template["exp_config_file"] = exp_config_file
    return template

def perform_eval_work(cloudlab_exps_file, cloudlab_exps_dir, analysis_exp_file, skip_app_setup=False):
    remote_experimental_data_dir = '/mydata/mimir_v2/experiment_coordinator/experimental_data/'
    experiment_sentinal_file = '/mydata/mimir_v2/experiment_coordinator/experiment_done.txt'
    remote_experimental_config_folder = "/users/jsev/"
    e2e_eval_configs_dir = cloudlab_exps_dir

    train_exp = None
    eval_exps = []

    cloudlab_exps_file = cloudlab_exps_dir + cloudlab_exps_file
    ## complication: (a) and (b) must be done simulataneously
    ## (a) run stuff on cloudlab
    #### current step:
    ####    (1) take the cloudlab_exps.json and assign experiments to cloudlab instances
    ####    (2) modify the template accordingly and ssh it into the cloudlab instance
    ####    (3) wait for it to run (simple call to run_exp_on_cloudlab.py)
    with open(cloudlab_exps_file, 'r') as f:
        cloudlab_config = json.loads(f.read())

    with open(cloudlab_exps_dir + analysis_exp_file, 'r') as f:
        analysis_config = json.loads(f.read())

    cloudlab_ips = cloudlab_config["minikube_ips_to_lifetime"].keys()
    experiments_running = True

    exp_name_to_localdir = {}
    experiment_name_to_status = {}
    for exp_name in cloudlab_config["name_to_diff_params"].keys():
        experiment_name_to_status[exp_name] = 0 # 0 indicates that this experiment has NOT been run yet.

    cloudlab_instances_to_status = {}
    for cloudlab_ip in cloudlab_ips:
        cloudlab_instances_to_status[cloudlab_ip] = 0 # 0 indicates NOT running an experiment
    number_cloudlab_instances = len(cloudlab_instances_to_status.keys())
    experiment_name_to_assigned_ip = {}

    cloudlab_instance_to_setup_status = {}
    for cloudlab_ip in cloudlab_ips:
        cloudlab_instance_to_setup_status[cloudlab_ip] = None # 0 indicates nothing was setup

    exp_name_to_mimir_config = {}
    processed_expriments = []
    jobs = []
    processing_jobs = []
    while 0 in experiment_name_to_status.values() or 1 in experiment_name_to_status.values():
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
                        template_changes["experiment_name"] = exp_name
                        experiment_name_to_assigned_ip[exp_name] = ip
                        break
                ## now that we have the next experiment selected, we must run it
                template_copy = copy.deepcopy(template)
                user = cloudlab_config["user"]
                private_key = cloudlab_config["private_key"]
                local_dir = cloudlab_config["local_dir"] + exp_name + '/'
                if cloudlab_instance_to_setup_status[ip] == template_copy['application_name'] or skip_app_setup:
                    ## in this case, the current application was already setup... can therefore move straight to exp
                    flags = {'skip_app_setup': True}
                else:
                    flags = False

                cur_args = [template_copy, template_changes, ip, flags, user, private_key, exp_name, local_dir,
                            remote_experimental_data_dir, experiment_sentinal_file, remote_experimental_config_folder,
                            e2e_eval_configs_dir]
                service = multiprocessing.Process(name=exp_name, target=run_new_experiment, args=cur_args)
                service.start()
                jobs.append(service)
                # and now change the statuses of everything appropriately...
                experiment_name_to_status[exp_name] = 1
                cloudlab_instances_to_status[ip] = 1
                cloudlab_instance_to_setup_status[ip] = template_copy['application_name']
                exp_name_to_localdir[exp_name] = local_dir

        # okay, if we made it out here then all the cloudlab instances must have assigned work.
        # we must wait for a job to complete now...
        while(len(jobs) > 0): # TODO: modify this to handle analysis
            jobs_to_remove = []
            for job in jobs:
                if not job.is_alive():
                    jobs_to_remove.append(job)
                    ## we need to modify the cloudlab_instances_to_status appropriately
                    ## and note that we have a new experiment ready for processing by the latter part
                    ## of the system...
                    finished_exp_name = job.name
                    freed_ip = experiment_name_to_assigned_ip[finished_exp_name]
                    experiment_name_to_status[finished_exp_name] = 2
                    cloudlab_instances_to_status[freed_ip] = 0
                    print "newly_freed_ip", freed_ip

                    ## create analysis json file from template and start local processing (if possible)
                    analysis_strategy = analysis_config["name_to_analysis_status"][finished_exp_name]
                    if analysis_strategy == 'eval':
                        eval_exps.append(finished_exp_name)
                        analysis_template_file = cloudlab_exps_dir + analysis_config["eval_template"]
                    elif analysis_strategy == 'train':
                        train_exp = finished_exp_name
                        analysis_template_file = cloudlab_exps_dir + analysis_config["train_template"]
                    with open(analysis_template_file, 'r') as f:
                        # using OrderedDict shoud ensure that it is in the order that I like...
                        analysis_template = json.loads(f.read(), object_pairs_hook=OrderedDict)
                        #analysis_template = json.loads(f.read())


                    local_dir = exp_name_to_localdir[finished_exp_name]
                    exp_config_file = local_dir + finished_exp_name + "/" + finished_exp_name + '_analysis.json'
                    mod_analysis_template = create_analysis_json_from_template(analysis_template, finished_exp_name, exp_config_file)
                    mod_analysis_template_loc = local_dir + finished_exp_name + "/" + finished_exp_name + '_exp_proc.json'
                    with open(mod_analysis_template_loc, 'w') as f:
                        json.dump(mod_analysis_template, f, indent=2)

                    # store the location... this also makes it easy for latter on if I want to decide to only process...
                    # (though the reason why I would not just use mimir is unclear... is the ability to generate from the
                    # template? then I wouldn't want to generate them here ... though if skipping then I suppose that
                    # I could generate them earlier...)
                    exp_name_to_mimir_config[finished_exp_name] = mod_analysis_template_loc
                    #'''

            '''
            ## TODO: SECOND part of the functionality should occur HERE... it's a big deal...
            ## TODO: modify this to handle local too (probably nest a check on local files before running existing code)
            ## tODO: ADD FLAG TO ONLY do local
            
            ## NOTE: THIS IS NOT THE CURRENT PROPOSED DESIGN. PLEASE SEE THE ACTUAL DESIGN DOCUMENT FOR HOW TO DO THIS. ##
            ## (i think start analysis immediately is over-rated... so just wait until all is complete and then use the 
            ## multi-experiment looper)
            
            service = multiprocessing.Process(name=finished_exp_name + '_analysis', target=run_analysis,
                                              args=cur_args)
            if len(processing_jobs) <= max_local_processing_instances:
                service.start()
                processing_jobs.append(service)
            else:
                pass
            # and now change the statuses of
            
            # this is pretty straightforward right?
            # (a) start processing of the training (so I can get the model)
            # (b) modify the analysis json
            # (c) call the other functions, making sure not to over-use system resources...
            ## HOWEVER, A **KEY** consideration is that it can integrate with existing data that I've acquired...
            ## ... so maybe check analysis_exps.kson against cloudlab_exps.json (or whatever the name is) and load
            ## up the system as required (and then follow the above system...)
            '''

            if jobs_to_remove == []:  # <-----
                time.sleep(60) # wait a min and check again
            else:
                for job_to_remove in jobs_to_remove:
                    jobs.remove(job_to_remove)
                break

    # for the moment, let's keep this REAL nice and simple... just make input file for multi_experiment_looper.py and call that....
    # exp_name_to_mimir_config[finished_exp_name] = mod_analysis_template_loc ;;;
    ## ??? -- ||| -- ???

    ## NOTE: MIGHT WANT TO TAKE THESE FROM A CONFIG FILE TOO... especially if I want to
    ## use a remote server... (such as for running the whole thing during the night...)
    multi_exp_looper_config = {}
    multi_exp_looper_config['model_config_file'] = exp_name_to_mimir_config[train_exp]
    multi_exp_looper_config['xlabel'] = "load (# instances of load generation)"
    multi_exp_looper_config['use_cached'] = False
    multi_exp_looper_config['exfil_rate'] = [10000000.0, 1000000.0, 100000.0, 10000.0],
    multi_exp_looper_config['timegran'] = 10
    multi_exp_looper_config['type_of_graph'] = "load"
    multi_exp_looper_config['graph_name'] = cloudlab_config['cur_name']

    eval_configs_to_xvals = {}
    for eval_exp in eval_exps:
        eval_configs_to_xvals[exp_name_to_mimir_config[eval_exp]] = 50 ## NOTE: this is WRONG but it WILL make the thing RUN!
    multi_exp_looper_config["eval_configs_to_xvals"] = eval_configs_to_xvals

    multi_experiment_config_file_path = cloudlab_exps_dir + cloudlab_config['cur_name'] + '_multi_config.json'
    with open(multi_experiment_config_file_path, 'w') as g:
        json.dump(multi_exp_looper_config, g, indent=2)

    print "calling run_looper with this config file...", multi_experiment_config_file_path
    run_looper(multi_experiment_config_file_path, True)

    #### (2) start LOCAL experiments... perhaps I can abstract the existing logic into a function???
        ## cause it'll be essentially equivalent for the multiprocessing part... however
            ## (1) ALSO need to autogenerate analysis json
            ## (2) ALSO need to collect the multi-process output
        ## (Note: probably no point in modifying that code that I wrote b/c I don't think it really does
        ## the same thing at atll...)

    ## (b) create analysis json + run stuff locally... this'll also require the same kinda multiprocess logic as above...
    ## create graphs

if __name__=="__main__":
    cloudlab_exps_file = 'cloudlab_exps.json'
    analysis_exp_file = 'analysis_exps.json'
    cloudlab_exps_dir = './e2e_eval_configs/'
    skip_app_setup = True
    perform_eval_work(cloudlab_exps_file, cloudlab_exps_dir, analysis_exp_file, skip_app_setup)