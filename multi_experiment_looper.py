from analysis_pipeline.mimir import run_analysis
import pickle
import matplotlib.pyplot as plt
import argparse
import json
from collections import OrderedDict
from experiment_coordinator.process_data_on_remote import process_on_remote
import multiprocessing
import time

def track_multiple_remotes(remote_ips, eval_experiment_names, training_experiment_name, exps_per_server,
                           name_to_config, remote_server_key, user, dont_retrieve_from_remote,
                           skip_install, skip_upload, mimir_num):
    '''
    This function will handle assigning the processing to the various remote servers
    and monitor / aggregate the results.

    Note: for now, we'll assume that there's only a single remote server. However, this server
    can still have multiple processes (maybe like 7 or so), so there's still some complexity here.
    '''
    evalconfigs_to_cm = {}

    # step (1): setup structures to track status of servers and experiments
    remote_instances_to_status = {}
    for remote_ip in remote_ips:
        remote_instances_to_status[remote_ip] = 0 # 0 indicates NO running experiments...

    experiment_name_to_status = {}
    for exp_name in eval_experiment_names:
        experiment_name_to_status[exp_name] = 0 # 0 indicates that this experiment has NOT been run yet.
    experiment_name_to_status[training_experiment_name] = 0
    # 0 = not running ; 1 = currrently running ; 2 = finished running

    # step (2): if existing servers have extra capacity, use it for processing
    experiment_name_to_assigned_ip = {}
    jobs = []
    can_run_additional_exps = True
    while 0 in experiment_name_to_status.values() or 1 in experiment_name_to_status.values():
        for ip,status in remote_instances_to_status.iteritems():
            update_config = True
            use_remote = True
            remote_server_ip = ip
            if status < exps_per_server and can_run_additional_exps:
                # if the training model isn't created yet, must do that before ANY other experiments...
                if experiment_name_to_status[training_experiment_name] != 2:
                    model_config_file = name_to_config[training_experiment_name]
                    list_of_eval_configs = [name_to_config[training_experiment_name]]
                    cur_args = [model_config_file, list_of_eval_configs, update_config, use_remote, remote_server_ip,
                                remote_server_key, user, dont_retrieve_from_remote] #, skip_install, skip_upload] #??
                    kwargs = {"skip_install":skip_install, "skip_upload": skip_upload, "mimir_num":mimir_num}
                    service = multiprocessing.Process(name=training_experiment_name, target=get_eval_results,
                                                      args=cur_args, kwargs=kwargs)
                    service.start()
                    jobs.append(service)

                    experiment_name_to_status[training_experiment_name] = 1
                    remote_instances_to_status[ip] += 1
                    experiment_name_to_assigned_ip[training_experiment_name] = ip
                    can_run_additional_exps = False
                else:
                    for exp_name in experiment_name_to_status.keys():
                        if experiment_name_to_status[exp_name] == 0:
                            # if we are here, then there is extra compute available + data that needs processing
                            model_config_file = name_to_config[training_experiment_name]
                            list_of_eval_configs = [name_to_config[exp_name]]
                            cur_args = [model_config_file, list_of_eval_configs, update_config, use_remote, remote_server_ip,
                                        remote_server_key, user, dont_retrieve_from_remote] #, skip_install, skip_upload] #??
                            kwargs = {"skip_install": skip_install, "skip_upload": skip_upload, "mimir_num":mimir_num}
                            service = multiprocessing.Process(name=exp_name, target=get_eval_results,
                                                              args=cur_args, kwargs=kwargs)
                            service.start()
                            jobs.append(service)

                            experiment_name_to_status[exp_name] = 1
                            remote_instances_to_status[ip] += 1
                            experiment_name_to_assigned_ip[exp_name] = ip

        # step 3: check if any current jobs are done. if they are, then modify the state appropriately.
        while (len(jobs) > 0):
            jobs_to_remove = []
            for job in jobs:
                if not job.is_alive():
                    jobs_to_remove.append(job)
                    finished_exp_name = job.name
                    finished_ip = experiment_name_to_assigned_ip[finished_exp_name]
                    remote_instances_to_status[finished_ip] -= 1
                    experiment_name_to_status[finished_exp_name] = 2

                    can_run_additional_exps = True # training experiment is done first... so if any is done, then it must be done

                    if finished_exp_name in eval_experiment_names:
                        evalconfigs_to_cm[finished_exp_name] = None ## TODO: should this be something????
            if jobs_to_remove == []:
                time.sleep(60) # wait a min and check again
            else:
                for job_to_remove in jobs_to_remove:
                    jobs.remove(job_to_remove)
                break

    # step 4: ???
    # once we get here, all the experiments must have finished running... so what's next
    # okay, well if everything ran, then once we get to here, we just want to returnt the correct stuff
    # evalconfigs_to_cm
    ### OR could i use: create_eval_graph ???? would it be possible???

def update_config_file(config_file_pth, if_trained_model):
    with open(config_file_pth, 'r') as f:
        # (a) read in the config.
        config_file = json.loads(f.read(), object_pairs_hook=OrderedDict)
    # (b) make appropriate modifications to the file
    if if_trained_model:
        config_file["get_endresult_from_memory"] = True

    config_file["make_edgefiles"] = False
    config_file["skip_graph_injection"] = True
    config_file["calc_vals"] = False
    config_file["calculate_z_scores"] = False

    # (c) write it back out...
    with open(config_file_pth, 'w') as f:
        json.dump(config_file, f, indent=2)

def get_eval_results(model_config_file, list_of_eval_configs, update_config, use_remote=False, remote_server_ip=None,
                     remote_server_key=None, user=None, dont_retrieve_from_remote=None, skip_install=False,
                     skip_upload=False, mimir_num = None):
    eval_config_to_cm = {}
    for eval_config in list_of_eval_configs:
        if not use_remote:
            eval_cm = run_analysis(model_config_file, eval_config=eval_config)
        else:
            # need to go into the config file and look @ exp_config_file and take the containing directory
            eval_analysis_config_file = eval_config
            with open(eval_analysis_config_file, 'r') as g:
                eval_conf = json.loads(g.read())

            print "multi_eval_analysis_config_file",eval_analysis_config_file
            #eval_conf_file = eval_conf["eval_conf"]
            eval_dir_with_data = "/".join(eval_analysis_config_file.split("/")[:-1]) + '/'
            print "multi_eval_analysis_config_file_after_split",eval_dir_with_data

            model_analysis_config_file = model_config_file
            with open(model_analysis_config_file, 'r') as g:
                model_conf = json.loads(g.read())
            #model_conf_dir = model_conf["eval_conf"]
            model_dir = "/".join(model_analysis_config_file.split("/")[:-1]) + '/'

            dont_retreive_train = False
            dont_retreive_eval = False
            #print "dont_retrieve_from_remote", dont_retrieve_from_remote
            #print "model_analysis_config_file",model_analysis_config_file
            #print "eval_analysis_config_file",eval_analysis_config_file
            #print  model_analysis_config_file in dont_retrieve_from_remote
            #print eval_analysis_config_file not in dont_retrieve_from_remote
            #exit(233)

            if dont_retrieve_from_remote is not None:
                if model_analysis_config_file in dont_retrieve_from_remote:
                    dont_retreive_train = True
                if eval_analysis_config_file in dont_retrieve_from_remote:
                    dont_retreive_eval = True

            # kwargs = {"skip_install": skip_install, "skip_upload": skip_upload}

            eval_cm = process_on_remote(remote_server_ip, remote_server_key, user, eval_dir_with_data, eval_analysis_config_file,
                              model_dir, model_analysis_config_file, skip_install=skip_install, skip_upload=skip_upload,
                                    dont_retreive_eval=dont_retreive_eval, dont_retreive_train=dont_retreive_train,
                                    mimir_num=mimir_num)

        eval_config_to_cm[eval_config] = eval_cm
        ## modify the config file so that you don't redo previously done experiments...
        if update_config:
            update_config_file(eval_config, if_trained_model=False)
            update_config_file(model_config_file, if_trained_model=True)

    return eval_config_to_cm

def cm_to_f1(cm, exfil_rate, timegran):
    cm = cm[exfil_rate][timegran]
    #print "cm", cm
    tn = 0.0
    fp = 0.0
    fn = 0.0
    tp = 0.0
    for index, row in cm.iterrows():
        tn += row['tn']
        fp += row['fp']
        fn += row['fn']
        tp += row['tp']
    f1_score = (2.0 * tp) / (2.0 * tp + fp + fn)
    return f1_score

def create_eval_graph(model_config_file, eval_configs_to_xvals, xlabel, use_cached, exfil_rates, timegran,
                      type_of_graph, graph_name, update_config_p, use_remote=False, remote_server_ip=None,
                      remote_server_key=None,user=None, dont_retrieve_from_remote=None):
    if use_cached:
        with open('./temp_outputs/' + graph_name + '_cached_looper.pickle', 'r') as f:
            evalconfigs_to_cm = pickle.loads(f.read())
    else:
        evalconfigs_to_cm = get_eval_results(model_config_file, eval_configs_to_xvals.keys(), update_config_p, use_remote,
                                             remote_server_ip=remote_server_ip, remote_server_key=remote_server_key,
                                             user=user, dont_retrieve_from_remote=dont_retrieve_from_remote)
        with open('./temp_outputs/' + graph_name + '_cached_looper.pickle', 'w') as f:
            f.write(pickle.dumps(evalconfigs_to_cm))

    return evalconfigs_to_cm


def generate_graphs(eval_configs_to_xvals, exfil_rates, evalconfigs_to_cm, timegran, type_of_graph, graph_name, xlabel):
    rate_to_xlist_ylist = {}
    for exfil_rate in exfil_rates:
        x_vals_list = []
        y_vals_list = []
        eval_to_prob_dist = {}
        for evalconfig,xval in eval_configs_to_xvals.iteritems():
            x_vals_list.append(xval)
            cur_cm = evalconfigs_to_cm[evalconfig]
            optimal_f1 = cm_to_f1(cur_cm, exfil_rate, timegran)
            y_vals_list.append( optimal_f1 )

            if type_of_graph == 'euclidean_distance':
                with open(evalconfig, 'r') as g:
                    config_stuff = json.loads(g.read())
                    with open(config_stuff['exp_config_file'], 'r') as z:
                        exp_config = json.loads(g.read())
                        eval_to_prob_dist[evalconfig] = None ## TODO
                        ## ## TODO: get the angles from the json configs. ## ##

            ## (step1) cache the results from get_eval_results (b/c gotta iterate on steps2&3) [[[done]]]
            ## (step2) put process cms (to get F1 scores)
            ## (step3) make actual graphs (can just stick into temp_outputs for now... I gues...)

        if type_of_graph == 'euclidean_distance':
            pass
        if type_of_graph == 'table':
            pass
        # then load is really straightforward

        print "x_vals_list", x_vals_list
        print "y_vals_list", y_vals_list
        x_vals_list, y_vals_list = zip(*sorted(zip(x_vals_list, y_vals_list)))

        plt.clf()
        plt.plot(x_vals_list, y_vals_list, marker='.', markersize=22)
        plt.xlabel(xlabel)
        plt.ylabel('f1 score')
        plt.show()
        plt.savefig('./multilooper_outs/' + graph_name + '_' + str(exfil_rate) + '.png')

        rate_to_xlist_ylist[exfil_rate] =  (x_vals_list, y_vals_list)

    ## TODO put them all on the same grid... (I'm just going to do it manually b/c hurry...)
    fig, axes = plt.subplots(nrows=1, ncols=len(exfil_rates), figsize=(50, 10))
    fig.suptitle(str(graph_name) + ' f1 vs load at various exfil rates')
    for counter,rate in enumerate(exfil_rates):
        x_vals,y_vals = rate_to_xlist_ylist[rate]
        axes[counter].plot(x_vals, y_vals, label=str(rate), marker='*', markersize=22)
        axes[counter].set_ylim(top=1.1,bottom=0.75) #  we can do it manually...
    fig.align_ylabels(axes)
    fig.savefig('./multilooper_outs/aggreg_' + graph_name + '.png')


def parse_config(config_file_pth):
    with open(config_file_pth, 'r') as f:
        #config_file = json.load(f)
        config_file = json.loads(f.read(), object_pairs_hook=OrderedDict)

        if 'model_config_file' in config_file:
            model_config_file = config_file['model_config_file']
        else:
            model_config_file = False

        if 'eval_configs_to_xvals' in config_file:
            eval_configs_to_xvals = config_file['eval_configs_to_xvals']
        else:
            eval_configs_to_xvals = False

        if 'xlabel' in config_file:
            xlabel = config_file['xlabel']
        else:
            xlabel = False

        if 'use_cached' in config_file:
            use_cached = config_file['use_cached']
        else:
            use_cached = False

        if 'exfil_rate' in config_file:
            exfil_rate = config_file['exfil_rate']
        else:
            exfil_rate = False

        if 'timegran' in config_file:
            timegran = config_file['timegran']
        else:
            timegran = False

        if 'type_of_graph' in config_file:
            type_of_graph = config_file['type_of_graph']
        else:
            type_of_graph = False

        if 'graph_name' in config_file:
            graph_name = config_file['graph_name']
        else:
            graph_name = False

        if 'use_remote' in config_file:
            use_remote = config_file['use_remote']
        else:
            use_remote = None

        if 'remote_server_ips' in config_file:
            remote_server_ips = config_file['remote_server_ips']
        else:
            remote_server_ips = None

        if 'remote_server_key' in config_file:
            remote_server_key = config_file['remote_server_key']
        else:
            remote_server_key = None

        if 'user' in config_file:
            user = config_file['user']
        else:
            user = None

        if 'dont_retrieve_from_remote' in config_file:
            dont_retrieve_from_remote = config_file['dont_retrieve_from_remote']
        else:
            dont_retrieve_from_remote = None

        if 'skip_install' in config_file:
            skip_install = config_file['skip_install']
        else:
            skip_install = False

        if 'skip_upload' in config_file:
            skip_upload = config_file['skip_upload']
        else:
            skip_upload = False

        if 'mimir_num' in config_file:
            mimir_num = config_file['mimir_num']
        else:
            mimir_num = False

    return model_config_file, eval_configs_to_xvals, xlabel, use_cached, exfil_rate, timegran, type_of_graph, \
           graph_name, use_remote, remote_server_ips, remote_server_key, user, dont_retrieve_from_remote, \
           skip_install, skip_upload, mimir_num

def run_looper(config_file_pth, update_config, use_remote):

    model_config_file, eval_configs_to_xvals, xlabel, use_cached, exfil_rate, timegran, type_of_graph, graph_name, \
    use_remote_from_config, remote_ips, remote_server_key, user, dont_retrieve_from_remote, skip_install, skip_upload, \
        mimir_num = parse_config(config_file_pth)

    if use_remote_from_config is not None:
        use_remote = use_remote_from_config or use_remote

    #print("type(eval_configs_to_xvals)", type(eval_configs_to_xvals))
    #exit(233)

    # DON'T FORGET ABOUT use_cached (it's very useful -- especially when iterating on graphs!!)
    use_cached = use_cached
    update_config = update_config
    use_remote = use_remote

    if use_remote:
        eval_experiment_names = eval_configs_to_xvals.keys()
        training_experiment_name = model_config_file
        exps_per_server = 4

        name_to_config = {}
        for eval_config in eval_configs_to_xvals.keys():
            name_to_config[eval_config] = eval_config
        name_to_config[training_experiment_name] = training_experiment_name

        evalconfigs_to_cm = track_multiple_remotes(remote_ips, eval_experiment_names, training_experiment_name,
                                                   exps_per_server, name_to_config, remote_server_key, user,
                                                   dont_retrieve_from_remote, skip_install, skip_upload, mimir_num)
    else:
        evalconfigs_to_cm = create_eval_graph(model_config_file, eval_configs_to_xvals, xlabel, use_cached, exfil_rate, timegran,
                                            type_of_graph, graph_name, update_config)

    generate_graphs(eval_configs_to_xvals, exfil_rate, evalconfigs_to_cm, timegran, type_of_graph, graph_name, xlabel)

if __name__=="__main__":
    print "RUNNING"

    parser = argparse.ArgumentParser(description='This can run multiple experiments in a row on MIMIR. Also makes graphs')
    parser.add_argument('--config_json', dest='config_json', default=None,
                        help='this is the configuration file used to run to loop through several experiments')
    parser.add_argument('--dont_update_config', dest='dont_update_config',
                        default=False, action='store_true')

    parser.add_argument('--use_remote_server', dest='use_remote',
                        default=False, action='store_true')

    args = parser.parse_args()

    if not args.config_json:
        #config_file_pth = "./multi_experiment_configs/wordpress_scale.json"
        #config_file_pth = "./analysis_pipeline/multi_experiment_configs/old_sockshop_angle_remote2.json"
        #config_file_pth = "./analysis_pipeline/multi_experiment_configs/old_sockshop_scale.json"
        config_file_pth = "./analysis_pipeline/multi_experiment_configs/sockshop_test_remote.json"
        #config_file_pth = "./analysis_pipeline/multi_experiment_configs/new_sockshop_scale.json"
        #config_file_pth = "./analysis_pipeline/multi_experiment_configs/old_sockshop_angle_remote2.json"
    else:
        config_file_pth = args.config_json

    run_looper(config_file_pth, (not args.dont_update_config), args.use_remote)