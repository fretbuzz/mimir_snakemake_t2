from mimir import run_analysis
import pickle
import matplotlib.pyplot as plt
import argparse
import json
from collections import OrderedDict
#from experiment_coordinator.process_data_on_remote import process_on_remote
import multiprocessing
import time
from tabulate import tabulate
import numpy as np
import ast
import os, errno
import time
import copy
import pandas as pd
plt.style.use('seaborn-paper')

method_to_legend = {'ensemble': 'Our Method',
                    'cilium': 'Communicating Services',
                    'ide': 'Eigenspace',
                    'decanter': 'Decanter',
                    'logistic_ide': 'logistic_ide',
                    'logistic': 'logistic',
                    'lasso': 'lasso',
                    'boosting_lasso': 'boosting_lasso',
                    'boosting_lasso_with_optimal_train_thresh': 'boosting_lasso_with_optimal_train_thresh',
                    'logistic_ide_with_optimal_train_thresh' : 'logistic_ide_with_optimal_train_thresh',
                    'logistic_with_optimal_train_thresh': 'logistic_with_optimal_train_thresh',
                    'lasso_with_optimal_train_thresh': 'lasso_with_optimal_train_thresh'}
B_in_KB = 1000.0

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

def handle_single_exp(model_config_file, eval_config, no_tsl, decanter_configs, live_p, update_config,
                      eval_config_to_cm, retrain_model_p, per_svc_exfil_model_p, exp_data_dir, eval_config_to_modelType_to_cm,
                      use_training_model_from_mem, no_cilium, dont_open_pdfs, use_all_results_from_mem, load_old_pipelines):

    eval_cm, perModel_eval_cm = run_analysis(False, model_config_file, eval_config=eval_config, no_tsl=no_tsl,
                           decanter_configs=decanter_configs, live=live_p, skip_to_calc_zscore=retrain_model_p,
                           per_svc_exfil_model_p=per_svc_exfil_model_p, exp_data_dir=exp_data_dir,
                           load_endresult_train=use_training_model_from_mem, no_cilium=no_cilium,
                           dont_open_pdfs=dont_open_pdfs, use_all_results_from_mem=use_all_results_from_mem,
                           load_old_pipelines=load_old_pipelines)

    if update_config:
        update_config_file(eval_config, if_trained_model=False)

    eval_config_to_cm[eval_config] = eval_cm
    eval_config_to_modelType_to_cm[eval_config] = perModel_eval_cm


def get_eval_results(model_config_file, list_of_eval_configs, update_config, retrain_model_p, per_svc_exfil_model_p,
                     use_remote=False, remote_server_ip=None,
                     remote_server_key=None, user=None, dont_retrieve_from_remote=None, only_finished_p=False,
                     no_tsl=False, decanter_configs=None, live_p=False, analyze_in_parallel=False, exp_data_dir=None,
                     use_training_model_from_mem=None, no_cilium=False, dont_open_pdfs = False,
                     use_all_results_from_mem=False, load_old_pipelines=False):
    manager = multiprocessing.Manager()


    eval_config_to_cm = manager.dict()
    eval_config_to_modelType_to_cm = manager.dict()
    ran_model_already = False
    running_analyses = []

    for eval_config in list_of_eval_configs:
        if not use_remote:
            if only_finished_p:
                if has_experiment_already_been_run(eval_config):
                    eval_cm, perModel_eval_cm = run_analysis(False, model_config_file, eval_config=eval_config, no_tsl=no_tsl,
                                           decanter_configs=decanter_configs, live=live_p,
                                           skip_to_calc_zscore=retrain_model_p, per_svc_exfil_model_p=per_svc_exfil_model_p,
                                           exp_data_dir=exp_data_dir, load_endresult_train=use_training_model_from_mem,
                                           no_cilium=no_cilium, dont_open_pdfs=dont_open_pdfs,
                                           use_all_results_from_mem=use_all_results_from_mem, load_old_pipelines=load_old_pipelines)
                else:
                    continue  # don't want to wait ---> so just pass over this one.
                pass
            else:
                ## TODO: probably wanna wrap in a call to multiprocess, to prevent problems with
                ## memory size and using swap space...
                if not ran_model_already:
                    print "training only the model...."
                    run_analysis(False, model_config_file, no_tsl=no_tsl,
                                 decanter_configs=decanter_configs, live=live_p,
                                 skip_to_calc_zscore=retrain_model_p, per_svc_exfil_model_p=per_svc_exfil_model_p,
                                 exp_data_dir=exp_data_dir, load_endresult_train=use_training_model_from_mem,
                                 no_cilium=no_cilium, dont_open_pdfs=dont_open_pdfs, use_all_results_from_mem=use_all_results_from_mem,
                                 load_old_pipelines=load_old_pipelines)
                    if update_config:
                        update_config_file(model_config_file, if_trained_model=True)
                    ran_model_already = True
                    print "model was trained... getting ready to run the testing data..."

                # run the analysis in several different process in parallel...
                if analyze_in_parallel:
                    # need to wait a bit to avoid unzipping several large pcaps at once if the following conditions hold:
                    # (1) is wordpress application (b/c only wordpress has such large pcaps that zipping is needed)
                    # (2) multiple eval traces are being analyzed (b/c a single one wouldn't be able to overload storage)
                    # (3) the pcaps are actually being analyzed (as opposed to using previously-generated edgefiles)
                    if 'wordpress' in model_config_file:
                        if len(list_of_eval_configs) > 1:
                            with open(eval_config) as json_file:
                                data = json.load(json_file)
                                make_edgefiles_p = data['make_edgefiles']
                            if make_edgefiles_p:
                                time.sleep(300)
                    handle_single_exp_args = (model_config_file, eval_config, no_tsl, decanter_configs, live_p, update_config,
                                      eval_config_to_cm, retrain_model_p, per_svc_exfil_model_p, exp_data_dir, eval_config_to_modelType_to_cm,
                                      use_training_model_from_mem, no_cilium, dont_open_pdfs, use_all_results_from_mem, load_old_pipelines)
                    p = multiprocessing.Process(target=handle_single_exp, args=handle_single_exp_args)
                    running_analyses.append(p)
                    p.start()
                else:
                    eval_cm, perModel_eval_cm = run_analysis(False, model_config_file, eval_config=eval_config, no_tsl=no_tsl,
                                           decanter_configs=decanter_configs, live=live_p, skip_to_calc_zscore=retrain_model_p,
                                           per_svc_exfil_model_p=per_svc_exfil_model_p, exp_data_dir=exp_data_dir,
                                            load_endresult_train=use_training_model_from_mem, no_cilium=no_cilium,
                                            dont_open_pdfs=dont_open_pdfs, use_all_results_from_mem=use_all_results_from_mem,
                                            load_old_pipelines=load_old_pipelines)

                    if update_config:
                        update_config_file(eval_config, if_trained_model=False)
        else:
            exit(322) # how would it even get here??

        if not analyze_in_parallel:
            eval_config_to_cm[eval_config] = eval_cm
            eval_config_to_modelType_to_cm[eval_config] = perModel_eval_cm

        ## modify the config file so that you don't redo previously done experiments...
        #if update_config:
        #    update_config_file(eval_config, if_trained_model=False)
        #    update_config_file(model_config_file, if_trained_model=True)

    if analyze_in_parallel:
        for proc in running_analyses:
            proc.join()

    eval_config_to_cm = eval_config_to_cm.copy()
    eval_config_to_modelType_to_cm = eval_config_to_modelType_to_cm.copy()

    with open('./check_this.txt', 'w') as f:
        f.write( pickle.dumps(eval_config_to_cm) )

    with open('./check_this_perModel.txt', 'w') as f:
        f.write( pickle.dumps(eval_config_to_modelType_to_cm) )

    return eval_config_to_cm, eval_config_to_modelType_to_cm

def aggregate_cm_vals_over_paths(cm, method=None):
    tn = 0.0
    fp = 0.0
    fn = 0.0
    tp = 0.0
    #print "cm.keys()", cm.keys()
    if type(cm) == dict:
        if method:
            cm = cm[method]
    else:
        pass
    for index, row in cm.iterrows():
        tn += row['tn']
        fp += row['fp']
        fn += row['fn']
        tp += row['tp']
    return tn,fp,fn,tp

def cm_to_f1(cm, exfil_rate, timegran,method=None):
    #print "cm", cm
    #print method, "method"
    if method:
        cm = cm[exfil_rate][timegran][method]
    else:
        cm = cm[exfil_rate][timegran]
    #print "cm",cm
    tn, fp, fn, tp = aggregate_cm_vals_over_paths(cm)
    f1_score = (2.0 * tp) / (2.0 * tp + fp + fn)
    return f1_score



def cm_to_exfil_rate_vs_f1(cm, evalconfig):
    timegran_to_method_to_rate_to_f1 = {}
    for rate, timegran_method_cm in cm.iteritems():
        for timegran, method_cm in timegran_method_cm.iteritems():
            methods = method_cm.keys()
            for method, confusion_matrix in method_cm.iteritems():
                tn, fp, fn, tp = aggregate_cm_vals_over_paths(confusion_matrix)
                f1_score = (2.0 * tp) / (2.0 * tp + fp + fn)

                if timegran not in timegran_to_method_to_rate_to_f1:
                    timegran_to_method_to_rate_to_f1[timegran] = {}
                if method not in timegran_to_method_to_rate_to_f1[timegran]:
                    timegran_to_method_to_rate_to_f1[timegran][method] = {}
                timegran_to_method_to_rate_to_f1[timegran][method][rate] = f1_score

    for timegan,method_to_rate_to_f1 in timegran_to_method_to_rate_to_f1.iteritems():
        plt.clf()
        plt.figure(figsize=(15, 20))
        fig, ax = plt.subplots()
        ax.set_ylim(top=1.1, bottom=-0.1)  # we can do it manually...
        for method in methods:
            rate_to_f1 = method_to_rate_to_f1[method]
            rates = [i / B_in_KB for i in rate_to_f1.keys()]
            f1s = rate_to_f1.values()
            rates, f1s = zip(*sorted(zip(rates, f1s)))

            # so rates would be x-axis and f1s would be the y-axis

            ax.plot(rates, f1s, label=str(method_to_legend[method]), linewidth=3.0, marker='.', markersize=22,)
            ax.set_title('F1 Score vs Exfiltration Rate', fontsize=18)
            ax.set_ylabel('F1 Scores', fontsize=15)
            ax.set_xscale('log')
            ax.set_xlabel('Log of Exfiltration Rate (log KB/min)', fontsize=15)
            ax.legend()

        evalconfig_name = evalconfig.split('/')[-1]
        f1_vs_exfil_rate_filname = "./multilooper_outs/" + evalconfig_name + '_' + str(timegan) + "_f1_vs_exfil_rate.png"
        print "f1_vs_exfil_rate_filname",f1_vs_exfil_rate_filname
        plt.tight_layout()
        fig.savefig(f1_vs_exfil_rate_filname)

    return timegran_to_method_to_rate_to_f1

def generate_secondary_cache_name(model_config_file, no_tsl, per_svc_exfil_model_p):
    sec_cache_name =  model_config_file[:-4]

    if not no_tsl:
        sec_cache_name += '_min_exfilrate_tsl_'
    if per_svc_exfil_model_p:
        sec_cache_name += '_persvc_exfil_model_'

    sec_cache_name += 'cached_evalconfigs_to_cm.pickle'
    return sec_cache_name

def get_evalconfigs_to_cm(model_config_file, eval_configs_to_xvals, xlabel, use_cached, exfil_rates, timegran,
                          type_of_graph, graph_name, update_config_p, only_finished_p,
                          retrain_model_p, per_svc_exfil_model_p, use_remote=False,
                          remote_server_ip=None, remote_server_key=None, user=None, dont_retrieve_from_remote=None,
                          no_tsl = False, decanter_configs=None, live_p=False, analyze_in_parallel = False,
                          exp_data_dir=None, use_training_model_from_mem=False, no_cilium=False, dont_open_pdfs=False,
                          use_all_results_from_mem=False, load_old_pipelines=False):
    # TODO: modify this function to use: retrain_model_p, per_svc_exfil_model_p

    cache_name = './temp_outputs/' + graph_name
    cache_name_cp = copy.copy(cache_name)
    if not no_tsl:
        cache_name += '_min_exfilrate_tsl_'
    #if per_svc_exfil_model_p:
    #    cache_name += '_persvc_exfil_model_'
    cache_name_persvc = cache_name_cp + '_persvc_exfil_model_'

    cache_name += '_cached_looper.pickle' # + 'ret' # if it doesn't help, remove the second part...

    # the idea of these line is to ensure that the cached results are in the an obvious location, so my looper can grab
    # them later on to make an average-results graph...
    ##secondary_cache_name = "/".join(model_config_file.split('/')[:-2]) + 'cached_evalconfigs_to_cm.pickle'
    secondary_cache_name = generate_secondary_cache_name(model_config_file, no_tsl, False)
    secondary_cache_name_persvc = generate_secondary_cache_name(model_config_file, False, True)

    if use_cached:
        #print "cache_name",cache_name
        with open(cache_name, 'r') as f:
            #cont = f.read()
            #print "cont",cont
            evalconfigs_to_cm = pickle.load(f)
            #evalconfigs_to_cm = pickle.loads(cont) #TODO<--renable!
            #evalconfigs_to_cm = {} # TODO: actually debug this whole part... (so I can re-enable the line above)
        #print "cache_name_persvc", cache_name_persvc
        with open(cache_name_persvc, 'r') as f:
            evalconfigs_to_model_to_cm = pickle.loads(f.read())
    else:
        list_of_eval_configs = []
        list_of_eval_sizes = []
        for config,size in eval_configs_to_xvals.iteritems():
            list_of_eval_configs.append(config)
            list_of_eval_sizes.append(size)

        # using solution from: https://stackoverflow.com/questions/9764298/is-it-possible-to-sort-two-listswhich-reference-each-other-in-the-exact-same-w
        # necessary for wordpress exps to be able to run on cloudlab... (b/c of space issues)...
        list_of_eval_sizes, list_of_eval_configs = (list(l) for l in zip(*sorted(zip(list_of_eval_sizes, list_of_eval_configs))))
        list_of_eval_configs.reverse()
        #print "list_of_eval_sizes", list_of_eval_sizes, "list_of_eval_configs", list_of_eval_configs

        evalconfigs_to_cm, evalconfigs_to_model_to_cm = get_eval_results(model_config_file, list_of_eval_configs, update_config_p,
                                             retrain_model_p, per_svc_exfil_model_p, use_remote,
                                             remote_server_ip=remote_server_ip, remote_server_key=remote_server_key,
                                             user=user, dont_retrieve_from_remote=dont_retrieve_from_remote,
                                             only_finished_p=only_finished_p, no_tsl=no_tsl, decanter_configs=decanter_configs,
                                             live_p = live_p, analyze_in_parallel = analyze_in_parallel, exp_data_dir=exp_data_dir,
                                             use_training_model_from_mem=use_training_model_from_mem, no_cilium=no_cilium,
                                             dont_open_pdfs=dont_open_pdfs, use_all_results_from_mem=use_all_results_from_mem,
                                             load_old_pipelines=load_old_pipelines)
        with open(cache_name, 'w') as f:
            f.write(pickle.dumps(evalconfigs_to_cm))
        with open(secondary_cache_name, 'w') as f:
            f.write(pickle.dumps(evalconfigs_to_cm))
        with open(cache_name_persvc, 'w') as f:
            f.write(pickle.dumps(evalconfigs_to_model_to_cm))
        with open(secondary_cache_name_persvc, 'w') as f:
            f.write(pickle.dumps(evalconfigs_to_model_to_cm))

    return evalconfigs_to_cm, evalconfigs_to_model_to_cm

def exfil_rate_vs_f1_at_various_timegran(timegran_to_method_to_rate_to_f1, evalconfig, method='ensemble'):
    plt.clf()
    plt.figure(figsize=(15, 20))
    fig, ax = plt.subplots()

    for timegran,method_to_rate_to_f1 in timegran_to_method_to_rate_to_f1.iteritems():
        rate_to_f1 = method_to_rate_to_f1[method]
        #rates = rate_to_f1.keys()
        rates = [i / B_in_KB for i in rate_to_f1.keys()]
        f1s = rate_to_f1.values()
        rates, f1s = zip(*sorted(zip(rates, f1s)))

        # so rates would be x-axis and f1s would be the y-axis
        #print "exfil_rate_vs_f1_at_various_timegran", timegran, rates, f1s

        if type(timegran) == tuple:
            timegran = 'multi-time'
        else:
            timegran = str(timegran) + ' sec'

        ax.plot(rates, f1s, label=str(timegran), marker='.', linewidth=3.0, markersize=15, alpha=0.8)
        ax.set_title('F1 Score vs Exfiltration Rate at Different Time Granularities', fontsize=12)
        ax.set_ylabel('F1 Scores', fontsize=11)
        ax.set_xscale('log')
        ax.set_xlabel('Log of Exfiltration Rate (log KB/min)', fontsize=11)
        ax.legend()

    evalconfig_name = evalconfig.split('/')[-1]
    ax.set_ylim(top=1.1, bottom=-0.1)  # we can do it manually...
    plt.tight_layout()
    f1_vs_exfil_rate_filname = "./multilooper_outs/diffTimeGran_" + evalconfig_name + "_f1_vs_exfil_rate.png"
    #print "f1_vs_exfil_rate_filname",f1_vs_exfil_rate_filname
    fig.savefig(f1_vs_exfil_rate_filname)

def has_experiment_already_been_run(config_file_pth):
    with open(config_file_pth, 'r') as f:
        config_file = json.loads(f.read(), object_pairs_hook=OrderedDict)
    # (b) make appropriate modifications to the file

    if config_file["get_endresult_from_memory"]:
        return True
    else:
        if (not config_file["make_edgefiles"]) and (config_file["skip_graph_injection"]) and (not config_file["calc_vals"]):
            # note: calculating z_scores is fast, so don't worry about it...
            return True
    return False

def get_prob_distr(processing_config):
    #print "processing_config", processing_config
    with open(processing_config, 'r') as g:
        config_stuff = json.loads(g.read())
        #print "file_made_by_experimental_apparatus", config_stuff['exp_config_file']
        with open(config_stuff['exp_config_file'], 'r') as z:
            z_cont = z.read()
            #print "g_cont", z_cont, len(z_cont)
            exp_config = json.loads(z_cont)
    return exp_config['prob_distro']

def convert_prob_distro_dict_to_array(prob_distro_dict, prob_distro_keys):
    prob_distro_list = []
    for cur_key in prob_distro_keys:
        prob_distro_list.append(prob_distro_dict[cur_key])
    prob_distro_vector = np.array(prob_distro_list)
    return prob_distro_vector

def generate_graphs(eval_configs_to_xvals, exfil_rates, evalconfigs_to_cm, timegran, type_of_graph, graph_name,
                    xlabel, model_config_file, no_tsl=False, model_xval=100, new_model=None, no_methods=False):
    method_to_rate_to_xlist_ylist = {}
    #print "evalconfigs_to_cm",evalconfigs_to_cm
    if new_model is not None:
        methods = [new_model]
    else:
        methods = evalconfigs_to_cm[eval_configs_to_xvals.keys()[0]][exfil_rates[0]][timegran].keys()
    angles_method_to_rate_to_xlist_ylist = {}

    try:
        os.makedirs('./multilooper_outs/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if not no_tsl:
        mutli_load_graph = './multilooper_outs/aggreg_' + str(timegran) + '_' + graph_name + '.png'
        multi_angle_graph = './multilooper_outs/euclidean_distance_' + str(timegran) + '_' + graph_name + '.png'
    else:
        mutli_load_graph = './multilooper_outs/aggreg_' + str(timegran) + '_' + graph_name + '_no_tsl' + '.png'
        multi_angle_graph = './multilooper_outs/euclidean_distance_' + str(timegran) + '_' + graph_name + '_no_tsl' + '.png'

    method_to_eval_to_f1 = {}
    for exfil_rate in exfil_rates:
        if not no_tsl:
            single_scale_load = './multilooper_outs/' + str(timegran) + '_' + graph_name + '_' + str(exfil_rate) + '.png'
        else:
            single_scale_load = './multilooper_outs/' + str(timegran) + '_' + graph_name + '_' + str(exfil_rate) + '_no_tsl' + '.png'

        method_to_x_vals_list = {}
        method_to_y_vals_list = {}
        for method in methods:
            x_vals_list = []
            y_vals_list = []
            eval_to_prob_dist = {}
            for evalconfig,xval in eval_configs_to_xvals.iteritems():
                x_vals_list.append(xval)

                #print "evalconfigs_to_cm[evalconfig]", evalconfigs_to_cm[evalconfig]
                #evalconfigs_to_cm[evalconfig] = evalconfigs_to_cm[evalconfig]["ensemble"]

                if evalconfig not in evalconfigs_to_cm:
                    continue
                else:
                    cur_cm = evalconfigs_to_cm[evalconfig]

                optimal_f1 = cm_to_f1(cur_cm, exfil_rate, timegran, method=method)
                y_vals_list.append( optimal_f1 )

                if method not in method_to_eval_to_f1:
                    method_to_eval_to_f1[method] = {}
                method_to_eval_to_f1[method][evalconfig] = optimal_f1

                if type_of_graph == 'angle':
                    prob_distro = get_prob_distr(evalconfig)
                    eval_to_prob_dist[evalconfig] = prob_distro
                #print "eval_to_prob_dist",eval_to_prob_dist

                timegran_to_method_to_rate_to_f1 = cm_to_exfil_rate_vs_f1(cur_cm, evalconfig)
                if new_model:
                    exfil_rate_vs_f1_at_various_timegran(timegran_to_method_to_rate_to_f1, evalconfig, method=new_model)
                else:
                    exfil_rate_vs_f1_at_various_timegran(timegran_to_method_to_rate_to_f1, evalconfig)

                ## (step1) cache the results from get_eval_results (b/c gotta iterate on steps2&3) [[[done]]]
                ## (step2) put process cms (to get F1 scores)
                ## (step3) make actual graphs (can just stick into temp_outputs for now... I gues...)

            method_to_x_vals_list[method] = x_vals_list
            method_to_y_vals_list[method] = y_vals_list


        #print "type_of_graph",type_of_graph
        if type_of_graph == 'angle':
            #print "---angle--"
            ## okay, we want to make the desired graph...
            ## x-axis will be euclidean distance...
            ## y-axis will be the F1 score...
            # step (1) : need to find the prob distro of the trained model
            model_prob_distro = get_prob_distr(model_config_file)
            prob_distro_keys = model_prob_distro.keys()
            model_prob_distro_vector = convert_prob_distro_dict_to_array(model_prob_distro, prob_distro_keys)

            # step (2) : then map name to distances
            evalconfig_to_distance_from_model = {}
            for eval,cur_prob_distro in eval_to_prob_dist.iteritems():
                prob_distr_vector = convert_prob_distro_dict_to_array(cur_prob_distro, prob_distro_keys)
                euclidean_dist = np.linalg.norm(model_prob_distro_vector - prob_distr_vector)
                evalconfig_to_distance_from_model[eval] = euclidean_dist

            # step (3) : create x_vals_list and y_vals_list
            angles_method_to_x_vals_list = {}
            angles_method_to_y_vals_list = {}
            for method in methods:
                angles_x_vals_list = []
                angles_y_vals_list = []
                for eval,euclidean_dist in evalconfig_to_distance_from_model.iteritems():
                    #print "euclidean_dist",euclidean_dist
                    angles_x_vals_list.append(euclidean_dist)
                    angles_y_vals_list.append(method_to_eval_to_f1[method][eval])
                    if method not in angles_method_to_rate_to_xlist_ylist:
                        angles_method_to_rate_to_xlist_ylist[method] = {}
                    angles_method_to_rate_to_xlist_ylist[method][exfil_rate] = (angles_x_vals_list, angles_y_vals_list)

        plt.clf()
        fig, ax = plt.subplots()
        ########
        for method in methods:
            x_vals_list = method_to_x_vals_list[method]
            x_vals_list = [i/float(model_xval) for i in x_vals_list]

            y_vals_list = method_to_y_vals_list[method]
            #print "x_vals_list", x_vals_list
            #print "y_vals_list", y_vals_list
            x_vals_list, y_vals_list = zip(*sorted(zip(x_vals_list, y_vals_list)))

            plt.plot(x_vals_list, y_vals_list, marker='.', markersize=22, label=method_to_legend[method])
            #plt.xlabel(xlabel)
            plt.xlabel('Ratio of Test Load to Train Load', fontsize=15)
            plt.ylabel('F1 Score', fontsize=15)
            plt.title('F1 Score vs Load')
            if method not in method_to_rate_to_xlist_ylist:
                method_to_rate_to_xlist_ylist[method] = {}
            method_to_rate_to_xlist_ylist[method][exfil_rate] = (x_vals_list, y_vals_list)
        ax.set_ylim(top=1.1,bottom=-0.1) #  we can do it manually...
        plt.legend()
        plt.show()
        plt.savefig(single_scale_load)
        ###################

    # [continuing on work for angle graphs] step (4) : finally create the graph (probably a scatterplot)
    if type_of_graph == 'angle':
        plt.clf()
        fig, axes = plt.subplots(nrows=1, ncols=len(exfil_rates), figsize=(20, 3.5))
        #fig.suptitle(str(graph_name) + ' f1 vs euclidean distance at various exfil rates')
        for counter, rate in enumerate(exfil_rates):
            for method, angles_rate_to_xlist_ylist in angles_method_to_rate_to_xlist_ylist.iteritems():
                if method != 'ensemble':
                    continue
                x_vals, y_vals = angles_rate_to_xlist_ylist[rate]
                #print "x_vals",x_vals
                #print "y_vals",y_vals
                axes[counter].scatter(x_vals, y_vals, marker='.', label=method_to_legend[method], s=275)
                axes[counter].set_ylim(top=1.1, bottom=-0.1)  # we can do it manually...
                axes[counter].set_ylabel("f1 score")
                axes[counter].set_xlabel("euclidean distance")
                BytesPerMegabyte = 1000000.0
                axes[counter].set_title(str(rate / BytesPerMegabyte) + ' MB/min Exfil')
                axes[counter].legend()
        fig.align_ylabels(axes)
        plt.tight_layout()
        fig.savefig(multi_angle_graph)

    ##  put them all on the same grid + y-axis scale... (I'm just going to do it manually b/c hurry...)
    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=len(exfil_rates), figsize=(50, 10))
    fig.suptitle(str(graph_name) + ' f1 vs load at various exfil rates')
    for counter,rate in enumerate(exfil_rates):
        for method, rate_to_xlist_ylist in method_to_rate_to_xlist_ylist.iteritems():
            x_vals,y_vals = rate_to_xlist_ylist[rate]
            axes[counter].plot(x_vals, y_vals, marker='*', markersize=22, label=method)
            axes[counter].set_ylim(top=1.1,bottom=-0.1) #  we can do it manually...
            axes[counter].set_ylabel("f1 score")
            axes[counter].set_xlabel("number of load generators")
            BytesPerMegabyte = 1000000.0
            axes[counter].set_title(str(rate / BytesPerMegabyte) + ' MB/min Exfil')
            axes[counter].legend()
    fig.align_ylabels(axes)
    fig.savefig(mutli_load_graph)

    # okay, we can leave the first two out... since I can just type it in with my HANDS.
    ## new plan: paper draft. use what we do have: old sock + wordpress. if we can fix hipsterstore, then we can use that.
    ## but i think processing will take too long (to get it by tomorrow...)

    # so just add this: TP  FN  TN  FP  TPR     FPR into a tuple and load the tuples into a list...
    # I guess we'll make one table per exfil rate per application...
    data = []
    for exfil_rate in exfil_rates:
        for evalconfig,xval in eval_configs_to_xvals.iteritems():
            if evalconfig not in  evalconfigs_to_cm:
                continue

            cur_cm = evalconfigs_to_cm[evalconfig][exfil_rate][timegran]
            for method in methods:
                tn, fp, fn, tp = aggregate_cm_vals_over_paths(cur_cm, method=method)
                #print "tp", tp, "fn", fn, "tn", tn, "fp",fp
                tpr = tp / float(tp + fn)
                fpr = fp / float(fp + tn)
                f1_score = (2.0 * tp) / (2.0 * tp + fp + fn)

                exp_name = evalconfig.split('/')[-1]
                cur_vals = (exfil_rate, exp_name, method, tp, fn, tn, fp, tpr, fpr, f1_score)
                data.append( cur_vals )

            ###
    #print "evalconfig",evalconfig,"exfil_rate",exfil_rate, "xval", xval, "\n"
    #data = [ cur_vals ]
    indicator_string = "results_table_for_"
    if new_model:
        indicator_string += 'NEW_' + new_model
    else:
        indicator_string += 'old_model'

    print indicator_string
    print "results_of_eval: " + '(at ' + str(timegran) + ' sec timegran)'
    print(tabulate(data, headers=['exfil_rate', 'exp_name', 'method', 'tp', 'fn', 'tn', 'fp', 'tpr', 'fpr', 'f1_score']))
    print "-----"
    print "\n"


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

            #print "timegran", timegran, type(timegran), type(timegran) != int
            if type(timegran) != int:
                #print "timegran_is_a_str... literal_eval..."
                timegran = ast.literal_eval(timegran)
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

        if 'no_tsl' in config_file:
            no_tsl = config_file['no_tsl']
        else:
            no_tsl = True

        if 'model_xval' in config_file:
            model_xval = config_file['model_xval']
        else:
            model_xval = 100

        decanter_configs = None
        if 'perform_decanter' in config_file:
            if config_file['perform_decanter']:
                try:
                    decanter_configs = {}
                    decanter_configs['train_gen_bro_log'] = config_file['train_gen_bro_log']
                    decanter_configs['test_gen_bro_log'] = config_file['test_gen_bro_log']
                    decanter_configs['gen_fingerprints_p'] = config_file['gen_fingerprints_p']
                    decanter_configs['fraction_of_training_pcap_to_use'] = config_file['fraction_of_training_pcap_to_use']
                    print "decanter_configs_formed", decanter_configs
                except:
                    raise Exception('If you want to peform decanter component, please include all Decanter-related configs!')

        if 'analyze_in_parallel' in config_file:
            analyze_in_parallel = config_file['analyze_in_parallel']
        else:
            analyze_in_parallel = False

    #print "orig_decanter_configs", decanter_configs
    #exit(2)

    return model_config_file, eval_configs_to_xvals, xlabel, use_cached, exfil_rate, timegran, type_of_graph, \
           graph_name, use_remote, remote_server_ips, remote_server_key, user, dont_retrieve_from_remote, no_tsl,\
            model_xval, decanter_configs, analyze_in_parallel

def run_looper(config_file_pth, update_config, use_remote, only_finished_p, live_p, retrain_model_p, min_exfil_rate_model_p,
               per_svc_exfil_model_p, exp_data_dir, use_training_model_from_mem, no_cilium, dont_open_pdfs,
               use_all_results_from_mem, load_old_pipelines):

    model_config_file, eval_configs_to_xvals, xlabel, use_cached, exfil_rate, timegran, type_of_graph, graph_name, \
    use_remote_from_config, remote_ips, remote_server_key, user, dont_retrieve_from_remote, no_tsl, model_xval, \
    decanter_configs, analyze_in_parallel =  parse_config(config_file_pth)

    if use_remote_from_config is not None:
        use_remote = use_remote_from_config or use_remote

    #print("type(eval_configs_to_xvals)", type(eval_configs_to_xvals))
    #exit(233)

    # DON'T FORGET ABOUT use_cached (it's very useful -- especially when iterating on graphs!!)
    use_cached = use_cached or use_all_results_from_mem
    update_config = update_config
    ##use_remote = use_remote
    only_finished_p = False #only_finished_p ## VERY USEFUL

    # retrain_model_p, min_exfil_rate_model_p,
    #                per_svc_exfil_model_p):

    no_tsl = not( (not no_tsl) or min_exfil_rate_model_p )

    evalconfigs_to_cm, evalconfigs_to_model_to_cm = get_evalconfigs_to_cm(model_config_file, eval_configs_to_xvals, xlabel, use_cached, exfil_rate, timegran,
                                              type_of_graph, graph_name, update_config, only_finished_p, retrain_model_p,
                                              per_svc_exfil_model_p, no_tsl=no_tsl, decanter_configs=decanter_configs,
                                              live_p=live_p, analyze_in_parallel=analyze_in_parallel, exp_data_dir=exp_data_dir,
                                               use_training_model_from_mem=use_training_model_from_mem, no_cilium=no_cilium,
                                               dont_open_pdfs=dont_open_pdfs, use_all_results_from_mem=use_all_results_from_mem,
                                               load_old_pipelines=load_old_pipelines)


    # this part handles displaying all of the new models...
    model_to_evalconfig_to_cm = {}
    for evalconfig, model_to_cm in evalconfigs_to_model_to_cm.items():
        for model,cm in model_to_cm.items():
            if len(cm[cm.keys()[0]]) == 0:
                continue
            if model not in model_to_evalconfig_to_cm.keys():
                model_to_evalconfig_to_cm[model] = {}
            model_to_evalconfig_to_cm[model][evalconfig] = cm

    '''
    new_evalconfig_to_rate_to_timegram_to_method_cm = {}
    for model, evalconfig_to_cm in model_to_evalconfig_to_cm.items():
        for evalconfig, rate_to_timegram_to_cm in evalconfig_to_cm.items():
            if evalconfig not in new_evalconfig_to_rate_to_timegram_to_method_cm:
                new_evalconfig_to_rate_to_timegram_to_method_cm[evalconfig] = {}
            for rate, timegram_to_cm in rate_to_timegram_to_cm.items():
                if rate not in new_evalconfig_to_rate_to_timegram_to_method_cm[evalconfig] and rate != {}:
                    new_evalconfig_to_rate_to_timegram_to_method_cm[evalconfig][rate] = {}
                for timegran, cm in timegram_to_cm.items():
                    if timegran not in new_evalconfig_to_rate_to_timegram_to_method_cm[evalconfig][rate]:
                        new_evalconfig_to_rate_to_timegram_to_method_cm[evalconfig][rate][timegran] = {}
                    new_evalconfig_to_rate_to_timegram_to_method_cm[evalconfig][rate][timegran][model] = cm
    '''

    # TODO: finish writing part related to evalconfigs_to_model_to_cm
    ############################################################
    print "\n\n-----"
    print "new model..."

    new_evalconfig_to_rate_to_timegram_to_method_cm = {}
    for model, evalconfig_to_cm in model_to_evalconfig_to_cm.items():
        for evalconfig, rate_to_timegram_to_cm in evalconfig_to_cm.items():
            if evalconfig not in new_evalconfig_to_rate_to_timegram_to_method_cm:
                new_evalconfig_to_rate_to_timegram_to_method_cm[evalconfig] = {}
            for rate, timegram_to_cm in rate_to_timegram_to_cm.items():
                if rate not in new_evalconfig_to_rate_to_timegram_to_method_cm[evalconfig] and rate != {}:
                    new_evalconfig_to_rate_to_timegram_to_method_cm[evalconfig][rate] = {}
                for cur_timegran, cm in timegram_to_cm.items():
                    if type(cur_timegran) != int and type(cur_timegran) != tuple:
                        continue
                    if cur_timegran not in new_evalconfig_to_rate_to_timegram_to_method_cm[evalconfig][rate]:
                        new_evalconfig_to_rate_to_timegram_to_method_cm[evalconfig][rate][cur_timegran] = {}
                    new_evalconfig_to_rate_to_timegram_to_method_cm[evalconfig][rate][cur_timegran][model] = cm

    #print "model_to_evalconfig_to_cm.keys()", model_to_evalconfig_to_cm.keys()
    for model, cur_evalconfig_to_cm in model_to_evalconfig_to_cm.iteritems():
        #generate_graphs(eval_configs_to_xvals, exfil_rate, cur_evalconfig_to_cm, timegran, type_of_graph,
        #                str(graph_name) + '_NEW_MODEL_' + str(model) + '_',
        #                xlabel, model_config_file, False, model_xval, new_model=model, no_methods=True)

        generate_graphs(eval_configs_to_xvals, exfil_rate, new_evalconfig_to_rate_to_timegram_to_method_cm, timegran, type_of_graph,
                        str(graph_name) + '_NEW_MODEL_' + str(model) + '_',
                        xlabel, model_config_file, False, model_xval, new_model = model, no_methods = True)

    turn_nested_results_dict_into_csv(new_evalconfig_to_rate_to_timegram_to_method_cm, eval_configs_to_xvals)

    print "\n\n-----"
    print "old model..."
    generate_graphs(eval_configs_to_xvals, exfil_rate, evalconfigs_to_cm, timegran, type_of_graph, graph_name, xlabel,
                    model_config_file, no_tsl, model_xval)

    #print "model_to_evalconfig_to_cm",model_to_evalconfig_to_cm

    ############################################################


    return eval_configs_to_xvals, exfil_rate, evalconfigs_to_cm, model_config_file

def turn_nested_results_dict_into_csv(new_evalconfig_to_rate_to_timegram_to_method_cm, eval_configs_to_xvals,
                                      output_dir='./multilooper_outs/'):
    print "starting_to_run_turn_nested_results_dict_into_csv..."

    results_df = None
    results_df_two = None
    for new_evalconfig, rate_to_timegran_to_method_cm in new_evalconfig_to_rate_to_timegram_to_method_cm.iteritems():
        # PLAN: MAKE A SINGLE row in df and then append rows: https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
        evalconfig_name = new_evalconfig.split('/')[-1]
        for rate, timegran_to_method_cm in rate_to_timegran_to_method_cm.iteritems():
            for timegran, method_cm in timegran_to_method_cm.iteritems():
                cur_f1_dict = {}
                for method, performnace_vals in method_cm.iteritems():
                    current_dict_two = {'exp_name': evalconfig_name, 'exfil_rate': rate, 'timegran': timegran, 'model_name': method}
                    current_dict_two.update(performnace_vals)

                    current_dict_two_cols = ['exp_name', 'exfil_rate', 'timegran']
                    current_dict_two_cols_other = [i for i in current_dict_two.keys() if i not in current_dict_two_cols]
                    current_dict_two_cols = current_dict_two_cols + current_dict_two_cols_other

                    if results_df_two is None:
                        results_df_two = pd.DataFrame(current_dict_two, columns=current_dict_two_cols, index=[0])
                    else:
                        results_df_two = results_df_two.append(current_dict_two, ignore_index=True)

                    tn, fp, fn, tp = aggregate_cm_vals_over_paths(performnace_vals)
                    f1_score = (2.0 * tp) / (2.0 * tp + fp + fn)
                    cur_f1_dict[method] = f1_score

                #print "eval_configs_to_xvals", eval_configs_to_xvals
                cur_load_level = eval_configs_to_xvals[new_evalconfig]
                current_dict = {'exp_name': evalconfig_name, 'exfil_rate': rate, 'timegran': timegran,
                                'load_level': cur_load_level}
                current_dict.update(cur_f1_dict)

                current_dict_cols = ['exp_name', 'exfil_rate', 'timegran']
                current_dict_cols_other = [i for i in current_dict.keys() if i not in current_dict]
                current_dict_cols = current_dict_cols + current_dict_cols_other
                current_dict_cols.append('load_level')

                # cm_to_f1(cm, exfil_rate, timegran,method=None)
                if results_df is None:
                    print "current_dict", current_dict
                    results_df = pd.DataFrame(current_dict, columns=current_dict_cols, index=[0])
                else:
                    results_df = results_df.append(current_dict, ignore_index=True)

    # TODO: save results_df and results_df_two somewhere meaningful...
    # TODO:
    #   (a) change order of columns to make sense for the given CSVs
    #   (b) change order of rows to make sense for the given CSVs
    results_df = results_df.sort_values(by=['timegran', 'exfil_rate', 'load_level'], ascending=[True, False, False])
    results_df.to_csv(output_dir + 'model_comparison.csv')
    results_df_two.to_csv(output_dir + 'timegran_comparison.csv')


if __name__=="__main__":
    print "RUNNING"

    parser = argparse.ArgumentParser(description='This can run multiple experiments in a row on MIMIR. Also makes graphs')
    parser.add_argument('--config_json', dest='config_json', default=None,
                        help='this is the configuration file used to run to loop through several experiments')

    parser.add_argument('--dont_update_config', dest='dont_update_config',
                        default=False, action='store_true')

    parser.add_argument('--use_remote_server', dest='use_remote',
                        default=False, action='store_true')

    parser.add_argument('--only_use_finished_exps', dest='only_finished_p',
                        default=False, action='store_true')

    parser.add_argument('--live', dest='live_p',
                        default=False, action='store_true')

    parser.add_argument('--retrain_model', dest='retrain_model_p',
                        default=False, action='store_true')

    parser.add_argument('--min_exfil_rate_model', dest='min_exfil_rate_model_p',
                        default=False, action='store_true',
                        help='return the min_exfil_rate_model (in which data is treated as a time series')

    parser.add_argument('--per_svc_exfil_model', dest='per_svc_exfil_model_p',
                        default=False, action='store_true',
                        help='[does not do anything, no reason to include this (not removing b/c might be convenient in the future...]')

    parser.add_argument('--exp_data_dir', dest='exp_data_dir', default=None,
                        help='if the experiment directory differs from the one listed in the config file, you can specify it here (useful for running locally)')

    parser.add_argument('--use_training_model_from_mem', dest='use_training_model_from_mem',
                        default=False, action='store_true',
                        help='get the end result of the training model from memory (useful for running locally)')

    parser.add_argument('--no_cilium', dest='no_cilium', default=False, action='store_true',
                        help='(for dev purposes) treats the config files as if they said not to do the cilium stuff')

    parser.add_argument('--dont_open_pdfs', dest='dont_open_pdfs', default=False, action='store_true',
                        help='(for dev purposes) no matter what it says in the config files, do not open the pdf files')

    parser.add_argument('--use_all_results_from_mem', dest='use_all_results_from_mem', default=False, action='store_true',
                        help='(for dev purposes) no matter what it says in the config files, treat it as if it wants to '
                             'use the endresults already in memory...')

    parser.add_argument('--load_old_pipelines', dest='load_old_pipelines', default=False, action='store_true',
                        help='[for dev purposes] loads the old pipelines (from statistical_analysis.py), so that the new one can be tested more easily')


    args = parser.parse_args()

    if not args.config_json:
        #############config_file_pth = "./multi_experiment_configs/wordpress_scale.json"
        #config_file_pth = "./analysis_pipeline/multi_experiment_configs/old_sockshop_angle_remote2.json"
        #config_file_pth = "./multi_experiment_configs/old_sockshop_scale.json"
        #############config_file_pth = "./multi_experiment_configs/old_sockshop_angle.json"
        #############config_file_pth = "./multi_experiment_configs/new_sockshop_angle.json"
        #config_file_pth = "./analysis_pipeline/multi_experiment_configs/sockshop_test_remote.json"
        ################config_file_pth = "./multi_experiment_configs/new_sockshop_scale.json"
        #config_file_pth = "./multi_experiment_configs/hipsterStore_scale.json"
        config_file_pth = "/Volumes/exM/tsting_e2e_repo/configs/new_sockshop_scale.json"
        ###config_file_pth = "../configs_to_reproduce_results/Data_Analysis/Sockshop/Scale/new_sockshop_scale.json"
    else:
        config_file_pth = args.config_json

    run_looper(config_file_pth, (not args.dont_update_config), args.use_remote, args.only_finished_p, args.live_p,
               args.retrain_model_p, args.min_exfil_rate_model_p, args.per_svc_exfil_model_p, args.exp_data_dir,
               args.use_training_model_from_mem, args.no_cilium, args.dont_open_pdfs, args.use_all_results_from_mem,
               args.load_old_pipelines)