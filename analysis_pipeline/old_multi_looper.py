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
                     remote_server_key=None, user=None, dont_retrieve_from_remote=None, only_finished_p=False, no_tsl=False):
    eval_config_to_cm = {}
    for eval_config in list_of_eval_configs:
        if not use_remote:
            if only_finished_p:
                if has_experiment_already_been_run(eval_config):
                    eval_cm = run_analysis(model_config_file, eval_config=eval_config, no_tsl=no_tsl)
                else:
                    continue  # don't want to wait ---> so just pass over this one.
                pass
            else:
                ## TODO: probably wanna wrap in a call to multiprocess, to prevent problems with
                ## memory size and using swap space...
                eval_cm = run_analysis(model_config_file, eval_config=eval_config, no_tsl=no_tsl)
        else:
            exit(322) # how would it even get here??

        eval_config_to_cm[eval_config] = eval_cm
        ## modify the config file so that you don't redo previously done experiments...
        if update_config:
            update_config_file(eval_config, if_trained_model=False)
            update_config_file(model_config_file, if_trained_model=True)

    return eval_config_to_cm

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
    if method:
        cm = cm[exfil_rate][timegran][method]
    else:
        cm = cm[exfil_rate][timegran]
    print "cm",cm
    tn, fp, fn, tp = aggregate_cm_vals_over_paths(cm)
    f1_score = (2.0 * tp) / (2.0 * tp + fp + fn)
    return f1_score

def create_eval_graph(model_config_file, eval_configs_to_xvals, xlabel, use_cached, exfil_rates, timegran,
                      type_of_graph, graph_name, update_config_p, only_finished_p, use_remote=False,
                      remote_server_ip=None, remote_server_key=None,user=None, dont_retrieve_from_remote=None,
                      no_tsl = False):
    if use_cached:
        with open('./temp_outputs/' + graph_name + '_cached_looper.pickle', 'r') as f:
            evalconfigs_to_cm = pickle.loads(f.read())
    else:
        evalconfigs_to_cm = get_eval_results(model_config_file, eval_configs_to_xvals.keys(), update_config_p, use_remote,
                                             remote_server_ip=remote_server_ip, remote_server_key=remote_server_key,
                                             user=user, dont_retrieve_from_remote=dont_retrieve_from_remote,
                                             only_finished_p=only_finished_p, no_tsl=no_tsl)
        with open('./temp_outputs/' + graph_name + '_cached_looper.pickle', 'w') as f:
            f.write(pickle.dumps(evalconfigs_to_cm))

    return evalconfigs_to_cm

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
    with open(processing_config, 'r') as g:
        config_stuff = json.loads(g.read())
        print "file_made_by_experimental_apparatus", config_stuff['exp_config_file']
        with open(config_stuff['exp_config_file'], 'r') as z:
            z_cont = z.read()
            print "g_cont", z_cont, len(z_cont)
            exp_config = json.loads(z_cont)
    return exp_config['prob_distro']

def convert_prob_distro_dict_to_array(prob_distro_dict, prob_distro_keys):
    prob_distro_list = []
    for cur_key in prob_distro_keys:
        prob_distro_list.append(prob_distro_dict[cur_key])
    prob_distro_vector = np.array(prob_distro_list)
    return prob_distro_vector

def generate_graphs(eval_configs_to_xvals, exfil_rates, evalconfigs_to_cm, timegran, type_of_graph, graph_name,
                    xlabel, model_config_file, no_tsl=False):
    method_to_rate_to_xlist_ylist = {}
    methods = evalconfigs_to_cm[eval_configs_to_xvals.keys()[0]][exfil_rates[0]][timegran].keys()
    angles_method_to_rate_to_xlist_ylist = {}

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
                optimal_f1 = cm_to_f1(cur_cm, exfil_rate, timegran, method=method)
                y_vals_list.append( optimal_f1 )

                if method not in method_to_eval_to_f1:
                    method_to_eval_to_f1[method] = {}
                method_to_eval_to_f1[method][evalconfig] = optimal_f1


                if type_of_graph == 'angle':
                    prob_distro = get_prob_distr(evalconfig)
                    eval_to_prob_dist[evalconfig] = prob_distro
                print "eval_to_prob_dist",eval_to_prob_dist

                ## (step1) cache the results from get_eval_results (b/c gotta iterate on steps2&3) [[[done]]]
                ## (step2) put process cms (to get F1 scores)
                ## (step3) make actual graphs (can just stick into temp_outputs for now... I gues...)

            method_to_x_vals_list[method] = x_vals_list
            method_to_y_vals_list[method] = y_vals_list


        print "type_of_graph",type_of_graph
        if type_of_graph == 'angle':
            print "---angle--"
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
        for method in methods:
            x_vals_list = method_to_x_vals_list[method]
            y_vals_list = method_to_y_vals_list[method]
            print "x_vals_list", x_vals_list
            print "y_vals_list", y_vals_list
            x_vals_list, y_vals_list = zip(*sorted(zip(x_vals_list, y_vals_list)))

            plt.plot(x_vals_list, y_vals_list, marker='.', markersize=22, label=method)
            plt.xlabel(xlabel)
            plt.ylabel('f1 score')
            if method not in method_to_rate_to_xlist_ylist:
                method_to_rate_to_xlist_ylist[method] = {}
            method_to_rate_to_xlist_ylist[method][exfil_rate] = (x_vals_list, y_vals_list)
        plt.legend()
        plt.show()
        plt.savefig(single_scale_load)

    # [continuing on work for angle graphs] step (4) : finally create the graph (probably a scatterplot)
    if type_of_graph == 'angle':
        plt.clf()
        fig, axes = plt.subplots(nrows=1, ncols=len(exfil_rates), figsize=(50, 10))
        fig.suptitle(str(graph_name) + ' f1 vs euclidean distance at various exfil rates')
        for counter, rate in enumerate(exfil_rates):
            for method, angles_rate_to_xlist_ylist in angles_method_to_rate_to_xlist_ylist.iteritems():
                x_vals, y_vals = angles_rate_to_xlist_ylist[rate]
                #print "x_vals",x_vals
                #print "y_vals",y_vals
                axes[counter].scatter(x_vals, y_vals, marker='*', label=method, s=700)
                axes[counter].set_ylim(top=1.1, bottom=-0.1)  # we can do it manually...
                axes[counter].set_ylabel("f1 score")
                axes[counter].set_xlabel("euclidean distance")
                BytesPerMegabyte = 1000000.0
                axes[counter].set_title(str(rate / BytesPerMegabyte) + ' MB/min Exfil')
                axes[counter].legend()
        fig.align_ylabels(axes)
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

            print "timegran", timegran, type(timegran), type(timegran) != int
            if type(timegran) != int:
                print "timegran_is_a_str... literal_eval..."
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


    return model_config_file, eval_configs_to_xvals, xlabel, use_cached, exfil_rate, timegran, type_of_graph, \
           graph_name, use_remote, remote_server_ips, remote_server_key, user, dont_retrieve_from_remote, no_tsl

def run_looper(config_file_pth, update_config, use_remote, only_finished_p):

    model_config_file, eval_configs_to_xvals, xlabel, use_cached, exfil_rate, timegran, type_of_graph, graph_name, \
    use_remote_from_config, remote_ips, remote_server_key, user, dont_retrieve_from_remote, no_tsl = parse_config(config_file_pth)

    if use_remote_from_config is not None:
        use_remote = use_remote_from_config or use_remote

    #print("type(eval_configs_to_xvals)", type(eval_configs_to_xvals))
    #exit(233)

    # DON'T FORGET ABOUT use_cached (it's very useful -- especially when iterating on graphs!!)
    use_cached = False #use_cached # TODO
    update_config = update_config
    ##use_remote = use_remote
    only_finished_p = False #only_finished_p ## VERY USEFUL

    evalconfigs_to_cm = create_eval_graph(model_config_file, eval_configs_to_xvals, xlabel, use_cached, exfil_rate, timegran,
                                        type_of_graph, graph_name, update_config, only_finished_p, no_tsl=no_tsl)

    generate_graphs(eval_configs_to_xvals, exfil_rate, evalconfigs_to_cm, timegran, type_of_graph, graph_name, xlabel,
                    model_config_file, no_tsl)

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

    args = parser.parse_args()

    if not args.config_json:
        config_file_pth = "./multi_experiment_configs/wordpress_scale.json"
        #config_file_pth = "./analysis_pipeline/multi_experiment_configs/old_sockshop_angle_remote2.json"
        #config_file_pth = "./multi_experiment_configs/old_sockshop_scale.json"
        ####config_file_pth = "./multi_experiment_configs/old_sockshop_angle.json"
        #config_file_pth = "./analysis_pipeline/multi_experiment_configs/sockshop_test_remote.json"
        #########config_file_pth = "./multi_experiment_configs/new_sockshop_scale.json"
    else:
        config_file_pth = args.config_json

    run_looper(config_file_pth, (not args.dont_update_config), args.use_remote, args.only_finished_p)