from mimir import run_analysis
import pickle
import matplotlib.pyplot as plt
import argparse
import json
from collections import OrderedDict

def get_eval_results(model_config_file, list_of_eval_configs):
    eval_config_to_cm = {}
    for eval_config in list_of_eval_configs:
        eval_cm = run_analysis(model_config_file, eval_config=eval_config)
        eval_config_to_cm[eval_config] = eval_cm
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
                      type_of_graph, graph_name):
    if use_cached:
        with open('./temp_outputs/cached_looper.pickle', 'r') as f:
            evalconfigs_to_cm = pickle.loads(f.read())
    else:
        evalconfigs_to_cm = get_eval_results(model_config_file, eval_configs_to_xvals.keys())
        with open('./temp_outputs/cached_looper.pickle', 'w') as f:
            f.write(pickle.dumps(evalconfigs_to_cm))

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
        plt.savefig('./temp_outputs/' + graph_name + '_' + str(exfil_rate) + '.png')

def parse_config(config_file_pth):
    with open(config_file_pth) as f:
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

    return model_config_file, eval_configs_to_xvals, xlabel, use_cached, exfil_rate, timegran, type_of_graph, graph_name

def run_looper(config_file_pth):
    model_config_file, eval_configs_to_xvals, xlabel, use_cached, exfil_rate, timegran, type_of_graph, graph_name = \
        parse_config(config_file_pth)

    #print("type(eval_configs_to_xvals)", type(eval_configs_to_xvals))
    #exit(233)

    # DON'T FORGET ABOUT use_cached (it's very useful -- especially when iterating on graphs!!)
    use_cached = use_cached
    create_eval_graph(model_config_file, eval_configs_to_xvals, xlabel, use_cached, exfil_rate, timegran,
                      type_of_graph, graph_name)


if __name__=="__main__":
    print "RUNNING"

    ### Okay, so the key here is to
    ## TODO: use tabulate to make table (should be easy enough...)
    ## TODO: add vector support (euclidean disance perhaps??? probably...)
    ### so what is the plan??
    ### (a) add config files!!!! [done]
    ### (b) add a params that specifies the type of graph/table [done]
    ### (c) add support for these graphs/tables [TODO --- and it's the hard part]

    parser = argparse.ArgumentParser(description='This can run multiple experiments in a row on MIMIR. Also makes graphs')
    parser.add_argument('--config_json', dest='config_json', default=None,
                        help='this is the configuration file used to run to loop through several experiments')
    args = parser.parse_args()

    if not args.config_json:
        #config_file_pth = "./multi_experiment_configs/wordpress_scale.json"
        config_file_pth = "./multi_experiment_configs/old_sockshop_angle.json"
    else:
        config_file_pth = args.config_json

    run_looper(config_file_pth)