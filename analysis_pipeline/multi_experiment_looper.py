from mimir import run_analysis
import pickle
import matplotlib.pyplot as plt


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

def create_eval_graph(model_config_file, eval_configs_to_xvals, xlabel, use_cached, exfil_rate, timegran):
    if use_cached:
        with open('./temp_outputs/cached_looper.pickle', 'r') as f:
            evalconfigs_to_cm = pickle.loads(f.read())
    else:
        evalconfigs_to_cm = get_eval_results(model_config_file, eval_configs_to_xvals.keys())
        with open('./temp_outputs/cached_looper.pickle', 'w') as f:
            f.write(pickle.dumps(evalconfigs_to_cm))

    x_vals_list = []
    y_vals_list = []
    for evalconfig,xval in eval_configs_to_xvals.iteritems():
        x_vals_list.append(xval)
        cur_cm = evalconfigs_to_cm[evalconfig]
        optimal_f1 = cm_to_f1(cur_cm, exfil_rate, timegran)
        y_vals_list.append( optimal_f1 )

        ## (step1) cache the results from get_eval_results (b/c gotta iterate on steps2&3) [[[done]]]
        ## (step2) put process cms (to get F1 scores)
        ## (step3) make actual graphs (can just stick into temp_outputs for now... I gues...)

    print "x_vals_list", x_vals_list
    print "y_vals_list", y_vals_list
    x_vals_list, y_vals_list = zip(*sorted(zip(x_vals_list, y_vals_list)))

    plt.clf()
    plt.plot(x_vals_list, y_vals_list, marker='.')
    plt.xlabel(xlabel)
    plt.ylabel('f1 score')
    plt.show()
    plt.savefig('./temp_outputs/aaa.png')

#eval_results = run_analysis('./analysis_json/sockshop_mk13.json', eval_config='./new_analysis_json/sockshop_mk22.json')
# run_analysis('./analysis_json/sockshop_mk13.json', eval_config = './new_analysis_json/sockshop_mk23.json')
# run_analysis('./analysis_json/sockshop_mk13.json', eval_config = './new_analysis_json/sockshop_mk24.json')
# run_analysis('./analysis_json/sockshop_mk13.json', eval_config = './new_analysis_json/sockshop_mk20.json')

if __name__=="__main__":
    print "RUNNING"
    #'''
    model_config_file = './analysis_json/sockshop_mk13.json'
    eval_configs_to_xvals = {'./new_analysis_json/sockshop_mk22.json': 60,
                             './new_analysis_json/sockshop_mk20.json' : 120,
                             './new_analysis_json/sockshop_mk23.json': 80,
                             './new_analysis_json/sockshop_mk24.json': 140,
                             './new_analysis_json/sockshop_auto_mk27.json' : 100}
                             #'./new_analysis_json/sockshop_auto_mk27.json': 100 } ## TODO: need to process this!!
    #'''
    '''
    model_config_file = './analysis_json/wordpress_one_3_auto_mk5.json'
    eval_configs_to_xvals = {'./new_analysis_json/wordpress_mk10.json' : 45,
                             './new_analysis_json/wordpress_mk22.json' : 65,
                             './new_analysis_json/wordpress_mk24.json' : 85}
    #'''
    ## TODO: reconfigure for the probability distributions
    '''
    model_config_file = './analysis_json/sockshop_mk13.json'
    eval_configs_to_xvals = {'./new_analysis_json/sockshop_three_mk5.json' : 100,
                             './new_analysis_json/sockshop_three_mk6.json' : 100,
                             './new_analysis_json/sockshop_three_mk7.json' : 100,
                             './new_analysis_json/sockshop_three_mk8.json' : 100,
                             './new_analysis_json/sockshop_three_mk9.json' : 100,
                             './new_analysis_json/sockshop_three_mk10.json' : 100,
                             './new_analysis_json/sockshop_three_mk11.json' : 100,
                             './new_analysis_json/sockshop_three_mk12.json' : 100,
                             './new_analysis_json/sockshop_three_mk13.json' : 100}
    '''

    xlabel = 'load (# instances of load generation)'
    use_cached = False
    exfil_rate = 1000000.0
    timegran = 10
    create_eval_graph(model_config_file, eval_configs_to_xvals, xlabel, use_cached, exfil_rate, timegran)