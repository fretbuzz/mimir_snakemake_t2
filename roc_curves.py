import matplotlib.pyplot as plt
import pickle
from analyze_traffix_matrixes import join_dfs, eigenspace_detection_loop, calc_tp_fp_etc
import numpy as np

#algo = 'control charts'
#algo = '3-tier control charts'

#### TODO: these are NOT proper ROC curves!!!
#### proper ROC curves would FIX the exfiltration rate
#### and just vary the parameters of the algorithm!!
# all_experimental_results is of the following form
# [(rep, exfil_amt)] = exp_results
# exp_results is in the form of
# {"FPR": number} (also TPR, etc.)
# TPR vs FPR
def roc_charts(all_experimental_results):
    tpr = []
    fpr = []
    tpr_fpr = []
    plt.figure(1)
    #print "exp results:",all_experimental_results
    for exp in all_experimental_results.values():
        #print exp[algo]  
        tpr_fpr.append( (exp[algo]['TPR'], exp[algo]['FPR'])  )
        #tpr.append( exp[algo]['TPR'] )
        #fpr.append( exp[algo]['FPR'] )

    fpr = sorted(list(set([fpr[1] for fpr in tpr_fpr])))

    print tpr_fpr
    for rate in fpr:
        tpr_total = 0
        total_rates = 0
        for item in tpr_fpr:
            if item[1] == rate:
                tpr_total += item[0]
                total_rates += 1
        tpr.append(float(tpr_total) / total_rates)

    print "tpr", tpr
    print "fpr", fpr
    roc_line,  = plt.plot(fpr, tpr)
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    #plt.show()

# x-axis: exfil_rate
# y_axis: tp rate
# see above function for format of arg
def tp_vs_exfil_rate(all_experimental_results, algo_name, algo):
    print all_experimental_results
    tpr = []
    fpr = []
    precision = []
    exfil_rate = []
    tpr_exfil = []
    fpr_exfil = []
    prec_exfil = []
    #plt.figure(2)
    for exp_settings, exp_results in all_experimental_results.iteritems():
        #print exp_settings, exp_results[algo]
        #tpr.append(exp_results[algo]['TPR'])
        #exfil_rate.append( exp_settings[1] )
        tpr_exfil.append( (exp_results[algo]['TPR'], exp_settings[1])  )
        fpr_exfil.append( (exp_results[algo]['FPR'], exp_settings[1])  )
        prec_exfil.append( (exp_results[algo]['Precision'], exp_settings[1]))

    exfil_rate = sorted(list(set([exfil[1] for exfil in tpr_exfil])))
    print exfil_rate 

    for rate in exfil_rate:
        tpr_total = 0
        fpr_total = 0
        total_rates = 0
        total_fpr_rates = 0
        precision_total = 0
        total_precision_rates = 0
        for item in tpr_exfil:
            if item[1] == rate:
                tpr_total += item[0]
                total_rates += 1
        for item in fpr_exfil:
            if item[1] == rate:
                fpr_total += item[0]
                total_fpr_rates += 1
        for item in prec_exfil:
            if item[1] == rate:
                precision_total += item[0]
                total_precision_rates += 1
        precision.append( float(precision_total) / total_precision_rates)
        tpr.append(float(tpr_total) / total_rates)
        fpr.append(float(fpr_total) / total_fpr_rates)

    tp_line, = plt.plot(exfil_rate, tpr, label = "TP Rate")
    fp_line, = plt.plot(exfil_rate, fpr, label = "FP Rate")
    f1_exfil = []
    for i in range(0, len(precision)):
        prec = precision[i]
        tp_rate = tpr[i]
        exfil = exfil_rate[i]
        if (prec + tp_rate) != 0:
            f1_exfil.append(((2 * (prec * tp_rate) / (prec + tp_rate)),exfil))
    
    #f1_line, = plt.plot([exfil[1] for exfil in f1_exfil],[f1[0] for f1 in f1_exfil], label = "f1")
    print "tpr", tpr
    print "exfil rate", exfil_rate
    plt.title(algo_name)
    plt.xlabel('amount exfiltrated in 5 sec period')
    plt.ylabel('rate')
    plt.legend(handles=[tp_line, fp_line])#, f1_line])
    plt.show()

    return tpr, fpr, exfil_rate
    ## f1-score = 2 * (precision * TPR) / (Precision + TPR)


### note: pickle_files should be a list of pickle files that I want to run this on
### here is some calling code:
###         python -c "from roc_curves import actual_roc; actual_roc()"
# parameter is if should perform the calculation right now (alternative would be to read from dict)
# calc_p is an integer
def actual_roc( ):
    # this function is just for tuning some parameters manually.
    # so I am just going to hardcode stuff b/c ATM I am not planning on
    # repurposing this for any additional work
    pickle_path = './experimental_data/test_15_long_small_incr/'
    pickle_files = [('rec_matrix_increm_5_rep_0.pickle', 'sent_matrix_increm_5_rep_0.pickle'),
                    ('rec_matrix_increm_5_rep_1.pickle', 'sent_matrix_increm_5_rep_1.pickle'),
                    ('rec_matrix_increm_5_rep_2.pickle', 'sent_matrix_increm_5_rep_2.pickle')]
    exfils = {80 : 0, 180 : 0}
    exp_time = 360
    start_analyze_time = 30

    relevant_tms = []
    for pf in pickle_files:
        relevant_tms.append( (pickle.load( open( pickle_path + pf[0], "rb" )), 
                            pickle.load( open( pickle_path + pf[1], "rb" )))  )
    
    joint_dfs = []
    for df_rec_sent in relevant_tms:
        joint_dfs.append( join_dfs(df_rec_sent[1], df_rec_sent[0]) )
    
    # now we have a list of the joint dfs. Lets loop through the params we have
    # available for the eigen-space based detection system
    experiment_results = {}
    i = 0
    for joint_df in joint_dfs:
        # let's try looping over a parameter!
        win_sz = 5
        beta_val = 0.05
        for crit_perc in np.arange(0,1, 0.005):
            print "critical percentage:", crit_perc
            eigenspace_warning_times = eigenspace_detection_loop(joint_df, critical_percent=crit_perc,
                    window_size= win_sz, beta = beta_val )
            eigenspace_performance_results = calc_tp_fp_etc(("eigenspace", i, crit_perc, win_sz, beta_val), exfils, eigenspace_warning_times, exp_time, start_analyze_time)
            #print eigenspace_performance_results
            experiment_results.update(eigenspace_performance_results)
            #print experiment_results 

        crit_perc = 0.01 # choosing arbitrarily for now
        for win_size in range(0,10):
            eigenspace_warning_times = eigenspace_detection_loop(joint_df, critical_percent=crit_perc,
                    window_size= win_size, beta = beta_val )
            eigenspace_performance_results = calc_tp_fp_etc(("eigenspace", i, crit_perc, win_size, beta_val), exfils, eigenspace_warning_times, exp_time, start_analyze_time)
            #print eigenspace_performance_results
            experiment_results.update(eigenspace_performance_results)
            #print experiment_results 

        for beta_value in np.arange(0, 1, 0.01):
            eigenspace_warning_times = eigenspace_detection_loop(joint_df, critical_percent=crit_perc,
                    window_size= win_sz, beta = beta_value )
            eigenspace_performance_results = calc_tp_fp_etc(("eigenspace", i, crit_perc, win_sz, beta_value), exfils, eigenspace_warning_times, exp_time, start_analyze_time)
            #print eigenspace_performance_results                                                
            experiment_results.update(eigenspace_performance_results)                                                            
            #print experiment_results 
        
        i += 1
    
    pickle.dump(experiment_results, 
            open( pickle_path + 'eigenspace_roc_increm_5.pickle', "wb" ) )
 
    return experiment_results

# example of how to call:
#   python -c "from roc_curves import graph_roc; graph_roc(None, './experimental_data/test_15_long_small_incr/eigenspace_roc_increm_5.pickle' )"
#  results_dict; is either passed from results of actual_roc() or is just None 
# if results_dict is real, then can just pass an empty string for roc_pickle_path
def graph_roc( results_dict, roc_pickle_path ):
    if not results_dict:
        results_dict = pickle.load( open( roc_pickle_path, "rb" ) )
    
    print results_dict

    # I am currently setting it up to take a gander at how varying crit_pert affects results
    # y-axis -> rate ; x-axis -> value of crit_pert
    # then I'd want to stick the values into the maplotlib graph generator function 
    crit_pert_to_rate = {} # val is (rate, contributers), where contributer is how many enteries contributed
    for key, val in results_dict.iteritems():
        print 'key', key, 'val', val
        if key[0] == "eigenspace" and key[3] == 5 and key[4] == 0.05:
            print "this value is under consideration"
            if key[2] in crit_pert_to_rate:
                print "this value already exists!"
                old_tpr = crit_pert_to_rate[key[2]][0][0]
                old_fpr = crit_pert_to_rate[key[2]][0][1]
                old_contrib = crit_pert_to_rate[key[2]][1]
                new_tpr = (old_tpr * old_contrib + val['TPR']) / (old_contrib + 1)
                new_fpr = (old_fpr * old_contrib + val['FPR']) / (old_contrib + 1)
                crit_pert_to_rate[key[2]] = ((new_tpr, new_fpr), old_contrib + 1)
            else:
                print "this value does not already exist; I am going to add it"
                crit_pert_to_rate[key[2]] = ((val['TPR'], val['FPR']), 1)
    
    print 'Mapping', crit_pert_to_rate
    crit_percents = []
    tpr = []
    fpr = []
    for key in sorted(crit_pert_to_rate):
        crit_percents.append(key)
        tpr.append(crit_pert_to_rate[key][0][0])
        fpr.append(crit_pert_to_rate[key][0][1])
            
        # print "%s: %s" % (key, mydict[key])

    #crit_percents = [crit_percent for crit_percent in crit_pert_to_rate.keys()]
    #tpr = [rates[0][0] for rates  in crit_pert_to_rate.values()]
    #fpr = [rates[0][1] for rates  in crit_pert_to_rate.values()]
    tp_line, = plt.plot(crit_percents, tpr, label = "TP Rate")
    fp_line, = plt.plot(crit_percents, fpr, label = "FP Rate")
    plt.legend(handles=[ tp_line, fp_line])
    
    plt.title("kinda roc")
    plt.xlabel('critical percentage value')
    plt.ylabel('rate')
    
    plt.show()
        

def load_exp(all_exp_results_loc):
    all_experimental_results = pickle.load( open( all_exp_results_loc, "rb" ) )
    #roc_charts(all_experimental_results)
    #plt.figure(2)
    #algo = 'control charts'
    #algo = '3-tier control charts'
    ms_tpr_eig, ms_fpr_eig, ms_exfil_rate_eig = tp_vs_exfil_rate(all_experimental_results, "MS Eigenspace", 'eigenspace')
    tt_tpr_eig, tt_fpr_eig, tt_exfil_rate_eig = tp_vs_exfil_rate(all_experimental_results, "TT Eigenspace", 'tt_eigenspace')
    ms_tpr, ms_fpr, ms_exfil_rate = tp_vs_exfil_rate(all_experimental_results, "Microservice Architecture", 'control charts')
    tt_tpr, tt_fpr, tt_exfil_rate = tp_vs_exfil_rate(all_experimental_results, "3-tier Architecture", '3-tier control charts')

    plt.figure(3)
    #print me_trp
    ms_tp_line_eig, = plt.plot(ms_exfil_rate_eig, ms_tpr_eig, label = "MS eig")
    tt_tp_line_eig, = plt.plot(tt_exfil_rate_eig, tt_tpr_eig, label = "3T eig")
    ms_tp_line, = plt.plot(ms_exfil_rate, ms_tpr, label = "Microservice")
    tt_tp_line, = plt.plot(tt_exfil_rate, tt_tpr, label = "3-Tier")
    plt.title("TP Rate")
    plt.xlabel('Bytes exfiltrated in 5 sec period')
    plt.ylabel('TP Rate')
    plt.legend(handles=[ms_tp_line, tt_tp_line, ms_tp_line_eig, tt_tp_line_eig])#, f1_line])
    plt.savefig('TP Rates' + '.png', bbox_inches='tight')
    plt.show()

    ms_fp_line, = plt.plot(ms_exfil_rate, ms_fpr, label = "Microservice")
    tt_fp_line, = plt.plot(tt_exfil_rate, tt_fpr, label = "3-Tier")
    ms_fp_line_eig = plt.plot(ms_exfil_rate_eig, ms_fpr_eig, label = "MS eig")
    tt_fp_line_eig = plt.plot(tt_exfil_rate_eig, tt_fpr_eig, label = "3T eig")
    
    plt.title("FP Rate")
    plt.xlabel('Bytes exfiltrated in 5 sec period')
    plt.ylabel('FP Rate')
    plt.legend(handles=[ms_fp_line, tt_fp_line, ms_tp_line_eig, tt_tp_line_eig])#, f1_line])
    plt.savefig('FP Rates' + '.png', bbox_inches='tight')
    plt.show()

