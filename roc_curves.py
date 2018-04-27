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
        beta_val = 0.215
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
        if key[0] == "eigenspace" and key[3] == 5 and key[4] == 0.215:
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

    plt.figure(1)
    #crit_percents = [crit_percent for crit_percent in crit_pert_to_rate.keys()]
    #tpr = [rates[0][0] for rates  in crit_pert_to_rate.values()]
    #fpr = [rates[0][1] for rates  in crit_pert_to_rate.values()]
    tp_line, = plt.plot(crit_percents, tpr, label = "TP Rate")
    fp_line, = plt.plot(crit_percents, fpr, label = "FP Rate")
    plt.legend(handles=[ tp_line, fp_line])
    
    plt.title("kinda roc")
    plt.xlabel('critical percentage value')
    plt.ylabel('rate')
    
    ### the above results make no sense (how does decreasing the thresh not have a stronger effect on 
    ### the FPR). The only conclusion that makes sense is that the algo is taking too long too reach 'steady
    ### state' with the current discount factor. Therefore alls the values seem very strange to the algo, so it keeps
    ### saying that an alert should be triggered.
    ### I am going to need to go w/ an unreasonably high discount factor (i.e. beta) b/c of the longer run_time
    ### note the second moment is just the variance (of z in this case). maybe I should modify the function
    ### to calc the var off of the collected data... I don't understand the math in that paper, and I am not
    ### going to pretend that I do. Just stick to modifying the parameters

    ### TODO: (1) per the above discussion, the beta value needs to be examined. First, get a graph of that going. 
    ### might want to use the code above as a starting point.
    ### (2) then the window size needs to be examined. Get a graph of that going. Try messing with the values and whatnot. See
    ### what happens. PCA is off the table as for the poster. 
    ### (3) Rerun analyze_traffic_matrix with the improved parameters (both w/ old data and new data)
    ### even if they do not really work, hey, it's something.
    ### (4) create graphs for poster
    ### (1) - (3) NEEDS to be done on Friday. (4) can wait until saturdy.

    # I am currently setting it up to take a gander at how varying crit_pert affects results
    # y-axis -> rate ; x-axis -> value of crit_pert
    # then I'd want to stick the values into the maplotlib graph generator function
    beta_to_rate = {} # val is (rate, contributers), where contributer is how many enteries contributed
    for key, val in results_dict.iteritems():
        #print 'key', key, 'val', val
        if key[0] == "eigenspace" and key[3] == 5 and key[2] == 0.01:
            print "this value is under consideration"
            print key, val
            if key[4] in beta_to_rate:
                print "this value already exists!"
                old_tpr = beta_to_rate[key[4]][0][0]
                old_fpr = beta_to_rate[key[4]][0][1]
                old_contrib = beta_to_rate[key[4]][1]
                new_tpr = (old_tpr * old_contrib + val['TPR']) / (old_contrib + 1)
                new_fpr = (old_fpr * old_contrib + val['FPR']) / (old_contrib + 1)
                beta_to_rate[key[4]] = ((new_tpr, new_fpr), old_contrib + 1)
            else:
                print "this value does not already exist; I am going to add it"
                beta_to_rate[key[4]] = ((val['TPR'], val['FPR']), 1)

    print 'Mapping', beta_to_rate
    crit_percents = []
    tpr = []
    fpr = []
    for key in sorted(beta_to_rate):
        crit_percents.append(key)
        tpr.append(beta_to_rate[key][0][0])
        fpr.append(beta_to_rate[key][0][1])
        # print "%s: %s" % (key, mydict[key])
    #crit_percents = [crit_percent for crit_percent in crit_pert_to_rate.keys()]
    #tpr = [rates[0][0] for rates  in crit_pert_to_rate.values()]
    #fpr = [rates[0][1] for rates  in crit_pert_to_rate.values()]
    plt.figure(2)
    print "tpr beta", tpr
    print "fpr beta", fpr
    tp_line, = plt.plot(crit_percents, tpr, label = "TP Rate")
    fp_line, = plt.plot(crit_percents, fpr, label = "FP Rate")
    plt.legend(handles=[ tp_line, fp_line])
    plt.title("kinda roc")

    plt.xlabel('beta value')
    plt.ylabel('rate')

    # now I shall do the same thing again w/ the window size
    window_to_rate = {}  # val is (rate, contributers), where contributer is how many enteries contributed
    for key, val in results_dict.iteritems():
        # print 'key', key, 'val', val
        if key[0] == "eigenspace" and key[4] == 0.215 and key[2] == 0.01:
            print "this value is under consideration"
            print key, val
            if key[3] in beta_to_rate:
                print "this value already exists!"
                old_tpr = window_to_rate[key[3]][0][0]
                old_fpr = window_to_rate[key[3]][0][1]
                old_contrib = beta_to_rate[key[3]][1]
                new_tpr = (old_tpr * old_contrib + val['TPR']) / (old_contrib + 1)
                new_fpr = (old_fpr * old_contrib + val['FPR']) / (old_contrib + 1)
                window_to_rate[key[3]] = ((new_tpr, new_fpr), old_contrib + 1)
            else:
                print "this value does not already exist; I am going to add it"
                window_to_rate[key[3]] = ((val['TPR'], val['FPR']), 1)

    print 'Mapping', beta_to_rate
    crit_percents = []
    tpr = []
    fpr = []
    for key in sorted(beta_to_rate):
        crit_percents.append(key)
        tpr.append(window_to_rate[key][0][0])
        fpr.append(window_to_rate[key][0][1])
        # print "%s: %s" % (key, mydict[key])
    # crit_percents = [crit_percent for crit_percent in crit_pert_to_rate.keys()]
    # tpr = [rates[0][0] for rates  in crit_pert_to_rate.values()]
    # fpr = [rates[0][1] for rates  in crit_pert_to_rate.values()]
    plt.figure(3)
    print "tpr beta", tpr
    print "fpr beta", fpr
    tp_line, = plt.plot(crit_percents, tpr, label="TP Rate")
    fp_line, = plt.plot(crit_percents, fpr, label="FP Rate")
    plt.legend(handles=[tp_line, fp_line])
    plt.title("kinda roc")

    plt.xlabel('window size')
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

# you can run using this code
#   python -c "from roc_curves import graph_final_roc; graph_final_roc()"
def graph_final_roc():
    results_path = './experimental_data/test_15_long_small_incr/' + 'all_experimental_results.pickle'
    all_results = pickle.load( open( results_path, "rb" )
    passimport matplotlib.pyplot as plt
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
        beta_val = 0.215
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
        if key[0] == "eigenspace" and key[3] == 5 and key[4] == 0.215:
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

    plt.figure(1)
    #crit_percents = [crit_percent for crit_percent in crit_pert_to_rate.keys()]
    #tpr = [rates[0][0] for rates  in crit_pert_to_rate.values()]
    #fpr = [rates[0][1] for rates  in crit_pert_to_rate.values()]
    tp_line, = plt.plot(crit_percents, tpr, label = "TP Rate")
    fp_line, = plt.plot(crit_percents, fpr, label = "FP Rate")
    plt.legend(handles=[ tp_line, fp_line])

    plt.title("kinda roc")
    plt.xlabel('critical percentage value')
    plt.ylabel('rate')

    ### the above results make no sense (how does decreasing the thresh not have a stronger effect on
    ### the FPR). The only conclusion that makes sense is that the algo is taking too long too reach 'steady
    ### state' with the current discount factor. Therefore alls the values seem very strange to the algo, so it keeps
    ### saying that an alert should be triggered.
    ### I am going to need to go w/ an unreasonably high discount factor (i.e. beta) b/c of the longer run_time
    ### note the second moment is just the variance (of z in this case). maybe I should modify the function
    ### to calc the var off of the collected data... I don't understand the math in that paper, and I am not
    ### going to pretend that I do. Just stick to modifying the parameters

    ### TODO: (1) per the above discussion, the beta value needs to be examined. First, get a graph of that going.
    ### might want to use the code above as a starting point.
    ### (2) then the window size needs to be examined. Get a graph of that going. Try messing with the values and whatnot. See
    ### what happens. PCA is off the table as for the poster.
    ### (3) Rerun analyze_traffic_matrix with the improved parameters (both w/ old data and new data)
    ### even if they do not really work, hey, it's something.
    ### (4) create graphs for poster
    ### (1) - (3) NEEDS to be done on Friday. (4) can wait until saturdy.

    # I am currently setting it up to take a gander at how varying crit_pert affects results
    # y-axis -> rate ; x-axis -> value of crit_pert
    # then I'd want to stick the values into the maplotlib graph generator function
    beta_to_rate = {} # val is (rate, contributers), where contributer is how many enteries contributed
    for key, val in results_dict.iteritems():
        #print 'key', key, 'val', val
        if key[0] == "eigenspace" and key[3] == 5 and key[2] == 0.01:
            print "this value is under consideration"
            print key, val
            if key[4] in beta_to_rate:
                print "this value already exists!"
                old_tpr = beta_to_rate[key[4]][0][0]
                old_fpr = beta_to_rate[key[4]][0][1]
                old_contrib = beta_to_rate[key[4]][1]
                new_tpr = (old_tpr * old_contrib + val['TPR']) / (old_contrib + 1)
                new_fpr = (old_fpr * old_contrib + val['FPR']) / (old_contrib + 1)
                beta_to_rate[key[4]] = ((new_tpr, new_fpr), old_contrib + 1)
            else:
                print "this value does not already exist; I am going to add it"
                beta_to_rate[key[4]] = ((val['TPR'], val['FPR']), 1)

    print 'Mapping', beta_to_rate
    crit_percents = []
    tpr = []
    fpr = []
    for key in sorted(beta_to_rate):
        crit_percents.append(key)
        tpr.append(beta_to_rate[key][0][0])
        fpr.append(beta_to_rate[key][0][1])
        # print "%s: %s" % (key, mydict[key])
    #crit_percents = [crit_percent for crit_percent in crit_pert_to_rate.keys()]
    #tpr = [rates[0][0] for rates  in crit_pert_to_rate.values()]
    #fpr = [rates[0][1] for rates  in crit_pert_to_rate.values()]
    plt.figure(2)
    print "tpr beta", tpr
    print "fpr beta", fpr
    tp_line, = plt.plot(crit_percents, tpr, label = "TP Rate")
    fp_line, = plt.plot(crit_percents, fpr, label = "FP Rate")
    plt.legend(handles=[ tp_line, fp_line])
    plt.title("kinda roc")

    plt.xlabel('beta value')
    plt.ylabel('rate')

    # now I shall do the same thing again w/ the window size
    window_to_rate = {}  # val is (rate, contributers), where contributer is how many enteries contributed
    for key, val in results_dict.iteritems():
        # print 'key', key, 'val', val
        if key[0] == "eigenspace" and key[4] == 0.215 and key[2] == 0.01:
            print "this value is under consideration"
            print key, val
            if key[3] in beta_to_rate:
                print "this value already exists!"
                old_tpr = window_to_rate[key[3]][0][0]
                old_fpr = window_to_rate[key[3]][0][1]
                old_contrib = beta_to_rate[key[3]][1]
                new_tpr = (old_tpr * old_contrib + val['TPR']) / (old_contrib + 1)
                new_fpr = (old_fpr * old_contrib + val['FPR']) / (old_contrib + 1)
                window_to_rate[key[3]] = ((new_tpr, new_fpr), old_contrib + 1)
            else:
                print "this value does not already exist; I am going to add it"
                window_to_rate[key[3]] = ((val['TPR'], val['FPR']), 1)

    print 'Mapping', beta_to_rate
    crit_percents = []
    tpr = []
    fpr = []
    for key in sorted(beta_to_rate):
        crit_percents.append(key)
        tpr.append(window_to_rate[key][0][0])
        fpr.append(window_to_rate[key][0][1])
        # print "%s: %s" % (key, mydict[key])
    # crit_percents = [crit_percent for crit_percent in crit_pert_to_rate.keys()]
    # tpr = [rates[0][0] for rates  in crit_pert_to_rate.values()]
    # fpr = [rates[0][1] for rates  in crit_pert_to_rate.values()]
    plt.figure(3)
    print "tpr beta", tpr
    print "fpr beta", fpr
    tp_line, = plt.plot(crit_percents, tpr, label="TP Rate")
    fp_line, = plt.plot(crit_percents, fpr, label="FP Rate")
    plt.legend(handles=[tp_line, fp_line])
    plt.title("kinda roc")

    plt.xlabel('window size')
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

# you can run using this code
#   python -c "from roc_curves import graph_final_roc; graph_final_roc()"
def graph_final_roc():
    results_path = './experimental_data/test_15_long_small_incr/' + 'all_experimental_results.pickle'
    all_results = pickle.load( open( results_path, "rb" ))

    # okay, so what do we want to here?
    # we want to create a 'mega' ROC, with all of our curves on there,
    # this will involve parsing the pickle file and getting things into shape
    # in regards to lambda val, don't worry ATM, just make one for each

    exfil_rate_to_graph = 5000 # this is arbitrary for now
    lambda_value_to_graph = 0.2
    algos_to_graph = ["naive MS control charts", "selective MS control charts", "naive 3-tier control charts"]

    for exp_params, result in all_results.iteritems():
        # exp_params has the format (rep, exfils.values()[0])
        # We want to fix the exfil_rate, so...
        if exp_params[1] == exfil_rate_to_graph:
            for algo, spec_results in result:
                # exp_params has the format (name, lambda value, stddev coef value)
                # spec_results has the format {'FPR': number, 'TPR': number, etc.}
                print exp_params, result, spec_results

                for cur_graph_algo in algos_to_graph:
                    if cur_graph_algo == algo[0]:

                        # We want to generate on ROC curve... okay, so what do we want to do...
                        # we need to fix (lambda_val) (b/c only one param should be varied, and
                        # that is stddev_coef_vale)
                        if algo[1] == lambda_value_to_graph:

                            # what next... so we need to create the lines so we can graph them...
                            # on the x-axis  we have the FP's...
                            # on the y-axis  we have the TP's...
                            # we need to generate these values... how to do this...
                            # look above, I do this a bunch of times... I think I need to just
                            # do the loop thing from above... Get an average val... wait that's
                            # my outer loop... hmm.... well.... I guess I need to go for a giant
                            # dictionary, similar to the above but bigger... it seems like the way
                            # to go
                            # TODO: (1) modify code from prev loop to compute dict w/ avg vals
                            # (2) in a loop outside this one, loop through a bunch of figures and
                            # stick the graphs in each (should be 5 b/c lambda varies)
                            pass




