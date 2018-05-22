import matplotlib.pyplot as plt
import pickle
from analyze_traffix_matrixes import join_dfs, eigenspace_detection_loop, calc_tp_fp_etc
import numpy as np
import time

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
    # NOTE: THIS SECTION SHOULD BE MODIFIED TO MAKE MY NEW ROC CURVES
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

    # END CODE I SHOULD BORROW TO MAKE MY ROC CURVES

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
    # so what are the variables being used here?
    # results_path, all_results_ratio, all_results_selective_ewma, results_ratio_exchanged_ineq

    #results_path = './experimental_data/test_15_long_small_incr/' + 'all_experimental_results.pickle'
    #results_path =  './experimental_data/cybermonday-target-hack/' + 'all_experimental_results_fixed_extended.pickle' #+ 'all_experimental_results.pickle'
    #results_path = './experimental_data/cybermonday-target-hack/' + 'all_experimental_results_maybe_fixed_bug_plus_lc_bug.pickle'
    results_path = './experimental_data/cybermonday-target-hack-take-3/' +'all_results_take_something_fixed.pickle'#+ 'all_results.pickle'

    #+ 'all_experimental_results_fixed_56.pickle'
    all_results = pickle.load( open( results_path, "rb" ))

    #print "all results", all_results

    #results_path_ratio = './experimental_data/cybermonday-target-hack/' + 'all_experimental_results_fixed_ratio.pickle'
    #results_path_ratio = './experimental_data/cybermonday-target-hack/' + 'all_experimental_results_fixed_ratio_extended.pickle'
    #results_path_ratio = './experimental_data/cybermonday-target-hack/' + '/all_experimental_results_ratio_all_coef.pickle'
    #all_results_ratio = pickle.load( open( results_path_ratio, "rb" ))
    all_results_ratio = all_results

    #results_bi_selective_ratio = './experimental_data/cybermonday-target-hack/' + 'all_experimental_results_fixed_just_bi.pickle'
    #all_results_selective_ewma = pickle.load( open( results_bi_selective_ratio, "rb" ))
    all_results_selective_ewma = all_results

    results_path_ratio_exchanged_ineq = './experimental_data/cybermonday-target-hack-take-3/' + 'all_results_more_lambdas.pickle'
    results_ratio_exchanged_ineq = pickle.load( open( results_path_ratio_exchanged_ineq, "rb" ))
    #print "ratio results", results_ratio_exchanged_ineq
    #time.sleep(5)
    #results_ratio_exchanged_ineq = all_results

    #results_eigen_path = './experimental_data/cybermonday-target-hack/'  + 'all_experimental_results_maybe_fixed_eigen_finally.pickle'
    #+'all_experimental_results_just_testing_v2.pickle'
    # + 'all_experimental_results_just_testing.pickle'#'all_experimental_results_just_eigen.pickle'
    #all_results_eigen = pickle.load( open( results_eigen_path, "rb" ))
    all_results_eigen = all_results

    #results_simple_selective_path = './experimental_data/cybermonday-target-hack-take-2/' + 'simple_just_selective.pickle'
    #all_results_simple_selective = pickle.load( open( results_simple_selective_path, "rb" ))
    all_results_simple_selective = all_results

    results_simple_linear_regression_path = './experimental_data/cybermonday-target-hack-take-3/' + 'simple_linear_regress_even_even_even_smaller_params.pickle'
    #'simple_linear_regress_even_even_smaller_params.pickle'
    #+ 'simple_linear_regress_even_smaller_params.pickle'
    #+ 'simple_linear_regress_dif_params.pickle'
    all_results_simple_linear_regression = pickle.load( open( results_simple_linear_regression_path, "rb" ))


    algos_to_graph = ["naive MS control charts", "selective MS control charts", "naive 3-tier control charts",
                      "ratio ewma"]

    #graphed_exfiltration_rate = 30000
    for key,val in all_results.iteritems():
        print key

    for graphed_exfiltration_rate in [10000, 20000, 30000]:

        #joint_line_ratio = return_joint_line(all_results_ratio, exfil_rate_to_graph = graphed_exfiltration_rate,
        #                                    algo_to_examine="ratio ewma", lambda_value=0.2)

        #print [point[2] for point in joint_line_ratio[0]]

        # lambda = 0.4 works best for all the training data
        joint_line_ratio_two = return_joint_line(all_results_ratio, exfil_rate_to_graph = graphed_exfiltration_rate,
                                            algo_to_examine="ratio ewma", lambda_value=0.4, sim_svc_ratio_exceeding_thresh=1,
                                            no_alert_if_any_svc_ratio_decreases=False)

        '''
        joint_line_ratio_three = return_joint_line(all_results_ratio, exfil_rate_to_graph = graphed_exfiltration_rate,
                                            algo_to_examine="ratio ewma", lambda_value=0.6)

        joint_line_ratio_four = return_joint_line(all_results_ratio, exfil_rate_to_graph = graphed_exfiltration_rate,
                                            algo_to_examine="ratio ewma", lambda_value=0.8)

        '''

        print "ratio", len(joint_line_ratio_two)

        joint_line_sewma = return_joint_line(all_results, exfil_rate_to_graph = graphed_exfiltration_rate,
                                            algo_to_examine="selective MS control charts", lambda_value=0.2)

        print "sewma", len(joint_line_sewma)


        joint_line_ttewma = return_joint_line(all_results, exfil_rate_to_graph = graphed_exfiltration_rate,
                                            algo_to_examine="naive 3-tier control charts", lambda_value=0.2)

        print "ttewma", len(joint_line_ttewma)
        print joint_line_ttewma

        joint_line_newma = return_joint_line(all_results, exfil_rate_to_graph = graphed_exfiltration_rate,
                                            algo_to_examine="naive MS control charts", lambda_value=0.2)

        print "newma", len(joint_line_newma)

        joint_line_bi_sewma = return_joint_line(all_results_selective_ewma, exfil_rate_to_graph=graphed_exfiltration_rate,
                                             algo_to_examine="bi-selective MS control charts", lambda_value=0.2)

        print "bi-sewma", len(joint_line_bi_sewma)

        joint_line_simple_selective = return_joint_line(all_results_simple_selective, exfil_rate_to_graph=graphed_exfiltration_rate,
                                             algo_to_examine="simple_selective", lambda_value=0)


        print "simple selective", len(joint_line_simple_selective)

        joint_line_simple_lg = return_joint_line(all_results_simple_linear_regression, exfil_rate_to_graph=graphed_exfiltration_rate,
                                             algo_to_examine="lin_reg", lambda_value=0)

        print "simple lg", len(joint_line_simple_lg)

        joint_line_ratio_exchanged = return_joint_line(results_ratio_exchanged_ineq, exfil_rate_to_graph=graphed_exfiltration_rate,
                                                 algo_to_examine="ratio ewma", lambda_value=0.1, sim_svc_ratio_exceeding_thresh=1,
                                            no_alert_if_any_svc_ratio_decreases=False)

        # note: I am going to mess w/ this
        joint_line_ratio_exchanged_two= return_joint_line(results_ratio_exchanged_ineq, exfil_rate_to_graph=graphed_exfiltration_rate,
                                                 algo_to_examine="ratio ewma", lambda_value=0.1, sim_svc_ratio_exceeding_thresh=1,
                                            no_alert_if_any_svc_ratio_decreases=False)

        joint_line_ratio_exchanged_six= return_joint_line(results_ratio_exchanged_ineq, exfil_rate_to_graph=graphed_exfiltration_rate,
                                                 algo_to_examine="ratio ewma", lambda_value=0.05, sim_svc_ratio_exceeding_thresh=1,
                                            no_alert_if_any_svc_ratio_decreases=False)

        joint_line_ratio_two_svc = return_joint_line(all_results_ratio, exfil_rate_to_graph = graphed_exfiltration_rate,
                                            algo_to_examine="ratio ewma", lambda_value=0.2, sim_svc_ratio_exceeding_thresh=2,
                                            no_alert_if_any_svc_ratio_decreases=False)

        joint_line_ratio_three_svc = return_joint_line(results_ratio_exchanged_ineq, exfil_rate_to_graph = graphed_exfiltration_rate,
                                            algo_to_examine="ratio ewma", lambda_value=0.1, sim_svc_ratio_exceeding_thresh=3,
                                            no_alert_if_any_svc_ratio_decreases=False)

        joint_line_ratio_two_no_decrease = return_joint_line(all_results_ratio, exfil_rate_to_graph=graphed_exfiltration_rate,
                                                     algo_to_examine="ratio ewma", lambda_value=0.4,
                                                     sim_svc_ratio_exceeding_thresh=1,
                                                     no_alert_if_any_svc_ratio_decreases=True)

        joint_line_ratio_two_svc_no_decre = return_joint_line(all_results_ratio, exfil_rate_to_graph=graphed_exfiltration_rate,
                                                       algo_to_examine="ratio ewma", lambda_value=0.4,
                                                       sim_svc_ratio_exceeding_thresh=2,
                                                       no_alert_if_any_svc_ratio_decreases=True)

        joint_line_ratio_three_svc_no_decre = return_joint_line(all_results_ratio, exfil_rate_to_graph=graphed_exfiltration_rate,
                                                       algo_to_examine="ratio ewma", lambda_value=0.4,
                                                       sim_svc_ratio_exceeding_thresh=3,
                                                       no_alert_if_any_svc_ratio_decreases=True)


        joint_line_ratio_linear_comb_ratio = return_joint_line(all_results_ratio, exfil_rate_to_graph=graphed_exfiltration_rate,
                                                       algo_to_examine="linear-comb ewma",lambda_value=0.4)


        #("eigenspace", crit_percent, windows_size, beta_val),
        window_size = 5
        beta_val = 0.05
        joint_line_eigenspace= return_joint_line(all_results_eigen, exfil_rate_to_graph=graphed_exfiltration_rate,
                                                       algo_to_examine="eigenspace", lambda_value=window_size,
                                                       sim_svc_ratio_exceeding_thresh=beta_val)

        print "ratio exchanged", joint_line_ratio_exchanged

        print "separate lines", joint_line_ratio_exchanged

        print "eigenspace", joint_line_eigenspace

        #print "joined lines", avg_lines(joint_line_ratio_exchanged)

        #joint_line_ratio_two = [avg_lines(joint_line_ratio_two)]
        #joint_line_sewma = [ avg_lines(joint_line_sewma)   ]
        #joint_line_ttewma = [ avg_lines(joint_line_ttewma)    ]
        #print "averaged ttewma", joint_line_ttewma
        #joint_line_newma = [  avg_lines(joint_line_newma)   ]
        #joint_line_bi_sewma = [  avg_lines(joint_line_bi_sewma)   ]
        joint_line_ratio_exchanged =   [ avg_lines(joint_line_ratio_exchanged)    ]
        joint_line_ratio_exchanged_two = [avg_lines(joint_line_ratio_exchanged_two)]
        joint_line_ratio_exchanged_six = [avg_lines(joint_line_ratio_exchanged_six)]
        joint_line_ratio_two_svc = [ avg_lines(joint_line_ratio_two_svc)    ]
        joint_line_ratio_three_svc =[ avg_lines(joint_line_ratio_three_svc)    ]
        #joint_line_ratio_two_no_decrease =[ avg_lines(joint_line_ratio_two_no_decrease)    ]
        #joint_line_ratio_two_svc_no_decre =[ avg_lines(joint_line_ratio_two_svc_no_decre)    ]
        #joint_line_ratio_three_svc_no_decre =[ avg_lines(joint_line_ratio_three_svc_no_decre)    ]
        joint_line_eigenspace = [avg_lines(joint_line_eigenspace)]
        joint_line_ratio_linear_comb_ratio = [avg_lines(joint_line_ratio_linear_comb_ratio)]
        joint_line_simple_selective = [avg_lines(joint_line_simple_selective)]
        joint_line_simple_lg = [avg_lines(joint_line_simple_lg)]
        joint_line_simple_lg[0].sort(key=lambda x: x[1])

        plt.figure(1)
        '''
        rat_one_tp, = plt.plot([ln[2] for ln in joint_line_ratio_two[0]],
                               [ln[0] for ln in joint_line_ratio_two[0]],
                               label='tp ratio')

        rat_one_fp, = plt.plot([ln[2] for ln in joint_line_ratio_two[0]],
                               [ln[1] for ln in joint_line_ratio_two[0]],
                               label='fp ratio')

        rat_one_tp_swema, = plt.plot([ln[2] for ln in joint_line_sewma[0]],
                                     [ln[0] for ln in joint_line_sewma[0]],
                                     label='swema tp')

        rat_one_fp_swema, = plt.plot([ln[2] for ln in joint_line_sewma[0]],
                                     [ln[1] for ln in joint_line_sewma[0]],
                                     label='swema fp')

        rat_one_tp_tt_nwema, = plt.plot([ln[2] for ln in joint_line_ttewma[0]],
                                        [ln[0] for ln in joint_line_ttewma[0]],
                                        label='ttewma tp')

        rat_one_fp_tt_nwema, = plt.plot([ln[2] for ln in joint_line_ttewma[0]],
                                        [ln[1] for ln in joint_line_ttewma[0]],
                                        label='ttewma fp')

        rat_one_tp_newma, = plt.plot([ln[2] for ln in joint_line_newma[0]],
                                     [ln[0] for ln in joint_line_newma[0]],
                                     label='newma tp')

        rat_one_fp_newma, = plt.plot([ln[2] for ln in joint_line_newma[0]],
                                     [ln[1] for ln in joint_line_newma[0]],
                                     label='newma fp')

        rat_one_tp_bi_swema, = plt.plot([ln[2] for ln in joint_line_bi_sewma[0]],
                                     [ln[0] for ln in joint_line_bi_sewma[0]],
                                     label='bi-swema tp')

        rat_one_fp_bi_swema, = plt.plot([ln[2] for ln in joint_line_bi_sewma[0]],
                                     [ln[1] for ln in joint_line_bi_sewma[0]],
                                     label='bi-swema fp')
        
        '''

        rat_one_tp_eigen, = plt.plot([ln[2] for ln in joint_line_simple_lg[0]],
                                     [ln[0] for ln in joint_line_simple_lg[0]],
                                     label='lg tp')

        rat_one_fp_eigen, = plt.plot([ln[2] for ln in joint_line_simple_lg[0]],
                                     [ln[1] for ln in joint_line_simple_lg[0]],
                                     label='lg fp')

        '''
        rat_one_tp_ratio_lc, = plt.plot([ln[2] for ln in joint_line_ratio_linear_comb_ratio[0]],
                                     [ln[0] for ln in joint_line_ratio_linear_comb_ratio[0]],
                                     label='ratio linear comb tp')

        rat_one_fp_ratio_lc, = plt.plot([ln[2] for ln in joint_line_ratio_linear_comb_ratio[0]],
                                     [ln[1] for ln in joint_line_ratio_linear_comb_ratio[0]],
                                     label='ratio linear comb fp')
        '''
        plt.legend(handles=[#rat_one_tp, rat_one_fp, rat_one_tp_swema, rat_one_fp_swema,
                            #rat_one_tp_tt_nwema, rat_one_fp_tt_nwema,
                            #rat_one_tp_newma, rat_one_fp_newma,
                            #rat_one_tp_bi_swema, rat_one_fp_bi_swema])
                            rat_one_tp_eigen, rat_one_fp_eigen])
                            #rat_one_tp_ratio_lc, rat_one_fp_ratio_lc])
        plt.title("coefficient of lg vs rates")
        plt.xlabel('coef')
        plt.ylabel('rate')



        ### this is the roc curve
        plt.figure(2)
        opacity = 0.8
        '''
        rat_one_tp, = plt.plot([ln[1] for ln in joint_line_ratio[0]],
                               [ln[0] for ln in joint_line_ratio[0]],
                               label='ratio')
                               
        rat_one_tp_two, = plt.plot([ln[1] for ln in joint_line_ratio_two[0]],
                               [ln[0] for ln in joint_line_ratio_two[0]],
                               label='ratio (0.6 lambda)', alpha=opacity)

        rat_one_tp_three, = plt.plot([ln[1] for ln in joint_line_ratio_exchanged_two[0]],
                               [ln[0] for ln in joint_line_ratio_exchanged_two[0]],
                               label='exch ratio (0.2 lambda)')

        rat_one_tp_four, = plt.plot([ln[1] for ln in joint_line_ratio_exchanged_six[0]],
                               [ln[0] for ln in joint_line_ratio_exchanged_six[0]],
                               label='exch ratio (0.6 lambda)')
        '''
        '''
        rat_one_tp_tt_nwema, = plt.plot([ln[1] for ln in joint_line_ttewma[0]],
                                        [ln[0] for ln in joint_line_ttewma[0]],
                                        label='Monolith Naive-mEWMA',alpha=opacity,
                                        linestyle='-.', marker='o')

        rat_one_tp_newma, = plt.plot([ln[1] for ln in joint_line_newma[0]],
                                     [ln[0] for ln in joint_line_newma[0]],
                                     label='Naive-mEWMA',alpha=opacity,
                                     linestyle='-', marker='^')
        '''
        '''
        rat_one_tp_swema, = plt.plot([ln[1] for ln in joint_line_sewma[0]],
                                     [ln[0] for ln in joint_line_sewma[0]],
                                     label='Selective EWMA', alpha=opacity,
                                     linestyle=':', marker='D')
        '''
        #'''
        # (for line below) -> lambda = 0.2
        rat_one_tp_two_reversed, = plt.plot([ln[1] for ln in joint_line_ratio_exchanged_two[0]],
                               [ln[0] for ln in joint_line_ratio_exchanged_two[0]],
                               label='Traffic-Ratio', alpha=opacity,
                                    linestyle='--', marker='*')
        #'''
        '''
        # (for line below) -> (3 services, lambda=0.2)
        rat_one_roc_two_three_svc, = plt.plot([ln[1] for ln in joint_line_ratio_three_svc[0]],
                               [ln[0] for ln in joint_line_ratio_three_svc[0]],
                               label='Coord-Traffic-Ratio', alpha=opacity,
                                linestyle=':', marker='+')
        '''

        lg_line, = plt.plot([ln[1] for ln in joint_line_simple_lg[0]],
                               [ln[0] for ln in joint_line_simple_lg[0]],
                               label='Linear Regression', alpha=opacity,
                                linestyle='-', marker='^')

        #'''
        '''
        rat_one_tp_ratio_linear_comb, = plt.plot([ln[1] for ln in joint_line_ratio_linear_comb_ratio[0]],
                                                 [ln[0] for ln in joint_line_ratio_linear_comb_ratio[0]],
                                                 label='Linear-Combination-Traffic-Ratio-mEWMA', alpha=opacity,
                                                 linestyle='-.', marker='o')

        '''
        '''
        rat_one_tp_ratio_linear_comb, = plt.plot([ln[1] for ln in joint_line_ratio_linear_comb_ratio[0]],
                                        [ln[0] for ln in joint_line_ratio_linear_comb_ratio[0]],
                                        label='Lin-Comb-Traffic-Ratio-mEWMA',alpha=opacity,
                                        linestyle='-.', marker='o')
        '''
        simple_selective_line, = plt.plot([ln[1] for ln in joint_line_simple_selective[0]],
                                        [ln[0] for ln in joint_line_simple_selective[0]],
                                        label='Simple Threshold',alpha=opacity,
                                        linestyle='-.', marker='s')

        '''
        #joint_line_eigenspace[0].sort(key=lambda x: x[1])
        rat_one_eigen, = plt.plot([ln[1] for ln in joint_line_eigenspace[0]],
                                        [ln[0] for ln in joint_line_eigenspace[0]],
                                        label='Eigenspace',alpha=opacity,
                                        linestyle='-.', marker='o')
        '''
        #'''

        #'''
        '''
        #rat_one_tp_bi_swema, = plt.plot([ln[1] for ln in joint_line_bi_sewma[0]],
        #                             [ln[0] for ln in joint_line_bi_sewma[0]],
        #                             label='bi-swema',alpha=opacity)
        '''
        '''

        rat_one_roc_two_two_svc, = plt.plot([ln[1] for ln in joint_line_ratio_two_svc[0]],
                               [ln[0] for ln in joint_line_ratio_two_svc[0]],
                               label='Microservice Traffic-Ratio-EWMA (simple, two_svc, lambda=0.2)', alpha=opacity,
                                linestyle='-', marker='^')
        '''
        joint_line_eigenspace[0].sort(key=lambda x: x[1])
        rat_one_eigen, = plt.plot([ln[1] for ln in joint_line_eigenspace[0]],
                                  [ln[0] for ln in joint_line_eigenspace[0]],
                                  label='Eigenspace', alpha=opacity,
                                  linestyle='-', marker='o')


        plt.legend(handles=[simple_selective_line,lg_line,
                            rat_one_tp_two_reversed,
                            rat_one_eigen])#, rat_one_roc_two_three_svc]),
                            #rat_one_tp_ratio_linear_comb, rat_one_eigen])#,
                            #rat_one_tp_two_reversed, rat_one_roc_two_two_svc])#,
                            #rat_one_tp_three, rat_one_tp_four])
                            #rat_one_tp, rat_one_tp_three, rat_one_tp_four])
        plt.title("ROC Curve for " + "{:,}".format(graphed_exfiltration_rate/1000/5) + " KB/s Exfiltration")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.savefig('ROC Curve Results' + '.png', bbox_inches='tight')
        plt.savefig('./roc_curve_' + str(graphed_exfiltration_rate) + '.eps', format='eps', dpi=1000)

        # this is going to be a figure just for ratio_ewma
        plt.figure(3)
        rat_one_tp_two_reversed, = plt.plot([ln[1] for ln in joint_line_ratio_exchanged_two[0]],
                               [ln[0] for ln in joint_line_ratio_exchanged_two[0]],
                               label='Microservice Traffic-Ratio-EWMA (simple, one_svc, lambda=0.1)', alpha=opacity,
                                linestyle='-', marker='^')
        rat_one_tp_two_reversed_two, = plt.plot([ln[1] for ln in joint_line_ratio_exchanged[0]],
                               [ln[0] for ln in joint_line_ratio_exchanged[0]],
                               label='Microservice Traffic-Ratio-EWMA (simple, one_svc, lambda=0.1)', alpha=opacity,
                                linestyle='-', marker='^')
        rat_one_tp_two_reversed_three, = plt.plot([ln[1] for ln in joint_line_ratio_exchanged_six[0]],
                               [ln[0] for ln in joint_line_ratio_exchanged_six[0]],
                               label='Microservice Traffic-Ratio-EWMA (simple, one_svc, lambda=0.05)', alpha=opacity,
                                linestyle='-', marker='^')

        '''
        rat_one_roc_two_two_svc, = plt.plot([ln[1] for ln in joint_line_ratio_two_svc[0]],
                               [ln[0] for ln in joint_line_ratio_two_svc[0]],
                               label='Microservice Traffic-Ratio-EWMA (simple, two_svc, lambda=0.2)', alpha=opacity,
                                linestyle='-', marker='^')

        rat_one_roc_two_three_svc, = plt.plot([ln[1] for ln in joint_line_ratio_three_svc[0]],
                               [ln[0] for ln in joint_line_ratio_three_svc[0]],
                               label='Microservice Traffic-Ratio-EWMA (simple, three_svc, lambda=0.2)', alpha=opacity,
                                linestyle='-', marker='^')
        '''
        '''
        rat_one_roc_two_no_decr, = plt.plot([ln[1] for ln in joint_line_ratio_two_no_decrease[0]],
                               [ln[0] for ln in joint_line_ratio_two_no_decrease[0]],
                               label='Microservice Traffic-Ratio-EWMA (no_decr, one_svc, lambda=0.4)', alpha=opacity,
                                linestyle='-', marker='^')

        rat_one_roc_two_two_svc_no_decr, = plt.plot([ln[1] for ln in joint_line_ratio_two_svc_no_decre[0]],
                               [ln[0] for ln in joint_line_ratio_two_svc_no_decre[0]],
                               label='Microservice Traffic-Ratio-EWMA (no_decr, two_svc, lambda=0.4)', alpha=opacity,
                                linestyle='-', marker='^')
        
        rat_one_roc_two_three_svc_no_decr, = plt.plot([ln[1] for ln in joint_line_ratio_three_svc_no_decre[0]],
                               [ln[0] for ln in joint_line_ratio_three_svc_no_decre[0]],
                               label='Microservice Traffic-Ratio-EWMA (no_decr, three_svc, lambda=0.4)', alpha=opacity,
                                linestyle='-', marker='^')
        '''
        plt.legend(handles=[rat_one_tp_two_reversed,rat_one_tp_two_reversed_two,rat_one_tp_two_reversed_three])#,
                            #rat_one_roc_two_no_decr,rat_one_roc_two_two_svc_no_decr])#,rat_one_roc_two_three_svc_no_decr])
        # rat_one_tp_three, rat_one_tp_four])
        # rat_one_tp, rat_one_tp_three, rat_one_tp_four])
        plt.title("ROC Curve")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plt.figure(4)
        '''
        rat_one_tp_tt_nwema, = plt.plot([ln[1] for ln in joint_line_ttewma[0]],
                                        [ln[0] for ln in joint_line_ttewma[0]],
                                        label='Monolith Naive-EWMA', alpha=opacity,
                                        linestyle='-.', marker='o')

        rat_one_tp_newma, = plt.plot([ln[1] for ln in joint_line_newma[0]],
                                     [ln[0] for ln in joint_line_newma[0]],
                                     label='Naive-EWMA', alpha=opacity)

        rat_one_tp_swema, = plt.plot([ln[1] for ln in joint_line_sewma[0]],
                                     [ln[0] for ln in joint_line_sewma[0]],
                                     label='Selective-EWMA', alpha=opacity,
                                     linestyle='--', marker='*')
        '''
        # (for line below) -> (lambda=0.2)
        rat_one_tp_two_reversed, = plt.plot([ln[1] for ln in joint_line_ratio_exchanged_two[0]],
                                            [ln[0] for ln in joint_line_ratio_exchanged_two[0]],
                                            label='Traffic-Ratio-mEWMA', alpha=opacity,
                                            linestyle='--', marker='*')

        # (for line below) -> (3 services, lambda=0.2)
        rat_one_roc_two_three_svc, = plt.plot([ln[1] for ln in joint_line_ratio_three_svc[0]],
                                              [ln[0] for ln in joint_line_ratio_three_svc[0]],
                                              label='Coord-Traffic-Ratio-mEWMA', alpha=opacity,
                                              linestyle='-', marker='^')

        rat_one_tp_ratio_linear_comb, = plt.plot([ln[1] for ln in joint_line_ratio_linear_comb_ratio[0]],
                                                 [ln[0] for ln in joint_line_ratio_linear_comb_ratio[0]],
                                                 label='Lin-Comb-Traffic-Ratio-mEWMA', alpha=opacity,
                                                 linestyle='-.', marker='o')
        # '''
        joint_line_eigenspace[0].sort(key=lambda x: x[1])
        rat_one_eigen, = plt.plot([ln[1] for ln in joint_line_eigenspace[0]],
                                  [ln[0] for ln in joint_line_eigenspace[0]],
                                  label='Eigenspace', alpha=opacity,
                                  linestyle='-.', marker='o')
        # '''

        # '''
        '''
        #rat_one_tp_bi_swema, = plt.plot([ln[1] for ln in joint_line_bi_sewma[0]],
        #                             [ln[0] for ln in joint_line_bi_sewma[0]],
        #                             label='bi-swema',alpha=opacity)
        '''
        # '''

        plt.legend(handles=[#rat_one_tp_tt_nwema, rat_one_tp_newma, rat_one_tp_swema,
                            rat_one_tp_two_reversed, rat_one_roc_two_three_svc, rat_one_tp_ratio_linear_comb, rat_one_eigen])  # ,
        # rat_one_tp_two_reversed, rat_one_roc_two_two_svc])#,
        # rat_one_tp_three, rat_one_tp_four])
        # rat_one_tp, rat_one_tp_three, rat_one_tp_four])
        plt.title("ROC Curve for " + "{:,}".format(graphed_exfiltration_rate/1000/5) + " KB/s Exfiltration")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.savefig('ROC Curve Results' + '.png', bbox_inches='tight')
        plt.savefig('./roc_curve_story_' + str(graphed_exfiltration_rate) + '.eps', format='eps', dpi=1000)

        # this figure is just to check if LR is working or not
        plt.figure(5)
        plt.title("ROC Curve")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        lg_line, = plt.plot([ln[1] for ln in joint_line_simple_lg[0]],
                               [ln[0] for ln in joint_line_simple_lg[0]],
                               label='Linear Regression', alpha=opacity,
                                linestyle='-', marker='^')


        plt.show()
    #plt.legend(handles=[rat_one_tp, rat_one_fp, rat_one_tp_swema, rat_one_fp_swema])

    '''
    rat_two, = plt.plot(coef_lines[1],
                        tp_lines[1],
                        label='two')
    rat_three, = plt.plot(coef_lines[2],
                          tp_lines[2],
                          label='three')
    '''
    #plt.show()


def return_joint_line(all_exp_results, exfil_rate_to_graph, algo_to_examine, lambda_value, sim_svc_ratio_exceeding_thresh=None,
                      no_alert_if_any_svc_ratio_decreases=None):

    tp_lines = []
    fp_lines = []
    coef_lines = []
    joint_lines = []

    for exp_params, result in all_exp_results.iteritems():
        # exp_params has the format (rep, exfils.values()[0])
        # We want to fix the exfil_rate, so...
        #print exp_params[1]
        if exp_params[1] == exfil_rate_to_graph:

            cur_tp_line = []
            cur_fp_line = []
            cur_coef_line = []
            cur_joint_line = []

            #print "found a matching experiment"
            for algo, spec_results in result.iteritems():
                # algo has the format (name, lambda value, stddev coef value)
                # ^^ this is no longer completely true
                # this is the format for svc-ratio: ("ratio ewma", lambda_values, ewma_stddev_coef_val,
                #                                    simulataneous_svc_ratio_exceeding_thresh,
                #                                    no_alert_if_any_svc_ratio_decreases),
                # spec_results has the format {'FPR': number, 'TPR': number, etc.}
                #print exp_params, result, spec_results

                # need to do the processing differently for eigenspace because I was stupid w/ the arangment of params in the tuple
                if algo_to_examine == 'eigenspace':
                    print "eigenspace match"
                    # ("eigenspace", crit_percent, windows_size, beta_val); want to iterate through crit_percent
                    # I'm going to overlead the params (i.e. use them in a way not implied by their name)
                    if algo_to_examine == algo[0] and lambda_value == algo[2] and sim_svc_ratio_exceeding_thresh == algo[3]:
                        cur_tp_line.append(spec_results['TPR'])
                        cur_fp_line.append(spec_results['FPR'])
                        cur_coef_line.append(algo[1])
                        cur_joint_line.append((spec_results['TPR'], spec_results['FPR'], algo[1]))
                    continue # do NOT also want to run the other if statements

                if algo_to_examine == algo[0] and lambda_value == algo[1]:
                    if algo_to_examine == "ratio ewma":
                        if sim_svc_ratio_exceeding_thresh != algo[3] or  no_alert_if_any_svc_ratio_decreases != algo[4]:
                            # if doesn't meet the requirements, try the next one
                            continue

                    #print "cur tpr", spec_results['TPR'], "fpr", spec_results['FPR'], "coef", algo[2]
                    cur_tp_line.append(spec_results['TPR'])
                    cur_fp_line.append(spec_results['FPR'])
                    cur_coef_line.append(algo[2])
                    cur_joint_line.append( ( spec_results['TPR'], spec_results['FPR'], algo[2])   )

            tp_lines.append(list(cur_tp_line))
            fp_lines.append(list(cur_fp_line))
            coef_lines.append(list(cur_coef_line))
            joint_lines.append(cur_joint_line)

    sorted_joint_lines = []
    for joint_ln in joint_lines:
        sorted_joint_ln = sorted(joint_ln, key=lambda x: x[2])
        sorted_joint_lines.append(sorted_joint_ln)

    return sorted_joint_lines

def avg_lines(joint_lines):
    results = []
    # picking the first one arbitrarily (they should all have similar values)
    for tup in joint_lines[0]:
        matching = []
        for joint_line in joint_lines:
            for joint_tup in joint_line:
                if joint_tup[2] == tup[2]:
                    matching.append(joint_tup)
        # should have three values
        #print matching

        # now actually calc the averges. I am going to do this a naive way
        avg_tpr = (matching[0][0] + matching[1][0] + matching[2][0]) / 3.0
        avg_fpr = (matching[0][1] + matching[1][1] + matching[2][1]) / 3.0
        stddev_coef = tup[2]

        aggreg_tuple = (avg_tpr, avg_fpr, stddev_coef)
        results.append(aggreg_tuple)
    #print "done averaging linges"
    return results