import matplotlib.pyplot as plt
import pickle
from analyze_traffix_matrixes import join_dfs

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
def actual_roc():
    pickle_path = './experimental_data/test_15_long_small_incr/'
    pickle_files = [('rec_matrix_increm_5_rep_0.pickle', 'sent_matrix_increm_5_rep_0.pickle'),
                    ('rec_matrix_increm_5_rep_1.pickle', 'sent_matrix_increm_5_rep_1.pickle'),
                    ('rec_matrix_increm_5_rep_2.pickle', 'sent_matrix_increm_5_rep_2.pickle')]

    relevant_tms = []
    for pf in pickle_files:
        relevant_tms.append( (pickle.load( open( pickle_path + pf[0], "rb" )), 
                            pickle.load( open( pickle_path + pf[1], "rb" )))  )
    
    joint_dfs = []
    for df_rec_sent in relevant_tms:
        joint_dfs.append( join_dfs(df_rec_sent[1], df_rec_sent[0]) )
    return joint_dfs

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
    #print ms_trp
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

