import matplotlib.pyplot as plt
import pickle

algo = 'control charts'

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

    fpr = list(set([fpr[1] for fpr in tpr_fpr]))

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
def tp_vs_exfil_rate(all_experimental_results):
    print all_experimental_results
    tpr = []
    exfil_rate = []
    tpr_exfil = []
    plt.figure(2)
    for exp_settings, exp_results in all_experimental_results.iteritems():
        #print exp_settings, exp_results[algo]
        #tpr.append(exp_results[algo]['TPR'])
        #exfil_rate.append( exp_settings[1] )
        tpr_exfil.append( (exp_results[algo]['TPR'], exp_settings[1])  )

    exfil_rate = list(set([exfil[1] for exfil in tpr_exfil]))
    print exfil_rate 

    for rate in exfil_rate:
        tpr_total = 0
        total_rates = 0
        for item in tpr_exfil:
            if item[1] == rate:
                tpr_total += item[0]
                total_rates += 1
        tpr.append(float(tpr_total) / total_rates)

    tp_line = plt.plot(exfil_rate, tpr)
    print "tpr", tpr
    print "exfil rate", exfil_rate
    plt.xlabel('exfil rate')
    plt.ylabel('tpr')
    plt.show()

def load_exp(all_exp_results_loc):
    all_experimental_results = pickle.load( open( all_exp_results_loc, "rb" ) )
    roc_charts(all_experimental_results)
    tp_vs_exfil_rate(all_experimental_results)
