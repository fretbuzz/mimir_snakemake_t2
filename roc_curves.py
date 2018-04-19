import matplotlib.pyplot as plt
import pickle

# all_experimental_results is of the following form
# [(rep, exfil_amt)] = exp_results
# exp_results is in the form of
# {"FPR": number} (also TPR, etc.)
# TPR vs FPR
def roc_charts(all_experimental_results):
    tpr = []
    fpr = []
    for exp in all_experimental_results.values():
        tpr.append( exp["TPR"] )
        fpr.append( exp["FPR"] )

    roc_line,  = plt.plot(fpr, tpr)
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.show()

# x-axis: exfil_rate
# y_axis: tp rate
# see above function for format of arg
def tp_vs_exfil_rate(all_experimental_results):
    tpr = []
    exfil_rate = []
    for exp_settings, exp_results in all_experimental_results.iteritems():
        tpr.append(exp_results["TRP"])
        exfil_rate.append( exp_settins[1] )
    
    tp_line = plt.plot(exfil_rate, tpr)
    plt.xlabel('exfil rate')
    plt.ylabel('tpr')
    plt.show()

def load_exp(all_exp_results_loc):
    all_experimental_results = pickle.load( open( all_exp_results_loc, "rb" ) )
    roc_charts(all_experimental_results)
    tp_vs_exfil_rate(all_experimental_results)
