import matplotlib.pyplot as plt

# all_experimental_results is of the following form
# [(rep, current_increment)] = exp_results
# exp_results is in the form of
# {"FPR": number} (also TPR, etc.)
# TPR vs FPR
def roc_charts(all_experimental_results):
    tpr = []
    fpr = []
    for exp in all_experimental_results:
        tpr.append( exp["TPR"] )
        fpr.append( exp["FPR"])

    roc_line,  = plt.plot(fpr, tpr)
    
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.show()

def tp_vs_exfil_rate(all_experimental_results):
    pass
