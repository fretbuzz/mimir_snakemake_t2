from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import math

# okay, what we want to do here is to construct
# x_vals should be FPR
# y_vals should be TPR
def construct_ROC_curve(x_vals, y_vals, title, plot_name):
    plt.figure()
    plt.ylim(-0.05,1.05)
    plt.xlim(-0.05,1.05)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(title)
    plt.plot(x_vals, y_vals)
    plt.savefig( plot_name + '.png', format='png', dpi=1000)
    plt.close()

def create_ROC_of_anom_score(jointDF, time_gran, ROC_path, calc_anom_score, title, plot_name):
    aggregated_anomly_scores = []
    attack_labels = []
    #print "###", jointDF, "###"
    print "\n", title
    for index,row in jointDF.iterrows():

        #print row
        anomaly_score_results = calc_anom_score(row)
        print (index, anomaly_score_results, row[0]),
        aggregated_anomly_scores.append(anomaly_score_results)
        attack_labels.append(row['labels'])

    tprs = []
    fprs = []

    thresholds_to_try = [i/10.0 for i in range(0, -100, -1)] + [i/100.0 for i in range(0,50,5)] +\
                        [i/100.0 for i in range(50,100,2)] + [i/10 for i in range(10,100,5)]
    print "\nthreshold_to_try", thresholds_to_try
    print '---\n'
    for threshold in thresholds_to_try:
        current_alerts = [int(i>=threshold) for i in aggregated_anomly_scores]
        time_gran_to_attack_labels = {}
        #print "time_gran", time_gran
        time_gran_to_attack_labels[time_gran] = attack_labels
        #print "current_attack_labels", time_gran_to_attack_labels[time_gran]
        current_tpr, current_fpr = calc_fp_and_tp(current_alerts, None, None, None, time_gran, time_gran_to_attack_labels)
        tprs.append(current_tpr)
        fprs.append(current_fpr)
        print (current_tpr, current_fpr),

    tprs, fprs = zip(*sorted(zip(tprs, fprs)))

    x_vals = fprs
    y_vals = tprs
    #title = 'Ensemble ROC curve at ' + str(time_gran) + ' Sec Granularity'
    #plot_name = ROC_path + 'aggreg_ROC_curve_' + str(time_gran) + '.csv'
    construct_ROC_curve(x_vals, y_vals, title, ROC_path + plot_name)

def generate_all_anom_ROCs(df_with_anom_features, time_gran, alert_file, sub_path, cur_alert_function, features_to_use):

    ROC_path = alert_file + sub_path + '_good_roc_'
    title = 'ROC Linear Combination of Features at ' + str(time_gran)
    plot_name = 'sub_roc_lin_comb_features_' + str(time_gran)
    create_ROC_of_anom_score(df_with_anom_features, time_gran, ROC_path, cur_alert_function, title,
                                            plot_name)

    # let's also try using each feature seperately
    for feature in features_to_use:
        title = 'ROC ' + feature + ' at ' + str(time_gran)
        plot_name = 'sub_roc_' + feature + '_' + str(time_gran)
        c#ur_alert_function = partial(alert_fuction, weights, [feature], 0.0)
        df_with_only_one_anom_feature = pd.DataFrame(0, index=df_with_anom_features.index, columns=df_with_anom_features.columns)
        df_with_only_one_anom_feature[feature] = df_with_anom_features[features]
        create_ROC_of_anom_score(df_with_only_one_anom_feature, time_gran, ROC_path, cur_alert_function,
                                                title, plot_name)

def determine_alert_function(df_anom_score_with_labels):
    ### TODO: this is going to actually determine the function, rather than just have one hardcoded in, eventually
    weights = {'New Class-Class Edges200_5__mod_z_score': 0.0009,
               'Communication Between Pods not through VIPs (no abs)200_5__mod_z_score': 0.0041,
               'DNS outside-to-inside ratio200_5__mod_z_score': 0.0008}
    bias = -0.0109

    cur_alert_function = partial(alert_fuction, weights, bias)
    return cur_alert_function, weights.keys()

def alert_fuction(weights , bias, row_from_csv):
    alert_score = 0
    #print "row_from_csv",row_from_csv
    for feature,weight in weights.iteritems():
        cur_contrib_to_alert_score = row_from_csv[feature] * weights
        #print feature, row_from_csv
        #print cur_contrib_to_alert_score
        if not math.isnan(cur_contrib_to_alert_score):
            alert_score += cur_contrib_to_alert_score
        else:
            alert_score += 0 # could also just pass
    return alert_score + bias

# def calc_fp_and_tp : ??? -> (TP, FP)_for_every_step, (TP, FP)_for_the_entire_interval
# this function calcuates the tp/fp. Okay, cool, but what counts as a TP/FP. Detecting once it the range?
# detecting at every time stamp (seperately). Detecting @ each time step, every time? What does it mean???
# oh, yah... this is kinda a tricky thing to ask...
# well, how about we take an afraid-to-commit approach... let's compute two vals: one for every time step and
# one for that period in general (I think maybe like that stratosphere IPS guys might be useful too??)
# actually lets only do (TP, FP)_for_every_step
def calc_fp_and_tp(alert_times, exfil_start, exfil_end, wiggle_room, time_granularity, time_gran_to_attack_labels):

    alerts_during_exfiltration = 0.0
    alerts_not_during_exfiltration = 0.0
    attack_labels = time_gran_to_attack_labels[time_granularity]
    #print "attack_labels", attack_labels
    for i in range(0,len(alert_times)):
        #print i, alert_times[i], attack_labels[i], '|',
        if alert_times[i]:
            if attack_labels[i]:
                alerts_during_exfiltration += 1.0
            else:
                alerts_not_during_exfiltration += 1.0
    #print '\n'

    #start_alert_time_index = exfil_start / time_granularity - int(math.floor( wiggle_room / time_granularity))
    #end_alert_time_index = exfil_end / time_granularity + int(math.floor( wiggle_room / time_granularity))
    #alerts_during_exfiltration = alert_times[start_alert_time_index:end_alert_time_index]
    #alerts_not_during_exfiltration = alert_times[0:start_alert_time_index] + alert_times[end_alert_time_index:]

    true_positives = alerts_during_exfiltration #sum(alerts_during_exfiltration)
    false_negatives = sum(attack_labels) - true_positives
    false_positives = alerts_not_during_exfiltration #sum(alerts_not_during_exfiltration)
    # number of negative vals in the experiment
    total_actual_negs = len(attack_labels) - sum(attack_labels)
    true_negatives = total_actual_negs - alerts_not_during_exfiltration # if alert[i] == 0 and attack_labels[i] == 0

    # tpr (a.k.a sensitivity) = proportion of actual positives that we succesfully identifies
    try:
        tpr = float(true_positives) / (true_positives + false_negatives)
    except ZeroDivisionError:
        #print "ZeroDivisionError!", true_positives, false_positives, true_negatives, false_negatives
        tpr = float('nan')

    # fpr = FP / (FP + TN)
    try:
        fpr = float(false_positives) / (true_negatives + false_positives )
    except ZeroDivisionError:
        fpr = float('nan')
    #print (tpr,fpr), (true_positives, false_positives), (false_positives, true_negatives), (total_actual_negs, alerts_not_during_exfiltration),(len(attack_labels), sum(attack_labels))
    #print sum(attack_labels), attack_labels
    #print '----'
    #print alert_times
    #print attack_labels
    #print "tpr,fpr",tpr,fpr

    return tpr, fpr