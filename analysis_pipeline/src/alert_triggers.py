import ast
import csv
import math

import numpy as np
import pandas
from pyod.models.hbos import HBOS

# func: multiple_outliers_in_window: time_series window_size num_outlier_vals_in_window -> list_of_alert_times
# time_series : 1-D list of 0/1, with 0 = no alert, 1 = alert
# window_size : number of values over which to sum outliers
# num_outlier_vals_in_window : must be >= this number of values in the window in order for an alert to be triggered
# note: cannot do node attribution here b/c we only have the 1-D time series, node attribution would need to be done seperately
from analysis_pipeline.generate_alerts import calc_fp_and_tp


# okay, so this file is going to contain some functions that actually trigger the alerts.
# in analyze_edgefiles, we have several techniques for analyzing the graphs by computing some metrics
# and making some (rather nice looking) graphs. But we end up with, essentially, several 1-D time series.
# we need to actually trigger alerts based on these time series. That's where the methods in this file
# come into play.
def multiple_outliers_in_window(time_series, window_size, num_outlier_vals_in_window):
    alerts = [float('nan') for i in range(1,window_size)]
    for i in range(window_size, len(time_series)):
        #print "i", i, "window_size", window_size
        current_window = time_series[i-window_size:i]
        num_exceeding_window_vals = sum(current_window)
        if num_exceeding_window_vals >= num_outlier_vals_in_window:
            alerts.append(1)
        else:
            alerts.append(0)
    return alerts

# def computer_outliers_via_percentile : time_series, percentile_threshold -> alerts_p
# time_series : 1-D list of real numbers/integers
# percentile_threshold : vals must be greater than this percentile to trigger an alert
# window_size -> number of vals immediately proceeding the next val that are NOT considered
        # this param exists to avoid immediate 'pollution' of dataset
# alerts_p : at each time step, either 0 or 1 (0 = no alert, 1 = alert)
def computer_outliers_via_percentile(time_series, percentile_threshold, window_size, training_window_size):
    # okay, step one is we want to iterate through our time series
    # at each step, we want to compute the upper limit on some percentile
    # and then record if the next value is greater than this
    # then compute the number of vals in the current_window that exceed the limit delineating this percentile
    # and let's not start calculating alerts until we have at least window_size vals to evaluate over
    alerts = [float('nan') for i in range(0,window_size)]
    for i in range(window_size, len(time_series)):# - 1):
        beginning_of_window = max(0, i - training_window_size)
        training_range = time_series[beginning_of_window:i - window_size]
        next_val = time_series[i]
        #print "training range", training_range
        #print "percentile_threshold", percentile_threshold
        #print "training range", training_range
        val_threshold = np.nanpercentile(training_range, percentile_threshold)
        #print "precentile_threshold", percentile_threshold, "val_threshold", val_threshold, "next_val", next_val, "trigger?", next_val > val_threshold
        if next_val > val_threshold:
            alerts.append(1)
        else:
            alerts.append(0)
    #a = raw_input('keep going?')
    return alerts

# there's a widely used rule of thumb called the 3-Sigma Rule, which I'm going to attempt to look at
# here. I'm going to compute (CurVal - Mean) / StdDev. Then applying a threshold at three could give me the
# the three sigma rule, even though it doesn't necessarily need to be three.
def compute_sigma_normalized_vals(time_series, window_size, min_training_window):
    normalized_vals = [float('nan') for i in range(0,min_training_window)]
    for i in range(min_training_window, len(time_series)-1):
        start_training_window = max(0, i - window_size)
        training_window = time_series[start_training_window:i]
        next_val = time_series[i]
        # now calc StdDev and Mean of the training window.
        stddev = np.std(training_window)
        mean = np.mean(training_window)
        try:
            normalized_next_val = (next_val - mean) / stddev
        except ZeroDivisionError:
            normalized_next_val = float('nan')
        normalized_vals.append(normalized_next_val)
    return normalized_vals

def compute_sigma_alerts(sigma_vals, sigma_value):
    alerts = []
    for sigma_val in sigma_vals:
        if abs(sigma_val) >= sigma_value:
            alerts.append(1)
        else:
            alerts.append(0)
    return alerts






###---

def hbos_anomaly_detection(feature_array, training_window_size, initial_training_vals):
    clf_hbos = HBOS()
    anomaly_scores = []
    for i in range(0,initial_training_vals):
        anomaly_scores.append(float('nan'))
    # -1 b/c we are going to look one ahead for our test value
    for i in range(initial_training_vals, len(feature_array[:,0]) - 1):
        beginning_of_window = max(0, i - training_window_size)
        current_feature_array_slice = feature_array[beginning_of_window:i,:]
        clf_hbos.fit( current_feature_array_slice )
        next_time_slice = feature_array[i+1, :]
        next_value_test_pred = clf_hbos.predict_proba(next_time_slice)
        anomaly_scores.append(next_value_test_pred)
    return anomaly_scores

# calc_fp_and_tp calcs (TP,FP) for a stream of alerts. We need to calculate it for all alerts
# okay, so it clearly needs a a set of alert times, plus ability to extract time_granularity,
# plus exfil_start, exfil_end, and wiggle room
# and what does it return?? it returns a dict of params->(TP,FP)
def calc_all_fp_and_tps(params_to_alerts, exfil_start, exfil_end, wiggle_room, time_gran_to_attack_labels):
    params_to_tp_fp = {}
    for params,alerts in params_to_alerts.iteritems():
        params_to_tp_fp[params] = calc_fp_and_tp(alerts, exfil_start, exfil_end, wiggle_room, int(params[2]),
                                                 time_gran_to_attack_labels)
    return params_to_tp_fp

# okay, so at this point we have (hopefully) a dict mapping (params) -> (TP/FP)
# now we need to construct those tables and also an RC curve
# note: we are going to need a separate table/curve for each tuple of
#       (current_metric_name, container_or_class, time_interval,window_size, num_outlier_vals_in_window)
#       (tho this is somewhat academic b/c time_interval,window_size, num_outlier_vals_in_window are fixed by me at this stage)
# and then we can vary the percentile_threshold in order to get the actual curve itself
# for reference; the tuple that goes into params is
#       (current_metric_name, container_or_class, time_interval, percentile_threshold, window_size, num_outlier_vals_in_window)
def organize_tpr_fpr_results(params_to_tp_fp, percentile_thresholds):
    # okay, let's start by extracting all possible relevant param values

    params_to_method_to_tpr_fpr = {}
    all_metrics = []
    all_container_or_class = []
    all_time_interval = []
    all_window_size = []
    all_num_outlier_vals_in_window = []
    all_anom_algos = []
    for params, tp_fp in params_to_tp_fp.iteritems():
        current_metric = params[0]
        current_container_or_class = params[1]
        current_time_interval = params[2]
        current_window_size = params[4]
        current_num_outlier_vals_in_window = params[5]
        current_anom_algo = params[6]

        if current_metric not in all_metrics:
            all_metrics.append(current_metric)
        if current_container_or_class not in all_container_or_class:
            all_container_or_class.append(current_container_or_class)
        if current_time_interval not in all_time_interval:
            all_time_interval.append( current_time_interval )
        if current_window_size not in all_window_size:
            all_window_size.append( current_window_size )
        if current_num_outlier_vals_in_window not in all_num_outlier_vals_in_window:
            all_num_outlier_vals_in_window.append( current_num_outlier_vals_in_window )
        if current_anom_algo not in all_anom_algos:
            all_anom_algos.append(current_anom_algo)

    # okay, now we know all the values that we should iterate over (not we are assuming that all possible combinations
    # exist, which is fine for now, but at some point it probably will not be...)
    for current_container_or_class in all_container_or_class:
        for current_time_interval in all_time_interval:
            for current_window_size_index in range(0,len(all_window_size)):
                current_window_size = all_window_size[current_window_size_index]
                current_num_outlier_vals_in_window = all_num_outlier_vals_in_window[current_window_size_index]
                #table_vals = {}
                for current_metric in all_metrics:

                    current_tpr = []
                    current_fpr = []

                    #print "all_num_outlier_vals_in_window", all_num_outlier_vals_in_window
                    #print "params_to_tp_fp", params_to_tp_fp
                    for percentile_threshold in percentile_thresholds:

                        current_key = (current_metric, current_container_or_class, current_time_interval,
                                                       percentile_threshold, current_window_size, current_num_outlier_vals_in_window, current_anom_algo)
                        #print params_to_tp_fp
                        if current_key in params_to_tp_fp:
                            current_tpr_val = params_to_tp_fp[current_key][0]

                            current_fpr_val = params_to_tp_fp[(current_metric, current_container_or_class, current_time_interval,
                                                           percentile_threshold, current_window_size, current_num_outlier_vals_in_window, current_anom_algo)][1]
                            current_tpr.append( current_tpr_val )
                            current_fpr.append( current_fpr_val )

                    relevant_params = (current_container_or_class, current_time_interval, current_window_size, current_num_outlier_vals_in_window, current_anom_algo)
                    # maybe we should use this opportunity to store this in a nice way
                    if relevant_params not in params_to_method_to_tpr_fpr:
                        params_to_method_to_tpr_fpr[relevant_params] = {}
                    params_to_method_to_tpr_fpr[relevant_params][current_metric] = (current_tpr, current_fpr)

    return params_to_method_to_tpr_fpr

def store_organized_tpr_fpr_results(params_to_method_to_tpr_fpr, tpr_fpr_file, percentile_thresholds):
    with open(tpr_fpr_file + 'tpr_fpr.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow([("percentile_thresholds"), percentile_thresholds])
        for value_name, value in params_to_method_to_tpr_fpr.iteritems():
            for method, tpr_fpr in value.iteritems():
                spamwriter.writerow([(value_name, method),
                                     ([i if not math.isnan(i) else (None) for i in tpr_fpr[0]],
                                      [i if not math.isnan(i) else (None) for i in tpr_fpr[01]])])

    # let's also put it into a csv file (that might be easier to look-at/processor-in-another-program)
    # let's make a dataframe of the results and then use the to_csv functionality of pandas...
    # index would be TPR or FPR
    # columns would be the method name (which includes things like percentile thresholds, etc.)
    column_names = []
    column_vals = []
    for params, method_to_tpr_fpr in params_to_method_to_tpr_fpr.iteritems():
        print method_to_tpr_fpr
        for method, tpr_fpr in method_to_tpr_fpr.iteritems():
            for tpr_index in range(0,len(tpr_fpr[0])):
                current_tpr = tpr_fpr[0][tpr_index]
                current_fpr = tpr_fpr[1][tpr_index]
                current_percentile = percentile_thresholds[tpr_index]
                current_column_name = method + '_' + params[0] + '_' + str(params[1]) + '_' + str(params[2]) + '_' + \
                                     str(params[3]) + '_' + str(params[4]) + str(current_percentile)
                column_names.append( current_column_name )
                #print "pre-csv vals", [current_tpr, current_fpr], tpr_index, current_column_name
                column_vals.append( [current_tpr, current_fpr] )

    index = ['TPR', 'FPR']
    tpr_fpr_array = np.array(column_vals)
    tpr_fpr_array = tpr_fpr_array.T
    tpr_fpr_dataframe = pandas.DataFrame(tpr_fpr_array, index=index, columns=column_names)
    tpr_fpr_dataframe.to_csv(tpr_fpr_file + 'tpr_fpr_easy_to_read.csv', na_rep='?')
    #time.sleep(300)


def make_roc_graphs(params_to_method_to_tpr_fpr, base_ROC_name):
    #print "params_to_method_to_tpr_fpr", type(params_to_method_to_tpr_fpr)#, params_to_method_to_tpr_fpr
    #print  "params_to_method_to_tpr_fpr[0]", type(params_to_method_to_tpr_fpr[0])#, params_to_method_to_tpr_fpr[0]
    params_to_method_to_tpr_fpr = params_to_method_to_tpr_fpr[0]
    for params, method_to_tpr_fpr in params_to_method_to_tpr_fpr.iteritems():
        for method, tpr_fpr in method_to_tpr_fpr.iteritems():
            current_metric = method
            current_container_or_class = params[0]
            current_time_interval = params[1]
            current_window_size = params[2]
            current_num_outlier_vals_in_window = params[3]
            current_anom_algo = params[4]
            # okay, this will definitly need to be modified in the future
            title = current_metric + "_" + current_container_or_class + "_" + str(current_time_interval) + "-sec_"\
                    + str(current_window_size) + "_" + str(current_num_outlier_vals_in_window) + '_' + current_anom_algo
            plot_name = base_ROC_name + "_" + title

            # we have don't wanna have tons of graphs where literally nothing happens
            tpr_changes = False
            fpr_changes = False
            for i in range(1,len(tpr_fpr[1])):
                if tpr_fpr[0][0] != tpr_fpr[0][i] or tpr_fpr[0][i] != 0:
                    tpr_changes = True
                    break
                if tpr_fpr[1][0] != tpr_fpr[1][i]:
                    fpr_changes = True
                    break

            if tpr_changes or fpr_changes:
                # another problem: we get wierd lines b/c fpr isn't in order
                # from https://stackoverflow.com/questions/9764298/is-it-possible-to-sort-two-listswhich-reference-each-other-in-the-exact-same-w/9764364
                print "old tpr_fpr", tpr_fpr
                fpr,tpr = zip(*sorted( zip(tpr_fpr[1], tpr_fpr[0])))
                print "new tpr_fpr", tpr,fpr

                construct_ROC_curve(list(fpr), list(tpr), title, plot_name)

def read_organized_tpr_fpr_file(alert_file):
    params_to_method_to_tpr_fpr = {}
    percentile_list = []
    with open(alert_file, 'r') as csvfile:
        csvread = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # note: effectively indexing in at 1 below b/c put percentile_thresholds in first line
        first = True
        for row in csvread:
            if first:
                first = False
                #print row
                print row[1]
                percentile_list = ast.literal_eval(row[1])
                print "percentile_list", percentile_list
                continue
            #print row
            #print row[0]
            row_zero = ast.literal_eval(row[0])
            row_one = ast.literal_eval(row[1])
            #print row_zero
            #print row_one
            params = row_zero[0]
            method = row_zero[1]
            tprs = row_one[0]
            fprs = row_one[1]
            #print "params,method", params,method
            #print "tprs,fprs", tprs,fprs
            tpr = [i if i != (None) else float('nan') for i in tprs ]
            fpr = [i if i != (None) else float('nan') for i in fprs ]

            if params not in params_to_method_to_tpr_fpr:
                params_to_method_to_tpr_fpr[params] = {}

            params_to_method_to_tpr_fpr[params][method] = (tpr, fpr)
    return params_to_method_to_tpr_fpr, percentile_list

# def choose_tpr_fpr_pair(tpr_fpr) : tuple_of_tprs_and_fprs -> (tpr, fpr)
# the input is a tuple of two lists. [0] is a list of TPRs and [1] is a list of FPRs
def choose_tpr_fpr_pair(tpr_fpr):
    tprs = tpr_fpr[0]
    fprs = tpr_fpr[1]
    # so we are going to use Youden's J statistic as a way to decide which tpr/fpr pair to use ATM
    # the formulat is TPR-TNR
    best_j_stat_so_far = -1.1 # this is worse than is even possible
    best_j_stat_index = -1
    for i in range(0, len(tprs)):
        current_j_stat = tprs[i] - fprs[i]
        if current_j_stat > best_j_stat_so_far:
            best_j_stat_so_far = current_j_stat
            best_j_stat_index = i
    return (tprs[best_j_stat_index], fprs[best_j_stat_index]), best_j_stat_index



# attacks_to_alerts is a dictionary that maps the name of an attack to the relevant file with the alert results
# (in order to make the graphs). This isn't really a normal part of the pipeline, so it is okay that we are assuming that
# that there is no reason we'd have the relevant info already loaded
# goal: attacks_to_params_to_method_to_tpr_fpr
def make_tables(attack_to_alert_file, base_table_name):
    attacks_to_params_to_method_to_tpr_fpr = {}
    attacks_to_percentile_thresh_list = {}
    for attack, alert_file in attack_to_alert_file.iteritems():
        if attack not in attacks_to_params_to_method_to_tpr_fpr:
            attacks_to_params_to_method_to_tpr_fpr[attack] = {}
        if attack not in attacks_to_percentile_thresh_list:
            attacks_to_percentile_thresh_list[attack] = {}

        # okay, so if we read in the file, then we get params_to_method_to_tpr_fpr automatically
        # then we can just stick it in a dictionary along with attacks_to_params_to_method_to_tpr_fpr...
        params_to_method_to_tpr_fpr, percentile_thresh_list = read_organized_tpr_fpr_file(alert_file)
        attacks_to_params_to_method_to_tpr_fpr[attack] = params_to_method_to_tpr_fpr
        attacks_to_percentile_thresh_list[attack] = percentile_thresh_list

    construct_tables(attacks_to_params_to_method_to_tpr_fpr, base_table_name, attacks_to_percentile_thresh_list)

def construct_tables(attacks_to_params_to_method_to_tpr_fpr, base_table_name, attacks_to_percentile_thresh_list):
    # gotta convert params_to_attacks_to_method_tpr_fpr to attacks_to_params_to_tp_fp
    # goal : params_to_attacks_toparams_to_attacks_to_method_tpr_fpr_method_tpr_fpr
    params_to_attacks_to_method_to_tpr_fpr = {}
    for attack, params_to_method_to_tpr_fpr in attacks_to_params_to_method_to_tpr_fpr.iteritems():
        for params, method_to_tpr_fpr in params_to_method_to_tpr_fpr.iteritems():
            if params not in params_to_attacks_to_method_to_tpr_fpr:
                params_to_attacks_to_method_to_tpr_fpr[params]= {}
            if attack not in params_to_attacks_to_method_to_tpr_fpr[params]:
                params_to_attacks_to_method_to_tpr_fpr[params][attack] = {}
            params_to_attacks_to_method_to_tpr_fpr[params][attack] = method_to_tpr_fpr

    actually_construct_tables(params_to_attacks_to_method_to_tpr_fpr, base_table_name, attacks_to_percentile_thresh_list)

# okay, wait this is a lot harder than I originally thought... b/c it involves integrating a bunch of attacks
# and bunch of anomaly detection methods... okay, let's start from the most concrete thing that I can think of
# and then work outwards
# NOTE: params_to_attacks_to_method_tpr_fpr is going to take some fancy manipulation in another function
def actually_construct_tables(params_to_attacks_to_method_tpr_fpr, base_table_name, attacks_to_percentile_thresh_list):
    attack_and_method_to_best_tpr_fpr = {} # (attack, method) -> best_(tpr, fpr)
    attack_and_method_to_params_for_best_tpr_fpr = {} # (attack, method) -> (params) (note: these params gave the line from above)
    #attack_and_method_to_best_percentile_thresh = {}
    for params, attacks_to_method_to_tpr_fpr in params_to_attacks_to_method_tpr_fpr.iteritems():
        for attacks, method_tpr_fpr in attacks_to_method_to_tpr_fpr.iteritems():
            #print "attacks_to_method_to_tpr_fpr", attacks_to_method_to_tpr_fpr
            #print "attacks", attacks
            # okay, so rows are attacks
            # and columns are tpr or fpr (need two tables)
            tpr_table = pandas.DataFrame(0.0, index = attacks_to_method_to_tpr_fpr.keys(), columns= method_tpr_fpr.keys())
            fpr_table = pandas.DataFrame(0.0, index = attacks_to_method_to_tpr_fpr.keys(), columns= method_tpr_fpr.keys())
            for method, tpr_fpr in method_tpr_fpr.iteritems():

                if (attacks, method) not in attack_and_method_to_best_tpr_fpr:
                    attack_and_method_to_best_tpr_fpr[(attacks, method)] = (-1.0, -1.0)
                    #attack_and_method_to_best_percentile_thresh[(attacks, method)] = -1
                #print "params_to_attacks_to_method_tpr_fpr", params_to_attacks_to_method_tpr_fpr
                #print "tpr_fpr", tpr_fpr
                #print "method", method, "tpr_fpr", tpr_fpr
                #print type(tpr_fpr)
                #print tpr_fpr[0]

                best_tpr_fpr_pair, best_tpr_fpr_index = choose_tpr_fpr_pair(tpr_fpr)
                best_tpr_fpr_percentile_thresh = attacks_to_percentile_thresh_list[attacks][best_tpr_fpr_index]

                print "tpr_fpr", tpr_fpr
                print method, "best_tpr_fpr_pair", best_tpr_fpr_pair, params

                tpr_table[attacks, method] = best_tpr_fpr_pair[0]
                fpr_table[attacks, method] = best_tpr_fpr_pair[1]

                # now gotta check if the best_tpr_fpr_pair is better than best result found so far for this (attack,method)
                new_tprs = [ best_tpr_fpr_pair[0], attack_and_method_to_best_tpr_fpr[(attacks, method)][0]]
                new_fprs = [ best_tpr_fpr_pair[1], attack_and_method_to_best_tpr_fpr[(attacks, method)][1]]
                new_tpr_fpr_selection_list = ( new_tprs, new_fprs )
                print "new_tpr_fpr_selection_list", new_tpr_fpr_selection_list, "params", params
                best_tpr_fpr_pair_over_params, best_tpr_fpr_pair_over_params_index = choose_tpr_fpr_pair( new_tpr_fpr_selection_list )
                #if best_tpr_fpr_pair_over_params_index == 0:
                #    attack_and_method_to_best_percentile_thresh[(attacks, method)] = best_tpr_fpr_percentile_thresh

                print "best_tpr_fpr_pair_over_params", best_tpr_fpr_pair_over_params, "method", method
                attack_and_method_to_best_tpr_fpr[(attacks, method)] = best_tpr_fpr_pair_over_params
                print "after (theoretical) swap", attack_and_method_to_best_tpr_fpr[(attacks, method)]

                if best_tpr_fpr_pair_over_params[0] ==  best_tpr_fpr_pair[0] and \
                                best_tpr_fpr_pair_over_params[1] == best_tpr_fpr_pair[1]:
                    attack_and_method_to_params_for_best_tpr_fpr[(attacks, method)] = [params, best_tpr_fpr_percentile_thresh] ## TODO <-- this is where the thresh would go
                    print "this one chosen!!!"
            # okay, so want I want to do is store the dataframe now...
            # note: there is no concept of something like a title, so that info will need to be
            # embedded into the filename...
            params_str = ''
            for param in params:
                params_str += str(param) + "_"
            tpr_table.to_html(base_table_name + '_tpr_' + params_str + '.html')
            fpr_table.to_html(base_table_name + '_tpr_' + params_str + '.html')
    # okay, well now attack_and_method_to_best_tpr_fpr should be filled. Recall that the key is (attack, method)
    # now we want to loop through the attacks and methods and construct a dataframe. Let's just make one and put the
    # j statistic in there.
    # okay, after that we need to elimate the useless methods (e.g., w/ tpr of 0 and whatnot)

    # okay, so it looks like the easiest way to get rid of the useless stuff is to do it now, before calling the constructor
    list_of_things_to_delete_from_dict = []
    for attack_method, tpr_fpr in attack_and_method_to_best_tpr_fpr.iteritems():
        # NOTE: I am intentionally deleting all of the 'simple_angles' stuff b/c
        #if tpr_fpr[0] - tpr_fpr[1] <= 0 or 'Simple Angle' in attack_method[1]:
        if 'Simple Angle' in attack_method[1] or 'Sum of Appserver Node Degrees' in attack_method[1]:
            list_of_things_to_delete_from_dict.append( attack_method)

    for item in list_of_things_to_delete_from_dict:
        del attack_and_method_to_best_tpr_fpr[item]

    list_of_attacks = list(set([i[0] for i in attack_and_method_to_best_tpr_fpr.keys()]))
    list_of_attacks.sort()
    list_of_methods = list(set([i[1] for i in attack_and_method_to_best_tpr_fpr.keys()]))
    print "list_of_methods", list_of_methods
    greatest_hits_youden_table = pandas.DataFrame(0.0, index=list_of_attacks, columns=list_of_methods )
    greatest_hits_tpr_table = pandas.DataFrame(0.0, index=list_of_attacks, columns=list_of_methods )
    greatest_hits_fpr_table = pandas.DataFrame(0.0, index=list_of_attacks, columns=list_of_methods )
    greatest_hits_param_table = pandas.DataFrame(0.0, index=list_of_attacks, columns=list_of_methods )
    greatest_hits_param_table = greatest_hits_param_table.astype(object)
    for attack_method, tpr_fpr in attack_and_method_to_best_tpr_fpr.iteritems():
        attack = attack_method[0]
        method = attack_method[1]
        params = attack_and_method_to_params_for_best_tpr_fpr[attack_method]
        print "attack", attack, "method", method, tpr_fpr[0] - tpr_fpr[1], params
        #print list_of_attacks
        #print list_of_methods
        greatest_hits_youden_table.loc[attack, method] = tpr_fpr[0] - tpr_fpr[1]
        greatest_hits_tpr_table.loc[attack, method] = tpr_fpr[0]
        greatest_hits_fpr_table.loc[attack, method] = tpr_fpr[1]
        # going to make a string of params for easier display
        #param_string = 'Node Granularity ' + str(params[0]) + '\n' + 'Time Granularity: ' + str(params[1]) + '\nAnomaly Window: '\
        #               + str(params[2]) +'\nAnomaly Alerts Threshold: ' + str(params[3])
        #display_tuple = ('Node Granularity: '+ str(params[0]), 'Time Granularity: ' + str(params[1]))
        display_tuple = (str(params[0][0]), str(params[0][1]), str(params[1]))
        greatest_hits_param_table.loc[attack, method] =  display_tuple
    print "attack_and_method_to_best_tpr_fpr"
    for key,val in attack_and_method_to_best_tpr_fpr.iteritems():
        print key,val

    largest_entry_in_column = []
    for column in greatest_hits_youden_table:
        print(greatest_hits_youden_table[column], type(greatest_hits_youden_table[column]),
              greatest_hits_youden_table[column].max())
        largest_entry_in_column.append( greatest_hits_youden_table[column].max() )

    print "largest_entry_in_column (before)", largest_entry_in_column
    print "list_of_methods", list_of_methods
    largest_entry_in_column, list_of_methods = zip(*sorted(zip(largest_entry_in_column, list_of_methods ), reverse=True))
    print "list of methods (right out)", list_of_methods
    #list_of_methods = list(list_of_methods)
    #list_of_methods.sort()
    #list_of_methods = sorted(list_of_methods)
    print "largest_entry_in_column (after)", largest_entry_in_column
    print "list_of_columns", list_of_methods
    greatest_hits_youden_table = greatest_hits_youden_table.reindex(list_of_methods, axis=1)
    greatest_hits_tpr_table = greatest_hits_tpr_table.reindex(list_of_methods, axis=1)
    greatest_hits_fpr_table = greatest_hits_fpr_table.reindex(list_of_methods, axis=1)
    greatest_hits_param_table = greatest_hits_param_table.reindex(list_of_methods, axis=1)

    greatest_hits_youden_table.to_html(base_table_name + '_youden_greatest_hits' + '.html')
    greatest_hits_tpr_table.to_html(base_table_name + '_tpr_greatest_hits' + '.html')
    greatest_hits_fpr_table.to_html(base_table_name + '_fpr_greatest_hits' + '.html')
    greatest_hits_param_table.to_html(base_table_name + '_params_greatest_hits' + '.html')

    with open(base_table_name + 'parameters_corresponding_to_youden_greatest_hits.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for value_name, value in attack_and_method_to_params_for_best_tpr_fpr.iteritems():
            spamwriter.writerow([value_name, value])


        # okay, so what do I want to do here??? Well, I have a bunch of these tables (one table per set of params)
    # but I don't actually really want that. What I want is for there to be ONE table, w/, say, a 'best hits' of
    # the different params. (also there's no point in showing ones that have a 0 TPR on every attack w/ every set
    # of params). Okay, so it looks like there are actually two components of this (1) choose the best set of params
    # for every metric. And then if that is STILL not good enough, we can get rid of it altogether.

    # okay, well let's keep deciding which metric is best via Youden's J statistic (TPR - FPR)

def save_alerts_to_csv(params_to_alerts, alert_file, time_gran_to_attack_labels):
    # okay, let's just construct a dataframe here and then we can save that.
    # okay, but how many dataframes/csv files do we actaully want???
    # well, here is the type of key for params_to_alerts
    # (current_metric_name, container_or_class, time_interval, percentile_threshold, window_size, num_outlier_vals_in_window,anomaly_detection_algo)

    # okay, so we are probably going to want to split the csv's by time interval... other than that I feel like we should just
    # stick it all in the name of the column. So I'm going to want to iterate through them by time_grans...
    counter = 0
    for time_gran in time_gran_to_attack_labels.keys():
        # okay, so the goal now is to compute the stuff that goes into alert_dataframe...
        column_names = []
        column_vals = []
        times = []
        for params, alerts in params_to_alerts.iteritems():
            if params[2] != time_gran:
                continue

            counter += 1
            # index = the times
            # columns = the names of the thing generating the alerts
            #if params[6] == 'exceeds_threshold_in_window':
            current_index_name = params[0] + '_' + params[1] + '_' + str(params[3]) + '_' + str(params[4]) + '_' + \
                                     str(params[5]) + '_' + params[6]
            #print "!", current_index_name, "!"
            #print "\t", len(alerts), alerts
            column_names.append(current_index_name)
            column_vals.append(alerts)
            times = [i * time_gran for i in range(0, len(alerts))]

        #print "column_names", column_names
        #print len(column_names), len(column_vals)
        #print "counter", counter
        #print column_vals
        array_column_vals = np.array(column_vals)
        #print array_column_vals
        array_column_vals = array_column_vals.T
        #print array_column_vals
        alert_dataframe = pandas.DataFrame(array_column_vals, index=times, columns=column_names)
        #print len(time_gran_to_attack_labels[time_gran]), len(times)
        relevant_attack_labels = time_gran_to_attack_labels[time_gran][:len(times)]
        alert_dataframe['label'] = relevant_attack_labels
        current_alert_csv_file = alert_file + '_alerts_at_' + str(time_gran) +'.csv'
        alert_dataframe.to_csv(current_alert_csv_file, na_rep='?')

def aggregate_feature_csvs(paths_to_csvs, aggregate_file):
    f_aggreg = open(aggregate_file, 'w')
    with open(paths_to_csvs[0], 'r') as f:
        for line in f:
            f_aggreg.write(line)
    for path_to_csv in paths_to_csvs[1:]:
        with open(path_to_csv, 'r') as f:
            f.next()
            for line in f:
                f_aggreg.write(line)
    f_aggreg.close()

def calc_anomaly_score(row):
    #print row['Change-Point Detection Node Degree50_5_unprocessed_container_mod_z_score']
    return -0.0002 * float(row['Change-Point Detection Node Degree50_5_unprocessed_container_mod_z_score']) +\
    0.0346 * float(row['Density50_5_unprocessed_container_mod_z_score']) +\
    0.004  * float(row['Communication Between Pods not through VIPs50_5_unprocessed_container_mod_z_score']) +\
    -0.0172 * float(row['Weighted Average Path Length50_5_unprocessed_container_mod_z_score']) +\
    -0.0002 * float(row['Weighted Overall Reciprocity50_5_unprocessed_container_mod_z_score']) +\
    -0.004  * float(row['Fraction of Communication Between Pods not through VIPs50_5_unprocessed_container_mod_z_score']) +\
    -0.0002 * float(row['Density50_5_class_mod_z_score']) +\
    -0.0318 * float(row['Change-Point Detection Node Instrength50_5_class_mod_z_score']) +\
    0.0131 * float(row['Weighted Average Path Length50_5_class_mod_z_score']) +\
    -0.0228 * float(row['Change-Point Detection Node Eigenvector_Centrality50_5_class_mod_z_score']) +\
    -0.0002 * float(row['Weighted Overall Reciprocity50_5_class_mod_z_score']) +\
    0.0002 * float(row['Change-Point Detection Node Non-Reciprocated In-Weight50_5_class_mod_z_score']) +\
    -0.0001 * float(row['Change-Point Detection Node Degree50_5_container_mod_z_score']) +\
    0.0008 * float(row['Weighted Average Path Length50_5_container_mod_z_score']) +\
    -0.0031 * float(row['Weighted Overall Reciprocity50_5_container_mod_z_score']) +\
    -0.0034 * float(row['Unweighted Average Path Length50_5_container_mod_z_score']) +\
    0.0032 * float(row['Unweighted Overall Reciprocity50_5_container_mod_z_score']) +\
    0      * float(row['Change-Point Detection Node Non-Reciprocated Out-Weight50_5_container_mod_z_score'])

def calc_anomaly_score_thirty_gran(row):
    return 0.0454 * float(row['Weighted Average Path Length50_5_unprocessed_container_mod_z_score']) + \
    0.0472 * float(row['Change-Point Detection Node Eigenvector_Centrality50_5_unprocessed_container_mod_z_score']) + \
    -0.0541 * float(row['Change-Point Detection Node Outstrength50_5_unprocessed_container_mod_z_score']) + \
    0.0401 * float(row['Change-Point Detection Node Betweeness Centrality50_5_unprocessed_container_mod_z_score']) +\
    0.1198 * float(row['Density50_5_container_mod_z_score']) +\
    0.0023 * float(row['Communication Between Pods not through VIPs50_5_container_mod_z_score']) +\
    0.0474 * float(row['Change-Point Detection Node Eigenvector_Centrality50_5_container_mod_z_score']) +\
    -0.0429 * float(row['Change-Point Detection Node Outstrength50_5_container_mod_z_score']) +\
    -0.1321 * float(row['Unweighted Average Path Length50_5_container_mod_z_score']) +\
     -0.0002 * float(row['Change-Point Detection Node Non-Reciprocated In-Weight50_5_container_mod_z_score']) +\
     -0.0002 * float(row['Change-Point Detection Node Non-Reciprocated Out-Weight50_5_container_mod_z_score']) +\
     -0.0022 * float(row['Fraction of Communication Between Pods not through VIPs50_5_container_mod_z_score']) +\
     -0.0001 * float(row['Density50_5_class_mod_z_score']) +\
     -0.0459 * float(row['Weighted Average Path Length50_5_class_mod_z_score']) +\
     -0.0614 * float(row['Change-Point Detection Node Outstrength50_5_class_mod_z_score']) +\
     -0.0652 * float(row['Weighted Overall Reciprocity50_5_class_mod_z_score'])

def calc_anomaly_score_sixty_gran(row):
      return -0.1915 * float(row['Density50_5_container_mod_z_score']) +\
      0.0027 * float(row['Communication Between Pods not through VIPs50_5_container_mod_z_score']) +\
      0.0092 * float(row['Unweighted Overall Reciprocity50_5_container_mod_z_score']) +\
      0.2325 * float(row['Density50_5_unprocessed_container_mod_z_score']) +\
     -0.0554 * float(row['Weighted Average Path Length50_5_unprocessed_container_mod_z_score']) +\
     -0.0484 * float(row['Change-Point Detection Node Load Centrality50_5_unprocessed_container_mod_z_score']) +\
     -0.0029 * float(row['Fraction of Communication Between Pods not through VIPs50_5_unprocessed_container_mod_z_score']) +\
      0.0007 * float(row['Density50_5_class_mod_z_score']) +\
     -0.0566 * float(row['Change-Point Detection Node Betweeness Centrality50_5_class_mod_z_score'])

def aggregate_csv_recipe():
    # okay, so what I am going to want to do here is loop through
    paths_to_csvs = [
        '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/alerts/wordpress_six_rep_2_',
        '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/alerts/wordpress_seven_rep_3_',
        '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/alerts/wordpress_eight_',
        '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/alerts/sockshop_eleven_',
        '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/alerts/sockshop_nine_better_exfil_']

    paths_to_csvs = [
        '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/alerts/wordpress_six_rep_4_',
        '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/alerts/sockshop_twelve_',
    ]

    paths_to_csvs = [
        '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/alerts/wordpress_eight_',
        '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/alerts/sockshop_nine_better_exfil_'
    ]

    paths_to_csvs = [
        '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_eleven_dns_1sec/alerts/wordpress_eleven_dns_1sec_',
        '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/wordpress_eleven/alerts/wordpress_eleven_',
        '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/alerts/wordpress_eight_',
        '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/alerts/sockshop_nine_better_exfil_',
        '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/alerts/wordpress_six_rep_4_',
        '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/alerts/sockshop_twelve_',
        '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/alerts/sockshop_eleven_'
    ]

    #aggregate_file = '/Volumes/Seagate Backup Plus Drive/experimental_data/aggregate_on_path_'
    aggregate_file = '/Volumes/Seagate Backup Plus Drive/experimental_data/aggregate_dns_path_'
    aggregate_file = '/Volumes/Seagate Backup Plus Drive/experimental_data/aggregate_dns_1sec_2sec_'
    aggregate_file = '/Volumes/Seagate Backup Plus Drive/experimental_data/aggregate_all_exps_'
    second_part_of_file_name = 'mod_z_score_sub_'
    time_grans = [10, 30, 60]
    for time_gran in time_grans:
        current_paths_to_csvs = [i + second_part_of_file_name +  str(time_gran) + '.csv' for i in paths_to_csvs]
        print current_paths_to_csvs
        current_aggregate_file = aggregate_file + str(time_gran) + '.csv'
        aggregate_feature_csvs(current_paths_to_csvs, current_aggregate_file)

###### ------ ####### -------- ####### -------- ######## -------- ######## -------- #######
#  [blank], and provided the [blank] are correct.okay, step for after lunch:
#   (1) finish construct_tables_and_graphs [[okay, well, good enough for now (complete pipeline really needs to be
            # built / run before I can test tho...]
#   (2) hook into pipeline (might need to change params and stuff)
#       (2b) should I finally make the json switch??
            # KEEP GOING FROM (2)/(3)
#   (3) actually try to run the whole pipeline (need to choose something w/ already existing metrics, tho, so
#       that only my additions to the pipeline will be tested)
#   (4) watch ML vid + finish HW f
#   (5) start arch. HW (prob by ready section of book on pthreads)