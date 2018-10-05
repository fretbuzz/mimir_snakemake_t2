import numpy as np
import csv
import math
import ast
import matplotlib.pyplot as plt
import pandas
import time

# okay, so this file is going to contain some functions that actually trigger the alerts.
# in analyze_edgefiles, we have several techniques for analyzing the graphs by computing some metrics
# and making some (rather nice looking) graphs. But we end up with, essentially, several 1-D time series.
# we need to actually trigger alerts based on these time series. That's where the methods in this file
# come into play.

# func: multiple_outliers_in_window: time_series window_size num_outlier_vals_in_window -> list_of_alert_times
# time_series : 1-D list of 0/1, with 0 = no alert, 1 = alert
# window_size : number of values over which to sum outliers
# num_outlier_vals_in_window : must be >= this number of values in the window in order for an alert to be triggered
# note: cannot do node attribution here b/c we only have the 1-D time series, node attribution would need to be done seperately
def multiple_outliers_in_window(time_series, window_size, num_outlier_vals_in_window):
    alerts = [0 for i in range(0,window_size)]
    for i in range(window_size, len(time_series) - window_size):
        #print "i", i, "window_size", window_size
        current_window = time_series[i:i+window_size]
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
    alerts = [0 for i in range(0,window_size+1)]
    for i in range(window_size+1, len(time_series) - 1):
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

# okay, cool, so what should I do now... well we need the actual function that'd go through these
# (making one of those curves would also be nice)... so it'd take our big ol' dictionary of vals,
# along with the params for our anomaly functions, and maybe whether or not to make graphs? then
# we can just crank through them (let's fix the window_size and number for now, b/c otherwise
# I feel like it'd take forever...)
# (also, there's some modification that's going to be needed earlier in the pipeline, but don't worry
# about it for now)

# def compute_alerts : calculated_vals, percentile_thresholds_to_try, window_size, num_outlier_vals_in_window,
#                   time_interval_lengths, alert_file, calc_alerts_p -> set of vals
# calculated_vals : dict mapping (time_gran, node_gran) to a dict of (metric name) -> (time series of metric vals)
# percentile_thresholds_to_try : list of percentile thresholds to give to the anom detection methods (see above)
# window_size : a param for the anom detection methods (see above funcs)
# num_outlier_vals_in_window : a param for the anom detection methods (see above funcs)
# time_interval_lengths : only perform calcs on these time granularities
# alert_file : file to save alert results to
# calc_alerts_p : should we calculate the results or just read them from the file
def compute_alerts(calculated_vals, percentile_thresholds_to_try, window_size, num_outlier_vals_in_window,
                   time_interval_lengths, alert_file, calc_alerts_p, training_window_size):
    alert_file = alert_file + 'alerts.csv'
    if calc_alerts_p:
        params_to_alerts = {}
        for label, metric_time_series in calculated_vals.iteritems():
            container_or_class = label[1]
            time_interval = int(label[0])
            if time_interval not in time_interval_lengths:
                continue

            # okay, now let's feed the beast
            for current_metric_name, current_metric_time_series in calculated_vals[label].iteritems():
                for percentile_threshold in percentile_thresholds_to_try:
                    alert_series = computer_outliers_via_percentile(current_metric_time_series, percentile_threshold, window_size, training_window_size)
                    alerts = multiple_outliers_in_window(alert_series, window_size, num_outlier_vals_in_window)
                    params_to_alerts[(current_metric_name, container_or_class, time_interval, percentile_threshold, window_size,
                                      num_outlier_vals_in_window)] = alerts
        # okay, so let's save these vals to a file, so I can calc TP/FP and make graphs without re-running all of them
        with open(alert_file, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for value_name, value in params_to_alerts.iteritems():
                spamwriter.writerow([value_name, [i if not math.isnan(i) else (None) for i in value]])

        return params_to_alerts
    else:
        params_to_alerts = {}
        with open(alert_file, 'r') as csvfile:
            csvread = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in csvread:
                #print row
                params_to_alerts[row[0]] = [i if i != (None) else float('nan') for i in ast.literal_eval(row[1])]
                print row[0], params_to_alerts[row[0]]
        return params_to_alerts

# def calc_fp_and_tp : ??? -> (TP, FP)_for_every_step, (TP, FP)_for_the_entire_interval
# this function calcuates the tp/fp. Okay, cool, but what counts as a TP/FP. Detecting once it the range?
# detecting at every time stamp (seperately). Detecting @ each time step, every time? What does it mean???
# oh, yah... this is kinda a tricky thing to ask...
# well, how about we take an afraid-to-commit approach... let's compute two vals: one for every time step and
# one for that period in general (I think maybe like that stratosphere IPS guys might be useful too??)
# actually lets only do (TP, FP)_for_every_step
def calc_fp_and_tp(alert_times, exfil_start, exfil_end, wiggle_room, time_granularity):
    start_alert_time_index = exfil_start / time_granularity - int( wiggle_room / time_granularity)
    end_alert_time_index = exfil_end / time_granularity + int( wiggle_room / time_granularity)
    alerts_during_exfiltration = alert_times[start_alert_time_index:end_alert_time_index]
    true_positives = sum(alerts_during_exfiltration)
    false_negatives = len(alerts_during_exfiltration) - true_positives
    alerts_not_during_exfiltration = alert_times[0:start_alert_time_index] + alert_times[end_alert_time_index:]
    false_positives = sum(alerts_not_during_exfiltration)
    true_negatives = len(alerts_not_during_exfiltration) - false_positives

    # tpr (a.k.a sensitivity) = proportion of actual positives that we succesfully identifies
    tpr = float(true_positives) / (true_positives + false_negatives)

    # fpr = FP / (FP + TN)
    fpr = float(false_positives) / (true_negatives + false_positives )

    return tpr, fpr

# calc_fp_and_tp calcs (TP,FP) for a stream of alerts. We need to calculate it for all alerts
# okay, so it clearly needs a a set of alert times, plus ability to extract time_granularity,
# plus exfil_start, exfil_end, and wiggle room
# and what does it return?? it returns a dict of params->(TP,FP)
def calc_all_fp_and_tps(params_to_alerts, exfil_start, exfil_end, wiggle_room):
    params_to_tp_fp = {}
    for params,alerts in params_to_alerts.iteritems():
        params_to_tp_fp[params] = calc_fp_and_tp(alerts, exfil_start, exfil_end, wiggle_room, int(params[2]))
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
    for params, tp_fp in params_to_tp_fp.iteritems():
        current_metric = params[0]
        current_container_or_class = params[1]
        current_time_interval = params[2]
        current_window_size = params[4]
        current_num_outlier_vals_in_window = params[5]

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

    # okay, now we know all the values that we should iterate over (not we are assuming that all possible combinations
    # exist, which is fine for now, but at some point it probably will not be...)
    for current_container_or_class in all_container_or_class:
        for current_time_interval in all_time_interval:
            for current_window_size in all_window_size:
                for current_num_outlier_vals_in_window in all_num_outlier_vals_in_window:
                    #table_vals = {}
                    for current_metric in all_metrics:

                        current_tpr = []
                        current_fpr = []

                        #print "all_num_outlier_vals_in_window", all_num_outlier_vals_in_window
                        #print "params_to_tp_fp", params_to_tp_fp
                        for percentile_threshold in percentile_thresholds:

                            current_tpr_val = params_to_tp_fp[(current_metric, current_container_or_class, current_time_interval,
                                                           percentile_threshold, current_window_size, current_num_outlier_vals_in_window)][0]

                            current_fpr_val = params_to_tp_fp[(current_metric, current_container_or_class, current_time_interval,
                                                           percentile_threshold, current_window_size, current_num_outlier_vals_in_window)][1]
                            current_tpr.append( current_tpr_val )
                            current_fpr.append( current_fpr_val )

                        relevant_params = (current_container_or_class, current_time_interval, current_window_size, current_num_outlier_vals_in_window)
                        # maybe we should use this opportunity to store this in a nice way
                        if relevant_params not in params_to_method_to_tpr_fpr:
                            params_to_method_to_tpr_fpr[relevant_params] = {}
                        params_to_method_to_tpr_fpr[relevant_params][current_metric] = (current_tpr, current_fpr)

    return params_to_method_to_tpr_fpr

def store_organized_tpr_fpr_results(params_to_method_to_tpr_fpr, tpr_fpr_file, percentile_thresholds):
    with open(tpr_fpr_file, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow([("percentile_thresholds"), percentile_thresholds])
        for value_name, value in params_to_method_to_tpr_fpr.iteritems():
            for method, tpr_fpr in value.iteritems():
                spamwriter.writerow([(value_name, method),
                                     ([i if not math.isnan(i) else (None) for i in tpr_fpr[0]],
                                      [i if not math.isnan(i) else (None) for i in tpr_fpr[01]])])

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
            # okay, this will definitly need to be modified in the future
            title = current_metric + "_" + current_container_or_class + "_" + str(current_time_interval) + "-sec_"\
                    + str(current_window_size) + "_" + str(current_num_outlier_vals_in_window)
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

# okay, what we want to do here is to construct
# x_vals should be FPR
# y_vals should be TPR
def construct_ROC_curve(x_vals, y_vals, title, plot_name):
    plt.figure()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(title)
    plt.plot(x_vals, y_vals)
    plt.savefig( plot_name + '.png', format='png', dpi=1000)
    plt.close()

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




###### ------ ####### -------- ####### -------- ######## -------- ######## -------- #######
#  [blank], and provided the [blank] are correct.okay, step for after lunch:
#   (1) finish construct_tables_and_graphs [[okay, well, good enough for now (complete pipeline really needs to be
            # built / run before I can test tho...]
#   (2) hook into pipeline (might need to change params and stuff)
#       (2b) should I finally make the json switch??
            # KEEP GOING FROM (2)/(3)
#   (3) actually try to run the whole pipeline (need to choose something w/ already existing metrics, tho, so
#       that only my additions to the pipeline will be tested)
#   (4) watch ML vid + finish HW
#   (5) start arch. HW (prob by ready section of book on pthreads)