import numpy as np
import csv
import math
import ast
import matplotlib.pyplot as plt
import pandas
import time
from pyod.models.hbos import HBOS


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

def calc_modified_z_score(time_series, window_size, min_training_window):
    if len(time_series) < min_training_window:
        return [float('nan') for i in range(0,len(time_series))]
    modified_z_scores = [float('nan') for i in range(0,min_training_window)]
    for i in range(0,min_training_window):
        print time_series[i]
    for i in range(min_training_window, len(time_series)):
        start_training_window = max(0, i - window_size)
        training_window = time_series[start_training_window:i]
        next_val = time_series[i]

        # now let's actually calculate the modified z-score
        median = np.nanmedian(training_window)
        # MAD = mean absolute deviation
        MAD = np.nanmedian([np.abs(val - median) for val in training_window])
        if MAD:
            next_modified_z_score = 0.6754 * (next_val - median) / MAD
            #print "No ZeroDivisionError!"
        else:
            #print "ZeroDivisionError!"
            if (next_val - median) == 0:
                next_modified_z_score = 0.0
            else:
                next_modified_z_score = float('inf')
        next_modified_z_score = abs(next_modified_z_score) ## TODO: remove???
        ## behavior is funny if there are inf's so, let's put an upper bound of 1000
        next_modified_z_score = min(next_modified_z_score, 1000)
        print "median", median, "MAD", MAD,"next_modified_z_score",next_modified_z_score, "val", next_val, type(median), type(MAD), type(next_val)
        modified_z_scores.append(next_modified_z_score)
    return modified_z_scores

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
def compute_alerts(calculated_vals, percentile_thresholds_to_try, window_sizes, list_num_outlier_vals_in_window,
                   time_interval_lengths, alert_file, calc_alerts_p, training_window_size, csv_path, minimum_training_window):
    alert_file = alert_file + 'alerts.csv'
    time_gran_to_feature_dataframe = {}
    time_gran_to_mod_z_score_dataframe = {}

    if calc_alerts_p:
        params_to_alerts = {}
        time_grans = []
        time_gran_to_list_of_anom_metrics_applied = {}
        time_gran_to_list_of_anom_values = {}

        for label, metric_time_series in calculated_vals.iteritems():
            container_or_class = label[1]
            time_interval = int(label[0])
            list_of_anom_metrics_applied = []
            list_of_anom_values = []
            if time_interval not in time_gran_to_list_of_anom_metrics_applied:
                time_gran_to_list_of_anom_metrics_applied[time_interval] = []
            if time_interval not in time_gran_to_list_of_anom_values:
                time_gran_to_list_of_anom_values[time_interval] = []

            if time_interval not in time_grans:
                time_grans.append(time_interval)
            if time_interval not in time_interval_lengths:
                continue

            # okay, now let's feed the beast
            for current_metric_name, current_metric_time_series in calculated_vals[label].iteritems():
                for percentile_threshold in percentile_thresholds_to_try:
                    # this is where the different anoamly detection methods could be tried.
                    for i in range(0,len(window_sizes)):
                        # these two vals are just params for the 'exceeds_threshold_in_window' anom. detec. algo.
                        window_size = window_sizes[i]
                        num_outlier_vals_in_window = list_num_outlier_vals_in_window[i]

                        alert_series = computer_outliers_via_percentile(current_metric_time_series, percentile_threshold,
                                                                        window_size, training_window_size)
                        alerts = multiple_outliers_in_window(alert_series, window_size, num_outlier_vals_in_window)

                        anomaly_detection_algo = 'exceeds_threshold_in_window'
                        params_to_alerts[(current_metric_name, container_or_class, time_interval, percentile_threshold, window_size,
                                          num_outlier_vals_in_window,anomaly_detection_algo)] = alerts # anomaly_detection_algo

                sigma_values = [i/float(10) for i in range(0,60,5)]
                sigma_min_training_window = minimum_training_window
                sigma_window_size = training_window_size
                number_nans_in_this_time_series = [str(i) for i in current_metric_time_series].count('nan')
                modified_min_z_score_training_window = number_nans_in_this_time_series + minimum_training_window

                #print "current_metric_time_series", len(current_metric_time_series)
                print '----'
                #if 'Density' not in current_metric_name:
                #    continue
                modified_z_scorese = calc_modified_z_score(current_metric_time_series, sigma_window_size,
                                                           modified_min_z_score_training_window)
                ## TODO: this function does not seem to work. I need to get the dataframe to be correctly created
                ## and stored. Then use aggregator function to make the training file and then use weka to test
                ## the linear combination. Then integrate the linear combination in here somehow, so that I can
                ## generate the relevant ROC curve... (for now, I gotta do a quick detour to meeting prep, then
                ## meeting with ray, then meeting with istio, then class, and then I can hopefully finish this...)
                ## TODO on monday: fix the z-score thing (probably want to start  with more training data) and
                ## then do the thing above [[ i think this note is from the week of 10/15)
                ## okie...
                if 'Density' in current_metric_name:
                    print current_metric_name, modified_z_scorese
                    #time.sleep(30)

                list_of_anom_metrics_applied.append(current_metric_name + str(sigma_window_size) + '_' + \
                                                    str(sigma_min_training_window) + '_'+ container_or_class +\
                                                    '_mod_z_score')
                list_of_anom_values.append(modified_z_scorese)
                if 'New' in current_metric_name:
                    print current_metric_name, "raw_time_series", current_metric_time_series, len(current_metric_time_series)


                sigma_vals = compute_sigma_normalized_vals(current_metric_time_series, sigma_window_size,
                                                             sigma_min_training_window)
                for sigma_value in sigma_values:
                    # okay, maybe integrate the three-sigma thing here???
                    sigma_threshold = sigma_value
                    sigma_alerts = compute_sigma_alerts(sigma_vals, sigma_value)
                    params_to_alerts[(current_metric_name, container_or_class, time_interval, sigma_threshold, '',
                                      '','sigma_normalized')] = sigma_alerts # anomaly_detection_algo

            time_gran_to_list_of_anom_metrics_applied[time_interval].extend(list_of_anom_metrics_applied)
            time_gran_to_list_of_anom_values[time_interval].extend(list_of_anom_values)

        for time_gran in time_grans:
            #print "list_of_anom_values",len(list_of_anom_values),len(list_of_anom_values[0]), list_of_anom_values
            list_of_anom_values = time_gran_to_list_of_anom_values[time_gran]
            list_of_anom_metrics_applied = time_gran_to_list_of_anom_metrics_applied[time_gran]
            print "list_of_anom_values", list_of_anom_values
            print "list_of_anom_metrics_applied",list_of_anom_metrics_applied
            mod_z_score_array = np.array(list_of_anom_values)
            #print "pre_rotate_shape", mod_z_score_array.shape,mod_z_score_array
            mod_z_score_array = mod_z_score_array.T
            #print "mod_z_score_array",mod_z_score_array,mod_z_score_array.shape
            #print list_of_anom_metrics_applied
            #print "mod_z_score_array", mod_z_score_array
            #print list_of_anom_metrics_applied
            #print mod_z_score_array.shape
            #print mod_z_score_array[0]
            p = 0
            for series in mod_z_score_array:
                print len(series),
                try:
                    print list_of_anom_metrics_applied[p]
                except:
                    print ''
                if len(series) < 90:
                    print series
                p+=1
            times = [i * time_gran for i in range(0, len(mod_z_score_array[:, 0]))]
            mod_z_score_dataframe = pandas.DataFrame(data=mod_z_score_array, columns=list_of_anom_metrics_applied, index=times)
            time_gran_to_mod_z_score_dataframe[time_gran] = mod_z_score_dataframe

        for time_gran in time_grans:
            list_of_metric_val_lists = []
            list_of_metric_names = []
            for label, metric_time_series in calculated_vals.iteritems():
                container_or_class = label[1]
                time_interval = int(label[0])
                if time_interval != time_gran:
                    continue

                # okay, so I think it might be a better idea to make a new 'detour' in the pipeline for the new methods...
                for current_metric_name, current_metric_time_series in calculated_vals[label].iteritems():
                    # okay, so I want to construct a matrix, where the row is the times and the columns are the
                    # feature values at those times
                    # let's make an array and then transpose it b/c if we transpose, then the rows are the feature vals
                    # and the columns are the times... this lends itself to being easily constructed... I think...

                    # TODO: problem: this supposes that a particular (node_gran, time_gran)... well, the time gran is fine
                    # but the node gran isn't... we want them to all be here but to embed the information in the names...
                    # okay, so now we are only considering values of a particular time granularity...
                    # now, we just need to make the matrix and to store the names
                    # let's start with storing the names and vals... then later we can actually make a matrix
                    list_of_metric_names.append(current_metric_name + '_' + container_or_class)
                    list_of_metric_val_lists.append(current_metric_time_series)

            # okay, so now that we have the lists with the values, we can make some matrixes (And then tranpose them :))
            feature_array = np.array(list_of_metric_val_lists)
            feature_array = feature_array.T

            # okay, so now we have the matrix along with the list we can do what we actually wanted to do:
            # (1) run some anom detection algos
            # (2) save in handy-csv format for processing by other software, potentially
            # okay, let's start with (2). Columns should be times
            times = [i * time_gran for i in range(0,len(feature_array[:,0]))]
            print feature_array
            feature_dataframe = pandas.DataFrame(data=feature_array, columns=list_of_metric_names, index=times)
            # okay, after lunch, save this dataframe as a csv and then sanity-check it and then start whipping through
            # the pyod functions. try to msg ray @ like 2
            #feature_dataframe.to_csv(csv_path + str(time_gran) + '.csv')
            time_gran_to_feature_dataframe[time_gran] = feature_dataframe
            ##hbos_alert_series = hbos_anomaly_detection(feature_array) ## TODO: give it the rest of the params
            # TODO: also note: I could probably do the other anom detection methods just by sending in the specific
            # model as a param and then calling it
            #print list_of_metric_val_lists
            #print feature_dataframe
            #time.sleep(300)

        # okay, so let's save these vals to a file, so I can calc TP/FP and make graphs without re-running all of them
        with open(alert_file, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for value_name, value in params_to_alerts.iteritems():
                spamwriter.writerow([value_name, [i if not math.isnan(i) else (None) for i in value]])

        return params_to_alerts, time_gran_to_feature_dataframe, time_gran_to_mod_z_score_dataframe
    else:
        params_to_alerts = {}
        with open(alert_file, 'r') as csvfile:
            csvread = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in csvread:
                #print row
                params_to_alerts[row[0]] = [i if i != (None) else float('nan') for i in ast.literal_eval(row[1])]
                print row[0], params_to_alerts[row[0]]
        return params_to_alerts, None

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
        tpr = float('nan')

    # fpr = FP / (FP + TN)
    try:
        fpr = float(false_positives) / (true_negatives + false_positives )
    except ZeroDivisionError:
        fpr = float('nan')
    #print '----'
    #print alert_times
    #print attack_labels
    #print "tpr,fpr",tpr,fpr

    return tpr, fpr

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

# exfil_rate used to determine if there should be a gap between exfil labels
def generate_attack_labels(time_gran, exfil_start, exfil_end, exp_length, sec_between_exfil_events=1):
    attack_labels = [0 for i in range(0, exfil_start / time_gran -1)]
    #attack_labels.extend([1 for i in range(0, (exfil_end - exfil_start)/time_gran)])

    # let's find the specific time intervals during the exfil period (note: potentially not all of the time intervals
    # actually have exfil occur during them)
    time_intervals_during_potential_exfil = [exfil_start + i * time_gran for i in range(0, int(math.ceil((exfil_end - exfil_start)/float(time_gran))) + 2)]
    # okay now let's find which ones to include/not-include
    specific_times_when_exfil_occurs = [exfil_start + i * sec_between_exfil_events for i in range(0, int(math.ceil((exfil_end - exfil_start)/sec_between_exfil_events)) + 1)]
    # okay, now we want to make sure that those specific exfil times occur during a time interval during the exfil period
    # before marking it as one in which exfil occurs
    attack_labels_during_exfil_period = []
    print "time_intervals_during_potential_exfil",time_intervals_during_potential_exfil
    print "specific_times_when_exfil_occurs",specific_times_when_exfil_occurs
    for i in range(0,len(time_intervals_during_potential_exfil)-1):
        start_of_interval = time_intervals_during_potential_exfil[i]
        end_of_interval = time_intervals_during_potential_exfil[i+1]
        found = False
        for specific_times in specific_times_when_exfil_occurs:
            # why geater-than-or-equal-to and less-than-or-equal-to? B/c it should happen slightly earlier (due to
            # timing mismatch of tcpdump) but it should get a response from the DNS server slightly later, which would
            # put it in the next camp. (This is really only meaningful for dnscat exfil, since using DET won't cause
            # gaps in the exfil line, necessarily)
            print start_of_interval, specific_times, end_of_interval, start_of_interval <= specific_times <= end_of_interval
            if start_of_interval <= specific_times <= end_of_interval:
                found = True
                break
        if found:
            attack_labels_during_exfil_period.append(1)
        else:
            attack_labels_during_exfil_period.append(0)

    attack_labels.extend(attack_labels_during_exfil_period)
    attack_labels.extend([0 for i in range(0, (exp_length - exfil_end)/time_gran)])

    return attack_labels

def generate_time_gran_to_attack_labels(time_gran_to_feature_dataframe, exfil_start, exfil_end, sec_between_exfil_events):
    time_gran_to_attack_lables = {}
    for time_gran, feature_dataframe in time_gran_to_feature_dataframe.iteritems():
        exp_length = feature_dataframe.shape[0] * time_gran
        attack_labels = generate_attack_labels(time_gran, exfil_start, exfil_end, exp_length, sec_between_exfil_events)
        time_gran_to_attack_lables[time_gran] = attack_labels
    return time_gran_to_attack_lables

def save_feature_datafames(time_gran_to_feature_dataframe, csv_path, time_gran_to_attack_labels):
    print "time_gran_to_feature_dataframe",time_gran_to_feature_dataframe.keys()
    for time_gran, feature_dataframe in time_gran_to_feature_dataframe.iteritems():
        attack_labels = time_gran_to_attack_labels[time_gran]
        print "feature_dataframe",feature_dataframe,feature_dataframe.index
        print "attack_labels",attack_labels
        feature_dataframe['labels'] = pandas.Series(attack_labels, index=feature_dataframe.index)
        feature_dataframe.to_csv(csv_path + str(time_gran) + '.csv', na_rep='?')

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

def create_ROC_of_joint_data(csv_path, time_gran,ROC_path):
    jointDF = pandas.read_csv(csv_path,na_values='?')
    aggregated_anomly_scores = []
    attack_labels = []
    for index,row in jointDF.iterrows():
        # this is the model that I want to calculate
        if time_gran == 10:
            aggregated_anomly_score = calc_anomaly_score(row)
        elif time_gran == 30:
            aggregated_anomly_score = calc_anomaly_score_thirty_gran(row)
        elif time_gran == 60:
            aggregated_anomly_score = calc_anomaly_score_sixty_gran(row)
        else:
            return "give a valid time gran!"
        aggregated_anomly_scores.append(aggregated_anomly_score)
        attack_labels.append(row['labels'])

    tprs = []
    fprs = []

    thresholds_to_try = [i/10.0 for i in range(0, -100, -1)] + [i/100.0 for i in range(0,50,5)] +\
                        [i/100.0 for i in range(50,100,2)] + [i/10 for i in range(10,100,5)]
    for threshold in thresholds_to_try:
        current_alerts = [int(i>=threshold) for i in aggregated_anomly_scores]
        time_gran_to_attack_labels = {}
        #print "time_gran", time_gran
        time_gran_to_attack_labels[time_gran] = attack_labels
        current_tpr, current_fpr = calc_fp_and_tp(current_alerts, None, None, None, time_gran, time_gran_to_attack_labels)
        tprs.append(current_tpr)
        fprs.append(current_fpr)

    # todo sort the tprs/fprs like before...
    tprs, fprs = zip(*sorted(zip(tprs, fprs)))

    x_vals = fprs
    y_vals = tprs
    title = 'Ensemble ROC curve at ' + str(time_gran) + ' Sec Granularity'
    plot_name = ROC_path + 'aggreg_ROC_curve_' + str(time_gran) + '.csv'
    construct_ROC_curve(x_vals, y_vals, title, plot_name)

def aggregate_csv_recipe():
    # okay, so what I am going to want to do here is loop through
    paths_to_csvs = [
        '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/alerts/wordpress_six_rep_2_',
        '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/alerts/wordpress_seven_rep_3_',
        '/Volumes/Seagate Backup Plus Drive/experimental_data/wordpress_info/alerts/wordpress_eight_',
        '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/alerts/sockshop_eleven_',
        '/Volumes/Seagate Backup Plus Drive/experimental_data/sockshop_info/alerts/sockshop_nine_better_exfil_']
    aggregate_file = '/Volumes/Seagate Backup Plus Drive/experimental_data/aggregate_z_mod_'
    second_part_of_file_name = 'mod_z_score_'
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