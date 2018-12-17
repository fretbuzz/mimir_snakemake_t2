import numpy as np
import csv
import math
import ast
import matplotlib.pyplot as plt
import pandas
import time
import scipy.stats

# TODO: NOTE: I'm 90% sure that this function is wrong...
# exfil_rate used to determine if there should be a gap between exfil labels
def generate_attack_labels(time_gran, exfil_start, exfil_end, exp_length, sec_between_exfil_events=1):
    #attack_labels = [0 for i in range(0, exfil_start / time_gran -1)]
    #attack_labels.extend([1 for i in range(0, (exfil_end - exfil_start)/time_gran)])

    # let's find the specific time intervals during the exfil period (note: potentially not all of the time intervals
    # actually have exfil occur during them)
    if exfil_start % time_gran == 0:
        # in this case, going to count the time interval right before too, since
        attack_labels = [0 for i in range(0, exfil_start / time_gran - 1)]
        time_intervals_during_potential_exfil = [exfil_start - time_gran] + [exfil_start + i * time_gran for i in range(0, int(math.ceil((exfil_end - exfil_start)/float(time_gran))) + 2)]
    else:
        attack_labels = [0 for i in range(0, exfil_start / time_gran)]
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
            print "attack found to occur at", start_of_interval, '-', end_of_interval
            attack_labels_during_exfil_period.append(1)
        else:
            attack_labels_during_exfil_period.append(0)

    print "attack_labels_during_exfil_period",attack_labels_during_exfil_period
    attack_labels.extend(attack_labels_during_exfil_period)
    attack_labels.extend([0 for i in range(0, (exp_length - exfil_end)/time_gran - 1)])

    return attack_labels

def generate_time_gran_to_attack_labels(time_interval_lengths, exfil_start, exfil_end, sec_between_exfil_events):
    time_gran_to_attack_lables = {}
    for time_gran in time_interval_lengths:
        exp_length = exfil_end - exfil_start
        attack_labels = generate_attack_labels(time_gran, exfil_start, exfil_end, exp_length, sec_between_exfil_events)
        time_gran_to_attack_lables[time_gran] = attack_labels
    return time_gran_to_attack_lables

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
            if str(next_val) == 'nan':
                next_modified_z_score = float('nan')
            elif (next_val - median) == 0:
                next_modified_z_score = 0.0
            else:
                next_modified_z_score = float('inf')
        next_modified_z_score = abs(next_modified_z_score) ## TODO: remove???
        ## behavior is funny if there are inf's so, let's put an upper bound of 1000
        next_modified_z_score = min(next_modified_z_score, 1000)
        print "median", median, "MAD", MAD,"next_modified_z_score",next_modified_z_score, "val", next_val, type(median), type(MAD), type(next_val)
        modified_z_scores.append(next_modified_z_score)
    #time.sleep(2)
    return modified_z_scores

def z_score(time_series, window_size, min_training_window):
    if len(time_series) < min_training_window:
        return [float('nan') for i in range(0,len(time_series))]
    z_scores = [float('nan') for i in range(0,min_training_window)]
    for i in range(0,min_training_window):
        print time_series[i]
    for i in range(min_training_window, len(time_series)):
        start_training_window = max(0, i - window_size)
        training_window = time_series[start_training_window:i]
        next_val = time_series[i]

        # now let's actually calculate the modified z-score
        mean = np.nanmean(training_window)
        stddev = np.nanstd(training_window)
        next_z_score = (next_val - mean) / stddev

        next_z_score = next_z_score
        ## behavior is funny if there are inf's so, let's put an upper bound of 1000
        z_scores.append(next_z_score)
    return z_scores

def calculate_modified_z_scores_of_grap_metrics(calculated_vals, label, minimum_training_window, training_window_size,
                                                time_gran_to_list_of_anom_metrics_applied, time_gran_to_list_of_anom_value,
                                                list_of_anom_metrics_applied, list_of_anom_values):
    # okay,
    for current_metric_name, current_metric_time_series in calculated_vals[label].iteritems():
        #
        sigma_min_training_window = minimum_training_window
        sigma_window_size = training_window_size
        # TODO: modify: I want nan's up until the first non-NAN value...
        first_non_NAN_index = 0
        for item in current_metric_time_series:
            if str(item) != 'nan':
                break
            else:
                first_non_NAN_index += 1
        number_nans_in_this_time_series = [str(i) for i in current_metric_time_series[0:first_non_NAN_index]].count(
            'nan')
        modified_min_z_score_training_window = number_nans_in_this_time_series + minimum_training_window

        # for the VIP metric, we only want to look at the non-zero values...
        if 'VIP' in current_metric_name:
            current_metric_time_series = [h if h else float('nan') for h in current_metric_time_series]

        modified_z_scorese = calc_modified_z_score(current_metric_time_series, sigma_window_size,
                                                   modified_min_z_score_training_window)
        list_of_anom_metrics_applied.append(current_metric_name + str(sigma_window_size) + '_' + \
                                            str(sigma_min_training_window) + '_' + label[1] + \
                                            '_mod_z_score')
        list_of_anom_values.append(modified_z_scorese)
        if 'New' in current_metric_name:
            print current_metric_name, "raw_time_series", current_metric_time_series, len(current_metric_time_series)

    return time_gran_to_list_of_anom_metrics_applied, time_gran_to_list_of_anom_value, list_of_anom_metrics_applied, list_of_anom_values

def generate_mod_z_score_dataframes(time_gran_to_list_of_anom_values, time_gran_to_list_of_anom_metrics_applied, time_grans):
    time_gran_to_mod_z_score_dataframe = {}
    for time_gran in time_grans:

        list_of_anom_values = time_gran_to_list_of_anom_values[time_gran]
        list_of_anom_metrics_applied = time_gran_to_list_of_anom_metrics_applied[time_gran]
        print "list_of_anom_values", list_of_anom_values
        print "list_of_anom_metrics_applied", list_of_anom_metrics_applied
        mod_z_score_array = np.array(list_of_anom_values)
        mod_z_score_array = mod_z_score_array.T
        p = 0
        for series in mod_z_score_array:
            print len(series),
            try:
                print list_of_anom_metrics_applied[p]
            except:
                print ''
            if len(series) < 90:
                print series
            p += 1
        print mod_z_score_array.shape
        times = [i * time_gran for i in range(0, len(mod_z_score_array[:, 0]))]
        mod_z_score_dataframe = pandas.DataFrame(data=mod_z_score_array, columns=list_of_anom_metrics_applied,
                                                 index=times)
        time_gran_to_mod_z_score_dataframe[time_gran] = mod_z_score_dataframe
    return time_gran_to_mod_z_score_dataframe

def generate_feature_dfs(calculated_vals, time_interval_lengths):
    time_gran_to_feature_dataframe = {}
    #for time_gran in time_grans:
    list_of_metric_val_lists = []
    list_of_metric_names = []
    time_grans = []
    for label, metric_time_series in calculated_vals.iteritems():
        container_or_class = label[1]
        time_interval = int(label[0])
        #if time_interval != time_gran:
        #    continue
        if time_interval not in time_interval_lengths:
            continue
        else:
            time_grans.append(time_interval)

        for current_metric_name, current_metric_time_series in calculated_vals[label].iteritems():
            list_of_metric_names.append(current_metric_name + '_' + container_or_class)
            list_of_metric_val_lists.append(current_metric_time_series)

    # okay, so now that we have the lists with the values, we can make some matrixes (And then tranpose them :))
    feature_array = np.array(list_of_metric_val_lists)
    feature_array = feature_array.T

    # okay, so now we have the matrix along with the list we can do what we actually wanted to do:
    # (1) run some anom detection algos
    # (2) save in handy-csv format for processing by other software, potentially
    # let's start start with (2). Columns should be times
    for time_gran in time_grans:
        times = [i * time_gran for i in range(0,len(feature_array[:,0]))]
        print feature_array
        feature_dataframe = pandas.DataFrame(data=feature_array, columns=list_of_metric_names, index=times)
        time_gran_to_feature_dataframe[time_gran] = feature_dataframe
    return time_gran_to_feature_dataframe

def calculate_mod_zscores_dfs(calculated_vals, minimum_training_window, training_window_size, time_interval_lengths):
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

        time_gran_to_list_of_anom_metrics_applied, time_gran_to_list_of_anom_value, list_of_anom_metrics_applied, list_of_anom_values =\
            calculate_modified_z_scores_of_grap_metrics(calculated_vals, label, minimum_training_window, training_window_size,
                                                        time_gran_to_list_of_anom_metrics_applied, time_gran_to_list_of_anom_values,
                                                        list_of_anom_metrics_applied, list_of_anom_values)
        time_gran_to_list_of_anom_metrics_applied[time_interval].extend(list_of_anom_metrics_applied)
        time_gran_to_list_of_anom_values[time_interval].extend(list_of_anom_values)

    time_gran_to_mod_z_score_dataframe = generate_mod_z_score_dataframes(time_gran_to_list_of_anom_values,
                                                                         time_gran_to_list_of_anom_metrics_applied, time_grans)
    return time_gran_to_mod_z_score_dataframe

def save_feature_datafames(time_gran_to_feature_dataframe, csv_path, time_gran_to_attack_labels):
    print "time_gran_to_feature_dataframe",time_gran_to_feature_dataframe.keys()
    for time_gran, feature_dataframe in time_gran_to_feature_dataframe.iteritems():
        attack_labels = time_gran_to_attack_labels[time_gran]
        print "feature_dataframe",feature_dataframe,feature_dataframe.index
        print "attack_labels",attack_labels
        feature_dataframe['labels'] = pandas.Series(attack_labels, index=feature_dataframe.index)
        feature_dataframe.to_csv(csv_path + str(time_gran) + '.csv', na_rep='?')

def calc_time_gran_to_zscore_dfs(time_gran_to_feature_dataframe, training_window_size,
                                 minimum_training_window):
    time_gran_z_score_dataframe = {}
    for time_interval, df in time_gran_to_feature_dataframe.iteritems():
        cols = cols = list(df.columns)
        df_zscore = pandas.DataFrame()
        for col in cols:
            current_metric_time_series = df[col]

            # TODO: for the case of the VIP I only wanna consider non-zero vals. Easy enough, just replace the 0s
            # with NANs. The problem, however, is that the thing will output Nan's, not 0s, cause 0 is defnitely not
            # an alarm worthy case. So what I'd probably have to do is run the damn thing and then replace the nan's
            # with 0's... though since I'm not going to trigger an alarm on a nan anyway (i think), maybe it doesn't
            # even matter...

            first_non_NAN_index = 0
            for item in current_metric_time_series:
                if str(item) != 'nan':
                    break
                else:
                    first_non_NAN_index += 1
            number_nans_in_this_time_series = [str(i) for i in current_metric_time_series[0:first_non_NAN_index]].count(
                'nan')
            mod_min_training_window = number_nans_in_this_time_series + minimum_training_window
            #print "current_metric_time_series",current_metric_time_series
            current_metric_time_series_list = current_metric_time_series.tolist()
            if 'ratio' in col:
                current_metric_time_series_list = [i if i else float('nan') for i in current_metric_time_series_list]
            current_col_z_scores = z_score(current_metric_time_series_list, training_window_size, mod_min_training_window)
            df_zscore[col + 'z_score'] = current_col_z_scores
        time_gran_z_score_dataframe[time_interval] = df_zscore
    return time_gran_z_score_dataframe