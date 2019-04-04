import numpy as np
import csv
import math
import ast
import matplotlib.pyplot as plt
import pandas
import time
import scipy.stats
import logging
from sklearn.preprocessing import RobustScaler

# TODO: NOTE: I'm 90% sure that this function is wrong...
# exfil_rate used to determine if there should be a gap between exfil labels
def generate_attack_labels(time_gran, exfil_start, exfil_end, exp_length, sec_between_exfil_events=1):
    #attack_labels = [0 for i in range(0, exfil_start / time_gran -1)]
    #attack_labels.extend([1 for i in range(0, (exfil_end - exfil_start)/time_gran)])

    # let's find the specific time intervals during the exfil period (note: potentially not all of the time intervals
    # actually have exfil occur during them)
    if exfil_start == exfil_end:
        return [0 for i in range(0, exp_length / time_gran)]
    else:
        attack_labels = [0 for i in range(0, exfil_start / time_gran)]
        #if exfil_start % time_gran == 0:
        #    # in this case, going to count the time interval right before too, since
        #    attack_labels = [0 for i in range(0, exfil_start / time_gran - 1)]
        #    time_intervals_during_potential_exfil = [exfil_start - time_gran] + [exfil_start + i * time_gran for i in range(0, int(math.ceil((exfil_end - exfil_start)/float(time_gran))) + 2)]
        #else:
        #    attack_labels = [0 for i in range(0, exfil_start / time_gran)]
        #    time_intervals_during_potential_exfil = [exfil_start + i * time_gran for i in range(0, int(math.ceil((exfil_end - exfil_start)/float(time_gran))) + 2)]

    # okay now let's find which ones to include/not-include
    ## note: this was to handle itermittent exfil (like dns w/ 5/10/15 sec spacing). I don't like it so I am going to
    ## comment it out and put in some simpler code.
    # TODO: I'm go to need to re-word all of this logic once I actually want to analyze physical exfiltrations again...
    '''
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
    print type(exp_length), type(exfil_end), type(time_gran)
    '''
    attack_labels.extend([0 for i in range(0, (exfil_start - exfil_end)/time_gran)])
    attack_labels.extend([0 for i in range(0, (exp_length - exfil_end)/time_gran )])

    return attack_labels

def generate_time_gran_to_attack_labels(time_interval_lengths, exfil_start, exfil_end, sec_between_exfil_events, exp_length):
    time_gran_to_attack_lables = {}
    for time_gran in time_interval_lengths:
        #exp_length = exfil_end - exfil_start
        attack_labels = generate_attack_labels(time_gran, exfil_start, exfil_end, exp_length, sec_between_exfil_events)
        time_gran_to_attack_lables[time_gran] = attack_labels
    return time_gran_to_attack_lables


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
    list_of_metric_val_lists = {}
    list_of_metric_names = {}
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
            if time_interval not in list_of_metric_val_lists:
                list_of_metric_val_lists[time_interval] = []
                list_of_metric_names[time_interval] = []

        for current_metric_name, current_metric_time_series in calculated_vals[label].iteritems():
            list_of_metric_names[time_interval].append(current_metric_name + '_' + container_or_class)
            list_of_metric_val_lists[time_interval].append(current_metric_time_series)

    for time_gran, metric_val_lists in list_of_metric_val_lists.iteritems():
        metric_names = list_of_metric_names[time_gran]
        # okay, so now that we have the lists with the values, we can make some matrixes (And then tranpose them :))
        feature_array = np.array(metric_val_lists)
        feature_array = feature_array.T

        # okay, so now we have the matrix along with the list we can do what we actually wanted to do:
        # (1) run some anom detection algos
        # (2) save in handy-csv format for processing by other software, potentially
        # let's start start with (2). Columns should be times

        #print feature_array
        #print list_of_metric_names
        for counter, feature_vector in enumerate(metric_val_lists):
            print metric_names[counter], feature_vector, len(feature_vector)

        times = [i * time_gran for i in range(0,len(feature_array[:,0]))]
        print feature_array
        feature_dataframe = pandas.DataFrame(data=feature_array, columns=metric_names, index=times)
        ##feature_dataframe.index.name = 'time' ## i think this should solve the problem of the time column not being labeled
        time_gran_to_feature_dataframe[time_gran] = feature_dataframe
    return time_gran_to_feature_dataframe

def save_feature_datafames(time_gran_to_feature_dataframe, csv_path, time_gran_to_attack_labels, time_gran_to_synthetic_exfil_paths_series,
                           time_gran_to_list_of_concrete_exfil_paths, time_gran_to_list_of_exfil_amts, end_of_training,
                           time_gran_to_new_neighbors_outside,
                           time_gran_to_new_neighbors_dns, time_gran_to_new_neighbors_all,
                           time_gran_to_list_of_amt_of_out_traffic_bytes, time_gran_to_list_of_amt_of_out_traffic_pkts):

    print "time_gran_to_feature_dataframe",time_gran_to_feature_dataframe.keys()
    for time_gran, attack_labels in time_gran_to_attack_labels.iteritems():
        print "time_gran", time_gran, "len of attack labels", len(attack_labels)
    max_time_gran = max(time_gran_to_feature_dataframe.keys())
    for time_gran, feature_dataframe in time_gran_to_feature_dataframe.iteritems():
        attack_labels = time_gran_to_attack_labels[time_gran]
        #print "feature_dataframe",feature_dataframe,feature_dataframe.index

        print time_gran_to_new_neighbors_outside[time_gran]
        print time_gran_to_new_neighbors_dns[time_gran]
        print time_gran_to_new_neighbors_dns[time_gran]
        feature_dataframe['new_neighbors_outside'] = pandas.Series(time_gran_to_new_neighbors_outside[time_gran], index=feature_dataframe.index)
        feature_dataframe['new_neighbors_dns'] = pandas.Series(time_gran_to_new_neighbors_dns[time_gran], index=feature_dataframe.index)
        feature_dataframe['new_neighbors_all ']= pandas.Series(time_gran_to_new_neighbors_all[time_gran], index=feature_dataframe.index)

        # make sure there's no stupid complex numbers here...
        for column in feature_dataframe:
            feature_dataframe[column] = feature_dataframe[column].apply(lambda x: np.real(x))

        #print "attack_labels",attack_labels, len(attack_labels), "time_gran", time_gran
        feature_dataframe['labels'] = pandas.Series(attack_labels, index=feature_dataframe.index)
        print "time_gran_to_synthetic_exfil_paths_series[time_gran]", time_gran_to_synthetic_exfil_paths_series[time_gran]
        if len(time_gran_to_synthetic_exfil_paths_series[time_gran].index.values) > len(feature_dataframe.index.values):
            time_gran_to_synthetic_exfil_paths_series[time_gran] = time_gran_to_synthetic_exfil_paths_series[time_gran][:len(feature_dataframe.index.values)]
        time_gran_to_synthetic_exfil_paths_series[time_gran].index = feature_dataframe.index[:len(time_gran_to_synthetic_exfil_paths_series[time_gran])]
        feature_dataframe['exfil_path'] = pandas.Series(time_gran_to_synthetic_exfil_paths_series[time_gran], index=feature_dataframe.index)
        feature_dataframe['concrete_exfil_path'] = pandas.Series(time_gran_to_list_of_concrete_exfil_paths[time_gran], index=feature_dataframe.index)
        feature_dataframe['exfil_weight'] = pandas.Series([i['weight'] for i in time_gran_to_list_of_exfil_amts[time_gran]], index=feature_dataframe.index)
        feature_dataframe['exfil_pkts'] = pandas.Series([i['frames'] for i in time_gran_to_list_of_exfil_amts[time_gran]], index=feature_dataframe.index)

        feature_dataframe['amt_of_out_traffic_bytes'] = pandas.Series(time_gran_to_list_of_amt_of_out_traffic_bytes[time_gran], index=feature_dataframe.index)
        feature_dataframe['amt_of_out_traffic_pkts'] = pandas.Series(time_gran_to_list_of_amt_of_out_traffic_pkts[time_gran], index=feature_dataframe.index)

        print "feature_dataframe", feature_dataframe

        ### now let's store an indicator of when the training set ends... end_of_training indicates the first member
        ### of the training dataset...
        print end_of_training, time_gran

        ## TODO: this function needs to be fixed so that everything can work.... okay. It's pretty clear to me that
        ## we need to determine the split based off of the max time granularity and then set the other thing srelative to that

        #max_time_gran

        current_end_of_train = int(end_of_training / max_time_gran) * time_gran

        test_period_list = [0 for i in range(0,int(current_end_of_train))] + \
                           [1 for i in range(int(current_end_of_train), len(feature_dataframe.index))]
        test_period_series = pandas.Series(test_period_list, index=feature_dataframe.index)
        feature_dataframe['is_test'] = test_period_series


        feature_dataframe.to_csv(csv_path + str(time_gran) + '.csv', na_rep='?')


def normalize_data_v2(time_gran_to_feature_dataframe, time_gran_to_attack_labels, end_of_training):
    time_gran_to_normalized_df = {}
    for time_gran, feature_dataframe in time_gran_to_feature_dataframe.iteritems():
        current_attack_labels = time_gran_to_attack_labels[time_gran]
        feature_dataframe['attack_labels'] = current_attack_labels
        last_label_in_training = int(math.floor(float(end_of_training) / time_gran))

        training_values = feature_dataframe.iloc[:last_label_in_training]
        training_noAttack_values = training_values.loc[training_values['attack_labels'] == 0]
        transformer = RobustScaler().fit(training_noAttack_values)

        # normalizes each column of the input matrix
        transformed_data = transformer.transform(feature_dataframe)
        transformed_training_noAttack_values = transformer.transform(training_noAttack_values)


        # might modify this at some point-- prob not the way to do it at the end...
        ## actually, several statistics professors tell me that this is indeed a property of the LASSO- you need to
        # bound it for extreme values.
        transformed_data = np.minimum(transformed_data, 100)

        time_gran_to_normalized_df[time_gran] = pandas.DataFrame(transformed_data, index=feature_dataframe.index,\
                                                                 columns=feature_dataframe.columns.values) #df_normalized

        # note whether or not I actually want to do this is TBD...
        # TODO: TEST TEST TEST
        time_gran_to_normalized_df[time_gran] = time_gran_to_normalized_df[time_gran].fillna( \
            pandas.DataFrame(transformed_training_noAttack_values, columns=feature_dataframe.columns.values).median())
        time_gran_to_normalized_df[time_gran] = time_gran_to_normalized_df[time_gran].dropna(axis=1)

    return time_gran_to_normalized_df