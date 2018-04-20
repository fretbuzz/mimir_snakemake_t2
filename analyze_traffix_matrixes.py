import pickle
import pandas as pd
import numpy as np
import sys
#from Tkinter import *
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
#from matplotlib.backend_bases import key_press_handler
#from matplotlib.figure import Figure
import parameters
from sklearn.decomposition import PCA
import scipy
from math import sqrt
'''
USAGE: python analyze_traffic_matrixes.py [recieved_matrix_location] [sent_matrix_location]

## Note: if the rec/sent matrix locations aren't given, will use defaults
## Other note: the import order above is important if I end up deciding to use Tkinter
'''

services = [
        'carts',
        'carts-db',
        'catalogue',
        'catalogue-db',
        'front-end',
        'orders',
        'orders-db',
        'payment',
        'queue-master',
        'rabbitmq',
        'session-db',
        'shipping',
        'user',
        'user-db'#,
        #'load-test',
        #'127.0.0.1', #it's always there, so I guess we should treat it like a normal thing
        #'172.17.0.1' # also this one too
]

# these services are NOT actually part of the microservice, so they should be ignored
services_to_ignore = [
    'load-test',
    '127.0.0.1', #it's always there, so I guess we should treat it like a normal thing
    '172.17.0.1' # also this one too
    ]

def main(rec_matrix_location, send_matrix_location):
    simulate_incoming_data(rec_matrix_location, send_matrix_location)

# This function reads pickle files corresponding to the send/received traffic matrices
# and then iterates through them by the time stamps, letting us pretend that the data
# is coming from an actively running system
def simulate_incoming_data(rec_matrix_location = './experimental_data/' + parameters.rec_matrix_location, 
        send_matrix_location = './experimental_data/' + parameters.sent_matrix_location, 
        display_sent_svc_pair = parameters.display_sent_svc_pair,
        display_rec_svc_pair  =  parameters.display_rec_svc_pair,
        graph_names = './experimental_data/' + parameters.graph_names,
        display_graphs = False,
        exfils = parameters.exfils,
        exp_time = parameters.desired_stop_time,
        start_analyze_time = parameters.start_analyze_time):
 
    experiment_results = {} # this is going to contain all the algo's performance metrics
    print "hello world"
    print "rec_matrix_loc", rec_matrix_location
    print "sent_matrix_loc", send_matrix_location
    df_sent = pd.read_pickle(send_matrix_location)
    df_rec = pd.read_pickle(rec_matrix_location)
    df_sent = df_sent.drop(services_to_ignore).drop(services_to_ignore, axis=1)
    df_rec = df_rec.drop(services_to_ignore).drop(services_to_ignore, axis=1)
    print "df_sent:", df_sent
    print "df_rec:", df_rec
    df_sent_time_slices = generate_time_slice_dfs(df_sent)
    df_rec_time_slices = generate_time_slice_dfs(df_rec)
    df_sent_control_stats = []
    df_rec_control_stats = []
    lambda_ewma = 0.2 # TODO don't arbitrarily pick lambda
    for df in df_sent_time_slices:
        prev_step_ewmas = control_charts(df, True,  prev_step_ewmas, lambda_ewma, services)
        df_sent_control_stats.append(prev_step_ewmas)
        #print df
    for df in df_rec_time_slices:
        prev_step_ewmas = control_charts(df, False,  prev_step_ewmas, lambda_ewma, services)
        df_rec_control_stats.append(prev_step_ewmas)
    #print df_sent_control_stats
    
    # aggregate the matrixes into what we would see in a corresponding 3-tier arch
    df_three_tier_sent = three_tier_time_aggreg(df_sent)
    df_three_tier_rec = three_tier_time_aggreg(df_rec)
    df_three_tier_sent_slices = generate_time_slice_dfs(df_three_tier_sent) 
    df_three_tier_rec_slices = generate_time_slice_dfs(df_three_tier_rec)
    three_tier_services = ['presentation', 'application', 'data']
    df_three_tier_sent_control_stats = []
    df_three_tier_rec_control_stats = []
    for df in df_three_tier_sent_slices:
        prev_step_ewmas = control_charts(df, True,  prev_step_ewmas, lambda_ewma, three_tier_services)
        df_three_tier_sent_control_stats.append(prev_step_ewmas)
    for df in df_three_tier_rec_slices:
        prev_step_ewmas = control_charts(df, False,  prev_step_ewmas, lambda_ewma, three_tier_services)
        df_three_tier_rec_control_stats.append(prev_step_ewmas)

    # check when control charts would give a warning
    # just going to use sent for now, could use reciever later
    times = get_times(df_sent)
    # starts at 1, b/c everyting has time stddev 0 at time 0, so everything would trigger a warning
    control_charts_warning_sent = []
    control_charts_warning_rec = []
    control_charts_warning_times_sent = []
    control_charts_warning_times_rec = []
    for time in range(1,len(times)-1):
        next_df_sent = df_sent[ df_sent['time'].isin([times[time]])]
        next_df_rec = df_rec[ df_rec['time'].isin([times[time]])]
        # TODO: pick a coefficient for the EWMA stddev that isn't arbitrary
        ewma_stddev_coef = 2
        warnings_sent,warning_times_sent = next_value_trigger_control_charts(next_df_sent, 
                df_sent_control_stats[time], times[time], ewma_stddev_coef)
        warnings_rec,warning_times_rec = next_value_trigger_control_charts(next_df_rec, 
                df_rec_control_stats[time], times[time], ewma_stddev_coef)
        control_charts_warning_sent.append(warnings_sent)
        control_charts_warning_rec.append(warnings_rec)
        control_charts_warning_times_sent += warning_times_sent
        control_charts_warning_times_rec += warning_times_rec
    print "these are the warnings from the control charts: (for data that is sent): "
    print control_charts_warning_sent,"just times:", warning_times_sent
    
    # combine the two sets of warning times (delete duplicates)
    all_control_chart_warning_times = list(set(control_charts_warning_times_sent + control_charts_warning_times_rec))
    print all_control_chart_warning_times
    
    # then calc TP/TN/FP/FN
    performance_results = calc_tp_fp_etc("control charts", exfils, all_control_chart_warning_times, 
                                        exp_time, start_analyze_time)
    experiment_results.update(performance_results)
    #print experiment_results

    ### TODO test this
    # now let's do the control charts test for the three-tier aggregation too
    three_tier_control_charts_warning_sent = []
    three_tier_control_charts_warning_rec = []
    three_tier_control_charts_warning_times_sent = []
    three_tier_control_charts_warning_times_rec = []
    for time in range(1,len(times)-1):
        next_df_sent = df_three_tier_sent[ df_three_tier_sent['time'].isin([times[time]])]
        next_df_rec = df_three_tier_sent[ df_three_tier_sent['time'].isin([times[time]])] 
        # TODO: pick a coefficient for the EWMA stddev that isn't arbitrary
        ewma_stddev_coef = 2
        warnings_sent,warning_times_sent = next_value_trigger_control_charts(next_df_sent,
                df_three_tier_sent_control_stats[time], times[time], ewma_stddev_coef)
        warnings_rec,warning_times_rec = next_value_trigger_control_charts(next_df_rec,
                df_three_tier_rec_control_stats[time], times[time], ewma_stddev_coef)
        three_tier_control_charts_warning_sent.append(warnings_sent)
        three_tier_control_charts_warning_rec.append(warnings_rec)
        three_tier_control_charts_warning_times_sent += warning_times_sent
        three_tier_control_charts_warning_times_rec += warning_times_rec
    print "these are the warnings from the 3-tier control charts: (for data that is sent): "
    print three_tier_control_charts_warning_sent,"just times:", three_tier_control_charts_warning_times_sent
    three_tier_all_control_chart_warning_times = list(set(three_tier_control_charts_warning_sent+three_tier_control_charts_warning_rec))
    three_tier_performance_results = calc_tp_fp_etc("3-tier control charts", exfils, three_tier_all_control_chart_warning_times,
            exp_time, start_analyze_time)
    experiment_results.update(three_tier_performance_results)

    '''  # see function def for why I think this is nonsense print pca_anom_scores
    # okay, we are going to try PCA-based analysis here
    print "about to try PCA anom detection!"
    pca_explained_vars = pca_anom_detect(df_sent, times)
    pca_anom_scores = detect_pca_anom(pca_explained_vars) 
    '''

    svc_pair_to_sent_control_charts = generate_service_pair_arrays(df_sent_control_stats, times)
    svc_pair_to_sent_bytes = traffic_matrix_to_svc_pair_list(df_sent)
    print svc_pair_to_sent_control_charts['front-end', 'user']
    sent_data_for_display = {'raw': svc_pair_to_sent_bytes, 'control-charts':svc_pair_to_sent_control_charts}
    generate_graphs(sent_data_for_display, times, display_sent_svc_pair, True, graph_names + "_sent_graphs")

    svc_pair_to_rec_control_charts = generate_service_pair_arrays(df_rec_control_stats, times)
    svc_pair_to_rec_bytes = traffic_matrix_to_svc_pair_list(df_rec)
    #print svc_pair_to_rec_control_charts['front-end', 'user']
    rec_data_for_display = {'raw': svc_pair_to_rec_bytes, 'control-charts':svc_pair_to_rec_control_charts}
    generate_graphs(rec_data_for_display, times, display_rec_svc_pair, False, graph_names + "_rec_graphs")

    if display_graphs:
        plt.show()
    
    # user sent to front end is particulary important to me b/c that's where data exfiltration
    # happens
    generate_graphs(sent_data_for_display, times, [['front-end', 'user' ]], True, graph_names + "_user_sent_front_graphs")

    # return experiment results, all ready for aggregation
    return experiment_results

# this function just generates graphs
# sent_data_for_display is a dictionary of data about the sent traffic matrixc
# currently the indexes are: 'control-charts' and 'raw'. Each of these is a dicitonary
# of the below form
# assumes the form {['src', 'dst']: [list of time-ordered values]
def generate_graphs(data_for_display, times, src_pairs_to_display, is_sent, graph_names):

    svc_pair_to_control_charts = data_for_display['control-charts'] 
    svc_pair_to_raw = data_for_display['raw']

    if len(src_pairs_to_display) == 1:
        columns,rows = 1,1
        plt.figure(figsize=(32, 20))
    elif len(src_pairs_to_display) == 2:
        rows = 2
        columns = 1
        plt.figure(figsize=(8, 7.5))
    elif len(src_pairs_to_display) == 4:
        columns = 2
        rows = 2
        plt.figure(figsize=(12, 7.5))
    else:
        print "about to crash because invalid size of list of objects to graph"    

    for i in range(0, len(src_pairs_to_display)):
        plt.subplot(rows,columns,i+1)

        cur_src_svc = src_pairs_to_display[i][0]
        cur_dst_svc = src_pairs_to_display[i][1]
        print cur_src_svc, cur_dst_svc, svc_pair_to_control_charts[cur_src_svc, cur_dst_svc]
        print cur_src_svc, cur_dst_svc, svc_pair_to_raw[cur_src_svc, cur_dst_svc]
        avg_line, = plt.plot(times, [item[0] for item in svc_pair_to_control_charts[cur_src_svc, cur_dst_svc]], label='mean')
        avg_plus_one_stddev = [item[0] + item[1] for item in svc_pair_to_control_charts[cur_src_svc, cur_dst_svc]]
        control_chart_above, = plt.plot(times, avg_plus_one_stddev, label='mean + 1 * stddev')
        avg_minus_one_stddev = [item[0] - item[1] for item in svc_pair_to_control_charts[cur_src_svc, cur_dst_svc]]
        control_chart_below, = plt.plot(times, avg_minus_one_stddev, label='mean - 1 * stddev')
        avg_plus_two_stddev = [item[0] + 2 * item[1] for item in svc_pair_to_control_charts[cur_src_svc, cur_dst_svc]]
        control_chart_two_above, = plt.plot(times, avg_plus_two_stddev, label='mean + 2 * stddev')
        avg_minus_two_stddev = [item[0] - 2 * item[1] for item in svc_pair_to_control_charts[cur_src_svc, cur_dst_svc]]
        control_chart_two_below, = plt.plot(times, avg_minus_two_stddev, label='mean - 2 * stddev')    
        raw_line, = plt.plot(times, svc_pair_to_raw[cur_src_svc, cur_dst_svc], label='sent bytes')
        graph_ready_times = [int(i) for i in times] # floats are hard to read
        plt.xticks(times, graph_ready_times)
        if is_sent:
            plt.title(cur_dst_svc + ' SENT TO ' + cur_src_svc) # not a typo, see Github issue #12
        else:
            plt.title(cur_dst_svc + ' RECIEVED FROM ' + cur_src_svc) # ^^
        plt.xlabel('seconds from start of experiment') 
        plt.ylabel('bytes')
        # some of the lines are obvious just by looking at it, so let's not show those
        #plt.legend(handles=[avg_line, control_chart_two_above, control_chart_two_below, control_chart_above, control_chart_below,  raw_line])
        plt.legend(handles=[avg_line, raw_line])
    plt.subplots_adjust(hspace=.3) # too close by default
    #plt.show()
    plt.savefig(graph_names + '.png', bbox_inches='tight')

# df -> {[src_svc, dst_svc] : [list of values in order of time]} 
def traffic_matrix_to_svc_pair_list(df):
    svcs_to_val_list = {}
    for src_svc in services:
        for dst_svc in services:
            svcs_to_val_list[src_svc, dst_svc] = df.loc[src_svc, dst_svc].tolist()
    return svcs_to_val_list

def three_tier_time_aggreg(dfs):
    aggreg_dfs = pd.DataFrame()

    times = get_times(dfs)
    df_time_slices = []
    for time_index in range(0,len(times)):
        time = times[time_index]
        current_df = dfs[ dfs['time'].isin([time])]
        df_time_slices.append(current_df)

    for df in df_time_slices:
        #print "df", df
        aggreg_df = aggregate_into_three_tier(df)
        aggreg_df['time'] = df['time'][0] # should all be the same anyway
        aggreg_dfs = aggreg_dfs.append(aggreg_df)
    return aggreg_dfs

# df -> (3x3)df
# this modifies the data so it matches what it would look like in the
# case of a traditional 3-tier web application
# problem: this only works on a single time-slice
def aggregate_into_three_tier(df):
    data = {'tier': ['presentation', 'application', 'data'], 'presentation': [0,0,0], 
            'application':[0, 0, 0], 'data':[0,0,0]}
    three_tier_df = pd.DataFrame(data)
    three_tier_df = three_tier_df.set_index('tier')
    #print three_tier_df
    not_core_svc = ['load-test', '127.0.0.1', '172.17.0.1']
    core_svc = [svc for svc in services if svc not in not_core_svc]
    for service in core_svc:
        #print three_tier_df
        if 'front-end' in service:
            ## it's presentation tier
            for dest_service in core_svc:
                if 'front-end' in dest_service:
                    pass # will not show up in aggregate
                elif 'db' in dest_service:
                    three_tier_df.set_value('presentation', 'data', 
                            three_tier_df.loc['presentation', 'data'] + df.loc[service, dest_service])
                else: # this will be application tier
                    three_tier_df.set_value('presentation', 'application',  
                            three_tier_df.loc['presentation', 'application'] 
                            + df.loc[service, dest_service])
        elif 'db' in service:
            ## it's data tier
            for dest_service in core_svc:
                if 'front-end' in dest_service:
                    three_tier_df.set_value('data', 'presentation',  
                            three_tier_df.loc['data', 'presentation'] + df.loc[service, dest_service])
                elif 'db' in dest_service:
                    pass
                else:
                    three_tier_df.set_value('data', 'application',  
                            three_tier_df.loc['data', 'application'] + df.loc[service, dest_service])
        else:
            ## it's application tier
            for dest_service in core_svc:
                if 'front-end' in dest_service:
                    three_tier_df.set_value('application', 'presentation',  
                            three_tier_df.loc['application', 'presentation'] + df.loc[service, dest_service])
                elif 'db' in dest_service:
                    three_tier_df.set_value('application', 'data',
                            three_tier_df.loc['application', 'data'] + df.loc[service, dest_service])
                else:
                    pass
    return three_tier_df

# result is {['src', 'dst'] : [list of values at the time intervals]}
# at the moment, it is going to deal soley with the control_chart_stats stuff
def generate_service_pair_arrays(stats, times):
    svc_to_vals = {}
    for sending_svc in services:
        for dst_svc in services:
            svc_to_vals[sending_svc, dst_svc] = []
    #print stats
    #print svc_to_vals
    for time_slice in stats:
        for svc_dst_pair, vals in time_slice.iteritems():
            #print svc_dst_pair, vals
            svc_to_vals[svc_dst_pair[0], svc_dst_pair[1]].append(vals)
    #print "I hope this worked!!!!"
    #print svc_to_vals
    return svc_to_vals

# DF(with lots of times) -> [DF(time A), DF(time A and B), DF(time A,B,C), etc.]
def generate_time_slice_dfs(df):
    times = get_times(df)
    elapsed_time = []
    #print times
    time_slices = []
    for time_index in range(0,len(times)):
        time = times[time_index]
        elapsed_time.append(time)
        df_so_far = df[ df['time'].isin(elapsed_time)]
        time_slices.append(df_so_far)
        #print elapsed_time, time_index, len(times)
        #print df_so_far
    return time_slices

#### TODO : modify this to use EWMA per #32
# this is the function to implement control channels
# i.e. compute mean and standard deviation for each pod-pair
# Note: is_send is 1 if it is the "send matrix", else zero
def control_charts(df, is_send, old_ewmas, lambda_ewma, df_services):
    ## going to return data in the form {[src_svc, dest_svc]: [ewma, ewma_stddev]}
    data_stats = {} #[]
    time = df.loc[:,'time'].max()
    for index_service in df_services:
        for column_service in df_services:
            relevant_traffic_values = df.loc[index_service, column_service]
            #print relevant_traffic_values, type(relevant_traffic_values)
            #if relevant_traffic_values.mean() != 0:
            #data_stats[index_service, column_service] = [relevant_traffic_values.mean(), relevant_traffic_values.std()]
            # so here is the implementation plan for EWMA
            new_ewma = old_ewmass[index_service, column_service] * (1 - lambda_ewma) + lambda_ewma * \
                    df.loc[df['time'] == time].loc[index_service, column_service]
            new_ewma_var = sqrt( ((lambda_ewma) / (2 - lambda_ewma)) * (relevant_traffic_values.std()**2))
            data_stats[index_service, column_service] = [new_ewma, new_ewma_var ]
    return data_stats

# might want to expand to a more generalized printing function at some stage
def print_control_charts_process(relevant_traffic_values, is_send, index_service, column_service):
    if is_send:
        print "\n", index_service, " SENT TO ", column_service
    else:
        print "\n", index_service, " RECEIVE FROM ", column_service
    #print relevant_traffic_values.describe()
    print "Mean: ", relevant_traffic_values.mean()
    print "Stddev: ", relevant_traffic_values.std()

def get_times(df):
    times = []
    for x in df.loc[:, 'time']:
        times.append(x)
    times = sorted(list(set(times)))
    return times

# this function uses the statistics that are in data_stats to
# see if the next value for a service pair causes 
# an alarm via control chart anomaly detection
def next_value_trigger_control_charts(next_df, data_stats, time, stddev_coef):
    warnings_triggered = []
    warning_times = []
    ## iterate through values of data_stats
    ## get value from traffic matrix
    ## if outside of bounds, print something in capital letters
    for src_dst, mean_stddev in data_stats.iteritems():
        #print "src_dst value: ", src_dst, "mean_stddev value: ", mean_stddev
        next_val = next_df.loc[ src_dst[0], src_dst[1] ]
        mean, stddev = mean_stddev[0], mean_stddev[1]
        if abs(next_val - mean) > (stddev_coef * stddev):
            #print "THIS IS THE POOR MAN'S EQUIVALENT OF AN ALARM!!", src_dst, mean_stddev
            warnings_triggered.append([src_dst[0], src_dst[1], time, mean_stddev])
            warning_times.append(time)
        #print "current warning times: ", warning_times
    return warnings_triggered, warning_times

# this function will determine how well PCA can fit the data
# okay, I'm going to follow the method from Ding's 'PCA-based
# Network Traffic Anomaly Detection', even though I am 90%
# sure that their method is nonsensical
def pca_anom_detect(df, times):
    explained_variances = []
    print df
    # arbitrarily choosing 5 for now, will investigate in more detail later
    n_components = 5
    for time in times:
        #print time
        current_df = df[ df['time'].isin([time])]
        pca = PCA(n_components=n_components)
        pca.fit(current_df)
        #pca_compon = pd.DataFrame(pca.transform(df), columns=['PCA%i' % i for i in range(n_components)], index=df.index)
        #print "pca components", pca_compon
        #print "pca explained var", pca.explained_variance_
        explained_variances.append(pca.explained_variance_)
    return explained_variances

# following the method in the same work as in the above function
# still think it is nonsense, but let's see how it does
def detect_pca_anom(pca_explained_vars):
    anom_scores = []
    first_component_explained_var_cusum = 0
    slices_so_far = 0
    for explained_var in pca_explained_vars:
        #print explained_var
        if explained_var[0] != 0:
            slices_so_far = slices_so_far + 1
            first_component_explained_var_cusum = first_component_explained_var_cusum + explained_var[0]
            #print "slices so far:", slices_so_far, "cusum: ", first_component_explained_var_cusum,"explained var: ", explained_var[0]
            denom = (1/float(slices_so_far)) * first_component_explained_var_cusum
            #print "denom", denom
            cur_anom_score = explained_var[0] / denom
            anom_scores.append( cur_anom_score ) 
        else:
            #print "welp, it's zero :("
            anom_scores.append('invalid')
    return anom_scores

def calc_tp_fp_etc(algo_name, exfils, warning_times, exp_time, start_analyze_time):
    print "calc_tp_fp_etc: ", algo_name, exfils, warning_times, exp_time, start_analyze_time
    attack_times = exfils.keys()
    print attack_times, warning_times
    total_attacks = len(attack_times)
    true_attacks_found = 0
    warning_times_after_strt_analyze = [time for time in warning_times if time >= start_analyze_time]
    for attack_time in attack_times:
        # need to add 5 b/c exfil starts at the attack_time, so it is recorded 5 sec later by Prometheus
        if (attack_time+5) in [int(time) for time in warning_times_after_strt_analyze]:
            true_attacks_found = true_attacks_found + 1
    true_attacks_missed = total_attacks - true_attacks_found
    false_attacks_found = len(warning_times_after_strt_analyze) - true_attacks_found
    total_negatives = ((exp_time-start_analyze_time)/5.0) - total_attacks
    true_negatives_found = total_negatives - false_attacks_found
    print "TPs", true_attacks_found, "FPs", false_attacks_found, "Total negs", total_negatives, "TNs",  true_negatives_found
    return {algo_name: {"TPR":  (float(true_attacks_found) / (true_attacks_found + true_attacks_missed)),
                        "FPR" : float(false_attacks_found) / (false_attacks_found + true_attacks_found), 
                        "FNR" : float(true_attacks_missed) / (true_negatives_found + true_attacks_missed),
                        "TNR" : float(true_negatives_found) / total_negatives}}

# following method in Lakhina's "Diagnosing
# Network-Wide Traffic Anomalies" (sigcomm '04)
def diagnose_anom_pca(old_dfs, cur_df, n_components):
    # first, convert dfs representation
    # to match the paper's representation
    # (rows = time slices of features, columns = time series
    # of a particular feature)
    mod_old_dfs = old_dfs.set_index(['time'], append = True )
    print mod_old_dfs
    converted_dfs = mod_old_dfs.unstack(level = 0)
    print converted_dfs
    mod_cur_df = cur_df.set_index(['time'], append = True)
    converted_cur_df = mod_cur_df.unstack(level = 0)
    print converted_cur_df

    # second, zero-mean the dfs (for each column)
    # note, not needed, fit_transform does this below
    #for column in converted_dfs:
    #    converted_dfs[column] = converted_dfs[column] - converted_dfs[column].mean()
    #converted_cur_df = converted_cur_df - converted_cur_df.mean()
    #print converted_dfs
    #print converted_cur_df
    
    # third, use PCA to determine old_dfs principal axis
    pca = PCA(n_components=n_components)
    converted_dfs_along_principal_components = pca.fit_transform(converted_dfs)
    print converted_dfs_along_principal_components
    #### TODO: is this a unit vector? it needs to be
    #### TODO: does PCA use StandardScaler (hence obviating the need for #2)

    # fourth, map data to principal axis
    # note: not needed,  handeled by fit_transform above

    # fifth, split data into two types of projections:
    # (1) normal, (2) anomalous, via threshold
    anom_thresh = -1
    cur_index = 0
    for index, row in converted_dfs_along_principal_components.iterrows():
        row_mean = row.mean()
        row_stddev = row.std()
        if (row > row_mean + 3 * row_stddev) or (row < row_mean + 3 * row_stddev):
            anom_thresh = cur_index
        cur_index = cur_index + 1
    if anom_thresh < 0:
        print "SOMETHING WENT WRONG"

    # sixth, project cur_df onto both projections
    # I could theoretically, get fancy with skikit-learn, but
    # it is probably better just to form the matrixes described
    # in the paper and carry out the designated computations
    # SO, take each principal component, and make it a row in the 
    # corresponding matrix
    P = np.array(pca.components_[0]) # this one has got to not be anomolous
    for i in range(1, anom_thresh):
        P = vstack(P, pca.components_[i])
    C = P * np.transpose(P)
    y_circum = C * mod_cur_df
    y_tilda = (1 - C) * mod_cur_df

    # seventh, compute threshold for the  squared prediction 
    # error (SPE) via the Q-statistic
    #phi_1 = ?
    #h_naught = 1 - ? 

    # eighth, compare SPE to threshold to determine if anomaly occurs

# following method given in Ide's "Eigenspace-based Anomaly Detection in
# Computer Systems"
def eigenvector_based_detector(old_u, current_tm, window_size, crit_bound, old_z_first_mom, old_z_sec_mom):
    #### TODO: change to using TruncatedSVD (b/c finding eig of singular matrix is kinda complicated)
    #### also, need to remove the uselss columns / rows from the TM (probably in the main loop above, tho)
    #### FURTHER TODO: I am pretty sure that the attack actually executes 5 seconds after we think it does
    #### b/c we start the attack right AFTER a measuremnt piont, so it doesn't show up until 5 seconds later
    # first, find the principle eigenvector of the traffic matrix
    print "shape: ", current_tm.shape, current_tm
    eigenvals, unit_eigenvect = np.linalg.eig(current_tm)   
    # principle eigenvector has largest associated eigenvalue
    largest_eigenval_index = np.argmax(eigenvals)
    princip_eigenvect = unit_eigenvect[largest_eigenval_index]
    print "pprincip_eigenvect", princip_eigenvect
    print "eigenvects", unit_eigenvect, eigenvals

    # second, obtain "typical pattern" of activity vector from old_u
    # this is the principal left singular vector
    print "size", np.shape(old_u), len(np.shape(old_u)), old_u
    if len(np.shape(old_u)) > 1:
        u,s,vh = np.linalg.svd(old_u)
        largest_signular_val_index = np.argmax(u)
        princip_left_singular_vect = u[largest_signular_val_index]

        # third, compute z(t), the dissimilarity between the principal left
        # singular vector and the principal eigenvector
        z = 1 - princip_left_singular_vect * princip_eigenvect # numpy auto transposes
        # note: is 1 if orthogonal, is 0 if identical

        # now compare z(t) with a threshold, using section 5.3
        # approximate the MLE algorithm for the vMF distribution
        beta = 0.005  ## TODO: determine via theory what this should be
        z_first_moment = (1 - beta) * old_z_first_mom + beta * z
        z_sec_moment = (1 - beta) * old_z_sec_mom + beta * (z ** 2)
        n = ((2 * z_first_moment ** 2) / (z_sec_moment - z_first_moment ** 2)) + 1
        sigma = (z_sec_moment - z_first_moment ** 2) / (2 * z_first_moment)
        # find the specific threshold value
        z_thresh = scipy.optimize.fsolve(vMF_thresh_func, 0, args=(n, sigma, crit_bound))
        #do the actual comparison, but first modify u
        if old_u.shape[1] >= window_size:
            np.delete(old_u, 0, 0)# get rid of old column (outside of window)
        old_u = np.hstack((old_u,princip_eigenvect))
        print np.size(old_u)
        if z > z_thresh:
            return 1, old_u, z_first_moment, z_sec_moment  # 1 = an alert
        else:
            return 0, old_u, z_first_moment, z_sec_moment  # 0 = no alert
    else:
        #old_u = princip_eigenvect
        old_u = np.hstack((old_u,princip_eigenvect))
        return 0, old_u, 0, 0  # 0 = no alert

def vMF_pdf_func(z, n, sigma):
    denom = (2* sigma) ** ((n-1)/2) * scipy.special.gamma((n-1)/2)
    return (np.exp((-1 * z) / (2 * sigma)) * z ** (((n-1)/2)-1)) / denom

def vMF_thresh_func(zth, n, sigma, crit_bound):
    return scipy.integrate.quad(vMF_pdf_func, zth, np.inf, args=(n,sigma)) - crit_bound

if __name__=="__main__":
    rec_matrix_location = './experimental_data/' + parameters.rec_matrix_location
    send_matrix_location = './experimental_data/' + parameters.sent_matrix_location
    if len(sys.argv) > 2:
        rec_matrix_location = sys.argv[1]
        send_matrix_location = sys.argv[2]
    main(rec_matrix_location, send_matrix_location)
