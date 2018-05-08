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
from sklearn.decomposition import PCA, SparsePCA
import scipy
#import scipy.sparse.linalg.eigs
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
    return simulate_incoming_data(rec_matrix_location, send_matrix_location)

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

    '''
    svc_pairs_to_consider = [ ('front-end', 'carts'), ('front-end', 'user'), \
                            ('front-end', 'catalogue'), #\, ('front-end', 'payment'), \
                            ('front-end', 'orders')  ]
    svc_pair_for_denom = [('front-end', 'user')]

    for ewma_stddev_coef_val in np.arange(0.0, 4.2,0.1):
        for lambda_values in np.arange(0.2, 0.8, 0.2):
            for simulataneous_svc_ratio_exceeding_thresh in range(1,4):
                for no_alert_if_any_svc_ratio_decreases in [False, True]:
            # now I need to give those new params that I added a workout
            # can just loop through possible values BUT do I need to chance the saving tuple that identifies it?

                    ratio_warning_times = ewma_svc_pair_ratio_loop(df_sent, lambda_ewma = lambda_values,
                                                                ewma_stddev_coef = ewma_stddev_coef_val,
                                                                ratio_denoms= svc_pair_for_denom,
                                                                svc_pair_list= svc_pairs_to_consider,
                                                                num_exceed=simulataneous_svc_ratio_exceeding_thresh,
                                                                dont_trigger_if_decrease=no_alert_if_any_svc_ratio_decreases,
                                                                linear_comb = False, abs_val = True)

                    print "the ratio alert triggered at these times", ratio_warning_times
                    ratio_performance_results = calc_tp_fp_etc(("ratio ewma", lambda_values, ewma_stddev_coef_val,
                                                                simulataneous_svc_ratio_exceeding_thresh,
                                                                no_alert_if_any_svc_ratio_decreases),
                                                               exfils, ratio_warning_times, exp_time, start_analyze_time)
                    #print "exfils took place at", exfils
                    #print "lamba", lambda_values, "ewma stddev coef", ewma_stddev_coef_val
                    #print "the ratio test did this well", ratio_performance_results
                    print ewma_stddev_coef_val, ratio_performance_results
                    experiment_results.update(ratio_performance_results)

    '''
    '''
    # this does the linear-combination-ewma
    for ewma_stddev_coef_val in np.arange(0.0, 4.2,0.1):
        for lambda_values in np.arange(0.2, 0.8, 0.2):
            # num_exceed and dont_trigger_if_decrease do not matter here b/c linear_comb inactivates thier code
            ratio_warning_times = ewma_svc_pair_ratio_loop(df_sent, lambda_ewma = lambda_values,
                                                               ewma_stddev_coef = ewma_stddev_coef_val,
                                                                ratio_denoms= svc_pair_for_denom,
                                                                svc_pair_list= svc_pairs_to_consider,
                                                                num_exceed=0,
                                                                dont_trigger_if_decrease=False,
                                                                linear_comb = True)

            print "the ratio alert triggered at these times", ratio_warning_times
            ratio_performance_results = calc_tp_fp_etc(("linear-comb ewma", lambda_values, ewma_stddev_coef_val),
                                                               exfils, ratio_warning_times, exp_time, start_analyze_time)
            #print "exfils took place at", exfils
            #print "lamba", lambda_values, "ewma stddev coef", ewma_stddev_coef_val
            #print "the ratio test did this well", ratio_performance_results
            print ewma_stddev_coef_val, ratio_performance_results
            experiment_results.update(ratio_performance_results)


    '''
    '''
    # this is for non-selective (on MS)
    for ewma_stddev_coef in np.arange(0 ,4.5,0.5):
        for lambda_values in np.arange(0.2, 0.4, 0.2):
            naive_all_control_chart_warning_times = control_charts_on_directional_dfs(df_sent, df_rec, services,
                                                                                      lambda_val=lambda_values,
                                                                                      stddev_coef=ewma_stddev_coef)
            print "all the control chart warning times", naive_all_control_chart_warning_times
            # format: (name, lambda value, stddev coef value)
            naive_performance_results = calc_tp_fp_etc(("naive MS control charts", lambda_values, ewma_stddev_coef), exfils,
                                                   naive_all_control_chart_warning_times,exp_time, start_analyze_time)
            experiment_results.update(naive_performance_results)

            print "the naive MS results are:"
            print naive_performance_results


    # this is for selective (on MS)
    print "ENTERING SELECTIVE TERRITORY"
    for ewma_stddev_coef in np.arange(0 ,4.5,0.5):
        for lambda_values in np.arange(0.2, 0.4, 0.2):
            selective_ewma_services = ['front-end', 'user']
            all_control_chart_warning_times = control_charts_on_directional_dfs(df_sent, df_rec, selective_ewma_services,
                                                                              lambda_val=lambda_values,
                                                                                stddev_coef=ewma_stddev_coef, also_rec=False)
            # then calc TP/TN/FP/FN
            performance_results = calc_tp_fp_etc(("selective MS control charts", lambda_values, ewma_stddev_coef),
                                                 exfils, all_control_chart_warning_times,exp_time, start_analyze_time)
            experiment_results.update(performance_results)
    '''
    '''
    # this is for selective (on MS)(bidirectional)
    print "ENTERING bi-SELECTIVE TERRITORY"
    for ewma_stddev_coef in np.arange(0, 4.5, 0.5):
        for lambda_values in np.arange(0.2, 0.4, 0.2):
            selective_ewma_services = ['front-end', 'user']
            all_control_chart_warning_times = control_charts_on_directional_dfs(df_sent, df_rec,
                                                                                selective_ewma_services,
                                                                                lambda_val=lambda_values,
                                                                                stddev_coef=ewma_stddev_coef)
            # then calc TP/TN/FP/FN
            performance_results = calc_tp_fp_etc(("bi-selective MS control charts", lambda_values, ewma_stddev_coef),
                                                 exfils, all_control_chart_warning_times, exp_time,
                                                 start_analyze_time)
            experiment_results.update(performance_results)

    
    '''
    '''
    # this is for 3-tier
    print "ENTERING 3-TIER TERRITORY"
    # aggregate the matrixes into what we would see in a corresponding 3-tier arch
    df_tt_sent = three_tier_time_aggreg(df_sent)
    df_tt_rec = three_tier_time_aggreg(df_rec)

    for ewma_stddev_coef in np.arange(0 ,4.5,0.5):
        for lambda_values in np.arange(0.2, 0.4, 0.2):
            three_tier_services = ['presentation', 'application', 'data']
            tt_all_control_chart_warning_times = control_charts_on_directional_dfs(df_tt_sent, df_tt_rec, three_tier_services,
                                                                                    lambda_val=lambda_values,
                                                                                   stddev_coef=ewma_stddev_coef)

            tt_performance_results = calc_tp_fp_etc(("naive 3-tier control charts", lambda_values, ewma_stddev_coef),
                                                    exfils, tt_all_control_chart_warning_times, exp_time, start_analyze_time)
            experiment_results.update(tt_performance_results)

    '''


    '''
    svc_pair_to_sent_control_charts = generate_service_pair_arrays(df_sent_control_stats, times, services)
    svc_pair_to_sent_bytes = traffic_matrix_to_svc_pair_list(df_sent, services)
    print svc_pair_to_sent_control_charts['front-end', 'user']
    sent_data_for_display = {'raw': svc_pair_to_sent_bytes, 'control-charts':svc_pair_to_sent_control_charts}
    #generate_graphs(sent_data_for_display, times, display_sent_svc_pair, True, graph_names + "_sent_graphs")
    #### TODO ^^ put back in
    
    svc_pair_to_rec_control_charts = generate_service_pair_arrays(df_rec_control_stats, times, services)
    svc_pair_to_rec_bytes = traffic_matrix_to_svc_pair_list(df_rec, services)
    #print svc_pair_to_rec_control_charts['front-end', 'user']
    rec_data_for_display = {'raw': svc_pair_to_rec_bytes, 'control-charts':svc_pair_to_rec_control_charts}
    #generate_graphs(rec_data_for_display, times, display_rec_svc_pair, False, graph_names + "_rec_graphs")
    #### TODO ^^ put back in

    if display_graphs:
        plt.show()
    
    # user sent to front end is particulary important to me b/c that's where data exfiltration
    # happens
    generate_graphs(sent_data_for_display, times, [['front-end', 'user' ]], True, graph_names + "_user_sent_front_graphs")
    generate_graphs(rec_data_for_display, times, [['front-end', 'user' ]], False, graph_names + "_user_rec_front_graphs")

    # let's make some graphs for the 3-tier aggregates (so I can tell what is going on)
    three_tier_svc_pair_to_sent_control_charts = generate_service_pair_arrays(df_three_tier_sent_control_stats, times, three_tier_services)
    three_tier_svc_pair_to_sent_bytes = traffic_matrix_to_svc_pair_list(df_three_tier_sent, three_tier_services)
    three_tier_sent_data_for_display = {'raw': three_tier_svc_pair_to_sent_bytes, 'control-charts': three_tier_svc_pair_to_sent_control_charts}
    
    generate_graphs(three_tier_sent_data_for_display, times, [['application', 'data']], True, graph_names + "_three_tier_sent_ad")
    #generate_graphs(three_tier_sent_data_for_display, times, [['presentation', 'application']], True, graph_names + "_three_tier_sent_pa")
    #generate_graphs(three_tier_sent_data_for_display, times, [['presentation', 'data']], True, graph_names + "_three_tier_sent_pd")
    # we need to do it for everything b/c there is a bizarre number of alerts
    three_tier_svc_pair_to_rec_control_charts = generate_service_pair_arrays(df_three_tier_rec_control_stats, times, three_tier_services)
    three_tier_svc_pair_to_rec_bytes = traffic_matrix_to_svc_pair_list(df_three_tier_rec, three_tier_services)
    three_tier_rec_data_for_display = {'raw': three_tier_svc_pair_to_rec_bytes, 'control-charts': three_tier_svc_pair_to_rec_control_charts}
    generate_graphs(three_tier_rec_data_for_display, times, [['application', 'data']], True, graph_names + "_three_tier_rec_ad")
    #generate_graphs(three_tier_rec_data_for_display, times, [['presentation', 'application']], True, graph_names + "_three_tier_rec_pa")
    #generate_graphs(three_tier_rec_data_for_display, times, [['presentation', 'data']], True, graph_names + "_three_tier_rec_pd")
    '''


     # per the discussion, this is not longer very important
        # will probably leave out in perpetuity
    # now, I think it is worth putting in, even if just to show as a 'state-of-the-art' that sucks
    joint_df = join_dfs(df_sent, df_rec)
    print joint_df

    ### TODO: wire it in. those three params are all that matters (I think it is wireed in)
    for crit_percent in np.append( np.arange(0.0, .51, .01), np.array([[0.001]]) ):
        for windows_size in np.arange(5, 6, 1): # honestly not going to worry about this param very much
            for beta_val in np.arange(0, 0.55, 0.05):

                    # orig val: critical_percent=0.0003, window_size=5, beta=0.05
                    eigenspace_warning_times = eigenspace_detection_loop(joint_df, critical_percent=crit_percent,
                                                                 window_size=windows_size, beta=beta_val)

                    print "the eigenspace alert triggered at these times", eigenspace_warning_times

                    eigenspace_performance_results = calc_tp_fp_etc(("eigenspace", crit_percent, windows_size, beta_val),
                                                               exfils, eigenspace_warning_times, exp_time, start_analyze_time)

                    print crit_percent, eigenspace_performance_results

                    experiment_results.update(eigenspace_performance_results)

    #''' # this code is old and has been replaced with the above loop
    '''
    eigenspace_warning_times = eigenspace_detection_loop(joint_df, critical_percent=0.0003, 
        window_size=5, beta = 0.05 )

    print eigenspace_warning_times
    eigenspace_performance_results = calc_tp_fp_etc("eigenspace", exfils, eigenspace_warning_times, exp_time, start_analyze_time)
    experiment_results.update(eigenspace_performance_results)
    '''
    '''

    #### TODO: get this working. the problem is in join_dfs. see comment inside for details
    #tt_joint_df = join_dfs(df_three_tier_sent, df_three_tier_rec)
    #print tt_joint_df
    #tt_eigenspace_warning_times = eigenspace_detection_loop(tt_joint_df, critical_percent=0.0003, 
    #                window_size=5, beta = 0.05 )
    tt_eigenspace_performance_results = calc_tp_fp_etc("tt_eigenspace", exfils, [], 
            exp_time, start_analyze_time)
    experiment_results.update(tt_eigenspace_performance_results)
    '''

    # return experiment results, all ready for aggregation
    print "the results dictionary is", experiment_results
    
    return experiment_results

# this function exists so that I don't have to keep calling stuff
# twice b/c I don't combine the matrixes
# if also_rec is False, then don't do it for df_rec. This is b/c data exfil is most noticable in df_sene
def control_charts_on_directional_dfs(df_sent, df_rec, services_to_check, lambda_val, stddev_coef, also_rec=True):

    control_charts_warning_times_sent = control_charts_loop(df_sent, True, lambda_ewma=lambda_val,
                                                            ewma_stddev_coef=stddev_coef,
                                                            services_to_monitor=services_to_check)
    if also_rec:
        control_charts_warning_times_rec = control_charts_loop(df_rec, False, lambda_ewma=lambda_val,
                                                           ewma_stddev_coef=stddev_coef,
                                                           services_to_monitor=services_to_check)
    else:
        control_charts_warning_times_rec = []

    # print "sent control chart warning times", control_charts_warning_times_sent
    # combine the two sets of warning times (delete duplicates)
    all_control_chart_warning_times = sorted(
        list(set(control_charts_warning_times_sent + control_charts_warning_times_rec)))
    print "all the control chart warning times", all_control_chart_warning_times

    return all_control_chart_warning_times

# control_charts = ewma
# services_to_monitor is wierd b/c python has mutuable default arguments
# returns a list of warning times
def control_charts_loop(dfs, is_sent, lambda_ewma = 0.2, ewma_stddev_coef = 3, services_to_monitor = None):
    if not services_to_monitor:
        services_to_monitor = ['front-end', 'user']


    df_time_slices = generate_time_slice_dfs(dfs)
    df_control_stats = []
    prev_step_ewmas = empty_ewmas(services_to_monitor, df_time_slices[0])
    for df in df_time_slices:
        prev_step_ewmas = control_charts(df, is_sent, prev_step_ewmas, lambda_ewma, services_to_monitor)
        df_control_stats.append(prev_step_ewmas)

    # check when control charts would give a warning
    # just going to use sent for now, could use reciever later
    times = get_times(dfs)
    print "THESE ARE THE TIMES", times
    # starts at 1, b/c everyting has time stddev 0 at time 0, so everything would trigger a warning
    control_charts_warning = []
    control_charts_warning_times = []
    for time in range(1, len(times) - 1):
        next_dfs = dfs[dfs['time'].isin([times[time]])]
        warnings, warning_times = next_value_trigger_control_charts(next_dfs,
                                                                              df_control_stats[time], times[time],
                                                                              ewma_stddev_coef)
        #print "Microsvc sent and rec warnings during loop", warnings
        control_charts_warning.append(warnings)
        control_charts_warning_times += warning_times
    print "these are the warnings from the control charts: (for data that is sent): "
    print control_charts_warning
    print "just times:", control_charts_warning_times

    return control_charts_warning_times

# ratio_denoms is a tuple (src_svc, dst_svc)
# if coordinated_p, then ALL the ratio's would need to increase (not just one)
def ewma_svc_pair_ratio_loop(dfs, lambda_ewma, ewma_stddev_coef, ratio_denoms, svc_pair_list, num_exceed=1,
                             dont_trigger_if_decrease=False, linear_comb=False, abs_val = False):
    times = get_times(dfs)
    # print times
    df_time_slices = []
    for time_index in range(0, len(times)):
        time = times[time_index]
        df_time_slices.append( dfs[ dfs['time'].isin([time]) ] )

    ratio_ewma_stats = []
    ## VV let's generate all the svc pair traffic vectors at once!!
    svc_pair_traffic_vectors = svc_pair_traffic_vectors_gen(df_time_slices, svc_pair_list)
    #print "these are the svc pair traffic vects", svc_pair_traffic_vectors

    prev_step_ewma_stats = empty_ewma_stats(svc_pair_list, ratio_denoms, svc_pair_traffic_vectors[times[0]])

    for cur_svc_pair_traffic_vector_index in sorted(svc_pair_traffic_vectors.keys()):
        cur_svc_pair_traffic_vector = svc_pair_traffic_vectors[ cur_svc_pair_traffic_vector_index ]
        #print "this is the current traffic vector", cur_svc_pair_traffic_vector, "time", cur_svc_pair_traffic_vector_index
        prev_step_ewma_stats = ewma_svc_pair_ratio(cur_svc_pair_traffic_vector, 
                                                    prev_step_ewma_stats, lambda_ewma,
                                                    ratio_denoms, svc_pair_list)
        #print "these are the ewma stats", prev_step_ewma_stats
        ratio_ewma_stats.append(prev_step_ewma_stats)

    # check when ewma would give a warning
    times = get_times(dfs)
    # starts at 1, b/c everyting has time stddev 0 at time 0, so everything would trigger a warning
    ewma_ratio_warning = []
    ewma_ratio_warning_times = []
    for time in range(1, len(times) - 1):
        relevant_svc_pair_traffic_vect = svc_pair_traffic_vectors[times[time]]
        warnings, warning_times = next_value_trigger_ewma_ratio(relevant_svc_pair_traffic_vect,
                                                ratio_ewma_stats[time], times[time],
                                                ewma_stddev_coef, ratio_denoms, svc_pair_list, num_exceed,
                                                dont_trigger_if_decrease, linear_comb, abs_val = abs_val)

        #print "service ratios sent and rec warnings during loop", warnings
        ewma_ratio_warning.append(warnings)
        ewma_ratio_warning_times += warning_times
        
    #print "these are the warnings from the control charts: (for data that is sent): "
    #print ewma_ratio_warning
    #print "just times:", ewma_ratio_warning_times

    return sorted(list(set(ewma_ratio_warning_times)))

# so what is the traffic vector?
# it is (front-end, app-tier): bytes
def svc_pair_traffic_vectors_gen(df_time_slices, svc_pair_list):
    traffic_vectors_over_time = {}
    for df in df_time_slices:
        #print "current df", df
        traffic_vector = {}
        for src_dst in svc_pair_list:
            traffic_vector[src_dst[0], src_dst[1]] = df.loc[src_dst[0], src_dst[1]]
        #print df.loc[svc_pair_list[0][0], 'time']
        traffic_vectors_over_time[df.loc[svc_pair_list[0][0], 'time']] = traffic_vector
    return traffic_vectors_over_time


def empty_ewma_stats(svc_pair_list, ratio_denoms, traffic_vector):
    data_stats = {}
    for denom_svc_pair in ratio_denoms:
        for src_dst in svc_pair_list:
            current_ratio = traffic_vector[src_dst[0], src_dst[1]] / traffic_vector[ denom_svc_pair[0], denom_svc_pair[1]]
            data_stats[src_dst, denom_svc_pair] = [current_ratio , 0]
    return data_stats
    
def ewma_svc_pair_ratio(cur_svc_pair_traffic_vector, old_ewmas, lambda_ewma, ratio_denoms, svc_pair_list):
    ## going to return data in the form {[(src_svc, dest_svc), (denom_src_svc, denom_dst_svc)]: [ewma, ewma_stddev]}
    data_stats = {}
                
    for src_dst in svc_pair_list:
        current_entry = cur_svc_pair_traffic_vector[src_dst[0], src_dst[1]]
        for denom_svc_pair in ratio_denoms:
            current_ratio = current_entry / cur_svc_pair_traffic_vector[ denom_svc_pair[0], denom_svc_pair[1]]
            new_ewma = old_ewmas[src_dst, denom_svc_pair][0] * (1 - lambda_ewma) + lambda_ewma * current_ratio
            
            old_ewma_var = old_ewmas[src_dst, denom_svc_pair][1]
            new_ewma_var = (1 - lambda_ewma) * (old_ewma_var + lambda_ewma * (current_ratio - new_ewma) ** 2 )
                        
            data_stats[src_dst, denom_svc_pair] = [new_ewma, new_ewma_var]
            
    return data_stats
    
def next_value_trigger_ewma_ratio(relevant_svc_pair_traffic_vect, ratio_ewma_stats, time,
                                    ewma_stddev_coef, ratio_denoms, svc_pair_list, num_exceed, dont_trigger_if_decrease,
                                  linear_comb=False, abs_val = False):
    warnings_triggered = []
    warning_times = []
    num_exceeded_so_far = 0
    linear_combo_current_ratio_sum = 0
    linear_combo_mean_ratio_sum = 0
    linear_combo_stddev_sum = 0
    for src_dst in svc_pair_list:
        for denoms_svc_pair in ratio_denoms:
            if src_dst != denoms_svc_pair:
                current_ewma_mean = ratio_ewma_stats[src_dst, denoms_svc_pair][0]
                current_ewma_stddev = sqrt( ratio_ewma_stats[src_dst, denoms_svc_pair][1]) # recall, I am storing the variance
                current_traffic_vector_value = relevant_svc_pair_traffic_vect[src_dst[0], src_dst[1]]
                current_denom_value = relevant_svc_pair_traffic_vect[denoms_svc_pair[0], denoms_svc_pair[1]]
                current_ratio = current_traffic_vector_value / current_denom_value
                #if src_dst[1] == 'catalogue':
                #print "pair", src_dst, time
                #print "will an alarm sounds?", current_ratio, current_ewma_mean, current_ratio - current_ewma_mean
                #print "??", ewma_stddev_coef, current_ewma_stddev, (ewma_stddev_coef * current_ewma_stddev),
                #print "will it??",(current_ratio - current_ewma_mean) > (ewma_stddev_coef * current_ewma_stddev), '\n'
                if not linear_comb:
                    if (current_ewma_mean - current_ratio) < 0:
                        if dont_trigger_if_decrease:
                            num_exceeded_so_far += -100 # and hence, it won't ever trigger anything ;)
                    if not abs_val:
                        if (current_ewma_mean - current_ratio) > (ewma_stddev_coef * current_ewma_stddev):
                            num_exceeded_so_far += 1
                    else:
                        if abs(current_ewma_mean - current_ratio) > (ewma_stddev_coef * current_ewma_stddev):
                            num_exceeded_so_far += 1
                    if num_exceeded_so_far == num_exceed:
                        warnings_triggered.append([src_dst, denoms_svc_pair,  time, current_ewma_mean,
                                        current_ewma_stddev, current_ratio])
                        warning_times.append(time)
                else:
                    # okay, so I want to do a simple linear combination here
                    linear_combo_current_ratio_sum += current_ratio
                    linear_combo_mean_ratio_sum += current_ewma_mean
                    linear_combo_stddev_sum += current_ewma_stddev
    if linear_comb:
        if (linear_combo_mean_ratio_sum - linear_combo_current_ratio_sum) > (ewma_stddev_coef * linear_combo_stddev_sum):
            # note, would need to modify this if I get fancier w/ the what ratio_denoms is
            warnings_triggered.append(["linear combo", ratio_denoms[0], time, linear_combo_mean_ratio_sum,
                                       linear_combo_stddev_sum, linear_combo_current_ratio_sum])
            warning_times.append(time)

    return warnings_triggered, warning_times

# reverses df_sent and then combines them
# (this is harder than it initially sounds)
def join_dfs(df_sent, df_rec):
    times = get_times(df_sent)
    #joint_df = df_rec.add(df_sent)
    # I am transposing df_sent b/c directionality is determined in relation to the destination pod
    # so the destination pod SENDS data in df_sent, so it is really the sender
    # note that destination pod is defined per the TCP handshake, not per individual packets
    #print df_sent
    time_groups = df_sent.groupby(['time'])
    joint_df = pd.DataFrame()
    for name,grp in time_groups:
        #print name
        current_time = grp['time']
        #print grp
        current_df_sent =  pd.concat([grp.drop(['time'], axis=1).T, current_time], axis=1)
        #print current_df_sent
        joint_df = pd.concat([joint_df, current_df_sent])
        #print joint_df
    # need to remove and then add back in the time column
    time_column = joint_df['time']
    #print "joint", joint_df
    #print "rec", df_rec.drop(['time'], axis=1)
    #### TODO: find of what is wrong with the line below. It doesn't work properly for the three-tier architecture
    #### (I already checked the rest of the func and it was fine)
    joint_df = joint_df.drop(['time'], axis=1).add(df_rec.drop(['time'], axis=1))
    #print "joint", joint_df, "times",  time_column
    joint_df = pd.concat([joint_df, time_column], axis = 1)
    #print "joint df", joint_df
    return joint_df

# this fuction runs the loop for the eignspace-detection method (over time)
# returns a list of times
def eigenspace_detection_loop(joint_df, critical_percent, window_size, beta):
    times = get_times(joint_df)
    old_u = np.array([])
    z_first_moment = 0 
    z_sec_moment = 0
    old_z_thresh = 0.5
    eigenspace_warning_times = []
    for time_index in range(0,len(times)):
        time = times[time_index]
        current_df = joint_df[ joint_df['time'].isin([time])]
        #print current_df
        alarm_p, old_u, z_first_moment, z_sec_moment,old_z_thresh = eigenvector_based_detector(old_u,
                current_df.drop(['time'], axis=1), window_size, critical_percent, z_first_moment, z_sec_moment, old_z_thresh, beta)
        print "z first moment", z_first_moment, "z second moment", z_sec_moment
        print alarm_p#, old_u, z_first_moment, z_sec_moment
        print type(alarm_p)
        if alarm_p:
            eigenspace_warning_times.append(time)
    return eigenspace_warning_times

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
        print [item[0] for item in svc_pair_to_control_charts[cur_src_svc, cur_dst_svc]]
        avg_line, = plt.plot(times, [item[0] for item in svc_pair_to_control_charts[cur_src_svc, cur_dst_svc]], label='mean')
        avg_plus_one_stddev = [item[0] + item[1] for item in svc_pair_to_control_charts[cur_src_svc, cur_dst_svc]]
        control_chart_above, = plt.plot(times, avg_plus_one_stddev, label='mean + 1 * stddev')
        avg_minus_one_stddev = [item[0] - item[1] for item in svc_pair_to_control_charts[cur_src_svc, cur_dst_svc]]
        control_chart_below, = plt.plot(times, avg_minus_one_stddev, label='mean - 1 * stddev')
        avg_plus_two_stddev = [item[0] + 3 * item[1] for item in svc_pair_to_control_charts[cur_src_svc, cur_dst_svc]]
        control_chart_two_above, = plt.plot(times, avg_plus_two_stddev, label='mean + 2 * stddev')
        avg_minus_two_stddev = [item[0] - 3 * item[1] for item in svc_pair_to_control_charts[cur_src_svc, cur_dst_svc]]
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
def traffic_matrix_to_svc_pair_list(df, svc):
    svcs_to_val_list = {}
    for src_svc in svc:
        for dst_svc in svc:
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
def generate_service_pair_arrays(stats, times, svcs):
    svc_to_vals = {}
    for sending_svc in svcs:
        for dst_svc in svcs:
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
            #print old_ewmas[index_service, column_service] 
            new_ewma = old_ewmas[index_service, column_service][0] * (1 - lambda_ewma) + lambda_ewma * \
                    df.loc[df['time'] == time].loc[index_service, column_service]
            #print index_service, column_service, new_ewma
            new_ewma_stddev = sqrt( ((lambda_ewma) / (2 - lambda_ewma)) * (relevant_traffic_values.std()**2))
            data_stats[index_service, column_service] = [new_ewma, new_ewma_stddev ]
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
    for src_dst, mean_stddev in data_stats.iteritems():
        #print "src_dst value: ", src_dst, "mean_stddev value: ", mean_stddev
        next_val = next_df.loc[ src_dst[0], src_dst[1] ]
        mean, stddev = mean_stddev[0], mean_stddev[1]
        ## NOTE: I am only checking for LARGER than (rather than also smaller) b/c attacker can only add traffic
        if (next_val - mean) > (stddev_coef * stddev):
            #print "THIS IS THE POOR MAN'S EQUIVALENT OF AN ALARM!!", src_dst, mean_stddev
            warnings_triggered.append([src_dst[0], src_dst[1], time, mean_stddev, next_val])
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
    #print "calc_tp_fp_etc: ", algo_name, exfils, warning_times, exp_time, start_analyze_time

    # this is a work around b/c on one / two experiments, the timer-thing got out of wack
    for wt in warning_times:
        if int(wt) % 5 > 0:
            return {}

    attack_times = exfils.keys()
    #print attack_times, warning_times
    total_positives = len(attack_times)
    true_positives = 0
    warning_times_after_strt_analyze = [time for time in warning_times if time >= start_analyze_time]
    for attack_time in attack_times:
        # need to add 5 b/c exfil starts at the attack_time, so it is recorded 5 sec later by Prometheus
        if (attack_time+5) in [int(time) for time in warning_times_after_strt_analyze]:
            true_positives = true_positives + 1
    false_negatives = total_positives - true_positives
    total_postives_signalled = len(warning_times_after_strt_analyze)
    false_positive = total_postives_signalled - true_positives
    total_negatives_signaled = ((exp_time - start_analyze_time)/5.0) - total_postives_signalled
    true_negatives = total_negatives_signaled - false_negatives

    ### I need to give the above problem more consideration, but I'm not actually using it at all
    ### and I am too tired to think straight and I gave it a lot of thought before
    print "TPs", true_positives, "FPs", false_positive, \
        "TNs",  true_negatives, "FNs", false_negatives ,"Total negs signalled", total_negatives_signaled

    # tpr = (true positives) / (condition positive)
    # fpr = (false positives) / (condition negative)
    # tnr = (true negatives) / (condition negative)
    # fnr = (false_negatives) / (condition positive)

    tpr = float(true_positives) / total_positives
    fpr = float(false_positive) / (true_negatives + false_positive)
    tnr = float(true_negatives) / (true_negatives + false_positive)
    fnr = float(true_negatives) / total_positives
    f_one = (2.0*true_positives)/(2*true_positives + false_positive + false_negatives)

    return {algo_name: {"TPR":  tpr,
                        "FPR" : fpr, 
                        "FNR" : fnr,
                        "TNR" : tnr,
                        "f_one": f_one }}

# following method in Lakhina's "Diagnosing
# Network-Wide Traffic Anomalies" (sigcomm '04)
# note: this is a real paper and the method is NOT nonsense ;)
# where it tests if it exceeds the 1 - alpha confidence level
# 1 - alpha is the percentile in the standard normal distro we're testing against
def diagnose_anom_pca(old_dfs, cur_df, n_components, alpha):
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
    pca = SparsePCA(n_components=n_components)
    converted_array_along_principal_components = pca.fit_transform(converted_dfs)
    #df = pd.DataFrame()
    converted_dfs_along_principal_components = pd.DataFrame(converted_array_along_principal_components)
    print converted_dfs_along_principal_components
    #print p
    #### TODO: is this a unit vector? it needs to be
    #### TODO: does PCA use StandardScaler (hence obviating the need for #2)

    # fourth, map data to principal axis
    # note: not needed,  handeled by fit_transform above

    # fifth, split data into two types of projections:
    # (1) normal, (2) anomalous, via threshold
    anom_thresh = -1
    cur_index = 0

    # TODO: this code is WRONG!!! For some reason it is iterating through the vectors taken at certain times
    # it should be iterating through the projections onto each individual vector space
    #for index, row in converted_dfs_along_principal_components.iterrows():
    for column_index in converted_dfs_along_principal_components:
        print "cur_index is", cur_index
        column = converted_dfs_along_principal_components[column_index]
        #print "index", index
        print "col", column
        row_mean = column.mean()
        row_stddev = column.std()
        print "col mean", row_mean, "col stddev", row_stddev
        print "row", column
        for col_item in column:
            #print row_item,
            if (col_item > row_mean + 3 * row_stddev) or (col_item < row_mean - 3 * row_stddev):
                anom_thresh = cur_index
        cur_index = cur_index + 1
        if anom_thresh != -1:
            break
    if anom_thresh == -1:
        anom_thresh = cur_index
    print "the anomalous threshold is", anom_thresh
    if anom_thresh <= 0:
        print "SOMETHING WENT WRONG"

    # sixth, project cur_df onto both projections
    # I could theoretically, get fancy with skikit-learn, but
    # it is probably better just to form the matrixes described
    # in the paper and carry out the designated computations
    # SO, take each principal component, and make it a row in the 
    # corresponding matrix
    #### TODO: Okay, this is wrong also. What the heck is in P??
    P = np.array(pca.components_[0]) # this one has got to not be anomolous
    print "P", P
    for i in range(1, anom_thresh):
        P = np.stack([P, pca.components_[i]], axis=0)

    print P.size, np.transpose(np.asmatrix(P)).size
    C = P * np.transpose(np.asmatrix(P)).size
    print C.size
    y_circum = C * mod_cur_df
    y_tilda = (1 - C) * mod_cur_df

    # seventh, compute threshold for the  squared prediction 
    # error (SPE) via the Q-statistic
    # TODO: finish implementing function by starting here. (this part done)
    # do steps 7 and 8, then run a couple tests to make sure it's fine,
    # and THEN investigate the TODO's up above
    phi = []
    r = anom_thresh  - 1
    m = np.size(pca.components_[0])
    for i in range(1,4):
        sigma = 0
        for j in range(r+1, m+1):
            sigma += converted_dfs_along_principal_components.explained_variance_[j] ** i
        phi.append(sigma)

    h_naught = 1 - (2 * (phi[0] * phi[2]) / (3 * phi[1]**2))

    # eighth, compare SPE to threshold to determine if anomaly occurs
    spe = np.linalg.norm(y_tilda)
    c_alpha = scipy.stats.norm.ppf(1 - alpha) # is the 1 - alpha percentile in a standard normal distribution
    first_term = c_alpha * sqrt(2 * phi[1] * h_naught**2) / phi[0]
    second_term = phi[1] * h_naught * (h_naught - 1) / phi[0]**2
    delta_a_squared = phi[0] * (first_term + 1 + second_term) ** (1.0 / h_naught)
    if spe > delta_a_squared:
        print "alarm is triggered"
        return True
    else:
        print "an alarm is NOT triggered"
        return False

# following method given in Ide's "Eigenspace-based Anomaly Detection in
# Computer Systems"
def eigenvector_based_detector(old_u, current_tm, window_size, crit_bound, old_z_first_mom, old_z_sec_mom, last_z_thresh, beta = 0.05):
    # first, find the principle eigenvector of the traffic matrix
    #print "shape: ", current_tm.shape, current_tm
    #eigenvals, unit_eigenvect = np.linalg.eig(current_tm)   
    eigenvals, unit_eigenvect = scipy.sparse.linalg.eigs(current_tm.as_matrix(), k=10, which="LM")
    
    # principle eigenvector has largest associated eigenvalue
    largest_eigenval_index = np.argmax(eigenvals)
    princip_eigenvect = unit_eigenvect[largest_eigenval_index]
    #print "pprincip_eigenvect", princip_eigenvect
    #print "eigenvects", unit_eigenvect
    #print "eigenvals", eigenvals

    # second, obtain "typical pattern" of activity vector from old_u
    # this is the principal left singular vector
    #print "size", np.shape(old_u), len(np.shape(old_u)), old_u
    if len(np.shape(old_u)) > 1:
        u,s,vh = np.linalg.svd(old_u)
        largest_signular_val_index = np.argmax(s)
        princip_left_singular_vect = u[largest_signular_val_index]

        # third, compute z(t), the dissimilarity between the principal left
        # singular vector and the principal eigenvector
        z = 1 - princip_left_singular_vect[np.newaxis] * np.transpose(np.asmatrix(princip_eigenvect)) # numpy auto transposes <- no it doesn't!!
        #print "z:", z
        z = np.absolute(z).tolist()[0][0]
        #print "z:", z
        # note: is 1 if orthogonal, is 0 if identical

        # now compare z(t) with a threshold, using section 5.3
        # approximate the MLE algorithm for the vMF distribution
        #print "old z_first_moment", old_z_first_mom
        # helps the system 'get up to speed'
        # nevermind, it just takes time to get up to speed
        #if old_z_first_mom == 0:
        #    old_z_first_mom = z+0.1
        #if old_z_sec_mom == 0:
        #    old_z_sec_mom = z **2
        z_first_moment = (1 - beta) * old_z_first_mom + beta * z
        #print "new z_first_moment", z_first_moment
        z_sec_moment = (1 - beta) * old_z_sec_mom + beta * (z ** 2)
        #print "z_sec_moment", z_sec_moment
        n = ((2 * z_first_moment ** 2) / (z_sec_moment - z_first_moment ** 2)) + 1
        sigma = (z_sec_moment - z_first_moment ** 2) / (2 * z_first_moment)
        # find the specific threshold value
        print "ARGS","n", n, "sigma", sigma,"crit bound", crit_bound
        z_thresh = scipy.optimize.fsolve(vMF_thresh_func, last_z_thresh, args=(n, sigma, crit_bound))
        print "z_thresh is", z_thresh
        z_thresh = z_thresh[0]
        print "z_thresh is", z_thresh 
        
        #do the actual comparison, but first modify u
        if old_u.shape[1] >= window_size:
            np.delete(old_u, 0, 0)# get rid of old column (outside of window)
        #print "old_u size", np.shape(old_u)
        #print "princip_eigenvect size", np.shape(np.transpose(princip_eigenvect[np.newaxis]))
        #old_u = np.stack([old_u,princip_eigenvect], axis=1)
        old_u = np.concatenate((old_u, np.transpose(princip_eigenvect[np.newaxis])), axis=1)
        #print np.size(old_u)
        print "z", z, "z_thresh", z_thresh
        if z > z_thresh:
            return 1, old_u, z_first_moment, z_sec_moment, z_thresh  # 1 = an alert
        else:
            return 0, old_u, z_first_moment, z_sec_moment, z_thresh  # 0 = no alert
    else:
        #old_u = princip_eigenvect
        if np.shape(old_u)[0] == 0:
            old_u = princip_eigenvect
        else:
            old_u = np.stack([old_u,princip_eigenvect],axis=1)
        return 0, old_u, 0, 0, 0.5  # 0 = no alert

def vMF_pdf_func(z, n, sigma):
    if z < 0:
        return 1000000 # this ensures that fsolve won't get negative 
    #if n < 1:
    #    return 1000000
    denom = (2* sigma) ** ((n-1)/2) * scipy.special.gamma((n-1)/2)
    return (np.exp((-1 * z) / (2 * sigma)) * z ** (((n-1)/2)-1)) / denom

def vMF_thresh_func(zth, n, sigma, crit_bound):
    unadjusted_bound, err  =  scipy.integrate.quad(vMF_pdf_func, zth, np.inf, args=(n,sigma)) 
    #print "unadjusetd bound", unadjusted_bound, "critical bound", crit_bound
    return unadjusted_bound- crit_bound

def empty_ewmas(svcs, df):
    data_stats = {}
    for index_service in svcs:
        for column_service in svcs:
            data_stats[index_service, column_service] = [df.loc[index_service, column_service], 0]
    return data_stats

if __name__=="__main__":
    rec_matrix_location = './experimental_data/' + parameters.rec_matrix_location
    send_matrix_location = './experimental_data/' + parameters.sent_matrix_location
    if len(sys.argv) > 2:
        rec_matrix_location = sys.argv[1]
        send_matrix_location = sys.argv[2]
    main(rec_matrix_location, send_matrix_location)
