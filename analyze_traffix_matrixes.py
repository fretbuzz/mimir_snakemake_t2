import pickle
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
'''
USAGE: python analyze_traffic_matrixes.py [recieved_matrix_location] [sent_matrix_location]

## Note: if the rec/sent matrix locations aren't given, will use defaults

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
        'user-db',
        'load-test',
        '127.0.0.1', #it's always there, so I guess we should treat it like a normal thing
        '172.17.0.1' # also this one too
]

def main(rec_matrix_location, send_matrix_location):
    simulate_incoming_data(rec_matrix_location, send_matrix_location)

# This function reads pickle files corresponding to the send/received traffic matrices
# and then iterates through them by the time stamps, letting us pretend that the data
# is coming from an actively running system
def simulate_incoming_data(rec_matrix_location, send_matrix_location):
    print "hello world"
    df_sent = pd.read_pickle(send_matrix_location)
    df_rec = pd.read_pickle(rec_matrix_location)
    #print "df_sent:", df_sent
    #print "df_rec:", df_rec
    df_sent_time_slices = generate_time_slice_dfs(df_sent)
    df_rec_time_slices = generate_time_slice_dfs(df_rec)
    df_sent_control_stats = []
    df_rec_control_stats = []
    for df in df_sent_time_slices:
        df_sent_control_stats.append(control_charts(df, True))
        print df
    for df in df_rec_time_slices:
        df_rec_control_stats.append(control_charts(df, False))
    print df_sent_control_stats
    
    # check when control charts would give a warning
    # just going to use sent for now, could use reciever later
    times = get_times(df_sent)
    # starts at 1, b/c everyting has time stddev 0 at time 0, so everything would trigger a warning
    for time in range(1,len(times)-1):
        next_df_sent = df_sent[ df_sent['time'].isin([times[time]]) ]
        next_value_trigger_control_charts(next_df_sent, df_sent_control_stats[time])
    
    # might want to make a function to generate graph-able arrays or something
    f_to_u = []
    for time_slice in df_sent_control_stats:
        print "f to u", time_slice['front-end', 'user']
        f_to_u.append(time_slice['front-end', 'user'])
        print "u to f", time_slice['user', 'front-end']
    print f_to_u
    plt.plot(f_to_u)
    plt.ylabel('bytes')
    plt.show()

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
def control_charts(df, is_send):
    ## going to return data in the form {[src_svc, dest_svc]: [mean, stddev]}
    data_stats = {} #[]
    for index_service in services:
        for column_service in services:
            # NOTE: this is where I'd condition on time values, if I wanted to do
            # like a moving average or something (well might be earlier actually)
            relevant_traffic_values = df.loc[index_service, column_service]
            #print relevant_traffic_values, type(relevant_traffic_values)
            #if relevant_traffic_values.mean() != 0:
                #print_control_charts_process(relevant_traffic_values, is_send, index_service, column_service)
            data_stats[index_service, column_service] = [relevant_traffic_values.mean(), relevant_traffic_values.std()]
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
def next_value_trigger_control_charts(next_df, data_stats):
    ## iterate through values of data_stats
    ## get value from traffic matrix
    ## if outside of bounds, print something in capital letters
    for src_dst, mean_stddev in data_stats.iteritems():
        #print "src_dst value: ", src_dst, "mean_stddev value: ", mean_stddev
        next_val = next_df.loc[ src_dst[0], src_dst[1] ]
        mean, stddev = mean_stddev[0], mean_stddev[1]
        if abs(next_val - mean) > (2 * stddev):
            print "THIS IS THE POOR MAN'S EQUIVALENT OF AN ALARM!!", entry

if __name__=="__main__":
    rec_matrix_location = './experimental_data/cumul_received_matrix.pickle'
    send_matrix_location = './experimental_data/cumul_sent_matrix.pickle'
    if len(sys.argv) > 2:
        rec_matrix_location = sys.argv[1]
        send_matrix_location = sys.argv[2]
    main(rec_matrix_location, send_matrix_location)
