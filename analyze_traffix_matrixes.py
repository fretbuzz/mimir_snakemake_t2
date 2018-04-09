import pickle
import pandas as pd
import numpy as np
import sys
'''
USAGE: python analyze_traffic_matrixes.py [recieved_matrix_location] [sent_matrix_location]

## Note: if the rec/sent matrix locations aren't given, will use defaults

'''
# TODO: see if the send-recieved pairs match up as we'd expect them to
# first statistic that I want: control chart. (need to write e2e tests, but other than that maybe fine?)
# second statistic that I want: PCA

# TODO: visualization + e2e tests + PCA
# plus maybe like multivariate linear regression?
# let's be more specific, what kinda visualization??
# well for now probably just a control chart? should probably be a 
# package that will do most of the work. Of course, I'm going to have
# a ton of charts... looks like pyspc is the way to go
# testing should probably be unit actually and could use PyUnit
# PCA should wait until I have fixed these other things, as otherwise I could
# be digging myself into a really deep hole
# TODO: right now I am still working on a service-to-service granularity
# perhaps I need to move to a level of abstraction beyond that
# aggregate into the "3-level-enterprise-app" abstraction
# it is actually called the "three-tier architecture"

# 1. 3-tier architecture
# 2. control charts
# 3. a couple of test cases, just to make sure everything isn't completely wrong
# 4. PCA
# 5. Integrate meaningful traffic data
# 6. investigate the send-recieve pair thing
# 7. re-organize run_experiment.py and it's related stuff
# 8. have the prometheus calling function specify a time-stamp
# 9. 

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

# DF(with lots of times) -> [DF(time A), DF(time A and B), DF(time A,B,C), etc.]
def generate_time_slice_dfs(df):
    times = get_times(df)
    elapsed_time = []
    print times
    time_slices = []
    #elapsed_time.append(times[0]) ## TODO: find a better solution that this
    for time_index in range(0,len(times)):
        print "starting loop...."
        time = times[time_index]
        elapsed_time.append(time)
        df_so_far = df[ df['time'].isin(elapsed_time)]

        #sent_stats = control_charts(df_sent_so_far, True)
        print elapsed_time, time_index, len(times)
        #print sent_stats
        print df_so_far
        '''
        # now let's check if it will trigger an alarm
        #print "SUP", df_sent
        next_sent_traffic_matrix = df_sent[ df_sent['time'].isin([times[time_index+1]]) ]
        #next_value_trigger_control_charts(next_sent_traffic_matrix, sent_stats)
        print "Finished displaying sent traffix matrix data..."
        #print "Here is the recieved traffic matrixes"
        #print df_rec
        #print "\nDisplaying recieved traffic matrix data..."
        rec_stats = control_charts(df_rec_so_far, False)
        #print rec_stats
        # now let's check if it will trigger an alarm
        next_rec_traffic_matrix = df_rec[df_rec['time'].isin([times[time_index+1]])]
        #next_value_trigger_control_charts(next_rec_traffic_matrix, rec_stats)
        #print "Finished displaying rec traffix matrix data..."
        '''
        time_slices.append(df_so_far)
        #time = times[time_index]
        #elapsed_time.append(time)
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
            if relevant_traffic_values.mean() != 0:
                '''
                if is_send:
                    print "\n", index_service, " SENT TO ", column_service
                else:
                    print "\n", index_service, " RECEIVE FROM ", column_service
                # this is where I could do something fancier than just mean and stddev
                # I could do this by being fancy with data_stats (perhaps cleverly unpacking later on)
                #print relevant_traffic_values.describe()
                print "Mean: ", relevant_traffic_values.mean()
                print "Stddev: ", relevant_traffic_values.std()
                '''
                data_stats[index_service, column_service] = [relevant_traffic_values.mean(), relevant_traffic_values.std()]
    return data_stats

def get_times(df):
    times = []
    for x in df.loc[:, 'time']:
        times.append(x)
    times = sorted(list(set(times)))
    return times

# this function uses the statistics that are in data_stats to
# see if the next value for a service pair causes 
# an alarm via control chart anomaly detection
def next_value_trigger_control_charts(df, data_stats):
    ## iterate through values of data_stats
    ## get value from traffic matrix
    ## if outside of bounds, print something in capital letters
    for entry in data_stats:
        print entry
        print df
        print df.loc[ entry[0], entry[1] ]
        next_val = df.loc[ entry[0], entry[1] ]
        mean, stddev = entry[2], entry[3]
        if abs(next_val - mean) > (2 * stddev):
            print "THIS IS THE POOR MAN'S EQUIVALENT OF AN ALARM!!", entry

if __name__=="__main__":
    rec_matrix_location = './experimental_data/cumul_received_matrix.pickle'
    send_matrix_location = './experimental_data/cumul_sent_matrix.pickle'
    if len(sys.argv) > 2:
        rec_matrix_location = sys.argv[1]
        send_matrix_location = sys.argv[2]
    main(rec_matrix_location, send_matrix_location)
