import pickle
import pandas as pd
import numpy as np

# TODO: see if the send-recieved pairs match up as we'd expect them to
# assume that I have the pickled data frames
# unpickle the frames
# find the differentials
# make the time column an index (maybe don't need to do this)
# first statistic that I want: control chart. 
# second statistic that I want: PCA

## Couple of things to talk about with the control charts. Right now it is just calculating
## the total statistics for the whole thing. But ideally, it'd be calculating them as it goes...
## maybe make the pickle-read function optional? Then I could call this from my pull_from_prom
## and get the relevant stats as we go...
## This might make more sense: right a function that "walks" through the traffic matrixes
## at each time step and then calculate the values. So have a special function that unpickles
## and goes through the matrixes and then calls the function that pull_from_prom would 
## hypothetically use


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

def main():
    simulate_incoming_data()

# This function reads pickle files corresponding to the send/received traffic matrices
# and then iterates through them by the time stamps, letting us pretend that the data
# is coming from an actively running system
def simulate_incoming_data():
    print "hello world"
    df_sent = pd.read_pickle('./experimental_data/cumul_sent_matrix.pickle')
    df_rec = pd.read_pickle('./experimental_data/cumul_received_matrix.pickle')


    ## TODO: I need to split the traffic matrixes by time
    times = get_times(df_sent)
    elapsed_time = []
    print times
    elapsed_time.append(times[0]) ## TODO: find a better solution that this
    for time in times[1:]:
        elapsed_time.append(time)
        df_sent_so_far = df_sent[ df_sent['time'].isin(elapsed_time)]
        df_rec_so_far = df_rec[ df_rec['time'].isin(elapsed_time)]
        #df_sent_so_far = df_sent
        #df_rec_so_far = df_rec


        #print df_sent_so_far
        #print df_rec_so_far
        #print "Here is the sent traffic matrixes"
        #print df_sent
        print "\nDisplaying sent traffic matrix data..."
        sent_stats = control_charts(df_sent_so_far, True)
        print sent_stats
        print "Finished displaying sent traffix matrix data..."

        #print "Here is the recieved traffic matrixes"
        #print df_rec
        print "\nDisplaying recieved traffic matrix data..."
        rec_stats = control_charts(df_rec_so_far, False)
        print rec_stats
        print "Finished displaying rec traffix matrix data..."

# this is the function to implement control channels
# i.e. compute mean and standard deviation for each pod-pair
# Note: direction is 1 if it is the "send matrix", else zero
def control_charts(df, is_send):
    ## going to return data in the form [src_svc, dest_svc, mean, stddev]
    data_stats = []
    for index_service in services:
        for column_service in services:
            # NOTE: this is where I'd condition on time values, if I wanted to do
            # like a moving average or something
            relevant_traffic_values = df.loc[index_service, column_service]
            #print relevant_traffic_values, type(relevant_traffic_values)
            if relevant_traffic_values.mean() != 0:
                if is_send:
                    print "\n", index_service, " SENT TO ", column_service
                else:
                    print "\n", index_service, " RECEIVE FROM ", column_service
                print relevant_traffic_values.describe()
                print "Mean: ", relevant_traffic_values.mean()
                print "Stddev: ", relevant_traffic_values.std()
                data_stats.append([index_service, column_service, relevant_traffic_values.mean(), relevant_traffic_values.std()])
    return data_stats

def get_times(df):
    times = []
    for x in df.loc[:, 'time']:
        times.append(x)
    times = sorted(list(set(times)))
    return times

if __name__=="__main__":
    main()
