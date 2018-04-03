import pickle
import pandas as pd
import numpy as np

# TODO: see if the send-recieved pairs match up as we'd expect them to
# assume that I have the pickled data frames
# unpickle the frames
# find the differentials
# make the time column an index (maybe don't need to do this)
# first statistic that I want: control chart. 
#   How to get this: 
#       (1) need to select relevant entries for each pod pair
#           need to condition on values OTHER than time
#           what values to select? need to select rows that match particular vlaues
#               then select some of the resulting columns
#       (2) for now just use .describe() to see a bunch of stats, can customize in a little while
# second statistic that I want:
#   PCA

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
    print "hello world"
    df_sent = pd.read_pickle('./experimental_data/cumul_sent_matrix.pickle')
    df_rec = pd.read_pickle('./experimental_data/cumul_received_matrix.pickle')

    #print "Here is the sent traffic matrixes"
    #print df_sent
    print "\nDisplaying sent traffic matrix data..."
    control_charts(df_sent, True)
    print "Finished displaying sent traffix matrix data..."

    #print "Here is the recieved traffic matrixes"
    #print df_rec
    print "\nDisplaying recieved traffic matrix data..."
    control_charts(df_rec, False)
    print "Finished displaying rec traffix matrix data..."

# this is the function to implement control channels
# i.e. compute mean and standard deviation for each pod-pair
# Note: direction is 1 if it is the "send matrix", else zero
def control_charts(df, is_send):
    for index_service in services:
        for column_service in services:
            # NOTE: this is where I'd condition on time values, if I wanted to do
            # like a moving average or something
            relevant_traffic_values = df.loc[index_service, column_service]
            #print relevant_traffic_values
            if relevant_traffic_values.mean() != 0:
                if is_send:
                    print "\n", index_service, " SENT TO ", column_service
                else:
                    print "\n", index_service, " RECEIVE FROM ", column_service
                print relevant_traffic_values.describe()
                print "Mean: ", relevant_traffic_values.mean()
                print "Stddev: ", relevant_traffic_values.std()

if __name__=="__main__":
    main()
