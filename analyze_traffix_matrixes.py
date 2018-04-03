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

    print "Here is the sent traffic matrixes"
    print df_sent
    for index_service in services:
        for column_service in services:
            # NOTE: this is where I'd condition on time values, if I wanted to do
            # like a moving average or something
            relevant_traffic_values = df_sent.loc[index_service, column_service]
            print relevant_traffic_values

    print "Here is the recieved traffic matrixes"
    print df_rec

if __name__=="__main__":
    main()

