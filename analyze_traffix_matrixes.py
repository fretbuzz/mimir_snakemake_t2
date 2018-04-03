import pickle
import pandas as pd
import numpy as np

# assume that I have the pickled data frames
# unpickle the frames
# find the differentials
# compute the statistics that I want:

def main():
    print "hello world"
    df_sent = pd.read_pickle('./experimental_data/cumul_sent_matrix.pickle')
    df_rec = pd.read_pickle('./experimental_data/cumul_received_matrix.pickle')

    print "Here is the sent traffic matrixes"
    print df_sent

    print "Here is the recieved traffic matrixes"
    print df_rec

if __name__=="__main__":
    main()

