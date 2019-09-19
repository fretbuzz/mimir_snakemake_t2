import pickle, sys
import pandas as pd

if __name__=="__main__":
    print "RUNNING"
    print sys.argv

    with open('./check_this.txt', 'r') as f:
        cont = f.read()

    pd_df = pickle.loads(cont)

    print "all done!"
