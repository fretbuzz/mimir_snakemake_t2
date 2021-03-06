import pandas as pd
from collections import Counter
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from textwrap import wrap

# dataframe, loop through the columns, determine the contribution of the various features, and store them into a
# dataframe, which is return and put into a seaborn heatmap.
# NOTE: could do all of this in a seperate thread while the other part is happening...
def generate_covariate_heatmap(coef_dict, X_test, exfil_type_series):
    #exfil_type_occurnces = Counter([tuple(i) if type(i)==list else i for i in exfil_type_series])
    #exfil_type_occurnces = Counter([str(i) if type(i)==list else i for i in exfil_type_series])
    exfil_type_occurnces = Counter([str(i) for i in exfil_type_series])
    exfil_types = exfil_type_occurnces.keys()
    coef_impact_df = pd.DataFrame(0.0, index=exfil_types, columns=coef_dict.keys())
    raw_feature_val_df = pd.DataFrame(0.0, index=exfil_types, columns=coef_dict.keys())
    counter = 0
    for index, row in X_test.iterrows() :
        #try:
        #    print("current_exfil_type",tuple(list(exfil_type_series)[counter]))
        #except:
        print("current_exfil_type",list(exfil_type_series)[counter])
        print("coef_impact_df.index", coef_impact_df.index.tolist())

        for column_name in row._index:
            #print "index", index
            #print "row", row
            #print "column_name", column_name
            column_val = row[column_name]
            #print list(exfil_type_series)
            #print counter
            #current_exfil_type = str(list(exfil_type_series)[counter]) # note: added str to solve indexing error...
            #urrent_exfil_type = list(exfil_type_series)[counter] # note: added str to solve indexing error...
            #try:
            #    current_exfil_type = tuple(list(exfil_type_series)[counter]) # note: added str to solve indexing error...
            #except:
            current_exfil_type = str(list(exfil_type_series)[counter])
            coef_impact_df[column_name][current_exfil_type] += column_val * coef_dict[column_name]
            raw_feature_val_df[column_name][current_exfil_type] += column_val
        counter += 1
    ## divide each row by the number of occurences of that type of exfilration
    for cur_exfil_type, occurences in exfil_type_occurnces.iteritems():
        print("cur_exfil_type",cur_exfil_type,type(cur_exfil_type), "occurences", occurences)
        print coef_impact_df.columns
        coef_impact_df.loc[[cur_exfil_type], :] = coef_impact_df.loc[[cur_exfil_type],:] / occurences
        raw_feature_val_df.loc[[cur_exfil_type], :] = raw_feature_val_df.loc[[cur_exfil_type],:] / occurences
    return coef_impact_df, raw_feature_val_df

def generate_heatmap(coef_impact_df, path, non_local_path):
    fig = plt.figure(figsize=(35,35))
    sns_plot = sns.heatmap(coef_impact_df, robust=True, annot=True)#,linewidths=.5)
    ax = fig.add_subplot(sns_plot)
    labels = coef_impact_df.index.values
    labels = ['\n'.join(wrap(l, 30)) for l in labels]

    x_labels = coef_impact_df.columns.values
    x_labels = ['\n'.join(wrap(str(l), 30)) for l in x_labels]

    plt.setp(ax.set_yticklabels(labels))
    plt.setp(ax.set_xticklabels(x_labels))

    fig.savefig(path)
    #fig.savefig(non_local_path)