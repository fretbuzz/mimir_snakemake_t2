
import json
from remote_experiment_runner import run_experiment
from analysis_pipeline.generate_paper_graphs import run_looper, generate_graphs, generate_secondary_cache_name
import pickle
import pandas as pd
import os

def get_eval_cm():
    pass

def main(experimental_directories, avg_results_filename):
    # step 1: find the result csvs
    results_csvs = []
    for experimental_directory in experimental_directories:
        results_csv = find_result_csvs(experimental_directory)
        results_csvs.append(results_csv)

    print "results_csvs", results_csvs

    list_of_result_dataframes = []
    sanity_check_alerts = []
    for counter, result_csv in enumerate(results_csvs):
        cur_df = pd.read_csv(result_csv)
        alert = sanity_check_results(cur_df, experimental_directories[counter])
        sanity_check_alerts.append(alert)
        list_of_result_dataframes.append( cur_df )

    # average the dataframes
    avg_results = pd.concat(list_of_result_dataframes).groupby(level=0).mean()

    # output the dataframes
    avg_results.to_csv('overall_results/' + avg_results_filename + '.csv')

    print "##########"
    for sanity_check_alert in sanity_check_alerts:
        if sanity_check_alert:
            print sanity_check_alert

def find_result_csvs(experimental_directory):
    if experimental_directories[-1] != '/':
        experimental_directory += '/'

    experimental_directory += 'multilooper_outs/'

    if os.path.isdir( experimental_directory + 'multilooper_outs/'):
        experimental_directory += 'multilooper_outs/'

    results_cvs_file =experimental_directory + 'model_comparison.csv'

    print "results_cvs_file", results_cvs_file

    if not os.path.isfile( results_cvs_file ):
        return None

    print "results_cvs_file", results_cvs_file
    return results_cvs_file

def sanity_check_results(results_df, exp_name):
    # TODO
    return None

if __name__=="__main__":
    print "RUNNING"
    experimental_directories = ['/Volumes/exM2/new_experimental_data/sockshop_scale_rep2',
                                '/Volumes/exM2/new_experimental_data/sockshop_scale_rep3',
                                '/Volumes/exM2/new_experimental_data/sockshop_scale_rep4']
    avg_results_filename = 'sockshop_scale'

    main(experimental_directories, avg_results_filename)