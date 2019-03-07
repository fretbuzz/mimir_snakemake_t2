from matplotlib import pyplot as plt
from jinja2 import FileSystemLoader, Environment
import datetime
import pdfkit
import subprocess
import numpy as np

def generate_aggregate_report(rate_to_timegran_to_methods_to_attacks_found_dfs,
                              rate_to_timegran_list_to_methods_to_attacks_found_training_df,
                              base_output_name, rates_to_experiment_info):

    date = str(datetime.datetime.now())
    env = Environment(
        loader=FileSystemLoader(searchpath="src")
    )
    aggreg_res_section = env.get_template("aggreg_res_section.html")
    sections = []

    ## STEP (1): composite bar graph
    ### so right now it is rate -> (time_gran  -> (method -> tp/fp/tn/fn on each attack))
    for rate, timegran_to_methods_to_attacks_found_dfs in rate_to_timegran_to_methods_to_attacks_found_dfs.iteritems():
        for timegran, methods_to_attacks_found_dfs in timegran_to_methods_to_attacks_found_dfs.iteritems():
            ## okay, one graph for each of these set of params should be made
            filename = "comp_bargraph_" + str(rate) + "_" + str(timegran) + ".png"
            temp_graph_loc = "./temp_outputs/" + filename
            graph_loc = base_output_name + filename
            per_attack_bar_graphs(methods_to_attacks_found_dfs, temp_graph_loc, graph_loc)

            sections.append(aggreg_res_section.render(
                time_gran = timegran,
                test_or_train = "test",
                avg_exfil_per_min = rates_to_experiment_info[rate]['avg_exfil_per_min'],
                exfil_per_min_variance = rates_to_experiment_info[rate]['exfil_per_min_variance'],
                avg_pkt_size = rates_to_experiment_info[rate]['avg_pkt_size'],
                pkt_size_variance = rates_to_experiment_info[rate]['pkt_size_variance'],
                composite_results_bargraph = '.' + temp_graph_loc
            ))

    # STEP (2): that other graph that I wanted [[TODO TODO TODO]]

    # Step (3) put it all into a handy-dandy report
    base_template = env.get_template("report_template.html")
    with open("mulval_inouts/aggregate_report.html", "w") as f:
        f.write(base_template.render(
            title = 'MIMIR AGGREGATE RESULTS',
            date = date,
            recipes_used = "see_below",
            sections=sections,
            avg_exfil_per_min = "see_below",
            avg_pkt_size ="see_below",
            exfil_per_min_variance = "see_below",
            pkt_size_variance="see_below"
        ))

    config = pdfkit.configuration(wkhtmltopdf="/usr/local/bin/wkhtmltopdf")

    aggregate_report_location = base_output_name + "_aggregate_report.pdf" # TODO: is this fine??
    pdfkit.from_file("mulval_inouts/aggregate_report.html", aggregate_report_location, configuration=config)
    out = subprocess.check_output(['open', aggregate_report_location])

def results_df_to_attack_fones(results_df):
    attacks = results_df.index
    attack_to_fone = {}

    for attack in attacks:
        tp = results_df['tp'][attack]
        fp = results_df['fp'][attack]
        precision = tp / (tp + fp)    # TP / (TP + FP)
        fn = results_df['fn'][attack]
        recall = tp / (tp + fn)    # TP / (TP + FN)
        cur_fOne = (2 * precision * recall) / (precision + recall)
        attack_to_fone[attack] = cur_fOne

    return attack_to_fone

def per_attack_bar_graphs(method_to_results_df, temp_location, file_storage_location):
    '''Taken more-or-less wholesale from https://matplotlib.org/gallery/statistics/barchart_demo.html'''
    fig, ax = plt.subplots()
    attacks = method_to_results_df[method_to_results_df.keys()[0]].index
    n_groups = len(attacks)
    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    colors_to_use = ['b', 'r']
    x_tick_labels = list(attacks)

    i = 0
    for cur_method, cur_results in method_to_results_df.iteritems():
        attack_to_fone = results_df_to_attack_fones(cur_results)
        rects1 = ax.bar(index + bar_width * i, attack_to_fone.values(), bar_width,
                        alpha=opacity, color=colors_to_use[i],
                        error_kw=error_config,
                        label='Men')
        i += 1


    ax.set_xlabel('Attacks')
    ax.set_ylabel('Optimal F1')
    ax.set_title('Optimal F1 per Attacks')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels( tuple(x_tick_labels) )
    ax.legend()

    fig.tight_layout()
    #plt.show()
    plt.savefig( temp_location )
    plt.savefig(file_storage_location)

