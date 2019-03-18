from matplotlib import pyplot as plt
from jinja2 import FileSystemLoader, Environment
import datetime
import pdfkit
import subprocess
import numpy as np
import pandas
import math

def generate_aggregate_report(rate_to_timegran_to_methods_to_attacks_found_dfs,
                              rate_to_timegran_list_to_methods_to_attacks_found_training_df,
                              base_output_name, rates_to_experiment_info):

    date = str(datetime.datetime.now())
    env = Environment(
        loader=FileSystemLoader(searchpath="src")
    )
    aggreg_res_section = env.get_template("aggreg_res_section.html")
    comp_res_section = env.get_template("comp_graph_section.html")
    sections = []

    ## STEP (1): composite bar graph
    ### so right now it is rate -> (time_gran  -> (method -> tp/fp/tn/fn on each attack))
    time_gran_to_attack_to_methods_to_f1s = {} # map to another dict, which maps attacks to another dict,
                                                # which maps attack to a list of f1s, one for each exfil rate @ a given time granularity
    time_gran_to_rate = {}
    time_gran_to_attack_to_methods_to_rates = {}

    # okay, first I need to calculate the dimensions of the grid of graphs..
    # well it's just the number of rates
    num_of_rates = len(rate_to_timegran_to_methods_to_attacks_found_dfs.keys())
    num_of_timegrans = len(rate_to_timegran_to_methods_to_attacks_found_dfs[rate_to_timegran_to_methods_to_attacks_found_dfs.keys()[0]].keys())
    num_of_attacks = None
    bargraph_locs = []

    # each plot corresponds to a certain rate... let's fix the columns at 3 and then do the necessary corresponding amt of rows.
    time_gran_to_comp_bargraph_info = {}
    for time_gran in rate_to_timegran_to_methods_to_attacks_found_dfs[rate_to_timegran_to_methods_to_attacks_found_dfs.keys()[0]].keys():
        filename = "subplots_comp-bargraph" + str(time_gran) + ".png"
        cur_lineGraph_loc = "./temp_outputs/" + filename

        nrows = int(math.ceil(num_of_rates / 3.0))
        ncolumns = 3
        print "nrows",nrows, "ncolumns",ncolumns, "num_of_rates",num_of_rates, math.ceil(num_of_rates / 3.0)
        bar_fig, bar_axes = plt.subplots(nrows=nrows, ncols=ncolumns, figsize=(26, 13), subplot_kw={ 'adjustable' : 'box'})
        time_gran_to_comp_bargraph_info[time_gran] = (bar_fig, bar_axes, cur_lineGraph_loc)

    df_attack_identites = None
    rate_counter = 0
    for rate, timegran_to_methods_to_attacks_found_dfs in rate_to_timegran_to_methods_to_attacks_found_dfs.iteritems():
        plt.clf()
        for timegran, methods_to_attacks_found_dfs in timegran_to_methods_to_attacks_found_dfs.iteritems():
            if timegran not in time_gran_to_attack_to_methods_to_f1s:
                time_gran_to_attack_to_methods_to_f1s[timegran] = {}
                time_gran_to_rate[timegran] = []
                time_gran_to_attack_to_methods_to_rates[timegran] = {}
            maybe_attack_found_df = methods_to_attacks_found_dfs[methods_to_attacks_found_dfs.keys()[0]]
            num_of_attacks = len(maybe_attack_found_df.index.values)



            ## okay, one graph for each of these set of params should be made
            filename = "comp_bargraph_" + str(rate) + "_" + str(timegran) + ".png"
            temp_graph_loc = "./temp_outputs/" + filename
            bargraph_locs.append(temp_graph_loc)
            graph_loc = base_output_name + filename

            # TODO: okay, so would want to define a figure here + probably pass it to per_attack_bar_graphs +
            # some indicator for the positionthat is should be located (so, would need to determine
            # all the sizing information before this loop starts and then just execute it in the inner part).
            # this'll also require modify rendering portion b/c we only want a single graph + better titles
            # obviously... ACTUALLY want one graph per time granularity, with one subfigure per exfiltration rates
            bar_axes = time_gran_to_comp_bargraph_info[timegran][1]
            cur_bar_axes = bar_axes[int(rate_counter / 3)][(rate_counter % 3)]

            df_attack_identites = per_attack_bar_graphs(methods_to_attacks_found_dfs, temp_graph_loc, graph_loc,
                                                        cur_bar_axes)

            # TODO: finish the current bar subgraphs
            ### TODO: STILL THESE ::://// This is at least like 3-4 hours of work... :::///
            ### TODO: PROBLEM THE ATTACKS MIGHT NOT BE IN THE SAME ORDER <--- top priority.
            #### ^^^ do this, and then can start on all the tasks below...
            ### TODO: (a) DEBUG THE GAPHS, (b) DEBUG IDE results, (c) make sure I get some (at least semi-) decent autoscaling results
            ### plus autoscaling graphs plz. (d) stick the new and improved graphs into a (very simple) aggregate report.
            ### TODO: NEED TO PLAN HOW MULTI-RATE IS GOING TO WORK!!!!
            ### AND GET SOCKSHOP WORKING!!! THAT's REALLY IMPORTANT!!!
            ##### lol... well, I kinda got (c).... okay, so I actually do need to get this done today, plus get autoscaling
            ##### sockshop to work too (since I'll need to re-do the exfil path element + modify viz's before the end
            ##### I am ready to report results)
            ## todo: (1) ide. is it fine? I understand it was probably working fine before b/c it was clearing edges, but you'd expect better
            ## todo: (2) wordpress autoscaling (analysis)
            ## todo: (3) sockshop autoscaling (analysis)
            ## todo: (4) overhaul the aggregate report w/ the new graph grids + change single to have a new page, using
            # that trick I learned...
            BytesPerMegabyte = 1000000.0
            cur_bar_axes.set_title(str(rate / BytesPerMegabyte ) + ' MB Per Minute')
            cur_bar_axes.set_ylabel('f1 scores')
            cur_bar_axes.set_xlabel('attack')
            #cur_bar_axes.legend()
            cur_bar_axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)


            #cur_bar_axes.set_xtick

            with pandas.option_context('display.max_colwidth', -1):
                df_attack_identites_html = df_attack_identites.to_html()

            ## note: I have some decent graphs, but I need to debug them still because some of the behavior seems off
            ## okay, this is where I'd want to create the other graph???? so what is the other graph again??
            ## well, it is f1 score vs exfiltration_rate... w/ one subfigure for each attack + one figure for
            ## each time gran (+ an aggregate graph of all the attacks just because that makes my life easier imho)
            ## steps: (1a): Need to get f1_vs_rate per attack
                        ## so, a dict mapping [time_gran] -> ([list_of_f1s], [list_of_rates_per_attack])
                        ## okay, so the order of these loops is suboptimal but that's okay. i'll just need to
                        ## store everything in a dict and then use it later
            update_attack_rate_linegraph_dicts(time_gran_to_attack_to_methods_to_f1s, timegran,
                                               methods_to_attacks_found_dfs, time_gran_to_attack_to_methods_to_rates, rate)
            time_gran_to_rate[timegran].append(rate)
            ##        (1b): actually make it into some graphs
            ##        (1c) adjust the rendering appropriately



            sections.append(aggreg_res_section.render(
                time_gran = timegran,
                test_or_train = "test",
                avg_exfil_per_min = rates_to_experiment_info[rate]['avg_exfil_per_min'],
                exfil_per_min_variance = rates_to_experiment_info[rate]['exfil_per_min_variance'],
                avg_pkt_size = rates_to_experiment_info[rate]['avg_pkt_size'],
                pkt_size_variance = rates_to_experiment_info[rate]['pkt_size_variance'],
                composite_results_bargraph = '.' + temp_graph_loc,
                df_attack_identites = df_attack_identites_html
            ))

        rate_counter += 1


    #  cur_bar_axes.set(adjustable='box', aspect='equal')
    for timegran, comp_bargraph_info in time_gran_to_comp_bargraph_info.iteritems():
        cur_axis_set = comp_bargraph_info[1]
        for cur_axis_row in cur_axis_set:
            for cur_axis in cur_axis_row:
                cur_axis.set(adjustable='box', aspect='auto')

    for time_gran, comp_graph_info in time_gran_to_comp_bargraph_info.iteritems():
        #comp_graph_info[0].tight_layout()
        comp_graph_info[0].savefig(comp_graph_info[2])
    #time_gran_to_comp_bargraph_info[timegran][0]
    #time_gran_to_comp_bargraph_info[time_gran].savefig
    #bar_fig.savefig(cur_lineGraph_loc)


    # STEP (2): that other graph that I wanted [[TODO TODO TODO]]
    # using: time_gran_to_attack_to_methods_to_f1s
    # and using: time_gran_to_rate
    plt.rcParams.update({'font.size': 62})
    timegran_to_linecomp_loc = {}
    for time_gran, attack_to_methods_to_f1s in time_gran_to_attack_to_methods_to_f1s.iteritems():
        plt.clf()
        filename = "comp_linegraph_" + str(time_gran) + ".png"
        cur_lineGraph_loc = "./temp_outputs/" + filename
        # for the outer loop, want to make a new figure (i.e. the whole grid)
        # (note: I'd still want it be 2D even if the theree was only enough to fill a single row...)
        fig, axes = plt.subplots(nrows=int(math.ceil(num_of_attacks /3.0)), ncols=3, figsize=(30, 25))
        fig.suptitle(str(time_gran) + ' sec time gran')
        j = 0
        markers = ['o', 's', '*']
        for attack, methods_to_f1s in attack_to_methods_to_f1s.iteritems():
            # okay, for this loop, want to make a new figure inside the grid
            # the figures are already created. so I just need to make the variables to index into axes
            m = 0
            for method,f1s in methods_to_f1s.iteritems():
                y = f1s # f1s go here
                #x = time_gran_to_rate[time_gran] # rates go here
                x = time_gran_to_attack_to_methods_to_rates[time_gran][attack][method]
                x = [math.log10(float(i)) for i in x]
                x, y = zip(*sorted(zip(x, y)))
                print "exfil rates", x

                axes[int(j / 3)][(j % 3)].plot(x,y, label=str(method), marker=markers[m], alpha=0.5, linewidth=3.0)
                axes[int(j / 3)][(j % 3)].set_title(str(attack), fontsize=15) # TODO: either want to modify names or use wrap-around
                axes[int(j / 3)][(j % 3)].set_ylabel('f1 scores', fontsize=25)
                axes[int(j / 3)][(j % 3)].set_xlabel('log rates', fontsize=25)
                axes[int(j / 3)][(j % 3)].legend()
                #axes[int(j / 3)][(j % 3)].set_xscale(value="log")

                axes[int(j / 3)][(j % 3)].set_ylim(top=1.1,bottom=-0.1)

                m+=1
            j += 1
        ## okay, well now I would probably want to store it somewhere...
        fig.savefig(cur_lineGraph_loc)
        timegran_to_linecomp_loc[time_gran] = cur_lineGraph_loc

        #'''
        sections.append(comp_res_section.render(
            time_gran=time_gran,
            comp_bargraph='.' + time_gran_to_comp_bargraph_info[time_gran][2],
            comp_linegraph='.' + cur_lineGraph_loc,
            df_attack_identites=df_attack_identites.to_html()
        ))
        #'''

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
    options={"print-media-type": None}
    pdfkit.from_file("mulval_inouts/aggregate_report.html", aggregate_report_location, configuration=config, options=options)
    out = subprocess.check_output(['open', aggregate_report_location])

def update_attack_rate_linegraph_dicts(time_gran_to_attack_to_methods_to_f1s, timegran,
                                       methods_to_attacks_found_dfs, time_gran_to_attack_to_methods_to_rates, rate):
    ## TODO: this function is probably the meat of adding the new kind of graphs... the actual function will
    ## be relatively straight-forward indexing (plus somethign complicated w/ figs/subfigs but that isn't super
    ## important IMHO)
    for method, attacks_found in methods_to_attacks_found_dfs.iteritems():
        attacks_to_fones = results_df_to_attack_fones(attacks_found)
        for attack,fones in attacks_to_fones.iteritems():
            #if attack not in time_gran_to_attack_to_methods_to_f1s[timegran]:
            #    time_gran_to_attack_to_methods_to_f1s[timegran][attack] = {}
            # print "perf", perf, "==endperf"
            if attack not in time_gran_to_attack_to_methods_to_f1s[timegran]:
                time_gran_to_attack_to_methods_to_f1s[timegran][attack] = {}
                time_gran_to_attack_to_methods_to_rates[timegran][attack] = {}
            if method not in time_gran_to_attack_to_methods_to_f1s[timegran][attack]:
                time_gran_to_attack_to_methods_to_f1s[timegran][attack][method] = []
                time_gran_to_attack_to_methods_to_rates[timegran][attack][method] = []
            time_gran_to_attack_to_methods_to_f1s[timegran][attack][method].append(fones)
            time_gran_to_attack_to_methods_to_rates[timegran][attack][method].append(rate)

def results_df_to_attack_fones(results_df):
    attacks = results_df.index
    attack_to_fone = {}

    #print "results_df",results_df
    for attack in attacks:
        tp = float(results_df['tp'][attack])
        fp = float(results_df['fp'][attack])
        if tp + fp == 0:
            precision = 1.0 # found everything perfectly, even though there was nothing
        else:
            precision = tp / (tp + fp)    # TP / (TP + FP)
        fn = float(results_df['fn'][attack])
        print "tp",tp,"fn",fn,"attack",attack

        # NOTE: AM i sure that this is right??
        if tp + fn == 0:
            recall = 1.0 # found everything, even tho there was noting
        else:
            recall = tp / (tp + fn)    # TP / (TP + FN)
        cur_fOne = (2 * precision * recall) / (precision + recall)
        attack_to_fone[attack] = cur_fOne

    return attack_to_fone

def per_attack_bar_graphs(method_to_results_df, temp_location, file_storage_location, relevant_subplots_axis):
    '''Taken more-or-less wholesale from https://matplotlib.org/gallery/statistics/barchart_demo.html'''
    fig, ax = plt.subplots()
    attacks = method_to_results_df[method_to_results_df.keys()[0]].index
    n_groups = len(attacks)
    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    colors_to_use = ['b', 'r']
    x_tick_labels = ('A', 'B', 'C', 'D', 'E', 'F', 'G',  'H', 'I')#list(attacks)
    df_attack_identites = {}
    print "attacks", attacks
    for counter,tick_val in enumerate(x_tick_labels):
        if counter < len(attacks):
            df_attack_identites[tick_val] = (attacks[counter],)

    i = 0
    for cur_method, cur_results in method_to_results_df.iteritems():
        attack_to_fone = results_df_to_attack_fones(cur_results)
        current_bar_locations = index + bar_width * i

        attack_fones = [attack_to_fone[attack] for attack in attacks]

        rects1 = ax.bar(current_bar_locations, attack_fones, bar_width,
                        alpha=opacity, color=colors_to_use[i],
                        error_kw=error_config,
                        label=cur_method)

        relevant_subplots_axis.bar(current_bar_locations, attack_fones, bar_width,
                        alpha=opacity, color=colors_to_use[i],
                        error_kw=error_config,
                        label=cur_method)
        i += 1


    ax.set_xlabel('Attacks')
    ax.set_ylabel('Optimal F1')
    ax.set_title('Optimal F1 per Attacks')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels( tuple(x_tick_labels) )
    #ax.legend()
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    fig.tight_layout()
    #plt.show()
    plt.savefig( temp_location )
    plt.savefig(file_storage_location)

    return pandas.DataFrame().from_dict(df_attack_identites).transpose() #, index=x_tick_labels, columns=['label', 'path'])

