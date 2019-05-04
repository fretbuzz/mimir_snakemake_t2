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
                              base_output_name, rates_to_experiment_info, rate_to_time_gran_to_outtraffic,
                              auto_open_pdfs_p):

    date = str(datetime.datetime.now())
    env = Environment(
        loader=FileSystemLoader(searchpath="./report_templates")
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
        bar_fig, bar_axes = plt.subplots(nrows=nrows, ncols=ncolumns, figsize=(26, 36), subplot_kw={ 'adjustable' : 'box'})
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
            try:
                cur_bar_axes = bar_axes[int(rate_counter / 3)][(rate_counter % 3)]
            except:
                cur_bar_axes = bar_axes[(rate_counter % 3)]

            attacks = methods_to_attacks_found_dfs[methods_to_attacks_found_dfs.keys()[0]].index
            n_groups = len(attacks)
            index = np.arange(n_groups)
            bar_width = 0.35

            df_attack_identites = per_attack_bar_graphs(methods_to_attacks_found_dfs, temp_graph_loc, graph_loc,
                                                        cur_bar_axes)

            BytesPerMegabyte = 1000000.0
            cur_bar_axes.set_title(str(rate / BytesPerMegabyte ) + ' MB Per Minute')
            cur_bar_axes.set_ylabel('accuracy')
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
    ## TODO: NOTE: These labels apply only for the wordpress experiments...
    #x_tick_labels = ('wp_VIP', 'no_VIP_DNS', 'just_DNS', 'Norm_Path', 'no_VIP_norm', 'No_Attack', 'Norm_DNS',  'Out', 'wp_VIP_DNS')#list(attacks)
    #x_tick_labels = ('A', 'B', 'C', 'D', 'E', 'No_Attack', 'G',  'H', 'I')#list(attacks)
    x_tick_labels = ('A', 'B', 'C', 'D', 'E', 'G',  'H', 'I')#list(attacks)

    for timegran, comp_bargraph_info in time_gran_to_comp_bargraph_info.iteritems():
        cur_axis_set = comp_bargraph_info[1]
        for cur_axis_row in cur_axis_set:
            try:
                for cur_axis in cur_axis_row:
                    cur_axis.set(adjustable='box', aspect='auto')
                    cur_axis.set_xticks(index + bar_width / 2)
                    cur_axis.set_xticklabels(x_tick_labels)
            except:
                cur_axis_row.set(adjustable='box', aspect='auto')

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


    # step (??): okay, I want another graph now... I want to show # of FPs at each exfil rate. (per method,
    # obviously)
    timegran_to_method_to_rates_to_fps = {}
    rate_order = []
    for rate, timegran_to_methods_to_attacks_found_dfs in rate_to_timegran_to_methods_to_attacks_found_dfs.iteritems():
        rate_order.append(rate)
        plt.clf()
        for timegran, methods_to_attacks_found_dfs in timegran_to_methods_to_attacks_found_dfs.iteritems():
            if timegran not in timegran_to_method_to_rates_to_fps:
                timegran_to_method_to_rates_to_fps[timegran] = {}
            for method, attacks_found_df in methods_to_attacks_found_dfs.iteritems():
                number_of_fps = find_num_fps(attacks_found_df)
                if method not in timegran_to_method_to_rates_to_fps[timegran]:
                    timegran_to_method_to_rates_to_fps[timegran][method] = []
                timegran_to_method_to_rates_to_fps[timegran][method].append(number_of_fps)

    #colors_to_use = ['b', 'r', 'g', 'y']
    colors_to_use = ['b', 'g', 'r', 'y']
    fp_comp_locs = {}
    bar_width = 0.2
    for timegran, method_to_rates_to_fps in timegran_to_method_to_rates_to_fps.iteritems():
        plt.clf()
        plt.figure(figsize=(15,15))
        fp_com_filename = "comp_fps_bar_" + str(timegran) + ".png"
        fp_com_loc = "./temp_outputs/" + fp_com_filename
        k = 0
        axe = plt.subplot()
        index = range(0, len(rate_order))
        for method in reversed(sorted(method_to_rates_to_fps.keys())):
            #for method, rates_to_fps in method_to_rates_to_fps.iteritems():
            rates_to_fps = method_to_rates_to_fps[method]
            current_bar_locations = [item + bar_width * k for item in index]
            print "method_ex", method, "current_bar_locations", current_bar_locations
            axe.bar(current_bar_locations, rates_to_fps, bar_width,
                    alpha=0.4, color=colors_to_use[k],
                    label=method)
            k += 1
        #axe.savefig(fp_com_loc)

        axe.set_xlabel('Exfil Rate (MB/s)')
        axe.set_ylabel('log FPs')
        axe.set_title('FPs per Method per Rate')
        #axe.set_xticks(len(rate_to_method_to_fps.keys()) + bar_width / 2)
        #axe.set_xticklabels(tuple(x_tick_labels))
        BytesPerMegabyte = 1000000.0
        print "rate_order", rate_order
        r_order = [r/BytesPerMegabyte for r in rate_order]
        axe.set_xticks(index)
        axe.set_xticklabels(tuple(r_order))
        # ax.legend()
        axe.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(fp_com_loc)
        fp_comp_locs[timegran] = fp_com_loc

        #'''
        sections.append(comp_res_section.render(
            time_gran=timegran,
            comp_bargraph='.' + time_gran_to_comp_bargraph_info[timegran][2],
            comp_linegraph='.' + timegran_to_linecomp_loc[timegran],
            df_attack_identites= df_attack_identites.to_html(),
            comp_fps_bar = '.' +  fp_comp_locs[timegran]
        ))
        #'''

    '''
    timegran_to_fp_loc = {}
    for time_gran, attack_to_methods_to_f1s in time_gran_to_attack_to_methods_to_f1s.iteritems():
        fig.savefig(cur_g_loc)
        timegran_to_fp_loc[time_gran] = cur_g_loc
    '''

    # Step (3) put it all into a handy-dandy report
    base_template = env.get_template("report_template.html")
    with open("report_templates/aggregate_report.html", "w") as f:
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
    try:
        options={"print-media-type": None}
        pdfkit.from_file("report_templates/aggregate_report.html", aggregate_report_location, configuration=config, options=options)
        if auto_open_pdfs_p:
            out = subprocess.check_output(['open', aggregate_report_location])
    except:
        pass

def update_attack_rate_linegraph_dicts(time_gran_to_attack_to_methods_to_f1s, timegran,
                                       methods_to_attacks_found_dfs, time_gran_to_attack_to_methods_to_rates, rate):
    ## this function is probably the meat of adding the new kind of graphs... the actual function will
    ## be relatively straight-forward indexing (plus somethign complicated w/ figs/subfigs but that isn't super
    ## important IMHO)
    for method, attacks_found in methods_to_attacks_found_dfs.iteritems():
        attacks_to_fones = results_df_to_attack_accuracy(attacks_found)
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

def find_num_fps(results_df):
    attacks = results_df.index
    attacks = [attack for attack in attacks]
    total_fps = 0

    #print "results_df",results_df
    for attack in attacks:
        fp = float(results_df['fp'][attack])
        total_fps += fp
    return total_fps

def results_df_to_attack_accuracy(results_df):
    attacks = results_df.index
    attacks = [attack for attack in attacks if attack != 'No Attack']
    attack_to_accuracy = {}

    #print "results_df",results_df
    for attack in attacks:
        tp = float(results_df['tp'][attack])
        fp = float(results_df['fp'][attack])
        #if tp + fp == 0:
        #    precision = 1.0 # found everything perfectly, even though there was nothing
        #else:
        #    precision = tp / (tp + fp)    # TP / (TP + FP)
        fn = float(results_df['fn'][attack])
        print "tp",tp,"fn",fn,"attack",attack

        ## NOTE: AM i sure that this is right??
        #if tp + fn == 0:
        #    recall = 1.0 # found everything, even tho there was noting
        #else:
        #    recall = tp / (tp + fn)    # TP / (TP + FN)
        #cur_fOne = (2 * precision * recall) / (precision + recall)
        curAccuracy = (tp) / (tp + fn)

        attack_to_accuracy[attack] = curAccuracy

    return attack_to_accuracy

def per_attack_bar_graphs(method_to_results_df, temp_location, file_storage_location, relevant_subplots_axis):
    '''Taken more-or-less wholesale from https://matplotlib.org/gallery/statistics/barchart_demo.html'''
    fig, ax = plt.subplots()
    attacks = method_to_results_df[method_to_results_df.keys()[0]].index
    attacks = [attack for attack in attacks if attack != 'No Attack']
    n_groups = len(attacks)
    index = np.arange(n_groups)
    bar_width = 0.2

    opacity = 0.3
    error_config = {'ecolor': '0.3'}
    #colors_to_use = ['b', 'r', 'g', 'y']
    colors_to_use = ['b', 'g', 'r', 'y']
    x_tick_labels = ('A', 'B', 'C', 'D', 'E', 'F', 'G',  'H', 'I')#list(attacks)

    ## TODO: NOTE: THESE APPLY ONLY TO THE WORDPRESS EXPERIMENTS!!!
    #x_tick_labels = ('wp_VIP', 'no_VIP_DNS', 'just_DNS', 'Normal_Path', 'no_VIP_normal', 'No_Attack', 'Normal_DNS',  'Straight_Out', 'wp_VIP_DNS')#list(attacks)


    df_attack_identites = {}
    print "attacks", attacks
    for counter,tick_val in enumerate(x_tick_labels):
        if counter < len(attacks):
            df_attack_identites[tick_val] = (attacks[counter],)

    i = 0
    for cur_method in reversed(sorted(method_to_results_df.keys())):
    #for cur_method, cur_results in method_to_results_df.iteritems():

        cur_results = method_to_results_df[cur_method]
        attack_to_accuracy = results_df_to_attack_accuracy(cur_results)

        current_bar_locations = index + bar_width * i

        print "attack_to_accuracy",attack_to_accuracy
        print "---"

        attack_fones = [attack_to_accuracy[attack] for attack in attack_to_accuracy.keys()]

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
    ax.set_ylabel('Accuracy')
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

