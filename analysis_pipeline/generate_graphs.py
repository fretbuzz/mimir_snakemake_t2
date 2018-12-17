import math

import numpy as np
from matplotlib import pyplot as plt

def generate_feature_multitime_boxplots(total_calculated_vals, basegraph_name, window_size, colors, time_interval_lengths, exfil_start, exfil_end, wiggle_room):
    time_grans = []
    node_grans = []
    metrics = []

    for label, calculated_values in total_calculated_vals.iteritems():
        ## gotta remove those dicts from calculated_values b/c it'll break our next function
        try:
            del calculated_values['non_reciprocated_in_weight']
        except:
            pass # I guess it wasn't there...

        try:
            del calculated_values['non_reciprocated_out_weight']
        except:
            pass # I guess it wasn't there...

        #####
        container_or_class = label[1]
        time_interval = label[0]
        if time_interval not in time_interval_lengths:
            continue
        time_grans.append(time_interval)
        node_grans.append(container_or_class)
        metrics.extend(calculated_values.keys())
    metrics = list(set(metrics))
    print "metrics", metrics

    # okay, so later on I am going to want to group by class/node granularity via color
    # and by time granularity via spacing... so each time granularity should be a seperatae
    # list and each of the class/node granularites should be a nested list (inside the corresponding list)
    # right now: (time gran, node gran) -> metrics -> vals

    node_grans = list(set(node_grans))
    print "node_grans", node_grans
    time_grans = list(set(time_grans))
    time_grans.sort()
    # okay, so what I want to do here is (time gran, node gran, metric) -> vals
    # or do I want to do (metric) -> (nested lists in order of the things above?)
    # well to do the covrariance matrix I am going to need (1) but in order to ddo the boxplots
    # I am going to need to do (2)
    # b/c then I can easily index in later

    # okay, so later on I am going to want to group by class/node granularity via color
    # and by time granularity via spacing... so each time granularity should be a seperatae
    # list and each of the class/node granularites should be a nested list (inside the corresponding list)
    # so below: (metric) -> (time gran) -> (nested list of node grans)
    #'''
    metrics_to_time_to_granularity_lists = {}
    fully_indexed_nans = {}
    fully_indexed_metrics = {}
    metrics_to_time_to_granularity_nans = {}
    for metric in metrics:
        metrics_to_time_to_granularity_lists[metric] = {}
        metrics_to_time_to_granularity_nans[metric] = {}
        for time_gran in time_grans:
            metrics_to_time_to_granularity_lists[metric][time_gran] = []
            metrics_to_time_to_granularity_nans[metric][time_gran] = []
            for node_gran in node_grans:
                try:
                    current_metric = total_calculated_vals[(time_gran, node_gran )][metric]
                    print "current_metric", current_metric
                except:
                    current_metric = []
                metrics_to_time_to_granularity_lists[metric][time_gran].append( current_metric )
                fully_indexed_metrics[(time_gran, node_gran, metric)] = current_metric

                nan_count = 0
                for val in current_metric:
                    #print val, math.isnan(val)
                    if math.isnan(val):
                        nan_count += 1
                metrics_to_time_to_granularity_nans[metric][time_gran].append(nan_count)

    for label, time_gran_to_val in metrics_to_time_to_granularity_nans.iteritems():
        for label_two, val in time_gran_to_val.iteritems():
            print 'nan', label, label_two, val
    #print metrics_to_time_to_granularity_nans

    # okay, so now I actually need to handle make those multi-dimensional boxplots
    for metric in metrics:
        if type(metric) == tuple:
            metric = ' '.join(metric)
        make_multi_time_boxplots(metrics_to_time_to_granularity_lists, time_grans, metric, colors,
                                 basegraph_name + metric + '_multitime_boxplot', node_grans, exfil_start, exfil_end,
                                 wiggle_room)

        make_multi_time_nan_bars(metrics_to_time_to_granularity_nans, time_grans, node_grans, metric,
                                 basegraph_name + metric + 'multi_nans')

    '''
    # If I decide to proceed with the covariance matrix idea, this code will come in handy
    print "about to make covariance matrix!"
    # delta_covariance_dataframe, abs_covariance_dataframe
    delta_covariance_dataframe, abs_covariance_dataframe = calc_covaraiance_matrix(fully_indexed_metrics)
    print "made covariance matrix! Now time to plot it!"
    print "delta correlation dataframe"
    print delta_covariance_dataframe
    print "abs correlation dataframe"
    print abs_covariance_dataframe
    plot_correlogram(delta_covariance_dataframe, abs_covariance_dataframe, basegraph_name)
    #'''


def make_multi_time_boxplots(metrics_to_time_to_granularity_lists, time_grans, metric, colors,
                             graph_name, node_grans, exfil_start, exfil_end, wiggle_room):
    fig = plt.figure()
    fig.clf()
    ax = plt.axis()
    #print "\n", metric
    #print "metrics_to_time_to_granularity_lists", metrics_to_time_to_granularity_lists
    #print "time_grans", time_grans

    max_yaxis = -1000000 # essentially -infinity
    min_yaxis = 1000000 # essentially infinity
    cur_pos = 1
    tick_position_list = []
    for time_gran in time_grans:
        current_vals = metrics_to_time_to_granularity_lists[metric][time_gran]
        current_vals = [[x for x in i if x is not None and not math.isnan(x)] for i in current_vals]
        number_nested_lists = len(current_vals)
        print "current_vals", current_vals
        print "number_nested_lists", number_nested_lists
        print "cur_pos", cur_pos
        number_positions_on_graph = range(cur_pos, cur_pos+number_nested_lists)
        tick_position = (float(number_positions_on_graph[0]) + float(number_positions_on_graph[-1])) / 2
        tick_position_list.append( tick_position )
        bp = plt.boxplot(current_vals, positions = number_positions_on_graph, widths = 0.6, sym='k.', showfliers=False)
        #print "number of boxplots just added:", len(metrics_to_time_to_granularity_lists[metric][time_gran]), " , ", number_nested_lists
        plt.title(metric)
        plt.xlabel('time granularity (seconds)')
        if 'Change-Point' in metric or 'Angle' in metric:
            plt.ylabel('Angle')
        else:
            plt.ylabel(metric)
        cur_pos += number_nested_lists + 1 # the +1 is so that there is extra space between the groups
        #set_boxplot_colors(bp, colors) # TODO: maybe wanna add back in at some point
        #print metric
        try:
            max_yaxis = max( max_yaxis, max([max(x) for x in current_vals]))
        except:
            pass
        try:
            min_yaxis = min( min_yaxis, min([min(x) for x in current_vals]))
        except:
            pass

        # I can iterate through the nested loops (in order to keep track of the x-axis values)
        # then I'll need a function that takes the 'deepest' list + granularity and exfil times
        # and then returns the specific points to plot
        i = 0
        for current_vals_at_certain_node_gran in metrics_to_time_to_granularity_lists[metric][time_gran]:
            # note that there is also an implicit time granularity from the for loop way up above
            # okay, so this is where it I'd call the function that looks for other stuff
            points_to_plot, start_index_ptp, end_index_ptp = get_points_to_plot(time_gran, current_vals_at_certain_node_gran, exfil_start, exfil_end, wiggle_room)
            non_exfil_points_to_plot = get_non_exfil_points_to_plot(current_vals_at_certain_node_gran, start_index_ptp, end_index_ptp)
            print "points_to_plot", metric, time_gran, points_to_plot, number_positions_on_graph[i]
            # plt.plot([1,1,1], [6,7,8], marker='o', markersize=3, color="red")
            for point in [x for x in non_exfil_points_to_plot if x is not None and not math.isnan(x)]:
                x_point = np.random.normal(loc=number_positions_on_graph[i], scale= 0.1, size=None)
                color_vector = [0.0, 0.0, 0.0, 0.7]
                plt.plot([x_point], [point], marker='o', markersize=4, color=color_vector)  # y=[point], style='g-', label='point')

            j = 0
            # plotting in reverse so that the newer ones are on top (and easier to see)
            for point in list(reversed([x for x in points_to_plot if x is not None and not math.isnan(x)])):
                # number_positions_on_graph[i]
                x_point = np.random.normal(loc=number_positions_on_graph[i], scale= 0.1,size=None)
                # let's just make the first 10% of exfil points one color and the next 90% another
                # wanna start with rgb(0,128,0) [green]
                # and end with rgb(124,252,0) [lawngreen]
                # in effect (including the reverse), early vals -> brighter, late vals -> darker
                if float(j)/len(points_to_plot) >= 0.9:
                    color_vector = [0.486, 0.988, 0.0, 0.8]
                else:
                    color_vector = [0.0, 0.5, 0.0, 0.8]
                #color_vector = [0.486 * (float(j)/len(points_to_plot)), 0.5 + 0.48  * (float(j)/len(points_to_plot)),
                #                0.0, 0.7]
                plt.plot([x_point], [point], marker='o', markersize=4, color=color_vector) #y=[point], style='g-', label='point')
                j +=1

            i += 1

    print "tick_position_list", tick_position_list
    yaxis_range = max_yaxis - min_yaxis
    #print "old min yaxis", min_yaxis
    #print "old max yaxis", max_yaxis
    #print "yaxis_range", yaxis_range
    min_yaxis = min_yaxis - (yaxis_range * 0.05)
    max_yaxis = max_yaxis + (yaxis_range * 0.25)
    #print "min_yaxis", min_yaxis
    #print "max_yaxis", max_yaxis
    plt.xlim(0, cur_pos)
    plt.ylim(min_yaxis, max_yaxis)
    plt.xticks(tick_position_list, [str(i) for i in time_grans])

    invisible_lines = []
    z = 0
    for color in colors:
        cur_line, = plt.plot([1,1], color, label=time_grans[z])
        invisible_lines.append(cur_line)
        z += 1
    plt.legend(invisible_lines, node_grans)
    for line in invisible_lines:
        line.set_visible(False)

    plt.savefig(graph_name + '.png', format='png')

# make_multi_time_nan_bars : ??
# this function finds the number of nan's present at each (time_gran, node_gran) and makes graphs that resemble
# the ones make in the multi_time_boxplots function
def make_multi_time_nan_bars(metrics_to_time_to_granularity_nans, time_grans, node_grans, metric, graph_name):
    fig = plt.figure(figsize=(20,8))
    fig.clf()
    ax = plt.axis()

    max_yaxis = -1000000  # essentially -infinity
    min_yaxis = 1000000  # essentially infinity
    cur_pos = 1
    tick_position_list = []
    for time_gran in time_grans:
        current_vals = metrics_to_time_to_granularity_nans[metric][time_gran]
        # yes, we'll display them in a bar graph.
        number_nested_lists = len(current_vals)
        number_positions_on_graph = range(cur_pos, cur_pos + number_nested_lists)
        tick_position = (float(number_positions_on_graph[0]) + float(number_positions_on_graph[-1])) / number_nested_lists
        tick_position_list.append(tick_position)
        #print current_vals, number_positions_on_graph

        print number_positions_on_graph, current_vals
        bp = plt.bar(number_positions_on_graph, current_vals)
        # print "number of boxplots just added:", len(metrics_to_time_to_granularity_lists[metric][time_gran]), " , ", number_nested_lists
        plt.title(metric)
        plt.xlabel('time granularity (seconds)')
        plt.ylabel('Number of nans')
        cur_pos += number_nested_lists + 1  # the +1 is so that there is extra space between the groups
        # print metric
        try:
            max_yaxis = max(max_yaxis, max(current_vals))
        except:
            print "problem with max_yaxis"
            pass
        try:
            min_yaxis = min(min_yaxis, min(current_vals))
        except:
            pass
    yaxis_range = max_yaxis - min_yaxis
    # print "old min yaxis", min_yaxis
    # print "old max yaxis", max_yaxis
    # print "yaxis_range", yaxis_range
    min_yaxis = min_yaxis - (yaxis_range * 0.05)
    max_yaxis = max_yaxis + (yaxis_range * 0.25)
    print "min_yaxis", min_yaxis, "max_yaxis", max_yaxis
    plt.xlim(0, cur_pos)
    plt.ylim(min_yaxis, max_yaxis)
    plt.xticks(tick_position_list, [str(i) for i in time_grans])

    plt.savefig(graph_name + '.png', format='png')

# get_points_to_plot : time_granularity values exfil_start_time exfil_end_time wiggle_room
#                   -> list_of_values_during_exfiltration_period index_at_start_of_exfil index_at_end_of_exfil
# this function calculates the values during exfiltration along with the start/end indexes for these values
# Ex: vals = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]; exfil_start = 30, exfil_end = 50, time_gran = 10, wiggle_room = 2
# floor((30-2)/10)=2, floor((50+2)/10) = 6; returns [20, 30, 40, 50, 60] with indexes 2,6
def get_points_to_plot(time_grand, vals, exfil_start, exfil_end, wiggle_room):
    print "exfil_start", exfil_start, "exfil_end", exfil_end, "time_grand", time_grand
    if exfil_start and exfil_end:
        start_index = int(float(exfil_start - wiggle_room) / time_grand)# * vals_per_time_interval
        end_index = int(float(exfil_end + wiggle_room) / time_grand) #* vals_per_time_interval
        return vals[start_index : end_index + 1], start_index, end_index
    else:
        return [], None, None

# get_non_exfil_points_to_plot : vals start_index end_index -> 'sub-list' of vals from start_index to end_index
# this function returns all the values that occured NOT during exfil
# note: this is a seperate function b/c I think that the logic might become more complicated
def get_non_exfil_points_to_plot(vals, start_index_ptp, end_index_ptp):
    if start_index_ptp == None or end_index_ptp == None:
        return vals
    return vals[:start_index_ptp] + vals[end_index_ptp + 1:]