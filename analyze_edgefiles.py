import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import statsmodels.api as sm
import math
import scipy
import time
import scipy.stats
import scipy.sparse.linalg
import pandas
import csv
import ast
import itertools

sockshop_ms_s = ['carts-db','carts','catalogue-db','catalogue','front-end','orders-db','orders',
        'payment','queue-master','rabbitmq','session-db','shipping','user-sim', 'user-db','user','load-test']

def pipeline_analysis_step(filenames, ms_s, time_interval, basegraph_name, calc_vals_p, window_size):
    list_of_graphs = []
    list_of_aggregated_graphs = [] # all nodes of the same class aggregated into a single node
    list_of_aggregated_graphs_multi = [] # the above w/ multiple edges
    total_calculated_values = {}
    counter= 0 # let's not make more than 50 images of graphs

    if calc_vals_p:
        for file_path in filenames:
            G = nx.DiGraph()
            print "path to file is ", file_path
            nx.read_edgelist(file_path,
                            create_using=G, delimiter=',', data=(('weight', float),))
            pos = graphviz_layout(G)
            nx.draw_networkx(G, pos, with_labels=True, arrows=True)
            if counter < 50: # keep # of network graphs to a reasonable amount
                plt.savefig(file_path.replace('.txt', '') + '_network_graph_container.png', format='png')
            #plt.show()
            list_of_graphs.append(G)
            aggreg_multi_G, aggreg_simple_G = aggregate_graph(G, ms_s)
            list_of_aggregated_graphs.append( aggreg_simple_G )
            list_of_aggregated_graphs_multi.append( aggreg_multi_G )

            plt.clf()
            pos = graphviz_layout(aggreg_simple_G)
            nx.draw_networkx(aggreg_simple_G, pos, with_labels=True, arrows=True)
            if counter < 50:
                plt.savefig(file_path.replace('.txt', '') + '_network_graph_class.png', format='png')

            counter += 1
    total_calculated_values[(time_interval, 'container')] = calc_graph_metrics(list_of_graphs, ms_s, time_interval,
                                                                               basegraph_name + '_container_', 'container',
                                                                               calc_vals_p, window_size)
    total_calculated_values[(time_interval, 'class')] = calc_graph_metrics(list_of_aggregated_graphs, ms_s, time_interval,
                                                                           basegraph_name + '_class_', 'class', calc_vals_p,
                                                                           window_size)
    return total_calculated_values

def calc_graph_metrics(G_list, ms_s, time_interval, basegraph_name, container_or_class, calc_vals_p, window_size):

    if calc_vals_p:
        average_path_lengths = []
        densities = []
        degree_dicts = []
        weight_recips = []
        weighted_average_path_lengths = []
        unweighted_overall_reciprocities = [] # defined per networkx definition (see their docs)
        weighted_reciprocities = [] # defined per the nature paper (see comment @ function definition)
        average_clusterings = []
        outstrength_dicts = []
        instrength_dicts = []
        eigenvector_centrality_dicts = []
        clustering_dicts = []
        betweeness_centrality_dicts = []
        load_centrality_dicts = []
        non_reciprocated_out_weight_dicts = []
        non_reciprocated_in_weight_dicts = []

        total_node_list = []
        for cur_g in G_list:
            for node in cur_g.nodes():
                total_node_list.append(node)
        total_node_list = list(set(total_node_list))

        for cur_G in G_list:
            # okay, so this is where to calculate those metrics from the excel document

            # first, let's do the graph-wide metrics (b/c it is simple) (these are only single values)
            # todo: this causes problems when one doesn't commmuncaite (e.g. sockshop queue-master
            # sometimes doesn't do anything)
            try:
                avg_path_length = nx.average_shortest_path_length(cur_G) #
            except:
                avg_path_length = float('nan')
                #for sub_G in (cur_G.subgraph(c) for c in sorted(nx.strongly_connected_components(cur_G), key=len, reverse=True)):
                #    avg_path_length += nx.average_shortest_path_length(sub_G)
                # not fully connected, going to add the average (shortest) path length in each strongly connected component
                # todo: need to weight by the # of components
                # todo: is this what I want to do here??
            try:
                recip = nx.overall_reciprocity(cur_G)  # if it is not one, then I cna deal w/ looking at dictinoarty
            except :
                recip = float('nan') # overall reciprocity not defined for empty graphs
            unweighted_overall_reciprocities.append(recip)

            #average_clusterings.append(nx.average_clustering(cur_G))

            #print nx.all_pairs_dijkstra_path_length(cur_G)
            sum_of_all_distances = 0
            for thing in nx.all_pairs_dijkstra_path_length(cur_G, weight=graph_distance):
                #print thing
                for key,val in thing[1].iteritems():
                    sum_of_all_distances += val
            try:
                weighted_avg_path_length = float(sum_of_all_distances) / (cur_G.number_of_nodes() * (cur_G.number_of_nodes() - 1))
            except ZeroDivisionError:
                weighted_avg_path_length = 0 # b/c number_of_nodes = 0/1
            weighted_average_path_lengths.append(weighted_avg_path_length)
            #input("check it")

            density = nx.density(cur_G)

            # now let's do the nodal metrics (and then aggregate them to node-class metrics)
            degree_dict = {}
            degree_dict_iterator = cur_G.out_degree()  # node-instance granularity (might wanna aggregate it or something)
            for val in degree_dict_iterator:
                #print val, val[0], val[1]
                degree_dict[val[0]] = val[1]
            #print "degree dict", degree_dict

            outstrength_dict = {}
            instrength_dict = {}
            for (u, v, data) in cur_G.edges(data=True):
                if u in outstrength_dict:
                    outstrength_dict[u] += data['weight']
                else:
                    outstrength_dict[u] = data['weight']
                if v in instrength_dict:
                    instrength_dict[v] += data['weight']
                else:
                    instrength_dict[v] = data['weight']
            try:
                eigenvector_centrality_dict = nx.eigenvector_centrality(cur_G)
            except nx.NetworkXPointlessConcept:
                # if graph is Null, then this metric is meaningless
                eigenvector_centrality_dict = None
            except nx.PowerIterationFailedConvergence:
                # if failed to converge, then we effectively know nothing
                eigenvector_centrality_dict = None

                #eigenvector_centrality_dict = {}
                #for node in total_node_list:
                #    eigenvector_centrality_dict[node] = 0
                # doesn't converge -> we now
                # "computes the centrality of a node based on the centrality
                # of its neighbors"


            #clustering_dict = nx.clustering(cur_G)
            try:
                betweeness_centrality_dict = nx.betweenness_centrality(cur_G)
            except nx.NetworkXPointlessConcept:
                betweeness_centrality_dict = None
                # if graph is Null, then this metric is meaningless
                #betweeness_centrality_dict = {}
                #betweeness_centrality_dict = {}
                #for node in total_node_list:
                #    betweeness_centrality_dict[node] = 0

                # "the sum of the fraction of all-pairs shortest paths that
                # pass through that node"

            try:
                load_centrality_dict = nx.load_centrality(cur_G)
            except nx.NetworkXPointlessConcept:
                load_centrality_dict = None
                # if graph is Null, then this metric is meaningless
                #load_centrality_dict = {}
                #for node in total_node_list:
                #    load_centrality_dict[node] = 0
                # "the fraction of all shortest paths that pass through that node":

            # prob wanna do weighted reciprocity per ms class (I'm thinking scatter plot) (tho I need to figure out the other axis)
            # note: in the nature paper they actually just graph in-strength vs out-strength (might be the way to go)
            # could also try a repeat of the angle-vector-measurement-trick that i did for degrees
            # (they also do weighted reciprocity vs time later on in the paper)
            weighted_reciprocity, non_reciprocated_out_weight_dict, non_reciprocated_in_weight_dict = network_weidge_weighted_reciprocity(cur_G)
            outstrength_dicts.append( outstrength_dict )
            instrength_dicts.append( instrength_dict )
            eigenvector_centrality_dicts.append( eigenvector_centrality_dict )
            #clustering_dicts.append( clustering_dict )
            betweeness_centrality_dicts.append( betweeness_centrality_dict )
            load_centrality_dicts.append( load_centrality_dict )
            non_reciprocated_out_weight_dicts.append( non_reciprocated_out_weight_dict )
            non_reciprocated_in_weight_dicts.append( non_reciprocated_in_weight_dict )

            weighted_reciprocities.append(weighted_reciprocity)
            # let's store these values to use again later
            average_path_lengths.append(avg_path_length)
            densities.append(density)
            degree_dicts.append(degree_dict)

        print "degrees", degree_dicts
        print "weighted recips", weight_recips
        #raw_input("Press Enter2 to continue...")
        # now going to perform leman method
        print "DOING ANGLES"
        print "degree dicts", degree_dicts

        # out degrees analysis
        node_degrees = turn_into_list(degree_dicts, total_node_list)
        angles_degrees = find_angles(node_degrees, window_size) #change_point_detection(degree_dicts, window_size=window_size)  # setting window size arbitrarily for now...
        print "angles degrees", type(angles_degrees), angles_degrees
        print node_degrees
        angles_degrees_eigenvector = change_point_detection(degree_dicts, window_size, total_node_list)
        print "angles degrees eigenvector", angles_degrees_eigenvector

        #######

        # outstrength analysis
        node_outstrengths = turn_into_list(outstrength_dicts, total_node_list)
        print "node_outstrengths", node_outstrengths
        outstrength_degrees = find_angles(node_outstrengths, window_size)
        outstrength_degrees_eigenvector = change_point_detection(outstrength_dicts, window_size, total_node_list)

        # instrength analysis
        node_instrengths = turn_into_list(instrength_dicts, total_node_list)
        print "node_instrengths", node_instrengths
        instrengths_degrees = find_angles(node_instrengths, window_size)
        instrengths_degrees_eigenvector = change_point_detection(instrength_dicts, window_size, total_node_list)

        # eigenvector centrality analysis
        node_eigenvector_centrality = turn_into_list(eigenvector_centrality_dicts, total_node_list)
        eigenvector_centrality_degrees = find_angles(node_eigenvector_centrality, window_size)
        eigenvector_centrality_degrees_eigenvector = change_point_detection(eigenvector_centrality_dicts, window_size, total_node_list)

        # clustering analysis (not implemented for directed type)
        #node_clustering = turn_into_list(clustering_dicts, total_node_list)
        #clustering_degrees = find_angles(node_clustering, window_size)
        #clustering_degrees_eigenvector = change_point_detection(node_clustering, window_size, total_node_list)


        # betweeness centrality analysis
        node_betweeness_centrality = turn_into_list(betweeness_centrality_dicts, total_node_list)
        betweeness_centrality_degrees = find_angles(node_betweeness_centrality, window_size)
        betweeness_centrality_degrees_eigenvector = change_point_detection(betweeness_centrality_dicts, window_size, total_node_list)

        # load centrality analysis
        node_load_centrality = turn_into_list(load_centrality_dicts, total_node_list)
        load_centrality_degrees = find_angles(node_load_centrality, window_size)
        load_centrality_degrees_eigenvector = change_point_detection(load_centrality_dicts, window_size, total_node_list)

        # non_reciprocated_out_weight analysis
        node_non_reciprocated_out_weight = turn_into_list(non_reciprocated_out_weight_dicts, total_node_list)
        non_reciprocated_out_weight_degrees = find_angles(node_non_reciprocated_out_weight, window_size)
        non_reciprocated_out_weight_degrees_eigenvector = change_point_detection(non_reciprocated_out_weight_dicts, window_size, total_node_list)


        # non_reciprocated_in_weight analysis
        node_non_reciprocated_in_weight = turn_into_list(non_reciprocated_in_weight_dicts, total_node_list)
        non_reciprocated_in_weight_degrees = find_angles(node_non_reciprocated_in_weight, window_size)
        non_reciprocated_in_weight_degrees_eigenvector = change_point_detection(non_reciprocated_in_weight_dicts, window_size, total_node_list)

        #######


        average_path_lengths_no_nan = []
        for val in average_path_lengths:
            if not math.isnan(val):
                average_path_lengths_no_nan.append(val)
            else:
                average_path_lengths_no_nan.append(0)

        weighted_average_path_lengths_no_nan = []
        for val in weighted_average_path_lengths:
            if not math.isnan(val):
                weighted_average_path_lengths_no_nan.append(val)
            else:
                weighted_average_path_lengths_no_nan.append(0)

        unweighted_overall_reciprocities_no_nan = []
        for val in unweighted_overall_reciprocities:
            if not math.isnan(val):
                unweighted_overall_reciprocities_no_nan.append(val)
            else:
                weighted_average_path_lengths_no_nan.append(0)

        weighted_reciprocities_no_nan = []
        for val in weighted_reciprocities:
            if not math.isnan(val):
                weighted_reciprocities_no_nan.append(val)
            else:
                weighted_reciprocities_no_nan.append(0)

        angles_degrees_no_nan = []
        for val in angles_degrees:
            if not math.isnan(val):
                angles_degrees_no_nan.append(val)
            else:
                angles_degrees_no_nan.append(0)

        appserver_sum_degrees = []
        for degree_dict in degree_dicts:
            appserver_degrees = []
            for key,val in degree_dict.iteritems():
                if 'appserver' in key:
                    appserver_degrees.append(val)
            appserver_sum_degrees.append( np.mean(appserver_degrees))

        print "graph density", densities
        densities_no_nan = []
        for val in densities:
            if not math.isnan(val):
                densities_no_nan.append(val)
            else:
                densities_no_nan.append(0)
        print densities_no_nan

        calculated_values = {}
        # abs values
        calculated_values['Unweighted Average Path Length'] = average_path_lengths
        calculated_values['average_path_lengths_no_nan'] = average_path_lengths_no_nan
        calculated_values['Weighted Average Path Length'] = weighted_average_path_lengths
        calculated_values['weighted_average_path_lengths_no_nan'] = weighted_average_path_lengths_no_nan
        calculated_values['Unweighted Overall Reciprocity'] = unweighted_overall_reciprocities
        calculated_values['unweighted_overall_reciprocities_no_nan'] = unweighted_overall_reciprocities_no_nan
        calculated_values['Weighted Overall Reciprocity'] = weighted_reciprocities
        calculated_values['weighted_reciprocities_no_nan'] = weighted_reciprocities_no_nan
        calculated_values['Density'] = densities
        calculated_values['densities_no_nan'] = densities_no_nan
        calculated_values['Sum of Appserver Node Degrees'] = appserver_sum_degrees
        #calculated_values['average_clusterings'] = average_clusterings

        # delta values
        calculated_values['Simple Angle Between Node Degree Vectors'] = angles_degrees
        calculated_values['angles_degrees_no_nan'] = angles_degrees_no_nan
        calculated_values['Change-Point Detection Node Degree'] = angles_degrees_eigenvector
        calculated_values['Simple Angle Between Node Outstrength Vectors'] = outstrength_degrees
        calculated_values['Change-Point Detection Node Outstrength']= outstrength_degrees_eigenvector
        calculated_values['Simple Angle Between Node Instrength Vectors'] = instrengths_degrees
        calculated_values['Change-Point Detection Node Instrength'] = instrengths_degrees_eigenvector
        calculated_values['Simple Angle Between Node Eigenvector_Centrality Vectors'] = eigenvector_centrality_degrees
        calculated_values['Change-Point Detection Node Eigenvector_Centrality'] = eigenvector_centrality_degrees_eigenvector
        #calculated_values['clustering_degrees'] = clustering_degrees
        #calculated_values['clustering_degrees_eigenvector'] = clustering_degrees_eigenvector
        calculated_values['Simple Angle Between Node Betweeness Centrality Vectors'] = betweeness_centrality_degrees
        calculated_values['Change-Point Detection Node Betweeness Centrality'] = betweeness_centrality_degrees_eigenvector
        calculated_values['Simple Angle Between Node Load Centrality Vectors'] = load_centrality_degrees
        calculated_values['Change-Point Detection Node Load Centrality'] = load_centrality_degrees_eigenvector
        calculated_values['Simple Angle Between Node Non-Reciprocated Out-Weight Vectors'] = non_reciprocated_out_weight_degrees
        calculated_values['Change-Point Detection Node Non-Reciprocated Out-Weight'] = non_reciprocated_out_weight_degrees_eigenvector
        calculated_values['Simple Angle Between Node Non-Reciprocated In-Weight'] = non_reciprocated_in_weight_degrees
        calculated_values['Change-Point Detection Node Non-Reciprocated In-Weight'] = non_reciprocated_in_weight_degrees_eigenvector

        # note: these are dictionaries
        #calculated_values['non_reciprocated_in_weight'] = non_reciprocated_in_weight_dicts
        #calculated_values['non_reciprocated_out_weight'] = non_reciprocated_out_weight_dicts


        with open(basegraph_name + '_processed_vales_' + container_or_class + '_' + '%.2f' % (time_interval) + '.txt', 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for value_name, value in calculated_values.iteritems():
                spamwriter.writerow([value_name, [i if not math.isnan(i) else (None) for i in value]])
    else:
        calculated_values = {}
        with open(basegraph_name + '_processed_vales_' + container_or_class + '_' + '%.2f' % (time_interval) + '.txt', 'r') as csvfile:
            csvread = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in csvread:
                print row
                # todo [check445]: this is a work-around for there being nan's in the strings of values.
                # now, what I should actually do is find out why the nan's are appearing in the strings and then fix
                # that, but since I am in a condensed time-frame, I am just going to ignore the values in which
                # there are nan's and fix them later
                # todo: best work around -> get rid of the nans when saving and then put back in when reading...
                # note: I think I implemented this below, but idk for sure...
                #try:
                calculated_values[row[0]] = [i if i != (None) else float('nan') for i in ast.literal_eval(row[1])]

                #except:
                #    calculated_values[row[0]] = []

    return calculated_values


# okay, so I guess 2 bigs things here: (1) I guess I should iterate through the all the calculated_vals
# dicts here? Also I need to refactor the combined boxplots such that they actually make sense...
def create_graphs(total_calculated_vals, basegraph_name, window_size, colors, time_interval_lengths, exfil_start, exfil_end):
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
        metrics = calculated_values.keys()
        #####

        average_path_lengths_no_nan = calculated_values['average_path_lengths_no_nan']
        weighted_average_path_lengths_no_nan = calculated_values['weighted_average_path_lengths_no_nan']
        unweighted_overall_reciprocities_no_nan = calculated_values['unweighted_overall_reciprocities_no_nan']
        weighted_reciprocities_no_nan = calculated_values['weighted_reciprocities_no_nan']
        densities_no_nan = calculated_values['densities_no_nan']
        angles_degrees_no_nan = calculated_values['angles_degrees_no_nan']

        #appserver_sum_degrees = calculated_values['Sum of Appserver Node Degrees']
        # change to 'Sum of Appserver Node Degrees'

        average_path_lengths = calculated_values['Unweighted Average Path Length']
        # change to: 'Unweighted Average Path Length'

        weighted_average_path_lengths = calculated_values['Weighted Average Path Length']
        # change to 'Weighted Average Path Length'

        unweighted_overall_reciprocities = calculated_values['Unweighted Overall Reciprocity']
        # change to 'Unweighted Overall Reciprocity'

        weighted_reciprocities = calculated_values['Weighted Overall Reciprocity']
        # change to 'Weighted Overall Reciprocity'

        densities = calculated_values['Density']
        # change to 'Density'

        ####
        ####
        angles_degrees = calculated_values['Simple Angle Between Node Degree Vectors']
        # change to 'Simple Angle Between Node Degree Vectors'
        #average_clusterings = calculated_values['average_clusterings']

        angles_degrees_eigenvector = calculated_values['Change-Point Detection Node Degree']
        # change to 'Change-Point Detection Node Degree'

        outstrength_degrees = calculated_values['Simple Angle Between Node Outstrength Vectors']
        # change to 'Simple Angle Between Node Outstrength Vectors'

        outstrength_degrees_eigenvector = calculated_values['Change-Point Detection Node Outstrength']
        # change to 'Change-Point Detection Node Outstrength'

        instrengths_degrees = calculated_values['Simple Angle Between Node Instrength Vectors']
        # change to 'Simple Angle Between Node Instrength Vectors'

        instrengths_degrees_eigenvector = calculated_values['Change-Point Detection Node Instrength']
        # change to 'Change-Point Detection Node Instrength'

        eigenvector_centrality_degrees = calculated_values['Simple Angle Between Node Eigenvector_Centrality Vectors']
        # change to 'Simple Angle Between Node Eigenvector_Centrality Vectors'

        eigenvector_centrality_degrees_eigenvector = calculated_values['Change-Point Detection Node Eigenvector_Centrality']
        # change to 'Change-Point Detection Node Eigenvector_Centrality'

        #clustering_degrees = calculated_values['clustering_degrees']
        #clustering_degrees_eigenvector = calculated_values['clustering_degrees_eigenvector']
        betweeness_centrality_degrees = calculated_values['Simple Angle Between Node Betweeness Centrality Vectors']
        # change to 'Simple Angle Between Node Betweeness Centrality Vectors'

        betweeness_centrality_degrees_eigenvector = calculated_values['Change-Point Detection Node Betweeness Centrality']
        # change to 'Change-Point Detection Node Betweeness Centrality'

        load_centrality_degrees = calculated_values['Simple Angle Between Node Load Centrality Vectors']
        # change to 'Simple Angle Between Node Load Centrality Vectors'

        load_centrality_degrees_eigenvector = calculated_values['Change-Point Detection Node Load Centrality']
        # change to 'Change-Point Detection Node Load Centrality'

        non_reciprocated_out_weight_degrees = calculated_values['Simple Angle Between Node Non-Reciprocated Out-Weight Vectors']
        # change to 'Simple Angle Between Node Non-Reciprocated Out-Weight Vectors'

        non_reciprocated_out_weight_degrees_eigenvector = calculated_values['Change-Point Detection Node Non-Reciprocated Out-Weight']
        # change to 'Change-Point Detection Node Non-Reciprocated Out-Weight'

        non_reciprocated_in_weight_degrees = calculated_values['Simple Angle Between Node Non-Reciprocated In-Weight']
        # change to 'Simple Angle Between Node Non-Reciprocated In-Weight'

        non_reciprocated_in_weight_degrees_eigenvector = calculated_values['Change-Point Detection Node Non-Reciprocated In-Weight']
        # change to 'Change-Point Detection Node Non-Reciprocated In-Weight'


        print "len average path lengths", average_path_lengths, "!!!!!", len(average_path_lengths)
        x = [i*time_interval for i in range(0, len(average_path_lengths))]

        print "avg path lengths", average_path_lengths, len(average_path_lengths)

        ''' # todo: re-enable
        make_graphs_for_val(x, average_path_lengths, time_interval, basegraph_name, 200,
                            'graph_avg_path_length', container_or_class, 'average path length (unweighted)', 'distance')


        make_graphs_for_val(x, weighted_average_path_lengths, time_interval, basegraph_name, 204,
                            '_graph_avg_weighted_path_length_', container_or_class, 'average path length (weighted)', 'distance')

        make_graphs_for_val(x, unweighted_overall_reciprocities, time_interval, basegraph_name, 207,
                            '_graph_unweighted_overall_reciprocity', container_or_class, 'unweighted overall reciprocity', 'reciprocity')

        make_graphs_for_val(x, weighted_reciprocities, time_interval, basegraph_name, 210,
                            '_graph_weighted_overall_reciprocity', container_or_class, 'weighted overall reciprocity',
                            'reciprocity (weighted)')

        make_graphs_for_val(x, densities, time_interval, basegraph_name, 213,
                            '_graph_overall_graph_density', container_or_class, 'overall graph density',
                            'density')

        make_graphs_for_val(x[window_size:], angles_degrees, time_interval, basegraph_name, 216,
                            '_graph_out_degree_simple_angles', container_or_class, 'out degree simple angles',
                            'angle')

        #plt.figure(6)
        try:
            plt.figure(12)
            plt.title("time vs app_server degrees")
            print appserver_sum_degrees
            appserver_sum_degrees = ast.literal_eval(appserver_sum_degrees)
            plt.plot(x, ast.literal_eval(appserver_sum_degrees))
            plt.savefig(basegraph_name + '_appserver_sum_degrees_' + '%.2f' % (time_interval) + '.png', format='png')
        except:
            pass
            #pass-ing b/c I don't really care...


        #make_graphs_for_val(x, average_clusterings, time_interval, basegraph_name, 50, '_graph_average_cluster_degree_',
        #                    container_or_class, 'average clustering simple degree', 'angle')

        print "old x", x
        starting_x = window_size+ (window_size-1)
        print "window_size", window_size, "so starting x:", starting_x
        x_simple_angle = x[window_size:]
        x = x[starting_x:] # b/c I don't calculate angles for the first window_size values...
        #for counter,val in enumerate(outstrength_degrees):
        #    print counter,val
        make_graphs_for_val(x, angles_degrees_eigenvector, time_interval, basegraph_name, 53,
                            '_graph_degrees_eigenvector_degree_', container_or_class, 'degrees eigenvector degree', 'angle')
        make_graphs_for_val(x_simple_angle, outstrength_degrees, time_interval, basegraph_name, 56,
                            '_graph_outstrength_degree_', container_or_class, 'outstrength simple degree', 'angle')
        make_graphs_for_val(x, outstrength_degrees_eigenvector, time_interval, basegraph_name, 59,
                            '_graph_outstrength_eigenvector_degree_', container_or_class, 'outstrength eigenvector degree', 'angle')
        make_graphs_for_val(x_simple_angle, instrengths_degrees, time_interval, basegraph_name, 62,
                            '_graph_instrength_degree_', container_or_class, 'instrength simple degree', 'angle')
        make_graphs_for_val(x, instrengths_degrees_eigenvector, time_interval, basegraph_name, 65,
                            '_graph_instrength_eigenvector_degree_', container_or_class, 'instrength eigenvector degree', 'angle')
        print 'HERE', label#, calculated_values
        try: # see check 445
            make_graphs_for_val(x_simple_angle, eigenvector_centrality_degrees, time_interval, basegraph_name, 68,
                            '_graph_eigenvector_centrality_degree_', container_or_class, 'eigenvector centrality simple degree', 'angle')
        except:
            pass
        make_graphs_for_val(x, eigenvector_centrality_degrees_eigenvector, time_interval, basegraph_name, 71,
                            '_graph_eigenvector_centrality_eigenvector_degree_', container_or_class, 'eigenvector centrality eigenvector degree',
                            'angle')
        #make_graphs_for_val(x, clustering_degrees, time_interval, basegraph_name, 74,
        #                    '_graph_clustering_degree_', container_or_class, 'clustering simple degree', 'angle')
        #make_graphs_for_val(x, clustering_degrees_eigenvector, time_interval, basegraph_name, 77,
        #                    '_graph_clustering_degrees_eigenvector_degree_', container_or_class, 'clustering eigenvector degree',
        #                    'angle')
        try: # see check 445
            make_graphs_for_val(x_simple_angle, betweeness_centrality_degrees, time_interval, basegraph_name, 80,
                            '_graph_betweeness_centrality_degree_', container_or_class, 'betweeness centrality degree', 'angle')
        except:
            pass
        make_graphs_for_val(x, betweeness_centrality_degrees_eigenvector, time_interval, basegraph_name, 83,
                            '_graph_betweeness_centrality_eigenvector_degree_', container_or_class, 'betweeness centrality eigenvector degree',
                            'angle')
        try:# see check 445
            make_graphs_for_val(x_simple_angle, load_centrality_degrees, time_interval, basegraph_name, 86,
                                '_graph_load_centrality_degree_', container_or_class, 'load centrality degree', 'angle')
        except:
            pass
        make_graphs_for_val(x, load_centrality_degrees_eigenvector, time_interval, basegraph_name, 89,
                            '_graph_load_centrality_eigenvector_degree_', container_or_class, 'load centrality eigenvector degree',
                            'angle')
        try: # see check 445
            make_graphs_for_val(x_simple_angle, non_reciprocated_out_weight_degrees, time_interval, basegraph_name, 92,
                            '_graph_non_reciprocated_out_weight_degree_', container_or_class, 'non-reciprocated outweight  degree', 'angle')
        except:
            pass
        make_graphs_for_val(x, non_reciprocated_out_weight_degrees_eigenvector, time_interval, basegraph_name, 95,
                            '_graph_non_reciprocated_out_weight_eigenvector_degree_', container_or_class, 'non-reciprocated outweight eigenvector degree',
                            'angle')
        try: # see check 445
            make_graphs_for_val(x_simple_angle, non_reciprocated_in_weight_degrees, time_interval, basegraph_name, 98,
                            '_graph_non_reciprocated_in_weight_degree_', container_or_class, 'non-reciprocated inweight degree', 'angle')
        except:
            pass
        make_graphs_for_val(x, non_reciprocated_in_weight_degrees_eigenvector, time_interval, basegraph_name, 101,
                            '_graph_non_reciprocated_in_weight_eigenvector_degree_', container_or_class, 'non-reciprocated inweight eigenvector degree',
                            'angle')
    '''

    # okay, so later on I am going to want to group by class/node granularity via color
    # and by time granularity via spacing... so each time granularity should be a seperatae
    # list and each of the class/node granularites should be a nested list (inside the corresponding list)
    # right now: (time gran, node gran) -> metrics -> vals

    node_grans = list(set(node_grans))
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
            for node_gran in node_grans:
                # todo: get rid of [1:] once I run an exp where I start the load generator bfore tcpdump
                metrics_to_time_to_granularity_lists[metric][time_gran].append(  total_calculated_vals[(time_gran, node_gran )][metric][1:] )
                fully_indexed_metrics[(time_gran, node_gran, metric)] = total_calculated_vals[(time_gran, node_gran )][metric]

                nan_count = 0
                for val in total_calculated_vals[(time_gran, node_gran )][metric]:
                    if math.isnan(val):
                        nan_count += 1
                metrics_to_time_to_granularity_nans[metric][time_gran].append(nan_count)

    # okay, so now I actually need to handle make those multi-dimensional boxplots
    for metric in metrics:
        make_multi_time_boxplots(metrics_to_time_to_granularity_lists, time_grans, metric, colors,
                                 basegraph_name + metric + '_multitime_boxplot', node_grans, exfil_start, exfil_end)

        make_multi_time_nan_bars(metrics_to_time_to_granularity_nans, time_grans, node_grans, metric,
                                 basegraph_name + metric + 'multi_nans')

    #'''
    '''
    # todo: next thing on the todo list is to create two seperate covariance matrices (one w/ deltas, one w/ absolutes),
    # note that this means I'll need to do like weighted average or something for the normal vals (i.e. not angles)
    # okay, so I am probably going to want to do that in calc_convariance_matrix, and then just have it return two
    # values...

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

# aggregate all nodes of the same class into a single node
# let's use a multigraph, so we can keep all the edges as intact...
def aggregate_graph(G, ms_s):
    H = nx.MultiDiGraph()
    mapping = {}
    mapping_node_to_ms = {}
    for ms in ms_s:
        mapping[ms] = []
    for node in G.nodes():
        for ms in ms_s:
            #print node
            if ms in node:
                mapping[ms].append(node)
                mapping_node_to_ms[node] = ms
                break
    print mapping_node_to_ms
    for (u,v,data) in G.edges(data=True):
        #print (u,v,data)
        try:
            H.add_edge(mapping_node_to_ms[u], mapping_node_to_ms[v], weight=data['weight'])
        except:
            print "this edge did NOT show up in the map!", (u,v,data)
            # this happens when the outside talking to the 'gateway' shows up in our pcaps
            if u == "1":
                H.add_edge("outside", mapping_node_to_ms[v], weight=data['weight'])
            elif v == "1":
                H.add_edge(mapping_node_to_ms[u], "outside", weight=data['weight'])
            else:
                print "I have no idea what is going on in the aggregate graph function..."
                exit(1)

    pos = graphviz_layout(H)
    nx.draw_networkx(H, pos, with_labels = True, arrows=True)
    #plt.show()

    # while we are at it, let's also return a simpler graph, which is just
    # the multigraph but with all the edges aggregated together
    M = nx.DiGraph()
    mapping_edge_to_weight = {}
    for node_one in H.nodes():
        for node_two in H.nodes():
            mapping_edge_to_weight[ (node_one, node_two)  ] = 0

    for (u,v,data) in H.edges.data(data=True):
        mapping_edge_to_weight[(u,v)] += data['weight']

    print "mapping_edge_to_weight", mapping_edge_to_weight

    for edges, weights in mapping_edge_to_weight.iteritems():
        if weights:
            M.add_edge(edges[0], edges[1], weight=weights)

    pos = graphviz_layout(M)
    nx.draw_networkx(M, pos, with_labels = True, arrows=True)
    #plt.show()

    return H, M

# returns dict of (ms_instance, ms_class) = (in, out)
# where 'in' is the total in-strength from all nodes in the ms_class to this ms_instance
# where 'out' is the total out-strength from the ms_instance to all the nodes in the ms_class
# TODO: there is some problem with the input data to pearsonr, causing the correlation value to be nan

def find_reciprocated_strength(G, ms_s):
    reciprocated_strength_dict = {}
    avg_strength_dict = {}
    max_strength_dict = {}
    eigenvector_ready_reciprocated_strength_dict = {}

    print G.nodes() # i think this should contain all the node

    for ms_instance in G.nodes():
        total_in = {}
        total_out = {}

        for ms_instance_two in G.nodes():
            total_in[ms_instance_two] = 0.0
            total_out[ms_instance_two] = 0.0
        #for edge in G.edges([ms_instance]):
        for edge in list(G.edges(data=True)): # todo: i tihnk this mehod does not exist in the current version of this program
            print "edge", edge
            # i think (src, dst, weight)
            if ms_instance in edge[0]:
                total_out[edge[1]] += edge[2]['weight']
            if ms_instance in edge[1]:
                total_in[edge[0]] += edge[2]['weight']

            if ms_instance in edge[0] and ms_instance in edge[1]:
                print "STRANGE!!"

            eigenvector_ready_reciprocated_strength_dict[edge[0], edge[1]] = edge[2]['weight']

        reciprocated_strength_dict[ms_instance] = (total_out, total_in)
        #for ms_instance_2, weight in total_out.iteritems():
        #    eigenvector_ready_reciprocated_strength_dict[ms_instance, ms_instance_2] = weight
        #for ms_instance_2, weight in total_in.iteritems():
        #    eigenvector_ready_reciprocated_strength_dict[ms_instance_2. ms_instance] = weight

    # now post-process the reciprocated-strength dictionary to aggregate to/from on the class level (tho the data
    # is still there for the node which the thing is about)
    processed_recip_strength_dict = {}
    for ms_instance in G.nodes():
        for ms in ms_s:
            processed_recip_strength_dict[ms_instance, ms] = [0,0] # first is in, second is out (in regard to ms_instance)

    for ms_instance, recip_strength_entry in reciprocated_strength_dict.iteritems():
        for out_dst, out_val in recip_strength_entry[0].iteritems():
            for ms in ms_s:
                if ms in out_dst:
                    #print "keys", ms_instance, out_dst
                    #print "result from keys", [ms_instance, ms]
                    #print "stuff!", recip_strength_entry
                    #print "incoming!!", processed_recip_strength_dict
                    processed_recip_strength_dict[ms_instance, ms][1] += out_val # yah, so out_dst seems to be wrong...
                    break
        for in_src, in_val in recip_strength_entry[1].iteritems():
            for ms in ms_s:
                if ms in in_src:
                    processed_recip_strength_dict[ms_instance, ms][0] += in_val
                    break
    return processed_recip_strength_dict, eigenvector_ready_reciprocated_strength_dict

# uh so what do I want to do here? i could do max/avg or max-avg, or something else
# let's do the divide I suppose?.... seems fine...
# note: going to do the heaviest out-weight... (doing in and out would have everything show up twice...)
# ^^^ not sure if that is true
# returns a dictionary (src, dst) -> [weights]
# src,dst are nodes but there also exist values for the ms_classes,
# so there's some (node, class) and (class, node) pairs in there too (so weights is a multiple element list in that case)
def find_heaviest_edge_vs_avg(G, ms_s):
    strength_dict = {}
    max_strength_dict = {}
    avg_strength_dict_aggregated = {}
    for node_src in G.node():
        for node_dst in G.nodes():
            strength_dict[node_src, node_dst] = []
        for ms in ms_s:
            strength_dict[node_src, ms] = []
            strength_dict[ms, node_src ] = []

    for edge in list(G.edges(data=True)):
        strength_dict[edge[0], edge[1]].append(edge[2]['weight'])
        # we're going to include aggregate components in this list (a.k.a. matrix)
        for ms in ms_s:
            if ms in edge[0]:
                strength_dict[ms, edge[1]].append(edge[2]['weight'])
                break
            if ms in edge[1]:
                strength_dict[edge[0], ms].append(edge[2]['weight'])
                break

    '''
    for edge_path, weights in avg_strength_dict.iteritems():
        for ms in ms_s:
            if ms in edge_path[0]:
                avg_strength_dict_aggregated[ms, e]
        print ms

    for edge_path, weights in avg_strength_dict.iteritems():
        max_strength_dict[edge_path] = max(weights)
    '''
    return strength_dict

# TODO
# needs to return two dicts; each is indexed by a node as well as the ms_class that it is connected to
# one dict returns the principal eigenvector for the weighted adjaceny matrix of the corresponding egonet
# one dict returns the total wieght of the egonet (w/ the corresponding exponent in the power lab TBD at
# a later stage of processing)
def find_dominant_pair(G, ms_s):
    princ_eigenvect = {}
    total_weight = {}

    class_to_instance = {}
    for ms in ms_s:
        class_to_instance[ms] = []
    # need a map of ms_class to ms_instances
    for node in G.nodes():
        for ms in ms_s:
            if ms in node:
                class_to_instance[ms].append(node)
                break

    for node in G.nodes():
        for ms in ms_s:
            # step 1: get the relevant ego
            relevant_egonet = nx.subgraph(G, class_to_instance[ms]+[node]) # todo: verify that this is correcr
            # step 2: find principal eigenvalue of the weighted adjancy matrix...
            adj_matrix = nx.adjacency_matrix(relevant_egonet)
            #print "dimensions of adjacency matrix", scipy.ndim(adj_matrix), "a shape: ", adj_matrix.shape[0]
            if adj_matrix.shape[0] > 1 or adj_matrix.shape[0] > 1 :  # some of these egonets will be essentially empty
                eigenval, eigenvect = scipy.sparse.linalg.eigs(adj_matrix, k=1) #scipy.linalg.eigh(adj_matrix)
                princ_eigenvect[node, ms] = eigenval[0]
                # step 3: the total weight of the egonet
                total_weight[node,ms] = scipy.sum(scipy.sum(adj_matrix, axis=1), axis=0)

    return princ_eigenvect, total_weight

# returns list of angles (of size len(tensor) - window_size)
# (b/c the first window_size angles do not exist in a meaningful way...)
# note: tensor is really a *list of dictionaries*, with keys of nodes_in_tensor
### NOTE: have this function be at least >4 if you want results that are behave coherently, ###
### though >= 6 is best ###
def change_point_detection(tensor, window_size, nodes_in_tensor):
    if window_size < 3:
        # does not make sense for window_size to be less than 3
        print "window_size needs to be >= 3 for pearson to work"
        exit(3)

    # let's outline what I gotta do here...
    # take a 'window' size time slice
    # for each pair of nodes in this window -> calculate correlation of time series
        # using pearson's rho
        # result is a correlation matrix
    # slide the window down the day, getting a time series of correlation matrices
    # for each correlation matrix, find the principal eigenvector
    # this is kinda a second pass of the window thing, but combine (via normal
        # average) all the eigenvectors in the window
    # find angle (seperate function)
    #print "tensor", tensor
    correlation_matrices = []
    p_value_matrices =[]
    correlation_matrix_eigenvectors = []
    # let's iterate through the times, pulling out slices that correspond to windows
    ####smallest_slice = 3 # 2 is guaranteed to get a pearson value of 1, even smaller breaks it
    nodes_under_consideration = []
    for i in range( window_size, len(tensor) + 1):
        start_of_window =  i - window_size # no +1 b/c of the slicing
        # compute average window (with what we have available)
        print "start_of_window", start_of_window
        #print "list slice window of tensor", tensor[start_of_window: i]
        tensor_window = tensor[start_of_window: i]

        # new node shows up -> append to end of list
        nodes_in_tensor_window = []
        for cur_tensor in tensor_window:
            if cur_tensor:
                nodes_in_tensor_window.extend(cur_tensor)
        nodes_in_tensor_window = list(set(nodes_in_tensor_window))
        print "nodes_in_tensor_window", nodes_in_tensor_window

        #nodes_in_tensor_window = [x[0] for x in (y.keys() for y in tensor_window)]
        #print [y.keys() for y in tensor_window]
        #print "nodes_in_tensor_window", nodes_in_tensor_window
        #nodes_in_tensor_window = list(set([x for x in (y.keys() for y in tensor_window)]))

        for node_in_tensor_window in nodes_in_tensor_window:
            if node_in_tensor_window not in nodes_under_consideration:
                nodes_under_consideration.append(node_in_tensor_window)

        # old node disspears completely -> remove from list??
        # todo: currently the size will just grow and grow...
        # the problem is we need to switch over all the eigenvectors at once
        # but since we are aggregating the angles later on, we'd have to
        # account for that somehow
        for node_under_consideration in nodes_under_consideration:
            if node_under_consideration not in nodes_in_tensor_window:
                pass

        correlation_matrix = pandas.DataFrame(0.0, index=nodes_under_consideration, columns=nodes_under_consideration)
        pearson_p_val_matrix = pandas.DataFrame(0.0, index=nodes_under_consideration, columns=nodes_under_consideration)

        for node_one in nodes_under_consideration:
            for node_two in nodes_under_consideration:
                # compute pearson's rho of the corresponding time series

                node_one_time_series = np.array([x[node_one] if x and node_one in x else float('nan') for x in tensor_window])
                node_two_time_series = np.array([x[node_two] if x and node_two in x else float('nan') for x in tensor_window])


                #print "node_one_time_series", node_one_time_series
                #print "node_two_time_series", node_two_time_series

                ''' don't really need this anymore...
                with open('./' + 'debugging.txt', 'a') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow([node_one, node_one_time_series])
                    spamwriter.writerow([node_two, node_two_time_series])
                  '''

                # remove Nan's from array before doing pearson analysis
                # note: np.isfinite will crash if there's a None in the arraay, but that's fine
                # cause I there shouldn't be any None's...
                #print "node_one_time_series", node_one_time_series
                #print "node_two_time_series", node_two_time_series
                invalid_node_one_time_series_entries = np.isfinite(node_one_time_series)
                invalid_node_two_time_series_entries = np.isfinite(node_two_time_series)
                invalid_time_series_entry = invalid_node_one_time_series_entries & invalid_node_two_time_series_entries
                pearson_rho = scipy.stats.pearsonr(node_one_time_series[invalid_time_series_entry],
                                                   node_two_time_series[invalid_time_series_entry])
                #print 'peasrson', pearson_rho, pearson_rho[0], node_one, node_two
                correlation_matrix.at[node_one, node_two] = pearson_rho[0]
                #print correlation_matrix
                pearson_p_val_matrix.at[node_one, node_two] = pearson_rho[1]

                #print node_one_time_series[invalid_time_series_entry], "|||", \
                #    node_two_time_series[invalid_time_series_entry], pearson_rho

                # this just shows no correlation... which is what we'd expect...
                # I don't think this'd cause any problems with respect to causing
                # problems with the angle
                if math.isnan(pearson_rho[0]):
                    correlation_matrix.at[node_one, node_two] = 0.0
                #'''
                # todo: does this make sense???? [[no, not really...]
                # todo: how about this: we leave the nan's here and then we handle it
                # when it comes to finding the eigenvector
                # note: this is a questionable edgecase. My reasoning is that
                # all the values are typically the same during each time interval, since
                # neither changes, we have no idea if there is (or isn't) a relation,
                # so to be safe let's say zero
                #if math.isnan(pearson_rho[0]):
                #    correlation_matrix.at[node_one, node_two] = 0.0
                #else:
                #    correlation_matrix.at[node_one, node_two] = pearson_rho[0]

                #'''
        print correlation_matrix
        correlation_matrices.append(correlation_matrix)
        p_value_matrices.append(pearson_p_val_matrix)

        eigen_vals, eigen_vects = scipy.linalg.eigh(correlation_matrix.values)
        # note: here we want the principal eigenvector, which is assocated with the
        # eigenvalue that has the largest magnitude
        print "eigenvalues", eigen_vals
        #print eigen_vects
        largest_mag_eigenvalue = max(eigen_vals, key=abs)
        #print "largest_mag_eigenvalue", largest_mag_eigenvalue
        largest_mag_eigenvalue_index = 0
        for counter, value in enumerate(eigen_vals):
            if value == largest_mag_eigenvalue:
                largest_mag_eigenvalue_index = counter
                break
        ''' # I thought this was going to be needed, but I am not so sure anymore...
        if len(largest_mag_eigenvalue_index) == 1:
            largest_mag_eigenvalue_index = largest_mag_eigenvalue_index[0]
        else:
            # if multiple identical eigenvalues, then we gotta
            # choose the largest eigenvector by comparing the magnitudes
            # of the eigenvectors
            largest_mag_eigenvector = -100000 # effectively infinity
            largest_mag_eigenvector_index = -1
            for possibly_largest_eigenvector_index in largest_mag_eigenvalue_index:
                # then just do the comparison
                if np.linalg.norm(eigen_vects[possibly_largest_eigenvector_index]) > largest_mag_eigenvector:
                    largest_mag_eigenvector = np.linalg.norm(eigen_vects[possibly_largest_eigenvector_index])
                    largest_mag_eigenvector_index = possibly_largest_eigenvector_index
            largest_mag_eigenvalue_index = largest_mag_eigenvector_index
        '''

        #print "eigenvectors", eigen_vects
        print "principal eigenvector", eigen_vects[largest_mag_eigenvalue_index]
        correlation_matrix_eigenvectors.append(eigen_vects[largest_mag_eigenvalue_index])

    #for correlation_matrix in correlation_matrices:
    #    print correlation_matrix.values, '\n'
    print "correlation eigenvects", correlation_matrix_eigenvectors
    angles = find_angles(correlation_matrix_eigenvectors, window_size)

    return angles

# as related to check445, this function returns too many nan's
# I think this is because it gives nan values when an all zero's
# vector is given (b/c cosine distance not defined for that)
# note: problem could potentially be in turn_into_list
def find_angles(list_of_vectors, window_size):

    angles = [] # first must be zero (nothing to compare to)
    for i in range(window_size, len(list_of_vectors)):
        print "angles is", angles
        start_of_window = i - window_size
        # compute average window (with what we have available)
        print "list slice window", list_of_vectors[start_of_window: i]
        # note: we also need to take care of size problems here, but putting zeros in the missing dimension
        max_size = max([x.shape[0] for x in list_of_vectors[start_of_window: i]])
        list_of_size_adjusted_vectors = []
        for vector in list_of_vectors[start_of_window : i]:
            size_difference =  max_size - vector.shape[0]
            for j in range(0, size_difference):
                vector = np.append(vector, [0.0])
            list_of_size_adjusted_vectors.append(vector)
        print "list slice window size adjusted", list_of_size_adjusted_vectors
        # NOTE: THIS IS THE ARITHMETIC AVERAGE OF THE NON-UNIT EIGENVECTORS
        # THIS MEANS THAT IT *WILL* BE MORE HEAVILY WEIGHTED TO THE LARGER ONES
        window_average = np.mean([x for x in list_of_size_adjusted_vectors if x != []], axis=0)
        #print "start of window", start_of_window, "window average", window_average

        # to compare angles, we should use unit vectors (and then calc the angle)
        # from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
        window_average_unit_vector = find_unit_vector(window_average)
        print "window_average_unit_vector", window_average_unit_vector
        print "window_average", window_average
        current_value_unit_vector = find_unit_vector(list_of_vectors[i])
        #print "window_average_unit_vector", window_average_unit_vector, "current_value_unit_vector", current_value_unit_vector

        # sometimes the first time interval has no activity
        if window_average_unit_vector.size == 0 or current_value_unit_vector.size == 0:
            #window_average_unit_vector = np.zeros(len(current_value_unit_vector))
            print "angle was", float('nan')
            angles.append(float('nan'))
            continue

        try:
            if math.isnan(window_average_unit_vector):
                angles.append(float('nan'))
                continue
        except:
            pass

        try:
            if math.isnan(current_value_unit_vector):
                angles.append(float('nan'))
                continue
        except:
            pass

        # ok, let's take care of the situation where the vectors are different
        # sizes. It sees to me that we can just extend the smaller one w/ a zero
        # value in the 'missing' dimension?
        size_difference = window_average_unit_vector.shape[0] - current_value_unit_vector.shape[0]
        print "size_difference", size_difference
        for i in range(0,abs(size_difference)):
            if size_difference > 0:
                # append to current_value_unit_vector
                current_value_unit_vector = np.append(current_value_unit_vector,[0])
            else:
                # append to window_average_unit_vector
                window_average_unit_vector = np.append(window_average_unit_vector,[0])

        print "window_average_unit_vector", window_average_unit_vector
        print "current_value_unit_vector", current_value_unit_vector

        angle = np.arccos( np.clip(np.dot(window_average_unit_vector, current_value_unit_vector), -1.0, 1.0)  )
        print "the angle was found to be ", angle
        angles.append(angle)

    print "angles", angles
    return angles

def find_unit_vector(vector):
    return vector / np.linalg.norm(vector)

def graph_distance(starting_point, ending_point, dictionary_of_edge_attribs):
    return float(1) / dictionary_of_edge_attribs['weight']

# see https://www.nature.com/articles/srep02729
# note, despite the name, I actually do both vertex-specific and network-wide here...
# further note, the paper also suggests a way to analyze it, which may be useful (see equation 11 in the paper)
def network_weidge_weighted_reciprocity(G):
    in_weights = {}
    out_weights = {}
    reciprocated_weight = {}
    non_reciprocated_out_weight = {}
    non_reciprocated_in_weight = {}

    for node in nx.nodes(G):
        reciprocated_weight[node] = 0
        non_reciprocated_out_weight[node] = 0
        non_reciprocated_in_weight[node] = 0
        in_weights[node] = 0
        out_weights[node] = 0

    for node in nx.nodes(G):
        for node_two in nx.nodes(G):
            if node != node_two:

                node_to_nodeTwo = G.get_edge_data(node, node_two)
                nodeTwo_to_node = G.get_edge_data(node_two, node)
                if node_to_nodeTwo:
                    node_to_nodeTwo = node_to_nodeTwo['weight']
                else:
                    node_to_nodeTwo = 0

                if nodeTwo_to_node:
                    nodeTwo_to_node = nodeTwo_to_node['weight']
                else:
                    nodeTwo_to_node = 0

                #print "edge!!!", edge
                #node_to_nodeTwo = G.out_edges(nbunch=[node, node_two])
                #nodeTwo_to_node = G.in_edges(nbunch=[node, node_two])
                print "len edges",  node_to_nodeTwo, nodeTwo_to_node
                reciprocated_weight[node] = min(node_to_nodeTwo, nodeTwo_to_node)
                non_reciprocated_out_weight[node] = max(node_to_nodeTwo - reciprocated_weight[node], 0)
                non_reciprocated_in_weight[node] = max(nodeTwo_to_node - reciprocated_weight[node], 0 )


    # only goes through out-edges (so no double counting)
    total_weight = 0
    for edge in G.edges(data=True):
        #print edge
        #input("Look!!!")
        try:
            total_weight += edge[2]['weight']
        except:
            pass # maybe the edge does not exist

    total_reicp_weight = 0
    for recip_weight in reciprocated_weight.values():
        total_reicp_weight += recip_weight

    if total_weight == 0:
        weighted_reciprocity = -1 # sentinal value
    else:
        weighted_reciprocity = float(total_reicp_weight) / float(total_weight)

    return weighted_reciprocity, non_reciprocated_out_weight, non_reciprocated_in_weight

# note: this works b/c items do not change order in lists
def turn_into_list(dicts, node_list):
    node_vals= []
    for dict in dicts:
        current_nodes = []
        # print G_list, len(G_list)
        for node in node_list:
            try:
                current_nodes.append(float(dict[node]))
            except:
                current_nodes.append(float('nan'))  # the current dict must not have an entry for node -> zero val
        node_vals.append(np.array(current_nodes))
    return node_vals

def make_graphs_for_val(x_vals, y_vals, time_interval, basegraph_name, fig_num, graph_name_extenstion,
                        container_or_class, graph_tile, y_axis_label):

    #y_vals = ast.literal_eval(y_vals)
    plt.figure(fig_num)
    plt.clf()
    plt.title(graph_tile + ', ' + '%.2f' % (time_interval))
    plt.ylabel(y_axis_label)
    plt.xlabel('time (sec)')
    print graph_name_extenstion
    print "x inputs", x_vals, type(x_vals)
    print "y inputs", y_vals, type(y_vals)
    plt.plot(x_vals, y_vals)
    plt.savefig(basegraph_name + graph_name_extenstion + '_' + container_or_class + '_' + '%.2f' % (time_interval) + '.png', format='png')

    plt.figure(fig_num + 1)
    plt.clf()
    plt.title(graph_tile + ', ' + '%.2f' % (time_interval))
    plt.ylabel(y_axis_label)
    plt.boxplot(y_vals, sym='k.', whis=[5, 95])
    plt.savefig(basegraph_name + graph_name_extenstion + '_boxplot_' + container_or_class + '_' + '%.2f' % (time_interval) + '.png', format='png')

# i think correlation matrix must be a pandas dataframe (with the appropriate labels)
def plot_correlogram(delta_covariance_dataframe, abs_covariance_dataframe, basegraph_name):

    # based off of example located at: https://seaborn.pydata.org/generated/seaborn.heatmap.html
    #don't think is needed: flights = sns.load_dataset("flights")
    #don't think is needed: flights = flights.pivot("month", "year", "passengers")

    # todo: I want to split this correlation matrix into several smaller dataframes
    # and then plot several correlation matrices

    #delta_covariance_dataframe.index.values[]

    print "heatmap about to be created!"
    print "delta_covariance dataframe", delta_covariance_dataframe
    fig = plt.figure(figsize=(20, 15))
    fig.clf()
    ax = sns.heatmap(delta_covariance_dataframe)
    ax.set_title('Delta Covariance Matrix')
    fig = ax.get_figure()
    #fig.show()
    fig.savefig(basegraph_name + 'delta_correlation_heatmap' + '.png', format='png')

    print "heatmap 2 about to be made!"
    fig3 = plt.figure(figsize=(20, 15))
    fig3.clf()
    ax3 = sns.heatmap(abs_covariance_dataframe)
    ax3.set_title('Abs Covariance Matrix')
    fig3 = ax3.get_figure()
    fig3.savefig(basegraph_name + 'abs_correlation_heatmap' + '.png', format='png')


    ''' # TOOD: RE-ENABLE
    # going to drop NaN's beforep plotting
    print "pairplot about to be created!"
    ax2 = sns.pairplot(delta_covariance_dataframe.dropna())# note: I hope dropna() doesn't mess the alignment up but it might
    ax2.set_title('Delta Pairplot Matrix')
    ax2.savefig(basegraph_name + 'delta_correlation_pairplot' + '.png', format='png')

    print "pairplot 2 about to be made!"
    ax4 = sns.pairplot(abs_covariance_dataframe.dropna()).set_title('Abs Pairplot Matrix') # note: I hope dropna() doesn't mess the alignment up but it might
    ax4.savefig(basegraph_name + 'abs_correlation_pairplot' + '.png', format='png')
    '''

    ''' Note: this'd take some work, but might be worth doing at some point
    # LIFTED: from https://python-graph-gallery.com/327-network-from-correlation-matrix/
    # Transform it in a links data frame (3 columns only):
    links = correlation_matrix.stack().reset_index()
    links.columns = ['var1', 'var2', 'value']
    # Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)
    links_filtered = links.loc[(links['value'] > 0.8) & (links['var1'] != links['var2'])]
    # Build your graph
    G = nx.from_pandas_dataframe(links_filtered, 'var1', 'var2')
    # Plot the network:
    nx.draw(G, with_labels=True, node_color='orange', node_size=400, edge_color='black', linewidths=1, font_size=15)
    '''

    #plt.show()

def calc_covaraiance_matrix(calculated_values):
    # todo: okay, so I guess here is where I want to calc the two different covariance matrixes (delta vs abs)
    # additionally, I'm going to prob want to calc the delta for the graph-wide metrics here (so that I don't
    # have to rerun all of the processing function, but I'll probably want ot move it over to that func at some
    # point).
    # so let's keep this really simple and just calc the change between two values (don't worry about doing like
    # EWMA or anything like that, keep it real simple)
    # so, this is how I think this should be done...
    # step 1: gotta determine which is in each
        # if neither degree nor no_nan is in title -> then abs value (everything else has degrees in the term...)
    # step 2: make the abs convariance input matrix
        # okay, so this has 2 components: (a) put into new dict, (b) remove from old dict
    #     for label, calculated_values in total_calculated_vals.iteritems():

    to_be_deleted = []
    for label in calculated_values.keys():
        if 'no_nan' in label[2]:
            to_be_deleted.append(label)
        elif 'degrees' in label[2] and 'eigen' not in label[2]:
            to_be_deleted.append(label)
    #print "to_be_deleted", to_be_deleted
    #print "keys to calculated_values", calculated_values.keys()

    for item in to_be_deleted:
        del calculated_values[item]

    abs_calculated_values = {}
    for label, values in calculated_values.iteritems():
        if 'degree' not in label and 'no_nan' not in label:
            abs_calculated_values[label] = values
    for label in abs_calculated_values.keys():
        del calculated_values[label]

    # step 3: one-by-one, calc the delta of the abs values
        # hmmm... how to do this...
    for label, values in abs_calculated_values.iteritems():
        # hm... so let's just calc the delta of the list of values... let's see if there is a function for this...
        delta_vals = [a - b for a,b in itertools.izip(values, values[1:])]
        # step 4: put these into the delta covariance input matrix, along with the other values
        calculated_values[(label[0], label[1], label[2] + '_delta')] = delta_vals

    # step 5: modify the thingee that actually calculates the covariance matrixes
    # step 6: modify graphing fuctions...

    # I want to remove all of the simple angle analysis here (but keep the
    # eigenvector analysis!)
    # todo: am I sure I don't these vals?
    parsed_calculated_values = {}
    for item, val in calculated_values.iteritems():
        if 'degree' in item:
            if 'eigenvector' in item:
                parsed_calculated_values[item] = val
        else:
            parsed_calculated_values[item] = val

    # using the method from: https://stackoverflow.com/questions/19736080/creating-dataframe-from-a-dictionary-where-entries-have-different-lengths
    #DataFrame(dict([ (k,Series(v)) for k,v in d.items() ]))
    #
    abs_covariance_matrix_input = pandas.DataFrame(dict([ (k,pandas.Series(v)) for k,v in abs_calculated_values.iteritems() ]))
    print "abs_covariance_matrix_input", abs_covariance_matrix_input
    print abs_covariance_matrix_input.shape

    delta_covariance_matrix_input = pandas.DataFrame(dict([ (k,pandas.Series(v)) for k,v in calculated_values.iteritems() ]))
    #delta_covariance_matrix_input = pandas.DataFrame(calculated_values)
    print "delta_covariance_matrix_input", delta_covariance_matrix_input
    print delta_covariance_matrix_input.shape
    # todo: is this square? ^^^ I think not b/c some values (computed via the vector/angle thingee) would
    # be missing some vals, compared to the simple graph-wide metrics
    # (so might wanna either pad or remove...)
    # NOTE: with the modification to using corr(), i don't think it needs to be a square anymore

    # must transpose b/c corr() finds covariance between columns, so it follows that each
    # column should have a seperate variable
    delta_covariance_dataframe = delta_covariance_matrix_input.corr()
    abs_covariance_dataframe = abs_covariance_matrix_input.corr()
    print delta_covariance_dataframe.shape, abs_covariance_dataframe.shape

    return delta_covariance_dataframe, abs_covariance_dataframe

# in the style of: https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots
def make_multi_time_boxplots(metrics_to_time_to_granularity_lists, time_grans, metric, colors,
                             graph_name, node_grans, exfil_start, exfil_end):
    fig = plt.figure()
    fig.clf()
    ax = plt.axis()

    max_yaxis = -1000000 # essentially -infinity
    min_yaxis = 1000000 # essentially infinity
    cur_pos = 1
    tick_position_list = []
    for time_gran in time_grans:
        current_vals = metrics_to_time_to_granularity_lists[metric][time_gran]
        # todo: is there a better way to handle Nan's? Maybe describe them somewhere??
        # yes, we'll display them in a bar graph.
        current_vals = [[x for x in i if x is not None and not math.isnan(x)] for i in current_vals]
        number_nested_lists = len(current_vals)
        number_positions_on_graph = range(cur_pos, cur_pos+number_nested_lists)
        tick_position = (float(number_positions_on_graph[0]) + float(number_positions_on_graph[-1])) / number_nested_lists
        tick_position_list.append( tick_position )
        bp = plt.boxplot(current_vals, positions = number_positions_on_graph, widths = 0.6, sym='k.')
        #print "number of boxplots just added:", len(metrics_to_time_to_granularity_lists[metric][time_gran]), " , ", number_nested_lists
        plt.title(metric)
        plt.xlabel('time granularity (seconds)')
        if 'Change-Point' in metric or 'Angle' in metric:
            plt.ylabel('Angle')
        else:
            plt.ylabel(metric)
        cur_pos += number_nested_lists + 1 # the +1 is so that there is extra space between the groups
        set_boxplot_colors(bp, colors)
        #print metric
        try:
            max_yaxis = max( max_yaxis, max([max(x) for x in current_vals]))
        except:
            pass
        try:
            min_yaxis = min( min_yaxis, min([min(x) for x in current_vals]))
        except:
            pass

        # todo: plot the points specifically during exfiltration times


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
    i = 0
    for color in colors:
        cur_line, = plt.plot([1,1], color, label=time_grans[i])
        invisible_lines.append(cur_line)
        i += 1
    plt.legend(invisible_lines, node_grans)
    for line in invisible_lines:
        line.set_visible(False)

    plt.savefig(graph_name + '.png', format='png')

def set_boxplot_colors(bp, colors):
    for counter, color in enumerate(colors):
        #print "counter", counter
        plt.setp(bp['boxes'][counter], color=color)
        plt.setp(bp['caps'][counter * 2], color=color)
        plt.setp(bp['caps'][counter * 2 + 1], color=color)
        plt.setp(bp['whiskers'][counter * 2], color=color)
        plt.setp(bp['whiskers'][counter * 2 + 1], color=color)
        ### might not necessarily have fliers (those are the points that show up outside
        ### of the bloxplot)
        try:
            plt.setp(bp['fliers'][counter * 2 ], color=color)
        except:
            pass
        try:
            plt.setp(bp['fliers'][counter * 2 + 1], color=color)
        except:
            pass
        ###
        plt.setp(bp['medians'][counter], color=color)

# todo: make this actually work out
def make_multi_time_nan_bars(metrics_to_time_to_granularity_nans, time_grans, node_grans, metric, graph_name):
    fig = plt.figure()
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
        bp = plt.bar(current_vals, positions=number_positions_on_graph, widths=0.6, sym='k.')
        # print "number of boxplots just added:", len(metrics_to_time_to_granularity_lists[metric][time_gran]), " , ", number_nested_lists
        plt.title(metric)
        plt.xlabel('time granularity (seconds)')
        plt.ylabel('Number of nans')
        cur_pos += number_nested_lists + 1  # the +1 is so that there is extra space between the groups
        # print metric
        try:
            max_yaxis = max(max_yaxis, max([max(x) for x in current_vals]))
        except:
            pass
        try:
            min_yaxis = min(min_yaxis, min([min(x) for x in current_vals]))
        except:
            pass
    yaxis_range = max_yaxis - min_yaxis
    # print "old min yaxis", min_yaxis
    # print "old max yaxis", max_yaxis
    # print "yaxis_range", yaxis_range
    min_yaxis = min_yaxis - (yaxis_range * 0.05)
    max_yaxis = max_yaxis + (yaxis_range * 0.25)
    # print "min_yaxis", min_yaxis
    # print "max_yaxis", max_yaxis
    plt.xlim(0, cur_pos)
    plt.ylim(min_yaxis, max_yaxis)
    plt.xticks(tick_position_list, [str(i) for i in time_grans])

    plt.savefig(graph_name + '.png', format='png')


##########################################
##########################################
##########################################
# todo tomorrow (saturday) (ideally, I get all of this done on saturday)
# (1) find replacement for dominantPair
    # morning (idk that one is really necessary?)
    # actually, i think this maybe fine? or just take the eigenvalue of the weighted adjaceny matrix...
    # i actually this is the way to go, but I am going ot need to only consider connections from a certain node
    # to a certain class of nodes (at a time)
    # okay, so this is more-or-less done...
# (2) implement leman-style tracking (+ simpler)
    # morning
# (3) make graphs of values from (2)
    # morning/afternoon
# (4) automate whole data processing pipeline
    # afternoon
# (5) get non-seastore graphs
    # afternoon

## rest of saturday plan
# automate/debug the existing data processing pipeline (the other 2 measures can be done monday/whne time is available)
# so at the end of today, I hope to have these things done
# (1) a script that'll take care of the whole data processing pipeline (or that I can grab pieces from to run part of it)
# (2) confidence that the pipeline I have so far is implemented correctly
# (3) graphs for seastore, preferably others as well (shouldn't be too bad if (1) goes smoothly)
# (4) play some ai war

# TODO: some other things to try
# using basic angle technique:
#   weighted reciprocity
#   maximum weight ratio on reciprocated edges egonet
        # from leman's paper, i think might be helpful to find the one-attack-link
# using full eigenspace technique:
#   out degree
#   outstrength
#   instrength
#   non-reciprocated weight
#
# actually we can all of these both ways, with node,class,and graph granularity!
#
# all these would take me to 10 total measures... that's a good number for now

# wed.
# (1) above graph metrics
    # prereq: finish implementing the leman method
        # I still need to work some more on this, b/c I think 'dirty' data (maybe nan's or something)
        # are being fed into the pearsonr function, and so it is returning nan
    # prereq: want another (aggregated by node class) graph
        # okay, I think this is more or less fine
    # need to modify it so edgesfiles for < 1 time intervals can be created
# (2) finish polishing graphs
    # big thing here is the multi-time resolution boxplots
# (3) develop concrete plan for simulating exfiltration
    # hopefully there is some pre-existing library that I can use
    # looks like DET (https://github.com/PaulSec/DET is my best bet here, tho
    # it may not be setup for multi-hop, tho it looks like no tool is setup for
    # that, so i guess that's just the way it is, wait holdup, it looks like
    # Proxy mode is a thing... yep, just need to configure the config.json file
    # btw, I will need a way to ssh into a large of these containers + install stuff
    # hm.... (see text document for more details...)


# (4) read dockerGuard code / technical details of paper
    # looks like most of it is in python, which is good
# (5) try running the dockerGuard code
    # might be tricky...