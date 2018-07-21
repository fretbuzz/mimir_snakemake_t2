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

sockshop_ms_s = ['carts-db','carts','catalogue-db','catalogue','front-end','orders-db','orders',
        'payment','queue-master','rabbitmq','session-db','shipping','user-sim', 'user-db','user','load-test']

def pipeline_analysis_step(filenames, ms_s, time_interval, basegraph_name, calc_vals_p, window_size):
    list_of_graphs = []
    list_of_aggregated_graphs = [] # all nodes of the same class aggregated into a single node
    list_of_aggregated_graphs_multi = [] # the above w/ multiple edges
    total_calculated_values = {}
    counter= 0 # let's not make more than 50 images of graphs

    for file_path in filenames:
        if calc_vals_p:
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
                avg_path_length = 0

            try:
                recip = nx.overall_reciprocity(cur_G)  # if it is not one, then I cna deal w/ looking at dictinoarty
            except :
                recip = -1 # overall reciprocity not defined for empty graphs
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
            except:
                eigenvector_centrality_dict = {}
                for node in total_node_list:
                    eigenvector_centrality_dict[node] = 0

            #clustering_dict = nx.clustering(cur_G)
            try:
                betweeness_centrality_dict = nx.betweenness_centrality(cur_G)
            except:
                betweeness_centrality_dict = {}
                for node in total_node_list:
                    betweeness_centrality_dict[node] = 0

            try:
                load_centrality_dict = nx.load_centrality(cur_G)
            except:
                load_centrality_dict = {}
                for node in total_node_list:
                    load_centrality_dict[node] = 0

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
            #input("stuff")
            weighted_reciprocity, weighted_reciprocity_eigenvector_ready = find_reciprocated_strength(cur_G, ms_s)
            weighted_reciprocity_processed = {}
            for key, val in weighted_reciprocity.iteritems():
                if val[0] == 0:
                    weighted_reciprocity_processed[key] = -1 # sentinal value
                else:
                    weighted_reciprocity_processed[key] = val[1] / val[0]  # out/in

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
        angles_degrees = find_angles(node_degrees, window_size) #eigenvector_analysis(degree_dicts, window_size=window_size)  # setting window size arbitrarily for now...
        print "angles degrees", type(angles_degrees), angles_degrees
        print node_degrees
        angles_degrees_eigenvector = eigenvector_analysis(degree_dicts, window_size, total_node_list)
        print "angles degrees eigenvector", angles_degrees_eigenvector

        #######

        # outstrength analysis
        node_outstrengths = turn_into_list(outstrength_dicts, total_node_list)
        print "node_outstrengths", node_outstrengths
        outstrength_degrees = find_angles(node_outstrengths, window_size)
        outstrength_degrees_eigenvector = eigenvector_analysis(outstrength_dicts, window_size, total_node_list)

        # instrength analysis
        node_instrengths = turn_into_list(instrength_dicts, total_node_list)
        print "node_instrengths", node_instrengths
        instrengths_degrees = find_angles(node_instrengths, window_size)
        instrengths_degrees_eigenvector = eigenvector_analysis(instrength_dicts, window_size, total_node_list)

        # eigenvector centrality analysis
        node_eigenvector_centrality = turn_into_list(eigenvector_centrality_dicts, total_node_list)
        eigenvector_centrality_degrees = find_angles(node_eigenvector_centrality, window_size)
        eigenvector_centrality_degrees_eigenvector = eigenvector_analysis(eigenvector_centrality_dicts, window_size, total_node_list)

        # clustering analysis (not implemented for directed type)
        #node_clustering = turn_into_list(clustering_dicts, total_node_list)
        #clustering_degrees = find_angles(node_clustering, window_size)
        #clustering_degrees_eigenvector = eigenvector_analysis(node_clustering, window_size, total_node_list)


        # betweeness centrality analysis
        node_betweeness_centrality = turn_into_list(betweeness_centrality_dicts, total_node_list)
        betweeness_centrality_degrees = find_angles(node_betweeness_centrality, window_size)
        betweeness_centrality_degrees_eigenvector = eigenvector_analysis(betweeness_centrality_dicts, window_size, total_node_list)

        # load centrality analysis
        node_load_centrality = turn_into_list(load_centrality_dicts, total_node_list)
        load_centrality_degrees = find_angles(node_load_centrality, window_size)
        load_centrality_degrees_eigenvector = eigenvector_analysis(load_centrality_dicts, window_size, total_node_list)

        # non_reciprocated_out_weight analysis
        node_non_reciprocated_out_weight = turn_into_list(non_reciprocated_out_weight_dicts, total_node_list)
        non_reciprocated_out_weight_degrees = find_angles(node_non_reciprocated_out_weight, window_size)
        non_reciprocated_out_weight_degrees_eigenvector = eigenvector_analysis(non_reciprocated_out_weight_dicts, window_size, total_node_list)


        # non_reciprocated_in_weight analysis
        node_non_reciprocated_in_weight = turn_into_list(non_reciprocated_in_weight_dicts, total_node_list)
        non_reciprocated_in_weight_degrees = find_angles(node_non_reciprocated_in_weight, window_size)
        non_reciprocated_in_weight_degrees_eigenvector = eigenvector_analysis(non_reciprocated_in_weight_dicts, window_size, total_node_list)

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
        calculated_values['average_path_lengths_no_nan'] = average_path_lengths_no_nan
        calculated_values['weighted_average_path_lengths_no_nan'] = weighted_average_path_lengths_no_nan
        calculated_values['unweighted_overall_reciprocities_no_nan'] = unweighted_overall_reciprocities_no_nan
        calculated_values['weighted_reciprocities_no_nan'] = weighted_reciprocities_no_nan
        calculated_values['angles_degrees_no_nan'] = angles_degrees_no_nan
        calculated_values['appserver_sum_degrees'] = appserver_sum_degrees
        calculated_values['densities_no_nan'] = densities_no_nan
        calculated_values['average_path_lengths'] = average_path_lengths
        calculated_values['weighted_average_path_lengths'] = weighted_average_path_lengths
        calculated_values['unweighted_overall_reciprocities'] = unweighted_overall_reciprocities
        calculated_values['weighted_reciprocities'] = weighted_reciprocities
        calculated_values['densities'] = densities
        calculated_values['angles_degrees'] = angles_degrees
        #calculated_values['average_clusterings'] = average_clusterings
        calculated_values['angles_degrees_eigenvector'] = angles_degrees_eigenvector
        calculated_values['outstrength_degrees'] = outstrength_degrees
        calculated_values['outstrength_degrees_eigenvector'] = outstrength_degrees_eigenvector
        calculated_values['instrengths_degrees'] = instrengths_degrees
        calculated_values['instrengths_degrees_eigenvector'] = instrengths_degrees_eigenvector
        calculated_values['eigenvector_centrality_degrees'] = eigenvector_centrality_degrees
        calculated_values['eigenvector_centrality_degrees_eigenvector'] = eigenvector_centrality_degrees_eigenvector
        #calculated_values['clustering_degrees'] = clustering_degrees
        #calculated_values['clustering_degrees_eigenvector'] = clustering_degrees_eigenvector
        calculated_values['betweeness_centrality_degrees'] = betweeness_centrality_degrees
        calculated_values['betweeness_centrality_degrees_eigenvector'] = betweeness_centrality_degrees_eigenvector
        calculated_values['load_centrality_degrees'] = load_centrality_degrees
        calculated_values['load_centrality_degrees_eigenvector'] = load_centrality_degrees_eigenvector
        calculated_values['non_reciprocated_out_weight_degrees'] = non_reciprocated_out_weight_degrees
        calculated_values['non_reciprocated_out_weight_degrees_eigenvector'] = non_reciprocated_out_weight_degrees_eigenvector
        calculated_values['non_reciprocated_in_weight_degrees'] = non_reciprocated_in_weight_degrees
        calculated_values['non_reciprocated_in_weight_degrees_eigenvector'] = non_reciprocated_in_weight_degrees_eigenvector

        # note: these are dictionaries
        #calculated_values['non_reciprocated_in_weight'] = non_reciprocated_in_weight_dicts
        #calculated_values['non_reciprocated_out_weight'] = non_reciprocated_out_weight_dicts


        with open(basegraph_name + '_processed_vales_' + container_or_class + '_' + '%.2f' % (time_interval) + '.txt', 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for value_name, value in calculated_values.iteritems():
                spamwriter.writerow([value_name, value])
    else:
        calculated_values = {}
        with open(basegraph_name + '_processed_vales_' + container_or_class + '_' + '%.2f' % (time_interval) + '.txt', 'r') as csvfile:
            csvread = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            try:
                for row in csvread:
                    print row
                    try:
                        calculated_values[row[0]] = ast.literal_eval(row[1])
                    except:
                        calculated_values[row[0]] = []
            except:
                print "that row was too long!!!"

    return calculated_values


# okay, so I guess 2 bigs things here: (1) I guess I should iterate through the all the calculated_vals
# dicts here? Also I need to refactor the combined boxplots such that they actually make sense...
def create_graphs(total_calculated_vals, basegraph_name, window_size, colors):
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
        time_grans.append(time_interval)
        node_grans.append(container_or_class)
        metrics = calculated_values.keys()
        #####

        average_path_lengths_no_nan = calculated_values['average_path_lengths_no_nan']
        weighted_average_path_lengths_no_nan = calculated_values['weighted_average_path_lengths_no_nan']
        unweighted_overall_reciprocities_no_nan = calculated_values['unweighted_overall_reciprocities_no_nan']
        weighted_reciprocities_no_nan = calculated_values['weighted_reciprocities_no_nan']
        angles_degrees_no_nan = calculated_values['angles_degrees_no_nan']
        appserver_sum_degrees = calculated_values['appserver_sum_degrees']
        densities_no_nan = calculated_values['densities_no_nan']
        average_path_lengths = calculated_values['average_path_lengths']
        weighted_average_path_lengths = calculated_values['weighted_average_path_lengths']
        unweighted_overall_reciprocities = calculated_values['unweighted_overall_reciprocities']
        weighted_reciprocities = calculated_values['weighted_reciprocities']
        densities = calculated_values['densities']
        angles_degrees = calculated_values['angles_degrees']

        #average_clusterings = calculated_values['average_clusterings']
        angles_degrees_eigenvector = calculated_values['angles_degrees_eigenvector']
        outstrength_degrees = calculated_values['outstrength_degrees']
        outstrength_degrees_eigenvector = calculated_values['outstrength_degrees_eigenvector']
        instrengths_degrees = calculated_values['instrengths_degrees']
        instrengths_degrees_eigenvector = calculated_values['instrengths_degrees_eigenvector']
        eigenvector_centrality_degrees = calculated_values['eigenvector_centrality_degrees']
        eigenvector_centrality_degrees_eigenvector = calculated_values['eigenvector_centrality_degrees_eigenvector']
        #clustering_degrees = calculated_values['clustering_degrees']
        #clustering_degrees_eigenvector = calculated_values['clustering_degrees_eigenvector']
        betweeness_centrality_degrees = calculated_values['betweeness_centrality_degrees']
        betweeness_centrality_degrees_eigenvector = calculated_values['betweeness_centrality_degrees_eigenvector']
        load_centrality_degrees = calculated_values['load_centrality_degrees']
        load_centrality_degrees_eigenvector = calculated_values['load_centrality_degrees_eigenvector']
        non_reciprocated_out_weight_degrees = calculated_values['non_reciprocated_out_weight_degrees']
        non_reciprocated_out_weight_degrees_eigenvector = calculated_values['non_reciprocated_out_weight_degrees_eigenvector']
        non_reciprocated_in_weight_degrees = calculated_values['non_reciprocated_in_weight_degrees']
        non_reciprocated_in_weight_degrees_eigenvector = calculated_values['non_reciprocated_in_weight_degrees_eigenvector']

        print "len average path lengths", average_path_lengths, "!!!!!", len(average_path_lengths)
        x = [i*time_interval for i in range(0, len(average_path_lengths))]

        print "avg path lengths", average_path_lengths, len(average_path_lengths)

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
        for counter,val in enumerate(outstrength_degrees):
            print counter,val
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
        make_graphs_for_val(x_simple_angle, eigenvector_centrality_degrees, time_interval, basegraph_name, 68,
                            '_graph_eigenvector_centrality_degree_', container_or_class, 'eigenvector centrality simple degree', 'angle')
        make_graphs_for_val(x, eigenvector_centrality_degrees_eigenvector, time_interval, basegraph_name, 71,
                            '_graph_eigenvector_centrality_eigenvector_degree_', container_or_class, 'eigenvector centrality eigenvector degree',
                            'angle')
        #make_graphs_for_val(x, clustering_degrees, time_interval, basegraph_name, 74,
        #                    '_graph_clustering_degree_', container_or_class, 'clustering simple degree', 'angle')
        #make_graphs_for_val(x, clustering_degrees_eigenvector, time_interval, basegraph_name, 77,
        #                    '_graph_clustering_degrees_eigenvector_degree_', container_or_class, 'clustering eigenvector degree',
        #                    'angle')
        make_graphs_for_val(x_simple_angle, betweeness_centrality_degrees, time_interval, basegraph_name, 80,
                            '_graph_betweeness_centrality_degree_', container_or_class, 'betweeness centrality degree', 'angle')
        make_graphs_for_val(x, betweeness_centrality_degrees_eigenvector, time_interval, basegraph_name, 83,
                            '_graph_betweeness_centrality_eigenvector_degree_', container_or_class, 'betweeness centrality eigenvector degree',
                            'angle')
        make_graphs_for_val(x_simple_angle, load_centrality_degrees, time_interval, basegraph_name, 86,
                            '_graph_load_centrality_degree_', container_or_class, 'load centrality degree', 'angle')
        make_graphs_for_val(x, load_centrality_degrees_eigenvector, time_interval, basegraph_name, 89,
                            '_graph_load_centrality_eigenvector_degree_', container_or_class, 'load centrality eigenvector degree',
                            'angle')
        make_graphs_for_val(x_simple_angle, non_reciprocated_out_weight_degrees, time_interval, basegraph_name, 92,
                            '_graph_non_reciprocated_out_weight_degree_', container_or_class, 'non-reciprocated outweight  degree', 'angle')
        make_graphs_for_val(x, non_reciprocated_out_weight_degrees_eigenvector, time_interval, basegraph_name, 95,
                            '_graph_non_reciprocated_out_weight_eigenvector_degree_', container_or_class, 'non-reciprocated outweight eigenvector degree',
                            'angle')
        make_graphs_for_val(x_simple_angle, non_reciprocated_in_weight_degrees, time_interval, basegraph_name, 98,
                            '_graph_non_reciprocated_in_weight_degree_', container_or_class, 'non-reciprocated inweight degree', 'angle')
        make_graphs_for_val(x, non_reciprocated_in_weight_degrees_eigenvector, time_interval, basegraph_name, 101,
                            '_graph_non_reciprocated_in_weight_eigenvector_degree_', container_or_class, 'non-reciprocated inweight eigenvector degree',
                            'angle')

    # okay, so later on I am going to want to group by class/node granularity via color
    # and by time granularity via spacing... so each time granularity should be a seperatae
    # list and each of the class/node granularites should be a nested list (inside the corresponding list)
    # right now: (time gran, node gran) -> metrics -> vals

    # todo: keep working on getting the multi-res boxplots (tho maybe check overlap with covariance matrix)
    node_grans = list(set(node_grans))
    time_grans = list(set(time_grans)).sort()
    # okay, so what I want to do here is (time gran, node gran, metric) -> vals
    # or do I want to do (metric) -> (nested lists in order of the things above?)
    # well to do the covrariance matrix I am going to need (1) but in order to ddo the boxplots
    # I am going to need to do (2)
    # b/c then I can easily index in later

    # okay, so later on I am going to want to group by class/node granularity via color
    # and by time granularity via spacing... so each time granularity should be a seperatae
    # list and each of the class/node granularites should be a nested list (inside the corresponding list)
    # so below: (metric) -> (time gran) -> (nested list of node grans)
    metrics_to_time_to_granularity_lists = {}
    fully_indexed_metrics = {}
    for metric in metrics:
        metrics_to_time_to_granularity_lists[metric] = {}
        for time_gran in time_grans:
            metrics_to_time_to_granularity_lists[metric][time_gran] = []
            for node_gran in node_grans:
                metrics_to_time_to_granularity_lists[metric][time_gran].append(  total_calculated_vals[(time_gran, node_gran )][metric] )
                fully_indexed_metrics[(time_gran, node_gran, metric)] = total_calculated_vals[(time_gran, node_gran )][metric]


    # okay, so now I actually need to handle make those multi-dimensional boxplots
    for metric in metrics:
        make_multi_time_boxplots(metrics_to_time_to_granularity_lists, time_grans, metric, colors, metric + '_multitime_boxplot')

    # todo: next thing on the todo list is to create two seperate covariance matrices (one w/ deltas, one w/ absolutes),
    # note that this means I'll need to do like weighted average or something for the normal vals (i.e. not angles)
    # okay, so I am probably going to want to do that in calc_convariance_matrix, and then just have it return two
    # values...

    print "about to make covariance matrix!"
    correlation_dataframe = calc_covaraiance_matrix(fully_indexed_metrics)
    print "made covariance matrix! Now time to plot it!"
    print "correlation dataframe"
    print correlation_dataframe
    plot_correlogram(correlation_dataframe, basegraph_name)

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
# note: tensor is really a list of dictionaries, with keys of nodes_in_tensor
# note: does not make sense for window_size to be less than 3
def eigenvector_analysis(tensor, window_size, nodes_in_tensor):
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
    for i in range( window_size, len(tensor) + 1):
        correlation_matrix = pandas.DataFrame(0.0, index=nodes_in_tensor, columns=nodes_in_tensor)
        pearson_p_val_matrix = pandas.DataFrame(0.0, index=nodes_in_tensor, columns=nodes_in_tensor)

        start_of_window =  i - window_size # no +1 b/c of the slicing
        # compute average window (with what we have available)
        print "start_of_window", start_of_window
        print "list slice window of tensor", tensor[start_of_window: i]
        tensor_window = tensor[start_of_window: i]

        # okay, now that we have the window, it is time to go through each pairing of nodes
        for node_one in nodes_in_tensor:
            for node_two in nodes_in_tensor:
                # compute pearson's rho of the corresponding time series
                node_one_time_series = [x[node_one] if node_one in x else 0 for x in tensor_window]
                node_two_time_series = [x[node_two] if node_two in x else 0 for x in tensor_window]

                print "node_one_time_series", node_one_time_series
                print "node_two_time_series", node_two_time_series

                ''' don't really need this anymore...
                with open('./' + 'debugging.txt', 'a') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow([node_one, node_one_time_series])
                    spamwriter.writerow([node_two, node_two_time_series])
                  '''

                pearson_rho = scipy.stats.pearsonr(node_one_time_series, node_two_time_series)
                print 'peasrson', pearson_rho, pearson_rho[0], node_one, node_two
                correlation_matrix.at[node_one, node_two] = pearson_rho[0]
                #print correlation_matrix
                pearson_p_val_matrix.at[node_one, node_two] = pearson_rho[1]
                #'''
                # todo: does this make sense????
                # note: this is a questionable edgecase. My reasoning is that
                # all the values are typically the same during each time interval, since
                # neither changes, we have no idea if there is (or isn't) a relation,
                # so to be safe let's say zero
                if math.isnan(pearson_rho[0]) and pearson_rho[1] == 1.0:
                    correlation_matrix.at[node_one, node_two] = 0.0
                else:
                    correlation_matrix.at[node_one, node_two] = pearson_rho[0]

                #'''
        print "correlation matrix\n", correlation_matrix
        correlation_matrices.append(correlation_matrix)
        p_value_matrices.append(pearson_p_val_matrix)

        eigen_vals, eigen_vects = scipy.linalg.eigh(correlation_matrix.values)
        # note: here we want the principal eigenvector, which is assocated with the
        # eigenvalue that has the largest magnitude
        print "eigenvalues", eigen_vals
        largest_mag_eigenvalue = max(eigen_vals, key=abs)
        print "largest_mag_eigenvalue", largest_mag_eigenvalue
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

        print "eigenvectors", eigen_vects
        print "principal eigenvector", eigen_vects[largest_mag_eigenvalue_index]
        correlation_matrix_eigenvectors.append(eigen_vects[largest_mag_eigenvalue_index])

    print "correlation eigenvects", correlation_matrix_eigenvectors
    angles = find_angles(correlation_matrix_eigenvectors, window_size)

    return angles

def find_angles(list_of_vectors, window_size):

    angles = [] # first must be zero (nothing to compare to)
    for i in range(window_size, len(list_of_vectors)):
        start_of_window = i - window_size
        # compute average window (with what we have available)
        print "list slice window", list_of_vectors[start_of_window: i]
        window_average = np.mean([x for x in list_of_vectors[start_of_window: i] if x != []], axis=0)
        #print "start of window", start_of_window, "window average", window_average

        # to compare angles, we should use unit vectors (and then calc the angle)
        # from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
        window_average_unit_vector = window_average / np.linalg.norm(window_average)
        current_value_unit_vector = list_of_vectors[i] / np.linalg.norm(list_of_vectors[i])
        #print "window_average_unit_vector", window_average_unit_vector, "current_value_unit_vector", current_value_unit_vector

        # sometimes the first time interval has no activity
        if window_average_unit_vector.size == 0:
            window_average_unit_vector = np.zeros(len(current_value_unit_vector))

        print "window_average_unit_vector", window_average_unit_vector
        print "current_value_unit_vector", current_value_unit_vector
        angle = np.arccos( np.clip(np.dot(window_average_unit_vector, current_value_unit_vector), -1.0, 1.0)  )
        angles.append(angle)

    return angles

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
                current_nodes.append(0.0)  # the current dict must not have an entry for node -> zero val
        node_vals.append(current_nodes)
        # print "degree angles", node_degrees
    return node_vals

def make_graphs_for_val(x_vals, y_vals, time_interval, basegraph_name, fig_num, graph_name_extenstion,
                        container_or_class, graph_tile, y_axis_label):

    #y_vals = ast.literal_eval(y_vals)
    plt.figure(fig_num)
    plt.clf()
    plt.title(graph_tile + ', ' + '%.2f' % (time_interval))
    plt.ylabel(y_axis_label)
    plt.xlabel('time (sec)')
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
def plot_correlogram(correlation_matrix, basegraph_name):

    # based off of example located at: https://seaborn.pydata.org/generated/seaborn.heatmap.html
    #don't think is needed: flights = sns.load_dataset("flights")
    #don't think is needed: flights = flights.pivot("month", "year", "passengers")

    ax = sns.heatmap(correlation_matrix)
    ax.savefig(basegraph_name + 'correlation_heatmap' + '.png', format='png')

    # going to drop NaN's beforep plotting
    ax2 = sns.pairplot(correlation_matrix.dropna()) # note: I hope dropna() doesn't mess the alignment up but it might
    ax2.savefig(basegraph_name + 'correlation_pairplot' + '.png', format='png')


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
    covariance_matrix_input = pandas.DataFrame(dict([ (k,pandas.Series(v)) for k,v in calculated_values.iteritems() ]))
    #covariance_matrix_input = pandas.DataFrame(calculated_values)
    print "covariance_matrix_input", covariance_matrix_input
    print covariance_matrix_input.shape
    # todo: is this square? ^^^ I think not b/c some values (computed via the vector/angle thingee) would
    # be missing some vals, compared to the simple graph-wide metrics
    # (so might wanna either pad or remove...)
    # NOTE: with the modification to using corr(), i don't think it needs to be the same angle anymore...

    # must transpose b/c corr() finds covariance between columns, so it follows that each
    # column should have a seperate variable
    covariance_dataframe = covariance_matrix_input.corr()
    print covariance_dataframe.shape

    return covariance_dataframe

# in the style of: https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots
def make_multi_time_boxplots(metrics_to_time_to_granularity_lists, time_grans, metric, colors, graph_name):
    fig = plt.figure()
    fig.clf()
    ax = plt.axis()

    cur_pos = 1
    tick_position_list = []
    for time_gran in time_grans:
        number_nested_lists = len(metrics_to_time_to_granularity_lists[metric][time_gran])
        number_positions_on_graph = range(cur_pos, cur_pos+number_nested_lists)
        tick_position = (float(number_positions_on_graph[0]) + float(number_positions_on_graph[-1])) / number_nested_lists
        tick_position_list.append( tick_position )
        bp = plt.boxplot(metrics_to_time_to_granularity_lists[metric][time_gran], positions = number_positions_on_graph, widths = 0.6)
        cur_pos += number_nested_lists + 1 # the +1 is so that there is extra space between the groups
        set_boxplot_colors(bp, colors)

    ax.xlim(0, cur_pos)
    ax.set_xticklabels([str(i) for i in time_grans])
    ax.set_xticks(tick_position_list)

    invisible_lines = []
    for color in colors:
        cur_line, = plt.plot([1,1], color)
        invisible_lines.append(cur_line.copy())
    plt.legend(invisible_lines, colors)
    for line in invisible_lines:
        line.set_visible(False)

    plt.savefig(graph_name + '.png', format='png')

def set_boxplot_colors(bp, colors):
    for counter, color in enumerate(colors):
        plt.setp(bp['boxes'][counter], color=color)
        plt.setp(bp['caps'][counter * 2], color=color)
        plt.setp(bp['caps'][counter * 2 + 1], color=color)
        plt.setp(bp['whiskers'][counter * 2], color=color)
        plt.setp(bp['whiskers'][counter * 2 + 1], color=color)
        plt.setp(bp['fliers'][counter * 2 ], color=color)
        plt.setp(bp['fliers'][counter * 2 + 1], color=color)
        plt.setp(bp['medians'][counter], color=color)

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