#######################
# NOTE: I really really really do not like this file.
# try to adapt the simplified_pipeline as needed, b/c this is horrible.
#
#
#
#########################



import networkx as nx
import seaborn as sns;

from analysis_pipeline.prepare_graph import prepare_graph, get_svc_equivalents

sns.set()
import scipy.stats
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import math
import scipy
import scipy.stats
import scipy.sparse.linalg
import pandas
import csv
import ast
import logging
import itertools
import gc
#from pympler import asizeof
import sys
from analysis_pipeline.next_gen_metrics import calc_neighbor_metric,generate_neig_dict,create_dict_for_dns_metric,\
    calc_dns_metric,find_angles,turn_into_list,calc_outside_inside_ratio_dns_metric

# okay, so things to be aware of:
# (a) we are assuming that if we cannot label the node and it is not loopback or in the '10.X.X.X' subnet, then it is outside

def pipeline_analysis_step(filenames, ms_s, time_interval, basegraph_name, calc_vals_p, window_size, container_to_ip,
                           is_swarm, make_net_graphs_p, infra_service):
    total_calculated_values = {}
    '''
    list_of_graphs = []
    list_of_aggregated_graphs = [] # all nodes of the same class aggregated into a single node
    list_of_aggregated_graphs_multi = [] # the above w/ multiple edges
    list_of_unprocessed_graphs = []
    counter= 0 # let's not make more than 50 images of graphs (per time_interval)

    svcs = get_svc_equivalents(is_swarm, container_to_ip)
    print "these services were found:", svcs
    if make_net_graphs_p or calc_vals_p:
        for file_path in filenames:
            G = nx.DiGraph()
            print "path to file is ", file_path
            nx.read_edgelist(file_path,
                            create_using=G, delimiter=',', data=(('weight', float),))

            # want an 'unprocessed' (just raw from pcaps) and a 'processed' graph (where I use orchestrator-specific knowledge)
            unprocessed_G = G.copy()
            print "going to use these services to do the mapping", svcs
            containers_to_ms = map_nodes_to_svcs(unprocessed_G, svcs)
            print "container to service mapping: ", containers_to_ms
            nx.set_node_attributes(unprocessed_G, containers_to_ms, 'svc')

            list_of_unprocessed_graphs.append(unprocessed_G)
            if is_swarm:
                mapping = {'192.168.99.1': 'outside'}
                try:
                    nx.relabel_nodes(unprocessed_G, mapping, copy=False)
                except KeyError:
                    pass  # maybe it's not in the graph?
            else:
                pass # note: I am not really processing the k8s case anyway (except to consolidate the outside nodes)

            if counter < 50: # keep # of network graphs to a reasonable amount
                filename = file_path.replace('.txt', '') + 'unprocessed_network_graph_container.png'
                print "about to make net graph w/ filename", filename
                make_network_graph(unprocessed_G, edge_label_p=True, filename=filename, figsize=(54,32), node_color_p=False,
                                   ms_s=ms_s)

            G,svcs = process_graph(G, is_swarm, container_to_ip, ms_s)
            if counter < 50: # keep # of network graphs to a reasonable amount
                filename = file_path.replace('.txt', '') + '_network_graph_container.png'
                make_network_graph(G, edge_label_p=True, filename=filename, figsize=(54, 32), node_color_p=False,
                                   ms_s=ms_s)
            list_of_graphs.append(G)

            aggreg_multi_G, aggreg_simple_G = aggregate_graph(G, ms_s)
            list_of_aggregated_graphs.append( aggreg_simple_G )
            list_of_aggregated_graphs_multi.append( aggreg_multi_G )
            if counter < 50:
                filename = file_path.replace('.txt', '') + '_network_graph_class.png'
                make_network_graph(aggreg_simple_G, edge_label_p=True, filename=filename, figsize=(16,10), node_color_p=False,
                                   ms_s=ms_s)
            counter += 1
    '''
    if is_swarm:
        svcs = get_svc_equivalents(is_swarm, container_to_ip)
    else:
        print "this is k8s, so using these sevices", ms_s
        svcs = ms_s

    total_calculated_values[(time_interval, 'container')] = calc_graph_metrics(filenames, time_interval,
                                                                               basegraph_name + '_container_', 'container',
                                                                               calc_vals_p, window_size, 'app_only',
                                                                               ms_s, container_to_ip, is_swarm, svcs, infra_service)

    total_calculated_values[(time_interval, 'class')] = calc_graph_metrics(filenames, time_interval,
                                                                           basegraph_name + '_class_', 'class', calc_vals_p,
                                                                           window_size, 'class', ms_s, container_to_ip,
                                                                           is_swarm, svcs, infra_service)

    total_calculated_values[(time_interval, 'unprocessed_container')] = calc_graph_metrics(filenames,
                                                        time_interval, basegraph_name + '_unprocessed_container_',
                                                       'unprocessed_container', calc_vals_p, window_size, 'none',
                                                        ms_s, container_to_ip, is_swarm, svcs, infra_service)
    ''' # todo: re-enable
    if is_swarm:
        print "about to calculate the expected structural characteristics for docker swarm..."
        total_calculated_values[(time_interval, 'container')].update(calc_service_specific_graph_metrics(list_of_graphs,
                                        svcs, calc_vals_p, basegraph_name + '_container_', 'container', time_interval))
        # let's also calculate these specific graph metrics for the raw graphs
        #total_calculated_values[(time_interval, 'unprocessed_container')].update(calc_service_specific_graph_metrics(list_of_unprocessed_graphs,
        #                                svcs, calc_vals_p, basegraph_name + '_unprocessed_container_', 'unprocessed_container', time_interval))
    #'''

    return total_calculated_values

# gotta keep the order of the nodes consistent when converting from a dictionary to a list
def get_total_node_list():
    pass

def calc_graph_metrics(filenames, time_interval, basegraph_name, container_or_class, calc_vals_p, window_size,
                       level_of_processing, ms_s, container_to_ip, is_swarm, svcs, infra_service):

    if calc_vals_p:
        average_path_lengths = []
        densities = []
        degree_dicts = []
        weighted_average_path_lengths = []
        unweighted_overall_reciprocities = [] # defined per networkx definition (see their docs)
        weighted_reciprocities = [] # defined per the nature paper (see comment @ function definition)
        outstrength_dicts = []
        instrength_dicts = []
        eigenvector_centrality_dicts = []
        betweeness_centrality_dicts = []
        load_centrality_dicts = []
        non_reciprocated_out_weight_dicts = []
        non_reciprocated_in_weight_dicts = []
        pod_comm_but_not_VIP_comms = []
        fraction_pod_comm_but_not_VIP_comms = []
        pod_comm_but_not_VIP_comms_no_abs = []
        fraction_pod_comm_but_not_VIP_comms_no_abs = []
        neighbor_dicts = []
        dns_metric_dicts = []
        '''
        total_node_list = []
        for cur_g in G_list:
            for node in cur_g.nodes():
                total_node_list.append(node)
        total_node_list = list(set(total_node_list))
        '''

        current_total_node_list = []

        #for cur_G in G_list:
        for counter, file_path in enumerate(filenames):
            gc.collect()
            G = nx.DiGraph()
            print "path to file is ", file_path
            nx.read_edgelist(file_path,
                            create_using=G, delimiter=',')
            cur_G = prepare_graph(G, svcs, level_of_processing, is_swarm, counter, file_path, ms_s, container_to_ip, infra_service)
            #print "right after graph is prepared", level_of_processing, list(cur_G.nodes(data=True))

            for node in cur_G.nodes():
                if node not in current_total_node_list:
                    current_total_node_list.append(node)

            neighbor_dicts.append( generate_neig_dict(cur_G) )
            dns_metric_dicts.append( create_dict_for_dns_metric(G, 'kube-dns_VIP') )

            #print "right before calc_VIP_metric", level_of_processing
            pod_comm_but_not_VIP_comm, fraction_pod_comm_but_not_VIP_comm = calc_VIP_metric(cur_G, True)
            pod_comm_but_not_VIP_comms.append(pod_comm_but_not_VIP_comm)
            fraction_pod_comm_but_not_VIP_comms.append(fraction_pod_comm_but_not_VIP_comm)
            pod_comm_but_not_VIP_comm_no_abs, fraction_pod_comm_but_not_VIP_comm_no_abs = calc_VIP_metric(cur_G, False)
            pod_comm_but_not_VIP_comms_no_abs.append( pod_comm_but_not_VIP_comm_no_abs )
            fraction_pod_comm_but_not_VIP_comms_no_abs.append( fraction_pod_comm_but_not_VIP_comm_no_abs )

            # TODO: remove lower line!!
            #continue

            # okay, so this is where to calculate those metrics from the excel document
            # first, let's do the graph-wide metrics (b/c it is simple) (these are only single values)
            try:
                avg_path_length = nx.average_shortest_path_length(cur_G) #
            except:
                avg_path_length = float('nan')
                # note: I could theoretically do something different here (e.g. calc for each subgraph and then
                # add together in a weighted manner, but I think I'll just do the easier method of placing a 'nan')

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
                # "computes the centrality of a node based on the centrality
                # of its neighbors"

            try:
                betweeness_centrality_dict = nx.betweenness_centrality(cur_G)
            except nx.NetworkXPointlessConcept:
                betweeness_centrality_dict = None
                # if graph is Null, then this metric is meaningless
                # "the sum of the fraction of all-pairs shortest paths that
                # pass through that node"

            try:
                load_centrality_dict = nx.load_centrality(cur_G)
            except nx.NetworkXPointlessConcept:
                load_centrality_dict = None
                # if graph is Null, then this metric is meaningless
                # "the fraction of all shortest paths that pass through that node":

            weighted_reciprocity, non_reciprocated_out_weight_dict, non_reciprocated_in_weight_dict = network_weidge_weighted_reciprocity(cur_G)
            outstrength_dicts.append( outstrength_dict )
            instrength_dicts.append( instrength_dict )
            eigenvector_centrality_dicts.append( eigenvector_centrality_dict )
            betweeness_centrality_dicts.append( betweeness_centrality_dict )
            load_centrality_dicts.append( load_centrality_dict )
            non_reciprocated_out_weight_dicts.append( non_reciprocated_out_weight_dict )
            non_reciprocated_in_weight_dicts.append( non_reciprocated_in_weight_dict )
            weighted_reciprocities.append(weighted_reciprocity)
            average_path_lengths.append(avg_path_length)
            densities.append(density)
            degree_dicts.append(degree_dict)
            print "size of betweeness centrality dicts", asizeof.asized(betweeness_centrality_dicts)
            print "size_of_cur_G", sys.getsizeof(cur_G.edges) + sys.getsizeof(cur_G.nodes)

            # TODO: would probably be a good idea to store these vals somewhere safe or something (I think
            # the program is holding all of the graphs in memory, which is leading to massive memory bloat)
            del G
            del cur_G

        #print "degrees", degree_dicts
        #print "weighted recips", weight_recips
        #print "degree dicts", degree_dicts

        #######

        print "About to perform vector-angle analysis_pipeline methods (i.e. DOING ANGLES)"

        # out degrees analysis_pipeline
        node_degrees = turn_into_list(degree_dicts, current_total_node_list)
        angles_degrees = find_angles(node_degrees, window_size) #change_point_detection(degree_dicts, window_size=window_size)  # setting window size arbitrarily for now...
        #print "angles degrees", type(angles_degrees), angles_degrees, node_degrees
        angles_degrees_eigenvector = change_point_detection(degree_dicts, window_size, current_total_node_list)
        #print "angles degrees eigenvector", angles_degrees_eigenvector

        # outstrength analysis_pipeline
        node_outstrengths = turn_into_list(outstrength_dicts, current_total_node_list)
        print "node_outstrengths", node_outstrengths
        outstrength_degrees = find_angles(node_outstrengths, window_size)
        outstrength_degrees_eigenvector = change_point_detection(outstrength_dicts, window_size, current_total_node_list)

        # instrength analysis_pipeline
        node_instrengths = turn_into_list(instrength_dicts, current_total_node_list)
        print "node_instrengths", node_instrengths
        instrengths_degrees = find_angles(node_instrengths, window_size)
        instrengths_degrees_eigenvector = change_point_detection(instrength_dicts, window_size, current_total_node_list)

        # eigenvector centrality analysis_pipeline
        node_eigenvector_centrality = turn_into_list(eigenvector_centrality_dicts, current_total_node_list)
        eigenvector_centrality_degrees = find_angles(node_eigenvector_centrality, window_size)
        eigenvector_centrality_degrees_eigenvector = change_point_detection(eigenvector_centrality_dicts, window_size, current_total_node_list)

        # betweeness centrality analysis_pipeline
        node_betweeness_centrality = turn_into_list(betweeness_centrality_dicts, current_total_node_list)
        betweeness_centrality_degrees = find_angles(node_betweeness_centrality, window_size)
        betweeness_centrality_degrees_eigenvector = change_point_detection(betweeness_centrality_dicts, window_size, current_total_node_list)

        # load centrality analysis_pipeline
        node_load_centrality = turn_into_list(load_centrality_dicts, current_total_node_list)
        load_centrality_degrees = find_angles(node_load_centrality, window_size)
        load_centrality_degrees_eigenvector = change_point_detection(load_centrality_dicts, window_size, current_total_node_list)

        # non_reciprocated_out_weight analysis_pipeline
        node_non_reciprocated_out_weight = turn_into_list(non_reciprocated_out_weight_dicts, current_total_node_list)
        non_reciprocated_out_weight_degrees = find_angles(node_non_reciprocated_out_weight, window_size)
        non_reciprocated_out_weight_degrees_eigenvector = change_point_detection(non_reciprocated_out_weight_dicts, window_size, current_total_node_list)

        # non_reciprocated_in_weight analysis_pipeline
        node_non_reciprocated_in_weight = turn_into_list(non_reciprocated_in_weight_dicts, current_total_node_list)
        non_reciprocated_in_weight_degrees = find_angles(node_non_reciprocated_in_weight, window_size)
        non_reciprocated_in_weight_degrees_eigenvector = change_point_detection(non_reciprocated_in_weight_dicts, window_size, current_total_node_list)

        # TODO TODO TODO
        # which_nodes can be ['all', 'outside', or 'kube-dns_VIP']
        # let's make the training time 5 minutes
        size_of_training_window = (5 * 60) / time_interval
        num_new_neighbors_outside = calc_neighbor_metric(neighbor_dicts, size_of_training_window, 'outside')
        num_new_neighbors_dns = calc_neighbor_metric(neighbor_dicts, size_of_training_window, 'kube-dns_VIP')
        num_new_neighbors_all = calc_neighbor_metric(neighbor_dicts, size_of_training_window, 'all')

        dns_angles = calc_dns_metric(dns_metric_dicts, current_total_node_list, window_size)
        dns_outside_inside_ratios,_,_ = calc_outside_inside_ratio_dns_metric(dns_metric_dicts)

        appserver_sum_degrees = []
        for degree_dict in degree_dicts:
            appserver_degrees = []
            for key,val in degree_dict.iteritems():
                if 'appserver' in key:
                    appserver_degrees.append(val)
            appserver_sum_degrees.append( np.mean(appserver_degrees))

        #########

        calculated_values = {}

        calculated_values['New Class-Class Edges'] = num_new_neighbors_all
        calculated_values['New Class-Class Edges with Outside'] = num_new_neighbors_outside
        calculated_values['New Class-Class Edges with DNS'] = num_new_neighbors_dns
        calculated_values['Angle of DNS edge weight vectors'] = dns_angles
        calculated_values['Fraction of Communication Between Pods not through VIPs (no abs)'] = fraction_pod_comm_but_not_VIP_comms_no_abs
        calculated_values['Communication Between Pods not through VIPs (no abs)'] = pod_comm_but_not_VIP_comms_no_abs
        calculated_values['DNS outside-to-inside ratio'] = dns_outside_inside_ratios

        # abs values:
        calculated_values['Unweighted Average Path Length'] = average_path_lengths
        calculated_values['Weighted Average Path Length'] = weighted_average_path_lengths
        calculated_values['Unweighted Overall Reciprocity'] = unweighted_overall_reciprocities
        calculated_values['Weighted Overall Reciprocity'] = weighted_reciprocities
        calculated_values['Density'] = densities
        calculated_values['Sum of Appserver Node Degrees'] = appserver_sum_degrees
        calculated_values['Fraction of Communication Between Pods not through VIPs'] = fraction_pod_comm_but_not_VIP_comms
        calculated_values['Communication Between Pods not through VIPs'] = pod_comm_but_not_VIP_comms
        # delta values:
        calculated_values['Simple Angle Between Node Degree Vectors'] = angles_degrees
        calculated_values['Change-Point Detection Node Degree'] = angles_degrees_eigenvector
        calculated_values['Simple Angle Between Node Outstrength Vectors'] = outstrength_degrees
        calculated_values['Change-Point Detection Node Outstrength']= outstrength_degrees_eigenvector
        calculated_values['Simple Angle Between Node Instrength Vectors'] = instrengths_degrees
        calculated_values['Change-Point Detection Node Instrength'] = instrengths_degrees_eigenvector
        calculated_values['Simple Angle Between Node Eigenvector_Centrality Vectors'] = eigenvector_centrality_degrees
        calculated_values['Change-Point Detection Node Eigenvector_Centrality'] = eigenvector_centrality_degrees_eigenvector
        calculated_values['Simple Angle Between Node Betweeness Centrality Vectors'] = betweeness_centrality_degrees
        calculated_values['Change-Point Detection Node Betweeness Centrality'] = betweeness_centrality_degrees_eigenvector
        calculated_values['Simple Angle Between Node Load Centrality Vectors'] = load_centrality_degrees
        calculated_values['Change-Point Detection Node Load Centrality'] = load_centrality_degrees_eigenvector
        calculated_values['Simple Angle Between Node Non-Reciprocated Out-Weight Vectors'] = non_reciprocated_out_weight_degrees
        calculated_values['Change-Point Detection Node Non-Reciprocated Out-Weight'] = non_reciprocated_out_weight_degrees_eigenvector
        calculated_values['Simple Angle Between Node Non-Reciprocated In-Weight'] = non_reciprocated_in_weight_degrees
        calculated_values['Change-Point Detection Node Non-Reciprocated In-Weight'] = non_reciprocated_in_weight_degrees_eigenvector

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
                calculated_values[row[0]] = [i if i != (None) else float('nan') for i in ast.literal_eval(row[1])]
                print row[0], calculated_values[row[0]]

    return calculated_values


# returns list of angles (of size len(tensor)). Note:
# i have decided to append window_size 'nans' to the front of the list of
# angles (so that the graphing goes easier b/c I won't have to worry about shifting stuff)
# (b/c the first window_size angles do not exist in a meaningful way...) -
# note: tensor is really a *list of dictionaries*, with keys of nodes_in_tensor
### NOTE: have this function be at least >4 if you want results that are behave coherently, ###
### though >= 6 is best ###
def change_point_detection(tensor, window_size, nodes_in_tensor):
    print "len(tensor)", len(tensor)
    if window_size < 3:
        # does not make sense for window_size to be less than 3
        print "window_size needs to be >= 3 for pearson to work"
        exit(3)
    if tensor == []:
        return []

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
            #print "cur_tensor", cur_tensor, type(cur_tensor)
            if cur_tensor != []:
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

                #print "node_one", node_one, "tensor_window", tensor_window
                node_one_time_series = np.array([x[node_one] if x and node_one in x else float('nan') for x in tensor_window])
                node_two_time_series = np.array([x[node_two] if x and node_two in x else float('nan') for x in tensor_window])


                #print "node_one_time_series", node_one_time_series
                #print "node_two_time_series", node_two_time_series

                # remove Nan's from array before doing pearson analysis_pipeline
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

        #print "eigenvectors", eigen_vects
        print "principal eigenvector", eigen_vects.T[largest_mag_eigenvalue_index]
        correlation_matrix_eigenvectors.append(eigen_vects.T[largest_mag_eigenvalue_index])

    #for correlation_matrix in correlation_matrices:
    #    print correlation_matrix.values, '\n'
    print "correlation eigenvects", correlation_matrix_eigenvectors
    angles = find_angles(correlation_matrix_eigenvectors, window_size)

    # note: padding front so that alignment is maintained
    for i in range(0, window_size - 1): # first window_size values becomes one value, hence want to add bakc window_size -1 vals
        angles.insert(0, float('nan'))

    return angles

# from : https://www.andrew.cmu.edu/user/lakoglu/pubs/EVENTDETECTION_AkogluFaloutsos.pdf
# returns list of angles (of size len(tensor)). Note:
# i have decided to append window_size 'nans' to the front of the list of
# angles (so that the graphing goes easier b/c I won't have to worry about shifting stuff)
# (b/c the first window_size angles do not exist in a meaningful way...) -
### NOTE: tensor is a list of adjacency dataframes
### NOTE: have this function be at least >4 if you want results that are behave coherently, ###
### though >= 6 is best ###
### NOTE: this relies on the the tensors being ordered the same (b/c otherwise the direction will swing around for
### no good reason)
def ide_angles(tensor, window_size, nodes_in_tensor):
    print "len(tensor)", len(tensor)
    if tensor == []:
        return []

    adjacency_matrix_eigenvectors = []
    # let's iterate through the times, pulling out slices that correspond to windows
    ####smallest_slice = 3 # 2 is guaranteed to get a pearson value of 1, even smaller breaks it
    for i in range( 0, len(tensor)):
        #start_of_window =  i - window_size # no +1 b/c of the slicing
        # compute average window (with what we have available)
        #print "start_of_window", start_of_window
        #print "list slice window of tensor", tensor[start_of_window: i]
        #tensor_window = tensor[start_of_window: i]

        adjacency_matrix = tensor[i]
        eigen_vals, eigen_vects = scipy.linalg.eig(adjacency_matrix.values)
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

        #print "eigenvectors", eigen_vects
        print "principal eigenvector", eigen_vects.T[largest_mag_eigenvalue_index]
        adjacency_matrix_eigenvectors.append(eigen_vects.T[largest_mag_eigenvalue_index])

    ## NOTE: OKAY, below here is fine. The key is load up the correlation_matrix_eeigenvectors (might want to change
    ## the name) with the wieghted adjacency matrix)

    #for correlation_matrix in correlation_matrices:
    #    print correlation_matrix.values, '\n'
    print "correlation eigenvects", adjacency_matrix_eigenvectors
    angles = find_angles(adjacency_matrix_eigenvectors, window_size)

    # note: padding front so that alignment is maintained
    #for i in range(0, window_size): # first window_size values becomes one value, hence want to add bakc window_size -1 vals
    #    angles.insert(0, float('nan'))

    return angles#, adjacency_matrix_eigenvectors


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
                #print "len edges",  node_to_nodeTwo, nodeTwo_to_node
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

    # I want to remove all of the simple angle analysis_pipeline here (but keep the
    # eigenvector analysis_pipeline!)
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


# okay, so this is a metric that applies only to kubernetes
# it says that if a container in service X sends data to a container in service Y
# then that data SHOULD go through either the VIP of X or VIP of Y (b/c NATing)
# so taking the DIFFERENCE has the potential to be a USEFUL metric
def calc_VIP_metric(G, abs_val_p):
    # okay, for now let's do a relatively simple, naive implementation
    pod_to_containers_in_other_svc = {}
    service_VIP_and_pod_comm = {} # in either direction (each direction seperately tho)
    print "calc_VIP_metric"
    attribs = nx.get_node_attributes(G, 'svc')
    logging.info("calc_VIP_attribs, " + str(attribs))
    for (node1, node2, data) in G.edges(data=True):
        #print node1#, nx.get_node_attributes(G, node1)
        #or node1 in G.nodes(data=True):
        #    for node2 in G.nodes(data=True):
        #        if node1 != node2:
        try:
            service1 = attribs[node1]
            service2 = attribs[node2] #node2[1]['svc']
            #print "calc_VIP_metric, svc to svc:", service1, service2
            #print node1, node2, '\n'

            data = data['weight']

            #print "services found!"

            if '_VIP' in node1 and '_VIP' in node2:
                print (node1, node2, data)
                print 'VIPs communicating??'
                exit(10)

            if '_VIP' in node1:
                service_VIP_and_pod_comm[service1, node2] = data
            elif '_VIP' in node2:
                service_VIP_and_pod_comm[node1, service2] = data
            else:
                # okay, so now we now it is pod-to-pod communication
                # I think I want to include both getting and recieving?? (so double-counting to a certain extent)
                #print "pod_to_containers_going", data
                if (node1, service2) in pod_to_containers_in_other_svc.keys():
                    #print "pod_to_containers entry found to exist"
                    pod_to_containers_in_other_svc[node1, service2] += data
                    #pod_to_containers_in_other_svc[node1, node2] += data
                else:
                    #print "pod_to_containers entry found to NOT exist"
                    pod_to_containers_in_other_svc[node1, service2] = data
                    #pod_to_containers_in_other_svc[node1, node2] = data

                if (service1, node2) in pod_to_containers_in_other_svc.keys():
                    #print "pod_to_containers entry2 found to exist"
                    pod_to_containers_in_other_svc[service1, node2] += data
                    #pod_to_containers_in_other_svc[node1, node2] += data
                else:
                    #print "pod_to_containers entry2 found to NOT exist"
                    pod_to_containers_in_other_svc[service1, node2] = data
                    #pod_to_containers_in_other_svc[node1, node2] = data

        except Exception as e:
            logging.info("calc_VIP_metric exception flagged!, " + str(node1) + ' ' + str(node2)+ ' ' + str(e))
    logging.info("service_VIP_and_pod_comm", service_VIP_and_pod_comm)
    logging.info("pod_to_containers_in_other_svc", pod_to_containers_in_other_svc)

    difference_between_pod_and_VIP = {}
    # okay, so now I'd like to calculate the difference.
    for comm_pair, bytes in service_VIP_and_pod_comm.iteritems():
        src = comm_pair[0]
        dest = comm_pair[1]
        logging.info(comm_pair)

        #try:
        pod_to_service_VIP = service_VIP_and_pod_comm[src,dest]
        #except:
        #    pod_to_service_VIP = 0
        #print pod_to_service_VIP

        try:
            pod_to_container =  pod_to_containers_in_other_svc[src,dest]
        except:
            # this is something that can happen (tho should only happen rarely)
            pod_to_container = 0
        #print pod_to_container


        difference_between_pod_and_VIP[src,dest] = pod_to_service_VIP - pod_to_container
    logging.info("difference_between_pod_and_VIP", difference_between_pod_and_VIP)
    total_difference_between_pod_and_VIP = 0
    for pair, data in difference_between_pod_and_VIP.iteritems():
        if abs_val_p:
            total_difference_between_pod_and_VIP += abs(data)
        else:
            total_difference_between_pod_and_VIP += data
        if abs(data) > 0:
            logging.info("pod_VIP_difference_not_zero " +  str(pair) + ' ' + str(data))
    sum_of_all_pod_to_container = sum(i for i in pod_to_containers_in_other_svc.values())
    logging.info("total_difference_between_pod_and_VIP, " +  str(total_difference_between_pod_and_VIP) + ', ' +
                 "total pod_to_container, " +  str(sum_of_all_pod_to_container))
    if sum_of_all_pod_to_container > 0:
        fraction_of_total_difference_between_pod_and_VIP = float(total_difference_between_pod_and_VIP) / sum_of_all_pod_to_container
    else:
        fraction_of_total_difference_between_pod_and_VIP = float('NaN')
    return total_difference_between_pod_and_VIP, fraction_of_total_difference_between_pod_and_VIP

# calc_service_specific_graph_metrics : ??
# this function calculates some graph metrics that are specific to each service or pair of services. These metrics are
# 'spread' of weight frm (svc,svc) traffic (direction matters) and 'spread' of in/out ratio of each container in the svc
# note: spead = (max(vals) - min(vals)) / (max(vals)), where vals is a list of values at that time stamp
def calc_service_specific_graph_metrics(G_list, svcs, calc_vals_p, basegraph_name, container_or_class, time_interval):
    if  calc_vals_p:
        calculated_values = {}
        map_svc_connection_to_weight_spread = {} # (svc_or_lb, svc_or_lb) -> list of integers (weight)
        map_svc_to_in_out_spread = {}

        for cur_G in G_list:
            map_svc_connection_to_weight_at_this_timestamp = {}
            map_svc_to_container_to_in = {}  # (svc_or_lb) -> (container) ->  list of integer (in weight)
            map_svc_to_container_to_out = {}  # (svc_or_lb) -> (container) -> list of integers (out weight)
            svc_to_list_of_in_out_ratio = {}  # (svc) -> [in_out_container1, in_out_container2, in_out_container3, ...]

            for svc in svcs:
                map_svc_to_container_to_in[svc] = {}
                map_svc_to_container_to_out[svc] = {}

            cur_G_src_attribs = nx.get_node_attributes(cur_G, 'svc')
            print "cur_G", cur_G
            for (u, v, data) in cur_G.edges(data=True):
                print (u,v,data)
                print "cur_G_src_attribs", cur_G_src_attribs
                if (cur_G_src_attribs[u],cur_G_src_attribs[v]) not in map_svc_connection_to_weight_at_this_timestamp:
                    map_svc_connection_to_weight_at_this_timestamp[(cur_G_src_attribs[u],cur_G_src_attribs[v])] = []
                map_svc_connection_to_weight_at_this_timestamp[(cur_G_src_attribs[u],cur_G_src_attribs[v])].append(data['weight'])

                if v in map_svc_to_container_to_in[cur_G_src_attribs[v]]:
                    map_svc_to_container_to_in[cur_G_src_attribs[v]][v] += data['weight']
                else:
                    map_svc_to_container_to_in[cur_G_src_attribs[v]][v] = data['weight']
                if u in map_svc_to_container_to_out[cur_G_src_attribs[u]]:
                    map_svc_to_container_to_out[cur_G_src_attribs[u]][u] += data['weight']
                else:
                    map_svc_to_container_to_out[cur_G_src_attribs[u]][u] = data['weight']
            for key_svc in list(set(map_svc_to_container_to_in.keys() + map_svc_to_container_to_out.keys())):
                for key_container in list(set(map_svc_to_container_to_in[key_svc].keys() +  map_svc_to_container_to_out[key_svc].keys() )):
                    if key_svc not in svc_to_list_of_in_out_ratio.keys():
                        svc_to_list_of_in_out_ratio[key_svc] = []
                    try:
                        current_in_val = map_svc_to_container_to_in[key_svc][key_container]
                        current_out_val = map_svc_to_container_to_out[key_svc][key_container]
                        svc_to_list_of_in_out_ratio[key_svc].append(current_in_val / current_out_val)
                    except:
                        svc_to_list_of_in_out_ratio[key_svc].append(float('NaN'))

            print "map_svc_connection_to_weight_spread", map_svc_connection_to_weight_spread
            for key,val in map_svc_connection_to_weight_at_this_timestamp.iteritems():
                if key not in map_svc_connection_to_weight_spread:
                    map_svc_connection_to_weight_spread[key] = []
                # I want some value that will capture the spread of the data in a single number
                spread = (max(val) - min(val)) / float(max(val))
                map_svc_connection_to_weight_spread[key].append(spread)
            for key,val in svc_to_list_of_in_out_ratio.iteritems():
                if key not in map_svc_to_in_out_spread:
                    map_svc_to_in_out_spread[key] = []
                spread = (max(val) - min(val)) / float(max(val))
                map_svc_to_in_out_spread[key].append(spread)

        calculated_values.update(map_svc_connection_to_weight_spread)
        calculated_values.update(map_svc_to_in_out_spread)

        with open(basegraph_name + '_processed_service_specific_vales_' + container_or_class + '_' + '%.2f' % (time_interval) + '.txt', 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for value_name, value in calculated_values.iteritems():
                if len(value) < 120000: # if larger than that, will not be able to read it in
                    spamwriter.writerow([value_name, [i if not math.isnan(i) else (None) for i in value]])
                else:
                    print value_name, " is really, really big!!"
    else:
        calculated_values = {}
        with open(basegraph_name + '_processed_service_specific_vales_' + container_or_class + '_' + '%.2f' % (time_interval) + '.txt',
                  'r') as csvfile:
            csvread = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in csvread:
                print row
                calculated_values[row[0]] = [i if i != (None) else float('nan') for i in ast.literal_eval(row[1])]
                print row[0], calculated_values[row[0]]

    print "service specific calculated values", calculated_values
    #time.sleep(20)
    return calculated_values