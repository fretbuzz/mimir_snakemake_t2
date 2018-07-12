# the ones I want to do are :
# reciprocity (this is more of a formality in my case b/c it is obviously one)
# density (ratio of edges-to-nodes)
# diameter (can be done for each pair of nodes, typically convyed via 'effective' diameter',
    # which is just the 90th percentile of the distance between every pair of connected nodes)
# clustering coefficient
    # there's an equation
# degree distribution
    # calc in and out degrees and use the tool to fit the distro
# joint degree distribution
    # see pseudocode
# in-flow vs out-flow (maybe?)

# okay, plan of attack:
# (1) read in edge file
# (2) graph it to make sure that it looks fine
# (3) start going down the list

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
#import seaborn as sns; sns.set()
import numpy as np
import statsmodels.api as sm
import math
import scipy
import time
import scipy.stats
import scipy.sparse.linalg
import pandas

sockshop_ms_s = ['carts-db','carts','catalogue-db','catalogue','front-end','orders-db','orders',
        'payment','queue-master','rabbitmq','session-db','shipping','user-sim', 'user-db','user','load-test']

number_boxplots = 0 # bad
boxplot_xtick_lables = []

def pipeline_analysis_step(filenames, ms_s, time_interval, basegraph_name):
    list_of_graphs = []
    list_of_aggregated_graphs = [] # all nodes of the same class aggregated into a single node
    list_of_aggregated_graphs_multi = [] # the above w/ multiple edges

    for file_path in filenames:
        G = nx.DiGraph()
        print "path to file is ", file_path
        nx.read_edgelist(file_path,
                        create_using=G, delimiter=',', data=(('weight', float),))
        list_of_graphs.append(G)
        aggreg_multi_G, aggreg_simple_G = aggregate_graph(G, ms_s)
        list_of_aggregated_graphs.append( aggreg_simple_G )
        list_of_aggregated_graphs_multi.append( aggreg_multi_G )

    calc_graph_metrics(list_of_graphs, ms_s, time_interval, basegraph_name + '_container_')
    calc_graph_metrics(list_of_aggregated_graphs, ms_s, time_interval, basegraph_name + '_class_')

def calc_graph_metrics(G_list, ms_s, time_interval, basegraph_name):
    global number_boxplots # this is a bad practice, but I am going to do it anyway
    global boxplot_xtick_lables

    average_path_lengths = []
    densities = []
    degree_dicts = []
    weight_recips = []
    weighted_average_path_lengths = []
    unweighted_overall_reciprocities = [] # defined per networkx definition (see their docs)
    weighted_reciprocities = [] # defined per the nature paper (see comment @ function definition)

    for cur_G in G_list:
        # okay, so this is where to calculate those metrics from the excel document

        # first, let's do the graph-wide metrics (b/c it is simple) (these are only single values)
        try:
            avg_path_length = nx.average_shortest_path_length(cur_G) #
        except:
            avg_path_length = 0

        try:
            recip = nx.overall_reciprocity(cur_G)  # if it is not one, then I cna deal w/ looking at dictinoarty
        except :
            recip = -1 # overall reciprocity not defined for empty graphs
        unweighted_overall_reciprocities.append(recip)

        # prob wanna do weighted reciprocity per ms class (I'm thinking scatter plot) (tho I need to figure out the other axis)
        # note: in the nature paper they actually just graph in-strength vs out-strength (might be the way to go)
        # could also try a repeat of the angle-vector-measurement-trick that i did for degrees
        # (they also do weighted reciprocity vs time later on in the paper)
        weighted_reciprocity, non_reciprocated_out_weight, non_reciprocated_in_weight = network_weidge_weighted_reciprocity(cur_G)
        weighted_reciprocities.append(weighted_reciprocity)

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
            print val, val[0], val[1]
            degree_dict[val[0]] = val[1]
        print "degree dict", degree_dict
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

    # okay, so now to do some simple analysis that'll lead to the creation of some graphs...
    # NOTE: slicing off the first value for a lot of these b/c i started tcpdump before the load generator...

    x = [i*time_interval for i in range(0, len(average_path_lengths))] #[1:] # I'm taking the first value now...
    print "avg path lengths", average_path_lengths, len(average_path_lengths)
    plt.figure(1)
    plt.clf()
    plt.title('unweighted average path length, ' + '%.2f' % (time_interval) )
    plt.ylabel('average path length (unweighted)')
    plt.xlabel('time (sec)')
    plt.plot(x, average_path_lengths) #[1:])
    plt.savefig(basegraph_name + '+avg_path_length_' + '%.2f' % (time_interval) + '.png', format='png')

    plt.figure(2)
    plt.clf()
    plt.title('unweighted average path length, ' + '%.2f' % (time_interval))
    plt.ylabel('average path length (unweighted)')
    #print ~np.isnan(average_path_lengths)
    average_path_lengths_no_nan = []
    for val in average_path_lengths:
        if not math.isnan(val):
            average_path_lengths_no_nan.append(val)
    #plt.boxplot(average_path_lengths[np.logical_not(np.isnan(average_path_lengths))])
    #plt.boxplot(average_path_lengths[~np.isnan(average_path_lengths)][1:] )
    plt.boxplot(average_path_lengths_no_nan, sym='k.', whis=[5, 95])
    plt.savefig(basegraph_name + '+avg_path_length_boxplot_' + '%.2f' % (time_interval) + '.png', format='png')

    fig = plt.figure(20)
    ax = plt.subplot(111)
    plt.title('unweighted average path length, multiple')
    plt.boxplot(average_path_lengths_no_nan, sym='k.', whis=[5, 95], labels=['%.2f' % (time_interval)],
                positions=[number_boxplots])#, names=[time_interval])
    ax.set_xlim(0, number_boxplots)
    boxplot_xtick_lables.append('%.2f' % (time_interval))
    plt.xticks(range(0,number_boxplots), boxplot_xtick_lables)
    number_boxplots +=1
    plt.savefig(basegraph_name + 'avg_path_length_boxplot_multiple_times_' + '%.2f' % (time_interval) + '.png', format='png')

    plt.figure(3)
    plt.clf()
    plt.ylabel('average weighted path length')
    plt.xlabel('time (sec)')
    plt.title('time vs (weighted) average path lengths, ' + '%.2f' % (time_interval))
    plt.plot(x, weighted_average_path_lengths)
    plt.savefig(basegraph_name + '_weighted_avg_path_length_' + '%.2f' % (time_interval) + '.png', format='png')

    plt.figure(4)
    plt.clf()
    plt.ylabel('average weighted path length')
    plt.title('time vs (weighted) average path lengths, ' + '%.2f' % (time_interval))
    weighted_average_path_lengths_no_nan = []
    for val in weighted_average_path_lengths:
        if not math.isnan(val):
            weighted_average_path_lengths_no_nan.append(val)
    plt.boxplot(weighted_average_path_lengths_no_nan, sym='k.', whis=[5, 95])
    #plt.boxplot(weighted_average_path_lengths[1:])
    plt.savefig(basegraph_name + '_weighted_avg_path_length_boxplot_' +'%.2f' % (time_interval) + '.png', format='png')

    plt.figure(5)
    plt.clf()
    plt.ylabel('overall reciprocity (unweighted)')
    plt.xlabel('time (sec)')
    # yah, so this could maybe detect like a udp thing (b/c no ack's?)
    plt.title('unweighted overall reciprocity, ' + '%.2f' % (time_interval))
    plt.plot(x, unweighted_overall_reciprocities)
    plt.savefig(basegraph_name + '_unweighted_overall_reciprocity_' + '%.2f' % (time_interval) + '.png', format='png')

    plt.figure(6)
    plt.clf()
    plt.ylabel('overall reciprocity (unweighted)')
    # yah, so this could maybe detect like a udp thing (b/c no ack's?)
    plt.title('unweighted overall reciprocity, ' + '%.2f' % (time_interval))
    unweighted_overall_reciprocities_no_nan = []
    for val in unweighted_overall_reciprocities:
        if not math.isnan(val):
            unweighted_overall_reciprocities_no_nan.append(val)
    plt.boxplot(unweighted_overall_reciprocities_no_nan, sym='k.', whis=[5, 95])
    #plt.boxplot(unweighted_overall_reciprocities[1:])
    plt.savefig(basegraph_name + '_unweighted_overall_reciprocity_boxplot_' + '%.2f' % (time_interval) + '.png', format='png')

    plt.figure(7)
    plt.clf()
    plt.ylabel('overall weighted reciprocity')
    plt.xlabel('time (sec)')
    plt.title('weighted reciprocity, ' + '%.2f' % (time_interval))
    plt.plot(x, weighted_reciprocities)
    plt.savefig(basegraph_name + '_weighted_overall_reciprocity_' + '%.2f' % (time_interval) + '.png', format='png')

    plt.figure(8)
    plt.clf()
    plt.ylabel('overall weighted reciprocity')
    plt.title('weighted reciprocity, ' + '%.2f' % (time_interval))
    weighted_reciprocities_no_nan = []
    for val in weighted_reciprocities:
        if not math.isnan(val):
            weighted_reciprocities_no_nan.append(val)
    plt.boxplot(weighted_reciprocities_no_nan, sym='k.', whis=[5, 95])
    plt.savefig(basegraph_name + '_weighted_overall_reciprocity_boxplot_' + '%.2f' % (time_interval) + '.png', format='png')

    plt.figure(9)
    plt.clf()
    plt.ylabel('overall graph density')
    plt.xlabel('time (sec)')
    plt.title('graph density, ' + '%.2f' % (time_interval))
    plt.plot(x, densities)
    plt.savefig(basegraph_name + '_graph_density_' + '%.2f' % (time_interval) + '.png', format='png')

    print "degrees", degree_dicts
    print "weighted recips", weight_recips
    #raw_input("Press Enter2 to continue...")
    # now going to perform leman method
    print "DOING ANGLES"
    window_size = 4
    node_degrees = []
    print "degree dicts", degree_dicts
    total_node_list = []
    for cur_g in G_list:
        for node in cur_g.nodes():
            total_node_list.append(node)
    total_node_list = list(set(total_node_list))
    for degree_dict in degree_dicts:
        current_nodes = []
        #print G_list, len(G_list)
        for node in total_node_list:
            try:
                current_nodes.append( degree_dict[node] )
            except:
                current_nodes.append(0) # the current degree_dict must not have an entry for node -> no comm -> degree zero
        if current_nodes not in node_degrees:
            print "new degree set", current_nodes
        node_degrees.append(current_nodes)
        #print "degree angles", node_degrees
    angles_degrees = find_angles(node_degrees, 4) #eigenvector_analysis(degree_dicts, window_size=window_size)  # setting window size arbitrarily for now...
    print "angles degrees", type(angles_degrees), angles_degrees
    print node_degrees
    #print "DOING WEIGHTED RECIPROCITY"

    angles_degrees_eigenvector = eigenvector_analysis(degree_dicts, 4, total_node_list)
    print "angles degrees eigenvector", angles_degrees_eigenvector

    plt.figure(10)
    plt.clf()
    plt.ylabel('angle between out-degree vectors')
    plt.xlabel('time (sec)')
    plt.title('angle between out-degree vectors, ' + '%.2f' % (time_interval))
    #x_after_window = x[window_size:]
    #if len(x_after_window) == 0: # problem with 1 time step...
    #    x_after_window = x
    #print "fig5", len(x_after_window), len(angles_degrees), angles_degrees
    plt.plot(x, angles_degrees)
    plt.savefig(basegraph_name + '_out_degree_angles_' + '%.2f' % (time_interval) + '.png', format='png')

    plt.figure(11)
    plt.clf()
    plt.ylabel('angle between out-degree vectors')
    plt.title('angle between out-degree vectors, ' + '%.2f' % (time_interval))
    angles_degrees_no_nan = []
    for val in angles_degrees:
        if not math.isnan(val):
            angles_degrees_no_nan.append(val)
    plt.boxplot(angles_degrees_no_nan, sym='k.', whis=[5, 95])
    plt.savefig(basegraph_name + '_out_degree_angles_boxplot_' + '%.2f' % (time_interval) + '.png', format='png')

    #plt.figure(6)
    plt.figure(12)
    plt.title("time vs app_server degrees")
    appserver_sum_degrees = []
    for degree_dict in degree_dicts:
        appserver_degrees = []
        for key,val in degree_dict.iteritems():
            if 'appserver' in key:
                appserver_degrees.append(val)
        appserver_sum_degrees.append( np.mean(appserver_degrees))
    plt.plot(x, appserver_sum_degrees)
    #plt.title('time vs angle for weighted reciprocity')
    #plt.plot(x, weighted_reciprocity_degrees)
    plt.show()

    print angles_degrees_no_nan

    plt.figure(13)
    plt.clf()
    plt.ylabel('overall graph density')
    plt.title('graph density, ' + '%.2f' % (time_interval))
    print "graph density", densities
    densities_no_nan = []
    for val in densities:
        if not math.isnan(val):
            densities_no_nan.append(val)
    print densities_no_nan
    plt.boxplot(densities_no_nan, sym='k.', whis=[5, 95])
    plt.savefig(basegraph_name + '_graph_density_boxplot_' + '%.2f' % (time_interval) + '.png', format='png')

    #weighted_reciprocity_degrees = eigenvector_analysis(weight_recips, window_size=4)  # todo: not sure if will work...


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
            if ms in node:
                mapping[ms].append(node)
                mapping_node_to_ms[node] = ms
                break
    for (u,v,data) in G.edges(data=True):
        H.add_edge(mapping_node_to_ms[u], mapping_node_to_ms[v], weight=data['weight'])
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

# returns list of angles
# TODO: fix (might need to rewrite a bunch from scratch...)
# todo: the values for each time stamp entry must be a dictinoary
# mapped by
def eigenvector_analysis(tensor, window_size,nodes_in_tensor):
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

    correlation_matrices = []
    p_value_matrices =[]
    correlation_matrix_eigenvectors = []
    # let's iterate through the times, pulling out slices that correspond to windows
    for i in range(0, len(tensor)):
        correlation_matrix = pandas.DataFrame(0, index=nodes_in_tensor, columns=nodes_in_tensor)
        pearson_p_val_matrix = pandas.DataFrame(0, index=nodes_in_tensor, columns=nodes_in_tensor)

        start_of_window = max(0, i - window_size + 1)
        # compute average window (with what we have available)
        print "list slice window of tensor", tensor[start_of_window: i]
        tensor_window = tensor[start_of_window: i]

        # okay, now that we have the window, it is time to go through each pair of nodes in
        # the window
        for node_one in nodes_in_tensor:
            for node_two in nodes_in_tensor:
                # compute pearson's rho of the corresponding time series
                try:
                    node_one_time_series = [x[node_one] for x in tensor_window]
                except:
                    continue
                try:
                    nodE_two_time_series = [x[node_two] for x in tensor_window]
                except:
                    continue
                pearson_rho = scipy.stats.pearsonr(node_one_time_series, nodE_two_time_series)
                print 'peasrson', pearson_rho
                if math.isnan(pearson_rho[0]) and pearson_rho[1] == 1.0:
                    correlation_matrix.at[node_one, node_two] = 1
                else:
                    if math.isnan(pearson_rho[0]):
                        correlation_matrix.at[node_one, node_two] = 0
                    else:
                        correlation_matrix.at[node_one, node_two] = pearson_rho[0]
                try:
                    pearson_p_val_matrix.at[node_one, node_two] = pearson_rho[1]
                except:
                    pearson_p_val_matrix.at[node_one, node_two] = -1


        correlation_matrices.append(correlation_matrix)
        p_value_matrices.append(pearson_p_val_matrix)

        eigen_vals, eigen_vects = scipy.linalg.eigh(correlation_matrix.values)
        correlation_matrix_eigenvectors.append(eigen_vects[0])

    print "correlation eigenvects", correlation_matrix_eigenvectors
    angles = find_angles(correlation_matrix_eigenvectors, window_size)

    return angles

def find_angles(list_of_vectors, window_size):

    angles = [0] # first must be zero (nothing to compare to)
    for i in range(1, len(list_of_vectors)):
        start_of_window = max(0, i - window_size)
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
        print edge
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