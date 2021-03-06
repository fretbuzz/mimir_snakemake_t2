import networkx as nx
import seaborn as sns
sns.set()
import math
import gc
import numpy as np
from next_gen_metrics import create_dict_for_dns_metric, \
    find_dns_node_name, reverse_svc_to_pod_dict, single_step_outside_inside_ratio_dns_metric, calc_VIP_metric
from prepare_graph import prepare_graph, is_ip, match_name_to_pod, map_nodes_to_svcs, \
    find_infra_components_in_graph,remove_infra_from_graph
import random
import logging
import time
import matplotlib
matplotlib.use('Agg',warn=False, force=True)
import matplotlib
import matplotlib.pyplot as plt
import os,errno
import copy
import cPickle as pickle
import pyximport
pyximport.install() # am I sure that I want this???
import multiprocessing
import numpy.random
import pandas as pd
import subprocess
from process_pcap import update_mapping

# okay, so things to be aware of:
# (a) we are assuming that if we cannot label the node and it is not loopback or in the '10.X.X.X' subnet, then it is outside

class injected_graph():
    def __init__(self, name, injected_graph_loc, non_injected_graph_loc, concrete_container_exfil_paths, exfil_amt,
                 svc_to_pod, pod_to_svc, total_edgelist_nodes, where_to_save_this_obj, counter, name_of_dns_pod_node,
                 current_total_node_list,
                 svcs, is_swarm, ms_s, container_to_ip, infra_instances, injected_class_graph_loc, name_of_injected_file,
                 nodeAttrib_injected_graph_loc, nodeAttrib_injected_graph_loc_class, pruned_graph_nodeAttrib_loc,
                 past_end_of_training, attack_happened_p, drop_infra_from_graph):
        self.drop_infra_from_graph = drop_infra_from_graph
        self.name = name
        self.injected_graph_loc = injected_graph_loc
        self.name_of_injected_file = name_of_injected_file
        self.non_injected_graph_loc = non_injected_graph_loc
        self.injected_class_graph_loc = injected_class_graph_loc
        self.concrete_container_exfil_paths = concrete_container_exfil_paths
        self.exfil_amt = exfil_amt
        self.svc_to_pod = svc_to_pod
        self.pod_to_svc = pod_to_svc
        self.total_edgelist_nodes = total_edgelist_nodes
        self.where_to_save_this_obj = where_to_save_this_obj
        self.cur_1si_G = None
        self.edgefile_folder_path = "/".join(injected_graph_loc.split('/')[:-1])
        self.metrics_file = self.edgefile_folder_path + '/metrics_for_' + name_of_injected_file + '.csv'
        self.name_of_dns_pod_node = name_of_dns_pod_node
        self.current_total_node_list = current_total_node_list
        self.graph_feature_dict = {}
        self.graph_feature_dict_keys = None
        self.nodeAttrib_injected_graph_loc = nodeAttrib_injected_graph_loc
        self.nodeAttrib_injected_graph_loc_class = nodeAttrib_injected_graph_loc_class
        self.past_end_of_training = past_end_of_training

        self.svcs = svcs
        self.counter = counter
        self.ms_s = ms_s
        self.container_to_ip = container_to_ip
        self.infra_instances = infra_instances

        self.cur_class_G = None
        self.pruned_graph_nodeAttrib_loc = pruned_graph_nodeAttrib_loc
        self.cur_1si_G_non_injected = None
        self.attack_happened_p = attack_happened_p

        # past_end_of_training, attack_happened_p

    def save(self):
        with open(self.where_to_save_this_obj, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def calc_single_step_metrics(self, sensitive_ms):
        self.graph_feature_dict = {}
        self._load_graph()
        self._load_nonInjected_graph()
        #self._create_class_level_graph()

        self.cur_1si_G = set_weight_inverse(self.cur_1si_G) ## ??
        undirected_class_G = create_undirected_graph(self.cur_class_G)
        # want undirected graph to have a single connected component...
        undirected_class_G = undirected_class_G.subgraph(max(nx.connected_components(undirected_class_G), key=len)).copy()
        undirected_class_G =  set_weight_inverse(undirected_class_G)
        ### ^^ note: this is used down below with random-walk betweeness...
        self.cur_class_G = normalize_graph(self.cur_class_G) # <--- Normalization!! Decide if I want to keep it or get rid of it!!!
        self.cur_class_G = set_weight_inverse(self.cur_class_G)

        undirected_pod_graph = create_undirected_graph(self.cur_1si_G)
        undirected_pod_graph = undirected_pod_graph.subgraph(max(nx.connected_components(undirected_pod_graph), key=len)).copy()


        density = nx.density(self.cur_1si_G)
        self.graph_feature_dict['pod_1si_density_list'] = density
        weight_into_dns_dict, weight_outof_dns_dict, _, _ = create_dict_for_dns_metric(self.cur_1si_G, self.name_of_dns_pod_node)
        self.graph_feature_dict['weight_into_dns_dict'] = weight_into_dns_dict
        try:
            self.graph_feature_dict['into_dns_from_outside_list'] = weight_into_dns_dict['outside']
        except:
            self.graph_feature_dict['into_dns_from_outside_list'] = 0.0
        self.graph_feature_dict['dns_out_metric_dicts'] = weight_outof_dns_dict

        pod_comm_but_not_VIP_comm, fraction_pod_comm_but_not_VIP_comm = calc_VIP_metric(self.cur_1si_G, True)
        self.graph_feature_dict['pod_comm_but_not_VIP_comms'] = pod_comm_but_not_VIP_comm
        self.graph_feature_dict['fraction_pod_comm_but_not_VIP_comms'] = fraction_pod_comm_but_not_VIP_comm

        pod_comm_but_not_VIP_comm_no_abs, fraction_pod_comm_but_not_VIP_comm_no_abs = calc_VIP_metric(self.cur_1si_G, False)
        self.graph_feature_dict['pod_comm_but_not_VIP_comms_no_abs'] = pod_comm_but_not_VIP_comm_no_abs
        self.graph_feature_dict['fraction_pod_comm_but_not_VIP_comms_no_abs'] = fraction_pod_comm_but_not_VIP_comm_no_abs

        svc_to_pod_with_outside = copy.deepcopy(self.svc_to_pod)
        svc_to_pod_with_outside['outside'] = ['outside']
        svc_pair_to_reciprocity, svc_pair_to_density, svc_pair_to_coef_of_var = \
                                                            pairwise_metrics(self.cur_1si_G, svc_to_pod_with_outside)

        for svc_pair,reciprocity in svc_pair_to_reciprocity.iteritems():
            cur_recip_key = str(svc_pair[0]) + '_to_' + str(svc_pair[1]) + '_reciprocity'
            cur_density_key = str(svc_pair[0]) + '_to_' + str(svc_pair[1]) + '_density'
            cur_svc_pair_of_coef = str(svc_pair[0]) + '_to_' + str(svc_pair[1]) + '_edge_coef_of_var'

            self.graph_feature_dict[cur_recip_key] = reciprocity
            self.graph_feature_dict[cur_density_key] = svc_pair_to_density[svc_pair]
            try:
                self.graph_feature_dict[cur_svc_pair_of_coef] = svc_pair_to_coef_of_var[svc_pair]
            except:
                self.graph_feature_dict[cur_svc_pair_of_coef] = 0.0

        #####

        betweeness_centrality_nodes = nx.betweenness_centrality(self.cur_1si_G, weight='inverse_weight',endpoints=True, normalized=True)
        betweeness_centrality_coef_var_of_classes,betweeness_centrality_mean, bc_max = \
            find_coef_of_var_for_nodes(betweeness_centrality_nodes, svc_to_pod_with_outside)

        self.graph_feature_dict = add_c_metric(self.graph_feature_dict, betweeness_centrality_coef_var_of_classes,
                                    betweeness_centrality_mean, bc_max, svc_to_pod_with_outside, 'betweeness_centrality')

        load_centrality_nodes = nx.load_centrality(self.cur_1si_G, weight='inverse_weight')
        load_centrality_coef_var_of_classes,load_centrality_mean, lc_max = \
            find_coef_of_var_for_nodes(load_centrality_nodes, svc_to_pod_with_outside)

        self.graph_feature_dict = add_c_metric(self.graph_feature_dict, load_centrality_coef_var_of_classes,
                                               load_centrality_mean, lc_max, svc_to_pod_with_outside, 'load_centrality')

        ## note: this doesn't make any sense...
        harmonic_centrality_nodes = nx.harmonic_centrality(self.cur_1si_G, distance='inverse_weight')
        harmonic_centrality_coef_var_of_classes, harmonic_centrality_mean, hc_max = \
            find_coef_of_var_for_nodes(harmonic_centrality_nodes, svc_to_pod_with_outside)

        self.graph_feature_dict = add_c_metric(self.graph_feature_dict, harmonic_centrality_coef_var_of_classes,
                                               harmonic_centrality_mean, hc_max, svc_to_pod_with_outside, 'harmonic_centrality')

        dict_of_clustering_coef = nx.clustering(self.cur_1si_G, weight='inverse_weight')
        clustering_coef_coef_of_var, clustering_coef_mean, clustering_coef_max = \
            find_coef_of_var_for_nodes(dict_of_clustering_coef, svc_to_pod_with_outside)
        self.graph_feature_dict = add_c_metric(self.graph_feature_dict, clustering_coef_coef_of_var,
                                               clustering_coef_mean, clustering_coef_max, svc_to_pod_with_outside, 'clustering_coef')

        #dict_of_cfbc = nx.current_flow_betweenness_centrality(undirected_pod_graph, normalized=False, weight='weight')
        try:
            cfbc_sub_pods = nx.current_flow_betweenness_centrality_subset(undirected_pod_graph, weight='weight',
                                                                          targets=['outside'],
                                                                          sources=svc_to_pod_with_outside[sensitive_ms],
                                                                          normalized=False)
            cfbc_coef_of_var, _, _ = find_coef_of_var_for_nodes(cfbc_sub_pods, svc_to_pod_with_outside)
        except:
            pass

        for service in svc_to_pod_with_outside.keys():
            try:
                self.graph_feature_dict['pods_cfbc_sub_coef_of_var_' + str(service)] = cfbc_coef_of_var[service]
            except:
                self.graph_feature_dict['pods_cfbc_sub_coef_of_var_' + str(service)] = float('NaN')

        ######
        # these metrics are new... might want to get rid of these or the previous ones, depending on how it goes...
        betweeness_centrality_nodes = nx.betweenness_centrality(self.cur_class_G, weight='inverse_weight',endpoints=True, normalized=True)
        for service in svc_to_pod_with_outside.keys():
            try:
                self.graph_feature_dict['class_betweeness_centrality_' + service] = betweeness_centrality_nodes[service]
            except:
                self.graph_feature_dict['class_betweeness_centrality_' + service] = float('NaN')

        load_centrality_nodes = nx.load_centrality(self.cur_class_G, weight='inverse_weight')
        for service in svc_to_pod_with_outside.keys():
            try:
                self.graph_feature_dict['class_load_centrality_' + service] = load_centrality_nodes[service]
            except:
                self.graph_feature_dict['class_load_centrality_' + service] = float('NaN')

        harmonic_centrality_nodes = nx.harmonic_centrality(self.cur_class_G, distance='inverse_weight')
        for service in svc_to_pod_with_outside.keys():
            try:
                self.graph_feature_dict['class_harmonic_centrality_' + service] = harmonic_centrality_nodes[service]
            except:
                self.graph_feature_dict['class_harmonic_centrality_' + service] = float('NaN')

        dict_of_clustering_coef = nx.clustering(self.cur_class_G, weight='inverse_weight')
        for service in svc_to_pod_with_outside.keys():
            try:
                self.graph_feature_dict['class_clustering_coef_' + service] = dict_of_clustering_coef[service]
            except:
                self.graph_feature_dict['class_clustering_coef_' + service] = float('NaN')

        #dict_of_cfbc = nx.current_flow_betweenness_centrality(self.cur_1si_G, normalized=True, weight='weight')
        #cfbc_coef_of_var, cfbc_mean, cfbc_max = find_coef_of_var_for_nodes(dict_of_cfbc, svc_to_pod_with_outside)
        #self.graph_feature_dict = add_c_metric(self.graph_feature_dict, cfbc_coef_of_var,
        #                                       cfbc_mean, cfbc_max, svc_to_pod_with_outside, 'current_flow_bc_')

        # normal weight instead of inverse weight b/c which path taken is proportional to the weight... which is what we want.
        #try:
        for (u,v,d) in undirected_class_G.edges(data=True):
            print "undir_class_G_edge:", u,v,d

        try:
            cfbc_nodes = nx.current_flow_betweenness_centrality(undirected_class_G, weight='weight')
        except:
            cfbc_nodes = None
        for service in svc_to_pod_with_outside.keys():
            try:
                self.graph_feature_dict['class_current_flow_bc_' + service] = cfbc_nodes[service]
            except:
                self.graph_feature_dict['class_current_flow_bc_' + service] = float('NaN')

        ######
        ######
        try:
            cfbc_sub_nodes = nx.current_flow_betweenness_centrality_subset(undirected_class_G, weight='weight',
                                                            targets=['outside'], sources=[sensitive_ms], normalized=True)
        except:
            pass # if cannot reach outside, there will be a key error...

        for service in svc_to_pod_with_outside.keys():
            try:
                self.graph_feature_dict['class_current_flow_bc_sub_' + service] = cfbc_sub_nodes[service]
            except:
                self.graph_feature_dict['class_current_flow_bc_sub_' + service] = float('NaN')

        print [i for i in self.cur_1si_G.nodes()]
        print [i for i in self.cur_class_G.nodes()]
        try:
            betweeness_centrality_subset_nodes = nx.betweenness_centrality_subset(self.cur_class_G, weight='inverse_weight',
                                                                              targets=['outside'], sources=[sensitive_ms],
                                                                              normalized=True)
        except:
            pass ## same reasoning as above (in the previous centrality measure...)

        for service in svc_to_pod_with_outside.keys():
            try:
                self.graph_feature_dict['class_betweeness_centrality_sub_' + service] = betweeness_centrality_subset_nodes[service]
            except:
                self.graph_feature_dict['class_betweeness_centrality_sub_' + service] = float('NaN')

        #cfbc_sub_pods = nx.current_flow_betweenness_centrality_subset(self.cur_1si_G, weight='weight',
        #                                                               targets=['outside'],
        #                                                               sources=svc_to_pod_with_outside(sensitive_ms),
        #                                                               normalized=True)

        #cfbc_sub_pods_coefvar, cfbc_sub_pods_mean, cfbc_sub_pods_max = find_coef_of_var_for_nodes(cfbc_sub_pods, svc_to_pod_with_outside)
        #self.graph_feature_dict = add_c_metric(self.graph_feature_dict, cfbc_sub_pods_coefvar,
        #                                       cfbc_sub_pods_mean, cfbc_sub_pods_max, svc_to_pod_with_outside, 'current_flow_bc_sub_')

        try:
            betweeness_centrality_subset_pods = nx.betweenness_centrality_subset(self.cur_1si_G, weight='inverse_weight',
                                                                              targets=['outside'],
                                                                             sources=svc_to_pod_with_outside[sensitive_ms],
                                                                              normalized=True)
        except:
            betweeness_centrality_subset_pods = {}
            for svc,pods in svc_to_pod_with_outside.iteritems():
                for pod in pods:
                    betweeness_centrality_subset_pods[pod] = float('NaN') #')#0.0

        bc_sub_pods_coefvar, bc_sub_pods_mean, bc_sub_pods_max = find_coef_of_var_for_nodes(betweeness_centrality_subset_pods, svc_to_pod_with_outside)
        self.graph_feature_dict = add_c_metric(self.graph_feature_dict, bc_sub_pods_coefvar,
                                               bc_sub_pods_mean, bc_sub_pods_max, svc_to_pod_with_outside, 'bc_sub_')

        #####
        #####w

        try: # this try-except exists b/c sockshop doesn't have this entry in the graph_feature_dict
            if math.isnan(self.graph_feature_dict['outside_to_wwwppp-wordpress_edge_coef_of_var']):
                print "huston, we have a problem over here"
        except:
            pass

        # yah, not so sure about this... need to store training/testing status in the graph object
        # (b/c if it is testing, then can use injected. else should use the fine ones)...
        self.graph_feature_dict['past_end_of_training'] = self.past_end_of_training
        print "past_end_of_training", self.past_end_of_training

        # not sure why I actually cared about this... but I cannot any real negatives... so let's just go with it, I guess...
        if self.past_end_of_training:
            adjacency_matri = nx.to_pandas_adjacency(self.cur_1si_G, nodelist=self.current_total_node_list)
        else:
            adjacency_matri = nx.to_pandas_adjacency(self.cur_1si_G_non_injected, nodelist=self.current_total_node_list)

        self.graph_feature_dict['adjacency_matrix'] = adjacency_matri

        dns_outside_inside_ratios, dns_list_outside, dns_list_inside = \
            single_step_outside_inside_ratio_dns_metric(weight_into_dns_dict, weight_outof_dns_dict)
        into_dns_ratio, into_dns_from_outside, into_dns_from_indeside = \
            single_step_outside_inside_ratio_dns_metric(weight_into_dns_dict, weight_into_dns_dict)

        self.graph_feature_dict['weight_into_dns_dict'] = weight_into_dns_dict
        self.graph_feature_dict['dns_outside_inside_ratios'] = dns_outside_inside_ratios
        self.graph_feature_dict['into_dns_ratio'] = into_dns_ratio

        self.graph_feature_dict['dns_list_outside'] = dns_list_outside
        self.graph_feature_dict['dns_list_inside'] = dns_list_inside

        self.graph_feature_dict_keys = self.graph_feature_dict.keys()

        return self.graph_feature_dict.keys()

    def save_metrics_dict(self):
        with open(self.metrics_file, 'wb') as f:  # Just use 'w' mode in 3.x
            f.write(pickle.dumps(self.graph_feature_dict))

    def _load_graph(self):
        self.cur_1si_G = nx.DiGraph()
        print "path to file is ", self.injected_graph_loc
        #f = open(self.injected_graph_loc, 'r')
        #lines = f.readlines()
        #nx.parse_edgelist(lines, delimiter=' ', create_using=self.cur_1si_G, data=[('frames',int), ('weight',int)])
        self.cur_1si_G = nx.read_gpickle( self.nodeAttrib_injected_graph_loc )

        self.cur_class_G = nx.DiGraph()
        #print "path to class file is ", self.injected_class_graph_loc
        #f = open(self.injected_class_graph_loc, 'r')
        #lines = f.readlines()
        #nx.parse_edgelist(lines, delimiter=' ', create_using=self.cur_class_G, data=[('frames',int), ('weight',int)])
        self.cur_class_G = nx.read_gpickle( self.nodeAttrib_injected_graph_loc_class )

    def _load_nonInjected_graph(self):
        self.cur_1si_G_non_injected = nx.read_gpickle( self.pruned_graph_nodeAttrib_loc )


    def load_metrics(self):
        print "metrics_file", self.metrics_file
        with open(self.metrics_file, mode='rb') as f:
            dict_contents = f.read()
            self.graph_feature_dict = pickle.loads(dict_contents)

    def _create_class_level_graph(self):
        self.cur_class_G = prepare_graph(self.cur_1si_G, self.svcs, 'class', 0, self.counter, self.injected_graph_loc,
                                    self.ms_s, self.container_to_ip, self.infra_instances)

def normalize_graph(G):
    total_weight = 0.0
    # doing in_edges so that stuff isn't double counted
    for (u,v,d) in G.in_edges(data=True):
        total_weight += d['weight']
    print "total_weight_in_graph", total_weight

    # then modify the edge weights accordingly...
    for (u,v,d) in G.in_edges(data=True):
        G[u][v]['weight'] = float(d['weight']) / total_weight

    return G

# heavier paths should actually be treated as LESS heavy w.r.t. to the graph community's definition of weight
def set_weight_inverse(G):
    for (u,v,d) in G.edges(data=True):
        if d['weight'] == 0:
            G.remove_edge(u,v)
        else:
            G[u][v]['inverse_weight'] = (1.0 / float(d['weight'])) # no, i don't think it does anything... * 1000 # TODO is the 1000 necessary for scaling problems???
    return G

def create_undirected_graph(G):
    G_undirected = copy.deepcopy(G)
    G_undirected = G_undirected.to_undirected()
    # want to make sure that none of the edges are over-counted...
    # first, set all edge weights to zero...
    for (u,v,d) in G.edges(data=True):
        G_undirected[u][v]['weight'] = 0
    # then add the values appropriately...
    # (only use in edges so that no values are counted twice...)
    for cur_node in G.nodes():
        for (u,v,d) in G.in_edges(cur_node, data=True):
            G_undirected[u][v]['weight'] += G[u][v]['weight'] #+ G[v][u]['weight']
    return G_undirected

def add_c_metric(feature_dict, coefvar_dict, mean_dict, max_dict, svc_to_pod, metric_name):
    for service in svc_to_pod.keys():
        feature_dict[metric_name + '_coef_of_var_' + str(service)] = coefvar_dict[service]

        feature_dict['avg_' + metric_name + '_'  + str(service)] = mean_dict[service]

        feature_dict['max_' + metric_name + '_' + str(service)] = max_dict[service]

    return feature_dict

class set_of_injected_graphs():
    def __init__(self, time_granularity, raw_edgefile_names,
                 svcs, ms_s, container_to_ip, infra_instances, synthetic_exfil_paths, initiator_info_for_paths,
                 attacks_to_times, collected_metrics_location, current_set_of_graphs_loc,
                 avg_exfil_per_min, exfil_per_min_variance, avg_pkt_size, pkt_size_variance,
                 end_of_training, pod_creation_log, processed_graph_loc, drop_infra_from_graph,
                 exfil_paths_series):
                 #sensitive_ms):#, out_q):

        #self.sensitive_ms = sensitive_ms
        self.drop_infra_from_graph = drop_infra_from_graph
        self.list_of_injected_graphs_loc = []
        self.time_granularity = time_granularity
        self.raw_edgefile_names = raw_edgefile_names
        self.svcs = svcs
        self.ms_s= ms_s
        self.container_to_ip = container_to_ip
        self.infra_instances =infra_instances
        self.synthetic_exfil_paths = synthetic_exfil_paths
        self.initiator_info_for_paths =initiator_info_for_paths
        self.attacks_to_times = attacks_to_times
        self.time_interval= time_granularity
        self.collected_metrics_location = collected_metrics_location
        self.current_set_of_graphs_loc = current_set_of_graphs_loc
        self.end_of_training = end_of_training
        #self.out_q = out_q
        self.pod_creation_log = pod_creation_log
        self.exfil_paths_series = exfil_paths_series

        self.calculated_values = {}
        self.calculated_values_keys = None

        self.list_of_concrete_container_exfil_paths = []
        self.list_of_logical_exfil_paths = []
        self.list_of_exfil_amts = []

        self.avg_exfil_per_min = avg_exfil_per_min
        self.exfil_per_min_variance =  exfil_per_min_variance
        self.avg_pkt_size  = avg_pkt_size
        self.pkt_size_variance =  pkt_size_variance
        self.list_of_amt_of_out_traffic_bytes = []
        self.list_of_amt_of_out_traffic_pkts = []

        self.current_total_node_list = None
        self.aggregate_csv_edgefile_loc = self.collected_metrics_location + '_aggregate_edgefile.csv'
        self.joint_col_list = None
        self.feature_graph_keys = None
        self.processed_graph_loc = processed_graph_loc

        # graphs
        # alerts

    def save(self):
        with open(self.current_set_of_graphs_loc, 'wb') as f:  # Just use 'w' mode in 3.x
            f.write(pickle.dumps(self))

    def calculate_cilium_performance(self, allowed_intersvc_comm):
        alert_vals = []
        for counter,injected_graph_loc in enumerate(self.list_of_injected_graphs_loc):
            trigger_alert = False
            print "injected_graph_loc (cilium)",injected_graph_loc

            with open(injected_graph_loc, 'rb') as pickle_input_file:
                injected_graph = pickle.load(pickle_input_file)
            class_edgefile = injected_graph.nodeAttrib_injected_graph_loc_class
            class_graph = nx.read_gpickle(class_edgefile)
            #container_graph = nx.read_gpickle(injected_graph.nodeAttrib_injected_graph_loc)

            #container_graph_non_injected = nx.DiGraph()
            #with open(injected_graph.non_injected_graph_loc, 'r') as f:
            #    lines = f.readlines()
            #    nx.parse_edgelist(lines, delimiter=' ', create_using=container_graph_non_injected, data=[('frames',int), ('weight',int)])

            for (u,v,d) in class_graph.edges(data=True):
                if d['frames'] > 0:
                    if (u,v) not in allowed_intersvc_comm:
                        if 'VIP' not in u and 'VIP' not in v and not is_ip(u) \
                        and not is_ip(v) and 'POD' not in u and 'POD' not in v and u != v:
                            trigger_alert=True

                            break

            if trigger_alert:
                alert_vals.append(1)
            else:
                alert_vals.append(0)

        return alert_vals

    def ide_calculations(self, calc_ide, ide_window_size):
        '''
        cur_out_q = multiprocessing.Queue()
        args = [self.aggregate_csv_edgefile_loc, self.joint_col_list, self.window_size, self.raw_edgefile_names,
                cur_out_q, calc_ide]
        ide_p = multiprocessing.Process(
            target=calc_ide_angles,
            args=args)
        ide_p.start()

        # okay, return these values so that we can do stuff with them later...
        return cur_out_q, ide_p
        '''
        return calc_ide_angles(self.aggregate_csv_edgefile_loc, self.joint_col_list, ide_window_size, self.raw_edgefile_names,
                               None, calc_ide)

    def calc_serialize_metrics(self, no_labeled_data=False):
        adjacency_matrixes = []
        dns_in_metric_dicts = []

        self.calculated_values = {}

        for injected_graph_loc in self.list_of_injected_graphs_loc:
            print "injected_graph_loc",injected_graph_loc
            with open(injected_graph_loc, 'rb') as pickle_input_file:
                injected_graph = pickle.load(pickle_input_file)
            injected_graph.load_metrics()
            current_graph_feature_dict = injected_graph.graph_feature_dict


            # need to calculate the metrics that require all data to be known here...
            print("current_graph_feature_dict.keys()", current_graph_feature_dict.keys())
            adjacency_matrixes.append( current_graph_feature_dict['adjacency_matrix'] )
            dns_in_metric_dicts.append( current_graph_feature_dict['weight_into_dns_dict'] )

            # ZXZX
            #'''
            if not no_labeled_data:
                # past_end_of_training, attack_happened_p
                if 'attack_labels' not in self.calculated_values.keys():
                    self.calculated_values['attack_labels'] = []
                self.calculated_values['attack_labels'].append(injected_graph.attack_happened_p)
            #'''

            #for value_name, value_value in current_graph_feature_dict.iteritems():
            for value_name in self.feature_graph_keys:
                if 'adjacency_matrix' in value_name or 'weight_into_dns_dict' in value_name or 'dns_out_metric_dicts' in value_name:
                    # these are handled seperately b/c we gotta get all the data then do some postprocessing
                    continue

                if value_name not in self.calculated_values.keys():
                    self.calculated_values[value_name] = []
                try:
                    self.calculated_values[value_name].append(current_graph_feature_dict[value_name])
                except:
                    self.calculated_values[value_name].append( float('NaN') )

            total_edgelist_nodes = injected_graph.total_edgelist_nodes
            current_total_node_list = injected_graph.current_total_node_list

        self.calculated_values['Fraction of Communication Between Pods not through VIPs (w abs)'] = \
            [abs(i) for i in self.calculated_values['fraction_pod_comm_but_not_VIP_comms']]
        self.calculated_values['Communication Between Pods not through VIPs (w abs)'] = \
            [abs(i) for i in self.calculated_values['pod_comm_but_not_VIP_comms_no_abs']]

        with open(self.collected_metrics_location, 'wb') as f:  # Just use 'w' mode in 3.x
            f.write(pickle.dumps(self.calculated_values))

    def put_values_into_outq(self, out_q):
        out_q.put(self.calculated_values)
        out_q.put(self.list_of_concrete_container_exfil_paths)
        out_q.put(self.list_of_exfil_amts)
        out_q.put([])  # new_neighbors_outside
        out_q.put([])  # new_neighbors_dns
        out_q.put([])  # new_neighbors_all
        out_q.put(self.list_of_amt_of_out_traffic_bytes)
        out_q.put(self.list_of_amt_of_out_traffic_pkts)
        out_q.put(self.exfil_paths_series)
        #out_q.put(self.container_to_ip)

    def load_serialized_metrics(self):
        with open(self.collected_metrics_location, mode='rb') as f:
            cur_contents = f.read()
            #print "cur_contents", cur_contents
            self.calculated_values = pickle.loads(cur_contents)

    def calcuated_single_step_metrics(self, sensitive_ms):
        print("self.list_of_injected_graphs_loc",self.list_of_injected_graphs_loc)
        overall_graph_feature_dict_keys = set()
        for counter, injected_obj_loc in enumerate(self.list_of_injected_graphs_loc):
            print("counter",counter)
            gc.collect()

            with open(injected_obj_loc, 'r') as input_file:
                injected_graph_obj = pickle.load(input_file)

            graph_feature_dict_keys = injected_graph_obj.calc_single_step_metrics(sensitive_ms)
            overall_graph_feature_dict_keys = overall_graph_feature_dict_keys.union(set(graph_feature_dict_keys))
            #injected_graph_obj.calc_existing_single_step_metrics()
            injected_graph_obj.save_metrics_dict()

            injected_graph_obj.save()
        self.feature_graph_keys = list(overall_graph_feature_dict_keys)

    def generate_aggregate_csv(self):
        #''''
        col_list = self.current_total_node_list
        joint_col_list = ['Col_' + str(i) for i in range(0,len(col_list))]
        #joint_col_list = [(col_item1 + '-to-' + col_item2) for col_item1 in col_list for col_item2 in col_list if col_item1 != col_item2]
        #joint_col_list +=  ['labels']
        out_df = pd.DataFrame(None, index=None, columns=joint_col_list)

        for injected_graph_loc in self.list_of_injected_graphs_loc:
            print "injected_graph_loc",injected_graph_loc
            with open(injected_graph_loc, 'rb') as pickle_input_file:
                injected_graph = pickle.load(pickle_input_file)
                injected_graph._load_graph()

            ## okay, well with the injected graph, it is go-time...
            # okay, let's take the networkx graph. [done]
            # let's make a dict. [done]
            # let's turn the dict into a dataframe (default value zero) [done]
            # let's append this dict to the out_dict [done]
            # at the end, let's print the dict to a file. [done]

            adjacency_matrix = nx.to_numpy_matrix(injected_graph.cur_1si_G, nodelist=col_list)
            eigenvalues,eigenvectors = numpy.linalg.eig(adjacency_matrix)
            # then find index of largest eigenvalue
            largest_eigenvalue_index = np.argmax(eigenvalues)
            # and then use that to index into the principal eigenvector
            principal_eigenvector = eigenvectors[:, largest_eigenvalue_index]
            # and then write it to a file...
            principal_eigenvector = np.real(principal_eigenvector.T)
            ## we know a prioir that the components of the eigenvector will all be real


            #print principal_eigenvector.shape
            cur_df = pd.DataFrame(principal_eigenvector, index=[1], columns=joint_col_list)

            '''
            adj_dict = {}
            adj_dict_of_dicts = nx.to_dict_of_dicts(injected_graph.cur_1si_G)
            for src_node,inner_dict in adj_dict_of_dicts.iteritems():
                for dest_node, edge_data in inner_dict.iteritems():
                    col_name = src_node + '-to-' + dest_node
                    adj_dict[col_name] = [edge_data['weight']]

            # eh, i don't think this is needed (if I want them, I can go do it manually)
            #adj_dict['labels'] = [injected_graph.attack_happened_p]

            cur_df = pd.DataFrame(adj_dict, columns=joint_col_list)
            cur_df = cur_df.fillna(0)
            '''
            #cur_df['192.168.99.100-outside'] = 0 # TODO: remove this from the graph entirely when I get a chance...maybe...
            # but for now, just set it equal to zero (it's so large that it causes scaling problems w.r.t. the other entries)

            #print "does it make here"
            out_df = out_df.append(cur_df, sort=True)

        #print "does it makeithere"
        self.joint_col_list = joint_col_list
        out_df.to_csv(path_or_buf=self.aggregate_csv_edgefile_loc)
        #'''
        #return

    def generate_injected_edgefiles(self):
        current_total_node_list = []
        svc_to_pod = {}
        node_attack_mapping = {}
        class_attack_mapping = {}
        total_edgelist_nodes = []
        avg_dns_weight = 0
        avg_dns_pkts = 0

        svcs = self.svcs
        is_swarm = 0
        ms_s = self.ms_s
        container_to_ip = self.container_to_ip
        infra_instances = self.infra_instances
        synthetic_exfil_paths = self.synthetic_exfil_paths
        initiator_info_for_paths = self.initiator_info_for_paths
        attacks_to_times = self.attacks_to_times
        time_interval = self.time_interval
        out_q = multiprocessing.Queue()
        name_of_dns_pod_node = None
        last_attack_injected = None
        carryover = 0
        avg_exfil_per_min = self.avg_exfil_per_min
        exfil_per_min_variance = self.exfil_per_min_variance
        avg_pkt_size = self.avg_pkt_size
        pkt_size_variance = self.pkt_size_variance
        pod_creation_log = self.pod_creation_log

        num_graphs_to_process_at_once = 40
        for counter in range(0, len(self.raw_edgefile_names), num_graphs_to_process_at_once):

            file_paths = self.raw_edgefile_names[counter: counter + num_graphs_to_process_at_once]
            processed_graph_loc = self.processed_graph_loc + '/' + 'time_gran_' + str(time_interval) + '/'
            try:
                os.makedirs(processed_graph_loc)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            args = [counter, file_paths, svcs, is_swarm, ms_s, container_to_ip, infra_instances,
                    synthetic_exfil_paths, initiator_info_for_paths, attacks_to_times,
                    time_interval, total_edgelist_nodes, svc_to_pod, avg_dns_weight, avg_dns_pkts,
                    node_attack_mapping, out_q, current_total_node_list, name_of_dns_pod_node, last_attack_injected,
                    carryover, avg_exfil_per_min, exfil_per_min_variance, avg_pkt_size, pkt_size_variance,
                    self.end_of_training, pod_creation_log, processed_graph_loc, self.drop_infra_from_graph]
            p = multiprocessing.Process(
                target=process_and_inject_single_graph,
                args=args)
            p.start()

            # these'll be lists (meant for storing)
            concrete_cont_node_path = out_q.get()
            pre_specified_data_attribs = out_q.get()
            injected_graph_obj_loc = out_q.get()

            # these'll be non-lists (meant for passing to next iteration)
            avg_dns_weight = out_q.get()
            avg_dns_pkts = out_q.get()
            #class_attack_mapping = out_q.get()
            node_attack_mapping = out_q.get()
            current_total_node_list = out_q.get()
            name_of_dns_pod_node = out_q.get()
            last_attack_injected = out_q.get()
            carryover = out_q.get()
            amt_of_out_traffic_bytes = out_q.get()
            amt_of_out_traffic_pkts = out_q.get()
            container_to_ip = out_q.get()
            attack_occuring_list = out_q.get()
            p.join()

            ## okay, literally the code above should be wrapped in a function call...
            ## however, you'd probably wanna process like 40-50 of these on a single call...
            self.list_of_concrete_container_exfil_paths.extend(concrete_cont_node_path)
            self.list_of_logical_exfil_paths.extend(attack_occuring_list)
            self.list_of_exfil_amts.extend(pre_specified_data_attribs)
            self.list_of_injected_graphs_loc.extend(injected_graph_obj_loc)
            if amt_of_out_traffic_bytes == 0:
                print "okay, something MUST be wrong!!"

            self.list_of_amt_of_out_traffic_bytes.extend(amt_of_out_traffic_bytes)
            self.list_of_amt_of_out_traffic_pkts.extend(amt_of_out_traffic_pkts)

            self.current_total_node_list = current_total_node_list

# not a great choice as a module becase then I can't run a multiprocess
def calc_ide_angles(aggregate_csv_edgefile_loc, joint_col_list, window_size, raw_edgefile_names, out_q, calc_ide):
    # okay, so what this'll probably be is just a way of interacting with common lisp...

    #if calc_ide: #False: #calc_ide:
    # step 1: setup the file with the params...
    with open('./clml_ide_params.txt', 'w') as f:
        # first thing: location of aggregatee-edgefile
        f.write(aggregate_csv_edgefile_loc + '\n')
        # second thing: number of columns ## NOTE: USED TO BE +1
        f.write(str(len(joint_col_list) + 1) + '\n') # plus one is for the unamed column (TODO: figure outwhat is it)
        # third thing: sliding window size
        f.write( str( window_size) + '\n')
        # fourth thing: total time
        f.write( str( len(raw_edgefile_names))  + '\n')
        # fifth thing: output file location
        f.write( aggregate_csv_edgefile_loc + '_clml_ide_results.txt' )

    time.sleep(10)

    # step 2: start sbcl on the appropriate script...
    #        out = subprocess.check_output(['sbcl', "--dynamic-space-size", "2560", "--script", "clml_ide.lisp"])
    print "calling sbcl now..."
    # note: http://quickdocs.org/clml/ indicates that a dynamic-space-size of 2560 should be sufficient, but
    # in my experience, that's actually not big enough (i.e. it'll crash unless you give it more)
    out = subprocess.check_output(['sbcl', "--dynamic-space-size", "5540", "--script", "clml_ide.lisp"])
    print "ide_out", out

    # step 3: copy the results into the appropriate location...
    ## okay, let's just store it in a seperate location, cause that'll be easier, I guess...
    try:
        with open(aggregate_csv_edgefile_loc + '_clml_ide_results.txt', 'r') as f:
            cont = f.read()
        cont_list = cont.split(" ")
        angles_list = []
        for i in cont_list:
            angles_list.append( i.replace("(", "").replace(")", "").rstrip().lstrip() )

        angles_list = [float('NaN') for i in range(0, window_size)] + angles_list

        angles_list = [float(i) for i in angles_list]
        if out_q:
            out_q.put(angles_list)
        else:
            return angles_list
    except:
        return [0 for i in range(0,len(raw_edgefile_names))]

def process_and_inject_single_graph(counter_starting, file_paths, svcs, is_swarm, ms_s, container_to_ip, infra_instances,
                                    synthetic_exfil_paths, initiator_info_for_paths, attacks_to_times,
                    time_interval, total_edgelist_nodes, svc_to_pod, avg_dns_weight, avg_dns_pkts,
                    node_attack_mapping, out_q, current_total_node_list,name_of_dns_pod_node,attack_injected, carryover,
                    avg_exfil_per_min, exfil_per_min_variance, avg_pkt_size, pkt_size_variance, end_of_training,
                     pod_creation_log, injection_rate_exp_path, drop_infra_from_graph):

    ## TODO: finish refining this function!!! I just need to move some logic around / modify for the upgraded
    ## experimental apparatus but the core logic is here.

    concrete_cont_node_path_list = []
    pre_specified_data_attribs_list = []
    injected_graph_obj_loc_list = []
    amt_of_out_traffic_bytes = []
    amt_of_out_traffic_pkts = []
    attack_occuring_list = []

    for counter_add, file_path in enumerate(file_paths):
        counter = counter_starting + counter_add
        cur_time = (counter * time_interval)
        past_end_of_training = cur_time > end_of_training
        print "cur_time",cur_time, "end_of_training",end_of_training, "cur_time > end_of_training",cur_time > end_of_training
        if past_end_of_training:
            print "wowsers!!"
        gc.collect()
        G = nx.DiGraph()
        print "path to file is ", file_path

        if counter == 6:
            print "let's walk through it  manually..."

        f = open(file_path, 'r')
        lines = f.readlines()
        nx.parse_edgelist(lines, delimiter=' ', create_using=G)

        potential_name_of_dns_pod_node = find_dns_node_name(G)
        if potential_name_of_dns_pod_node != None:
            name_of_dns_pod_node = potential_name_of_dns_pod_node
        logging.info("name_of_dns_pod_node, " + str(name_of_dns_pod_node))
        print "name_of_dns_pod_node", name_of_dns_pod_node


        ### TODO: want to update::: container_to_ip using the pod/creation log (this is called: pod_creation_log)
        ## UPDATE: CORRECT. THAT'S WHY I TOOK IT OUT. can remove all these comments (included commented out code, at some point...)
        ## okay, testing this ATM. if it works, then I can delete the line above
        ###### WAIT, I DON'T THINK THIS ACTUALLY DOES ANYTHING!!! ##########
        ##### I need this now b/c it is useful to have infra_instances #####
        container_to_ip, infra_instances = update_mapping(container_to_ip, pod_creation_log, time_interval, counter, infra_instances)

        ###print "cur_time", cur_time, "container_to_ip_zz", container_to_ip

        cur_1si_G = prepare_graph(G, None, 'app_only', is_swarm, counter, file_path, ms_s,
                                  container_to_ip, infra_instances, drop_infra_p=drop_infra_from_graph)

        into_outside_bytes, into_outside_pkts = find_amt_of_out_traffic(cur_1si_G)
        amt_of_out_traffic_bytes.append(into_outside_bytes)
        amt_of_out_traffic_pkts.append(into_outside_pkts)


        name_of_file = file_path.split('/')[-1]

        prefix_for_inject_params = 'avg_exfil_' + str(avg_exfil_per_min) + '_var_' + str(exfil_per_min_variance) #\
                                   #+ '_avg_pkt_' + str(avg_pkt_size) + ':' + str(pkt_size_variance) + '_'

        name_of_injected_file =  prefix_for_inject_params +  file_path.split('/')[-1]

        edgefile_pruned_folder_path = injection_rate_exp_path + '/pruned_edgefiles/'
        graph_obj_folder_path = injection_rate_exp_path #+ '/graph_objs/'

        # let's save a copy of the edgefile for the graph w/ the injected attack b/c that'll help with debugging
        # the system...
        edgefile_injected_folder_path = injection_rate_exp_path + '/injected_edgefiles/'

        ## if the graph object folder directory doesn't currently exist, then we'd want to create it...
        ## using the technique from https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python

        try:
            os.makedirs(graph_obj_folder_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        try:
            os.makedirs(edgefile_injected_folder_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        ## if the pruned folder directory doesn't currently exist, then we'd want to create it...
        ## using the technique from https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python
        try:
            os.makedirs(edgefile_pruned_folder_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        nx.write_edgelist(cur_1si_G, edgefile_pruned_folder_path + name_of_file, data=['frames', 'weight'])
        nx.write_gpickle(cur_1si_G, edgefile_pruned_folder_path + 'with_nodeAttribs' + name_of_file)


        ### NOTE: I think this is where we'd want to inject the synthetic attacks...
        for node in cur_1si_G.nodes():
            if node not in current_total_node_list:
                current_total_node_list.append(node)

        # print "right after graph is prepared", level_of_processing, list(cur_G.nodes(data=True))
        logging.info("svcs, " + str(svcs))

        cur_1si_G, node_attack_mapping, pre_specified_data_attribs, concrete_cont_node_path,carryover,attack_injected = \
        inject_synthetic_attacks(cur_1si_G, synthetic_exfil_paths, attacks_to_times, counter, node_attack_mapping,
                                 name_of_dns_pod_node, carryover, attack_injected, time_interval, avg_exfil_per_min,
                                 exfil_per_min_variance, avg_pkt_size, pkt_size_variance, container_to_ip)
        if attack_injected:
            attack_occuring_list.append(synthetic_exfil_paths[attack_injected])
        else:
            attack_occuring_list.append(0)

        # new nodes exist now... so need to do a limited amt of re-processing...
        containers_to_ms,svcs = map_nodes_to_svcs(cur_1si_G, svcs, container_to_ip)
        nx.set_node_attributes(cur_1si_G, containers_to_ms, 'svc')
        if drop_infra_from_graph:
            infra_nodes = find_infra_components_in_graph(cur_1si_G, infra_instances)
            cur_1si_G = remove_infra_from_graph(cur_1si_G, infra_nodes)

        ## note this is actually fine, I think...
        for thing in cur_1si_G.nodes(data=True):
            logging.info(thing)
            try:
                # print thing[1]['svc']
                cur_svc = thing[1]['svc']
                if 'VIP' not in thing[0] and cur_svc not in svc_to_pod:
                    svc_to_pod[cur_svc] = [thing[0]]
                else:
                    if 'VIP' not in thing[0] and thing[0] not in svc_to_pod[cur_svc]:
                        svc_to_pod[cur_svc].append(thing[0])
            except:
                # print "there was a svc error"
                pass

        pod_to_svc = reverse_svc_to_pod_dict(svc_to_pod)


        attack_happened_p = 0
        if attack_injected:
            attack_happened_p = 1

        if counter == 5:
            pass
        cur_class_G = prepare_graph(cur_1si_G, svcs, 'class', is_swarm, counter, file_path, ms_s, container_to_ip,
                                    infra_instances, drop_infra_p=drop_infra_from_graph)

        ## if the injected folder directory doesn't currently exist, then we'd want to create it...
        ## using the technique from https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python
        try:
            os.makedirs(edgefile_injected_folder_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        nx.write_edgelist(cur_1si_G, edgefile_injected_folder_path + name_of_injected_file,
                          data=['frames', 'weight'])

        nx.write_edgelist(cur_class_G, edgefile_injected_folder_path + 'class_' + name_of_injected_file,
                          data=['frames', 'weight'])

        nx.write_gpickle(cur_1si_G, edgefile_injected_folder_path + 'with_nodeAttribs' + name_of_injected_file)
        nx.write_gpickle(cur_class_G, edgefile_injected_folder_path + 'class_' + 'with_nodeAttribs' + name_of_injected_file)

        # injected_filenames.append(edgefile_injected_folder_path+name_of_file)
        total_edgelist_nodes = update_total_edgelist_nodes_if_needed(cur_1si_G, total_edgelist_nodes)
        # injected_filenmames[counter] = edgefile_injected_folder_path + name_of_file

        name = name_of_file
        injected_graph_obj_loc = graph_obj_folder_path + 'graph_obj_' + name_of_injected_file
        print("injected_graph_obj_loc", injected_graph_obj_loc)
        injected_graph_obj = injected_graph(name, edgefile_injected_folder_path + name_of_injected_file,
                                            edgefile_pruned_folder_path + name_of_file,
                                            concrete_cont_node_path,
                                            pre_specified_data_attribs, svc_to_pod, pod_to_svc,
                                            total_edgelist_nodes,
                                            injected_graph_obj_loc, counter, name_of_dns_pod_node,
                                            current_total_node_list,
                                            svcs, 0, ms_s, container_to_ip, infra_instances,
                                            edgefile_injected_folder_path + 'class_' + name_of_injected_file,
                                            name_of_injected_file,
                                            edgefile_injected_folder_path + 'with_nodeAttribs' + name_of_injected_file,
                                            edgefile_injected_folder_path + 'class_' + 'with_nodeAttribs' + name_of_injected_file,
                                            edgefile_pruned_folder_path + 'with_nodeAttribs' + name_of_file,
                                            past_end_of_training, attack_happened_p,
                                            drop_infra_from_graph)

        injected_graph_obj.save()
        # at 53: 4.04 GB

        concrete_cont_node_path_list.append(concrete_cont_node_path)
        pre_specified_data_attribs_list.append(pre_specified_data_attribs)
        injected_graph_obj_loc_list.append(injected_graph_obj_loc)

        print "into_outside_bytes", into_outside_bytes
        if into_outside_bytes == 0:
            for (u,v,d) in G.edges(data=True):
                print (u,v,d)
                if 'outside' in u or 'outside' in v:
                    print (u,v,d)
            print "into_outside_bytes equals ZERO!! CRAZY!!!"
            #exit(222) # it can happen sometimes...

        del injected_graph_obj  # help??
        cur_1si_G.clear()
        del cur_1si_G  # help??
        cur_class_G.clear()
        del cur_class_G  # help
        G.clear()
        del G  # help

    out_q.put(concrete_cont_node_path_list)
    out_q.put(pre_specified_data_attribs_list)
    out_q.put(injected_graph_obj_loc_list)
    out_q.put(avg_dns_weight)
    out_q.put(avg_dns_pkts)
    #out_q.put(class_attack_mapping)
    out_q.put(node_attack_mapping)
    out_q.put(current_total_node_list)
    out_q.put(name_of_dns_pod_node)
    out_q.put(attack_injected)
    out_q.put(carryover)
    out_q.put(amt_of_out_traffic_bytes)
    out_q.put(amt_of_out_traffic_pkts)
    out_q.put(container_to_ip)
    out_q.put(attack_occuring_list)

def find_amt_of_out_traffic(cur_1si_G):
    into_outside_bytes = 0
    into_outside_pkts = 0
    edges_into_outside = cur_1si_G.in_edges('outside',data=True)
    for (u,v,d) in edges_into_outside:
        into_outside_bytes += d['weight']
        into_outside_pkts +=  d['frames']
    return into_outside_bytes,into_outside_pkts

def inject_synthetic_attacks(graph, synthetic_exfil_paths, attacks_to_times, graph_number, attack_number_to_mapping,
                            name_of_dns_pod_node,old_carryover, last_attack, time_gran, avg_exfil_per_min, exfil_per_min_variance,
                             avg_pkt_size, pkt_size_variance, container_to_ip):
    # (1) identify whether a synthetic attack is injected here
    # (2) identify whether this is the first occurence of injection... if it was injected
    ## earlier, then we need to re-use the mappings...
    # (3) add the weights...

    current_time = graph_number
    attack_occuring = None
    #fraction_of_pkt_min = 0
    #fraction_of_weight_min = 0

    # step (1) : identify whether an attack needs to be injected here
    #print "attacks_to_times", attacks_to_times, type(attacks_to_times), current_time, node_granularity
    for counter, attack_ranges in enumerate(attacks_to_times):
        if current_time >= attack_ranges[0] and current_time < attack_ranges[1]:
            # then the attack occurs during this interval....
            attack_occuring = counter % len(synthetic_exfil_paths)
            print "attack in range found!"
            break

    if attack_occuring == None:
        return graph, attack_number_to_mapping, {'weight': 0,
                                                 'frames': 0}, [], 0, None

    # step (2): is this the first contiguous time iteration that this kind of attack occurs?
    # if yes -> must determine the node mapping now
    # if no -> use node mapping from before (requires no work b/c needed info already in dict)
    if attack_occuring not in attack_number_to_mapping.keys() or last_attack != attack_occuring:
        current_mapping = {} # abstract_node -> concrete_node
        for node in synthetic_exfil_paths[attack_occuring]:
            # if node not in current_mapping.keys() # doesn't actually matter...
            if node == 'kube_dns_pod':
                print "kube_dns_pod found in mapping function!!", name_of_dns_pod_node
                current_mapping['kube_dns_pod'] = name_of_dns_pod_node
            else:
                print "node_to_map", node
                concrete_node = abstract_to_concrete_mapping(node, graph, [], container_to_ip)
                if 'dns' in node:
                    print "new_mapping!", node, concrete_node
                current_mapping[node] = concrete_node
        attack_number_to_mapping[attack_occuring] = current_mapping

    concrete_node_path = determine_concrete_node_path(synthetic_exfil_paths[attack_occuring], attack_number_to_mapping[attack_occuring])
    print "abstract node exfil path", synthetic_exfil_paths[attack_occuring]
    print "concrete_node_exfil_path", concrete_node_path

    fraction_of_weight_min, fraction_of_pkt_min = determine_exfiltration_amt(avg_exfil_per_min, exfil_per_min_variance,
                                                                             avg_pkt_size, pkt_size_variance, time_gran,
                                                                             old_carryover)

    print "fraction_of_weight_min_", fraction_of_weight_min, "fraction_of_pkt_min_",fraction_of_pkt_min

    ## Step (4): use the previously calculated exfiltration rate to actually make the appropriate modifications
    ## to the graph.

    # 4a. if the fraction_of_weight_min < 40, then it's too small to be in any packet. In fact, let's make the
    # minimum size 60 since otherwise it seems kinda stupid
    if fraction_of_weight_min < 60:
        return graph, attack_number_to_mapping, {'weight':0, 'frames': 0}, \
               ['exfil amt too small, not doing'], fraction_of_weight_min, attack_occuring

    for concrete_node_pair in concrete_node_path:
        graph = add_edge_weight_graph(graph, concrete_node_pair[0], concrete_node_pair[1],
                                      fraction_of_weight_min, fraction_of_pkt_min)

    print "modifications_to_graph...", concrete_node_path, fraction_of_weight_min, fraction_of_pkt_min

    return graph, attack_number_to_mapping, {'weight':fraction_of_weight_min, 'frames': fraction_of_pkt_min}, \
           concrete_node_path, 0, attack_occuring

def determine_concrete_node_path(current_exfil_path, current_abstract_to_physical_mapping):
    concrete_node_path = []
    node_one_loc = 0
    while node_one_loc < (len(current_exfil_path) - 1):
        abstract_node_pair = (current_exfil_path[node_one_loc], current_exfil_path[node_one_loc + 1])
        concrete_possible_dst = current_abstract_to_physical_mapping[abstract_node_pair[1]]
        print "abstract_node_pair", abstract_node_pair
        print "concrete_possible_dst", concrete_possible_dst

        ### there are two subcases of this first case.
        ### (1): first exp_support_scripts initiates flow [pod (DST) and vip are same service]
        ### (2). dst initiates flow [pod (exp_support_scripts) and vip are same service
        if 'VIP' in concrete_possible_dst:
            ## in this case, we need to compensate for the VIP re-direction that occurs
            ## in the Kubernetes VIP.
            concrete_node_src_one = current_abstract_to_physical_mapping[abstract_node_pair[0]]
            concrete_node_src_two = current_abstract_to_physical_mapping[abstract_node_pair[1]]
            abstract_node_dst = current_exfil_path[node_one_loc + 2]
            concrete_node_dst = current_abstract_to_physical_mapping[abstract_node_dst]
            print "vip_located_xx", concrete_node_src_one, concrete_node_src_two, concrete_node_dst
            print current_exfil_path
            print current_abstract_to_physical_mapping
            print "concrete_node_path", node_one_loc, concrete_node_path
            if abstract_node_pair_same_service_p(abstract_node_pair[0], abstract_node_pair[1]):
                node_one_loc += 1  # b/c we're modifying two edges here, we need to increment the counter one more time...
                concrete_node_path.append((concrete_node_src_one, concrete_node_dst))
                concrete_node_path.append((concrete_node_src_two, concrete_node_dst))
                print "concrete_node_path", node_one_loc, concrete_node_path
            elif abstract_node_pair_same_service_p(abstract_node_dst, abstract_node_pair[1]):
                node_one_loc += 1  # b/c we're modifying two edges here, we need to increment the counter one more time...
                concrete_node_path.append((concrete_node_src_one, concrete_node_src_two))
                concrete_node_path.append((concrete_node_src_one, concrete_node_dst))
            else:
                print "apparently a vip in the path doesn't belong to either service??"
                exit(544)
        else:
            # this case does not involve any redirection via the kubernetes network model, so it is simple
            concrete_node_src = current_abstract_to_physical_mapping[abstract_node_pair[0]]
            concrete_node_dst = current_abstract_to_physical_mapping[abstract_node_pair[1]]
            concrete_node_path.append((concrete_node_src, concrete_node_dst))
            print "concrete_node_path", node_one_loc, concrete_node_path, concrete_node_src, concrete_node_dst
        node_one_loc += 1

    return concrete_node_path


# abstract_to_concrete_mapping: abstract_node graph -> concrete_node (in graph)
def abstract_to_concrete_mapping(abstract_node, graph, excluded_list, container_to_ip):
    #print "abstract_to_concrete_mapping", abstract_node, graph.nodes(),node_granularity
    ## okay, so there's a couple of things that I should do???
    if abstract_node == 'internet':
        abstract_node = 'outside'
    else:
        #if node_granularity == 'class':
        #    abstract_node = abstract_node.replace('_pod', '').replace('_vip', '').replace('_','-')
        #else:
        if '_pod' in abstract_node:
            abstract_node = abstract_node.replace('_pod', '')
            abstract_node = abstract_node.replace('_', '-')
        elif '_vip' in abstract_node:
            abstract_node = abstract_node.replace('_vip', '')
            abstract_node = abstract_node.replace('_', '-')
            abstract_node = abstract_node + '_VIP'
        else:
            print "neither internet nor pod nor vip"
            exit(453)

    print "modified abstract_node", abstract_node
    #matching_concrete_nodes = [node for node in graph.nodes() if abstract_node in node if node not in excluded_list]

    matching_concrete_nodes = []
    for node in graph.nodes(data=True):
        try:
            svc = node[1]['svc']
        except:
            svc = None
        if abstract_node == 'user-db' and node[0] == 'user-db_VIP':
            pass
        if match_name_to_pod(abstract_node, node[0], svc=svc) and node[0] not in excluded_list:
            matching_concrete_nodes.append(node[0])

    #matching_concrete_nodes = [node[0] for node in graph.nodes(data=True) if match_name_to_pod(abstract_node, node[0],svc=node[1]['svc']) if node[0] not in excluded_list]
    print "matching_concrete_nodes", matching_concrete_nodes
    try:
        concrete_node = random.choice(matching_concrete_nodes)
    except:
        if 'VIP' in abstract_node or 'outside' in abstract_node:
            concrete_node = abstract_node # must be a node that isn't present in the graph
        else:
            # the pod is not present in the graph... this is kinda problematic, but in terms of labelling, let's
            # just take it from the stored names
            matching_concrete_nodes = [node[0] for node in container_to_ip.values() if match_name_to_pod(abstract_node, node[0], svc=node[4]) if
                                       node[0] not in excluded_list]
            if matching_concrete_nodes == []:
                print "abstract", abstract_node
                print "abstract_to_concrete_mapping of made-up node"
                exit(237)

            concrete_node = random.choice(matching_concrete_nodes)
    print "concrete_node", concrete_node, "abstract_node", abstract_node
    return concrete_node


def add_edge_weight_graph(graph, concrete_node_src, concrete_node_dst, fraction_of_weight_median,
                          fraction_of_pkt_median):
    if concrete_node_src == None:
        print "Error: concrete_node_src is None"
        exit(455)
    if concrete_node_dst == None:
        print "Error: concrete_node_dst is None"
        exit(344)

    ack_packet_size = 40  # bytes
    if concrete_node_src in graph and concrete_node_dst in graph:
        pass  # don't need to do anything b/c the nodes already exist...
    elif concrete_node_src in graph:
        # concrete_node_dst not in graph --> need to add a node
        graph.add_node(concrete_node_dst)
    elif concrete_node_dst in graph:
        # need to add a node
        graph.add_node(concrete_node_src)
    else:
        print "concrete_node_dst",concrete_node_dst
        graph.add_node(concrete_node_dst)
        graph.add_node(concrete_node_src)

    # if packet bytes is > 60, let' sgo ahead and add it
    # if no edge exists, then we need to add one...
    if not graph.has_edge(concrete_node_src, concrete_node_dst):
        graph.add_edge(concrete_node_src, concrete_node_dst, weight=0, frames=0)
    graph[concrete_node_src][concrete_node_dst]['weight'] += fraction_of_weight_median
    graph[concrete_node_src][concrete_node_dst]['frames'] += fraction_of_pkt_median

    # now need to account for the acks...
    if not graph.has_edge(concrete_node_dst, concrete_node_src):
        graph.add_edge(concrete_node_dst, concrete_node_src, weight=0, frames=0)
    graph[concrete_node_dst][concrete_node_src]['weight'] += (fraction_of_pkt_median * ack_packet_size)
    graph[concrete_node_dst][concrete_node_src]['frames'] += fraction_of_pkt_median

    return graph

def abstract_node_pair_same_service_p(abstract_node_one, abstract_node_two):
    ## okay, well, let's give this a shot
    abstract_node_one_core = abstract_node_one.replace('_pod', '').replace('_vip', '')
    abstract_node_two_core = abstract_node_two.replace('_pod', '').replace('_vip', '')
    return abstract_node_one_core == abstract_node_two_core

def avg_behavior_into_dns_node(pre_injection_weight_into_dns_dict, pre_inject_packets_into_dns_dict, pod_to_svc):
    avg_dns_weight = 0.0
    avg_dns_pkts = 0.0
    non_null_edges = 0
    #app_pods = pod_to_svc.keys()
    for node in pre_injection_weight_into_dns_dict.keys():
        #print("pre_injection_weight_into_dns_dict",pre_injection_weight_into_dns_dict)
        #print("node", node, node in pod_to_svc, pod_to_svc.keys())
        if node in pod_to_svc:
            weight = pre_injection_weight_into_dns_dict[node]
            pkts = pre_inject_packets_into_dns_dict[node]
            if pkts != 0 or weight != 0:
                avg_dns_weight += weight
                avg_dns_pkts += pkts
                non_null_edges += 1
    if non_null_edges != 0:
        avg_dns_weight = avg_dns_weight / non_null_edges
        avg_dns_pkts = avg_dns_pkts / non_null_edges
    #else:
    #    avg_dns_weight = 0.0
    #    avg_dns_pkts = 0.0

    return avg_dns_weight, avg_dns_pkts

#'''
def determine_exfiltration_amt(avg_exfil_per_min, exfil_per_min_variance, avg_pkt_size, pkt_size_variance, time_gran, carryover):
    exfil_amt = numpy.random.normal(loc=avg_exfil_per_min, scale=exfil_per_min_variance) + carryover
    pkt_size = numpy.random.normal(loc=avg_pkt_size, scale=pkt_size_variance)

    print "exfil_amt",exfil_amt,"pkt_size",pkt_size
    # okay, so the exfil amt is what it'd be over a whole minute so
    exfil_amt_in_this_time_interval = (exfil_amt) * (1/60.0)  * time_gran
    exfil_amt_in_this_time_interval = int(math.floor(exfil_amt_in_this_time_interval))
    pkts_in_this_time_interval = int(math.floor(exfil_amt_in_this_time_interval / pkt_size))

    print "exfil_amt_in_this_time_interval",exfil_amt_in_this_time_interval,"pkts_in_this_time_interval",pkts_in_this_time_interval
    return exfil_amt_in_this_time_interval, pkts_in_this_time_interval

def pairwise_metrics(G, svc_to_nodes):
    svc_pair_to_reciprocity = {}
    svc_pair_to_density = {}
    svc_pair_to_coef_of_var = {}
    svc_triple_to_degree_coef_of_var = {}

    for svc_one,nodes_one in svc_to_nodes.iteritems():
        for svc_two,nodes_two in svc_to_nodes.iteritems():
            orig_nodes_one = copy.deepcopy(nodes_one)
            orig_nodes_two = copy.deepcopy(nodes_two)
            nodes_one_with_vip = nodes_one + [svc_one + '_VIP']
            nodes_two_with_vip = nodes_two + [svc_two + '_VIP']
            if svc_one != svc_two:
                ## okay, so let's calculate the actual  metrics here
                ## (a) add VIPs to the lists
                ## (b) make subgraph [done]
                subgraph = G.subgraph(nodes_one_with_vip + nodes_two_with_vip).copy()

                '''
                print "nodes_one_with_vip", nodes_one_with_vip
                print "nodes_two_with_vip", nodes_two_with_vip
                print [node for node in G.nodes]
                print subgraph.nodes
                for (u, v, weight) in G.edges(data='weight'):
                    print (u, v, weight)
                print "@@@"
                for (u, v, weight) in subgraph.edges(data='weight'):
                    print (u, v, weight)
                print "subgraph-end"
                '''

                subgraph = make_bipartite(subgraph, nodes_one_with_vip, nodes_two_with_vip)
                ## (c) calculate subgraph values [done]
                # bipartite_density = bipartite.density(subgraph, nodes_one_with_vip)
                bipartite_density = nx.density(subgraph)
                weighted_reciprocity, _, _ = network_weidge_weighted_reciprocity(subgraph)
                #print svc_one, "to", svc_two
                coef_of_var = find_coef_of_variation(subgraph, nodes_one)
                # ^^^ NOTE: using nodes_one instead of nodes_one_with_vip b/c we don't want the vip in the
                # coef_of_variation calculation b/c that value is different than the normal container-to-container
                # connections

                ## (d) store them somewhere accessible and return [done]
                '''TODO: don't wan to put the equiv ones both ways...'''
                if svc_one < svc_two:
                    svc_pair_to_density[(svc_one, svc_two)] = bipartite_density
                    svc_pair_to_reciprocity[(svc_one,svc_two)] = weighted_reciprocity
                    svc_pair_to_coef_of_var[(svc_one,svc_two)] = coef_of_var
                #print "between_stuff", svc_one, svc_two, len(subgraph.edges(data=True)), subgraph.edges(data=True)

                '''
                service_one_degrees = subgraph.degree(nbunch=orig_nodes_one)
                service_one_degrees_list = []
                for svc_one_node_node_degree_tuple in service_one_degrees:
                    svc_one_node = svc_one_node_node_degree_tuple[0]
                    node_degree = svc_one_node_node_degree_tuple[1]
                    service_one_degrees_list.append( node_degree )
                try:
                    svc_triple_to_degree_coef_of_var[(svc_one, svc_two, svc_one)] = np.std(service_one_degrees_list) / np.mean(service_one_degrees_list)
                except:
                    svc_triple_to_degree_coef_of_var[(svc_one, svc_two, svc_one)] = 0.0

                service_two_degrees = subgraph.degree(nbunch=orig_nodes_two)
                service_two_degrees_list = []
                for svc_two_node_node_degree_tuple in service_two_degrees:
                    svc_two_node = svc_two_node_node_degree_tuple[0]
                    node_degree = svc_two_node_node_degree_tuple[1]
                    service_two_degrees_list.append( node_degree )
                #print("service_two_degrees_list", service_two_degrees_list, np.std(service_two_degrees_list), np.mean(service_two_degrees_list))
                try:
                    svc_triple_to_degree_coef_of_var[(svc_one, svc_two, svc_two)] = np.std(service_two_degrees_list) / np.mean(service_two_degrees_list)
                except:
                    svc_triple_to_degree_coef_of_var[(svc_one, svc_two, svc_two)] = 0.0
                '''

                #### TODO: remove when I am done w/ looking at visualizations...
                ''''
                plt.figure(figsize=(12, 12))  # todo: turn back to (27, 16)
                plt.title(svc_one + '_' + svc_two + '___' + str(bipartite_density) + '___' + str(weighted_reciprocity))
                pos = graphviz_layout(subgraph)
                for key in pos.keys():
                    pos[key] = (pos[key][0] * 4, pos[key][1] * 4)  # too close otherwise
                nx.draw_networkx(subgraph, pos, with_labels=True, arrows=True, font_size=8, font_color='b')
                edge_labels = nx.get_edge_attributes(subgraph, 'weight')
                nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=7, label_pos=0.3)
                '''

            else:
                subgraph = G.subgraph(nodes_one_with_vip)
                svc_pair_to_density[(svc_one, svc_two)] = nx.density(subgraph)
                weighted_reciprocity, _, _ = network_weidge_weighted_reciprocity(subgraph)
                svc_pair_to_reciprocity[(svc_one, svc_two)] = weighted_reciprocity
                #print "self_stuff", svc_one, subgraph.edges(data=True)

                '''
                service_one_degrees = subgraph.degree(nbunch=orig_nodes_one)
                service_one_degrees_list = []
                for svc_one_node_node_degree_tuple in service_one_degrees:
                    svc_one_node = svc_one_node_node_degree_tuple[0]
                    node_degree = svc_one_node_node_degree_tuple[1]
                    service_one_degrees_list.append( node_degree )
                '''
                #svc_triple_to_degree_coef_of_var[(svc_one, svc_two, svc_one)] = np.std(service_one_degrees_list) / np.mean(service_one_degrees_list)


    ###plt.show()

    return svc_pair_to_reciprocity, svc_pair_to_density, svc_pair_to_coef_of_var#, svc_triple_to_degree_coef_of_var

def make_bipartite(G, node_set_one, node_set_two):
    for node_one in node_set_one:
        for node_two in node_set_one:
            try:
                G.remove_edge(node_one, node_two)
            except:
                pass
            try:
                G.remove_edge(node_two, node_one)
            except:
                pass


    for node_one in node_set_two:
        for node_two in node_set_two:
            try:
                G.remove_edge(node_one, node_two)
            except:
                pass
            try:
                G.remove_edge(node_two, node_one)
            except:
                pass
    for node in G.nodes():
        if node not in node_set_two and node not in node_set_one:
            G.remove_node(node)

    #nx.draw(G)
    #plt.show()
    #plt.savefig('./bipartite_example.png')

    #print "node_set_one", node_set_one
    #print "node_set_two", node_set_two
    #print [i for i in G.nodes()]
    #exit(34)
    return G

# https://en.wikipedia.org/wiki/Coefficient_of_variation
def find_coef_of_variation(G, start_nodes):
    edge_weights = []
    for (u,v,weight) in G.edges(data='weight'):
        #print (u,v,weight)
        if u in start_nodes:
            edge_weights.append(weight)
    # now find the coef of var...
    # recall: is std_dev / mean
    stddev_of_edge_weights = np.std(edge_weights)
    mean_of_edge_weights = np.mean(edge_weights)

    #print "start_nodes", start_nodes
    #print "edge_weights",edge_weights
    #print "stddev_of_edge_weights",stddev_of_edge_weights
    #print "mean_of_edge_weights",mean_of_edge_weights,
    #print "coef_of_var", stddev_of_edge_weights / mean_of_edge_weights
    #print "-----"

    return stddev_of_edge_weights / mean_of_edge_weights


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

                # print "edge!!!", edge
                # node_to_nodeTwo = G.out_edges(nbunch=[node, node_two])
                # nodeTwo_to_node = G.in_edges(nbunch=[node, node_two])
                # print "len edges",  node_to_nodeTwo, nodeTwo_to_node
                reciprocated_weight[node] = min(node_to_nodeTwo, nodeTwo_to_node)
                non_reciprocated_out_weight[node] = max(node_to_nodeTwo - reciprocated_weight[node], 0)
                non_reciprocated_in_weight[node] = max(nodeTwo_to_node - reciprocated_weight[node], 0)

    # only goes through out-edges (so no double counting)
    total_weight = 0
    for edge in G.edges(data=True):
        # print edge
        # input("Look!!!")
        try:
            total_weight += edge[2]['weight']
        except:
            pass  # maybe the edge does not exist

    total_reicp_weight = 0
    for recip_weight in reciprocated_weight.values():
        total_reicp_weight += recip_weight

    if total_weight == 0:
        weighted_reciprocity = -1  # sentinal value
    else:
        weighted_reciprocity = float(total_reicp_weight) / float(total_weight)

    return weighted_reciprocity, non_reciprocated_out_weight, non_reciprocated_in_weight

def make_edgelist_dict(cur_1si_G, total_edgelist_nodes):
    edgelist  = nx.to_edgelist(cur_1si_G)
    edgelist_dict = {}
    for start,stop,val in edgelist:
        edgelist_dict[(start,stop)] = val['weight']
    for edge_pair in total_edgelist_nodes:
        if edge_pair not in edgelist_dict:
            edgelist_dict[edge_pair] = 0
    #print "edgelist_dict",
    return edgelist_dict

def find_coef_of_var_for_nodes(node_feature_val_dict, svc_to_pod):
    class_to_coef_var_dict = {}
    class_to_mean_dict = {}
    class_to_max_dict = {}
    for svc,list_of_pods in svc_to_pod.iteritems():
        current_feature_list = []
        for pod in list_of_pods:
            try:
                current_feature_list.append(node_feature_val_dict[pod])
            except:
                pass

        # coef of var is (standard_deviation / mean)
        feature_stddev = np.std(current_feature_list)
        feature_mean = np.mean(current_feature_list)
        #print "current_feature_list",current_feature_list
        try:
            feature_max = np.max(current_feature_list)
        except:
            feature_max = float('NaN')
        class_to_coef_var_dict[svc] = feature_stddev / feature_mean
        class_to_mean_dict[svc] = feature_mean
        class_to_max_dict[svc] = feature_max

    return class_to_coef_var_dict, class_to_mean_dict, class_to_max_dict


def update_total_edgelist_nodes_if_needed(cur_1si_G, total_edgelist_nodes):
    for node_one in cur_1si_G.nodes():
        for node_two in cur_1si_G.nodes():
            if (node_one,node_two) not in total_edgelist_nodes:
                total_edgelist_nodes.append( (node_one,node_two) )
    return total_edgelist_nodes

