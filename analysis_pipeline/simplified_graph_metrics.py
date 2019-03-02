import networkx as nx
import seaborn as sns;
sns.set()
import seaborn as sns;
sns.set()
import math
import csv
import ast
import gc
import numpy as np
from analysis_pipeline.next_gen_metrics import calc_neighbor_metric, generate_neig_dict, create_dict_for_dns_metric, \
    calc_dns_metric, calc_outside_inside_ratio_dns_metric, find_dns_node_name, sum_max_pod_to_dns_from_each_svc,\
    reverse_svc_to_pod_dict,turn_into_list, single_step_outside_inside_ratio_dns_metric
from analysis_pipeline.src.analyze_edgefiles import calc_VIP_metric, change_point_detection, ide_angles
from analysis_pipeline.prepare_graph import prepare_graph, get_svc_equivalents
import random
import copy
import logging
import time
import matplotlib
matplotlib.use('Agg',warn=False, force=True)
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import os,errno
from networkx.algorithms import bipartite
import copy
import process_control_chart
#plt.switch_backend('gtkagg')
import cPickle as pickle
import pyximport
pyximport.install() # am I sure that I want this???
import multiprocessing
import numpy.random

# okay, so things to be aware of:
# (a) we are assuming that if we cannot label the node and it is not loopback or in the '10.X.X.X' subnet, then it is outside

class injected_graph():
    def __init__(self, name, injected_graph_loc, non_injected_graph_loc, concrete_container_exfil_paths, exfil_amt,
                 svc_to_pod, pod_to_svc, total_edgelist_nodes, where_to_save_this_obj, counter, name_of_dns_pod_node,
                 current_total_node_list,
                 svcs, is_swarm, ms_s, container_to_ip, infra_service, injected_class_graph_loc, name_of_injected_file,
                 nodeAttrib_injected_graph_loc, nodeAttrib_injected_graph_loc_class):
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

        self.svcs = svcs
        self.is_swarm = is_swarm
        self.counter = counter
        self.ms_s = ms_s
        self.container_to_ip = container_to_ip
        self.infra_service = infra_service

        self.cur_class_G = None

    def save(self):
        with open(self.where_to_save_this_obj, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def calc_single_step_metrics(self):
        self._load_graph()
        #self._create_class_level_graph()

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
        svc_pair_to_reciprocity, svc_pair_to_density, svc_pair_to_coef_of_var, svc_triple_to_degree_coef_of_var = \
                                                            pairwise_metrics(self.cur_1si_G, svc_to_pod_with_outside)

        ''' # this level of craftsmenship is too low...
        self.graph_feature_dict['list_of_svc_pair_to_reciprocity'] = svc_pair_to_reciprocity
        self.graph_feature_dict['list_of_svc_pair_to_density'] = svc_pair_to_density
        self.graph_feature_dict['list_of_svc_pair_to_coef_of_var'] = svc_pair_to_coef_of_var
        '''

        for svc_pair,reciprocity in svc_pair_to_reciprocity.iteritems():
            self.graph_feature_dict[str(svc_pair[0]) + '_to_' + str(svc_pair[1]) + '_reciprocity'] = reciprocity
            self.graph_feature_dict[str(svc_pair[0]) + '_to_' + str(svc_pair[1]) + '_density'] = svc_pair_to_density[svc_pair]
            self.graph_feature_dict[str(svc_pair[0]) + '_to_' + str(svc_pair[1]) + '_edge_coef_of_var'] = svc_pair_to_coef_of_var[svc_pair]

        eigenvector_centrality_classes = nx.eigenvector_centrality(self.cur_class_G, weight='weight')
        #information_centrality_classes = nx.information_centrality(self.cur_class_G, weight='weight') # not implemented for directed
        betweeness_centrality_classes = nx.betweenness_centrality(self.cur_class_G, weights='weight')
        current_flow_betweenness_centrality_classes = nx.current_flow_betweenness_centrality(self.cur_class_G, weight='weight')
        load_centrality_classes = nx.load_centrality(self.cur_class_G, weight='weight')
        harmonic_centrality_classes = nx.harmonic_centrality(self.cur_class_G, distance='weight')

        degree_classes_list = list(nx.degree(self.cur_class_G)) # all we can get is the list
        for service in self.svc_to_pod.keys():
            self.graph_feature_dict['eigenvector_centrality_of_' + str(service)] = eigenvector_centrality_classes[service]
            #self.graph_feature_dict['information_centrality_of_' + str(service)] = information_centrality_classes[service]
            self.graph_feature_dict['betweeness_centrality_of_' + str(service)] = betweeness_centrality_classes[service]
            self.graph_feature_dict['current_flow_betweeness_centrality_of_' + str(service)] = current_flow_betweenness_centrality_classes[service]
            self.graph_feature_dict['load_centrality_of_' + str(service)] = load_centrality_classes[service]
            self.graph_feature_dict['harmonic_centrality_classes_of_' + str(service)] = harmonic_centrality_classes[service]

        for class_val_degree_val_tuple in degree_classes_list:
            class_val = class_val_degree_val_tuple[0]
            degree_val = class_val_degree_val_tuple[1]
            if class_val in self.svc_to_pod.keys():
                self.graph_feature_dict['class_degree_of_' + str(class_val)] = degree_val

        eigenvector_centrality_nodes = nx.eigenvector_centrality(self.cur_1si_G, weight='weight')
        eigenvector_centrality_coef_var_of_classes = find_coef_of_var_for_nodes(eigenvector_centrality_nodes, self.svc_to_pod)
        #information_centrality_nodes = nx.information_centrality(self.cur_1si_G, weight='weight')
        #information_centrality_coef_var_of_classes = find_coef_of_var_for_nodes(information_centrality_nodes, self.svc_to_pod)
        betweeness_centrality_nodes = nx.betweenness_centrality(self.cur_1si_G, weights='weight')
        betweeness_centrality_coef_var_of_classes = find_coef_of_var_for_nodes(betweeness_centrality_nodes, self.svc_to_pod)
        current_flow_betweenness_centrality_nodes = nx.current_flow_betweenness_centrality(self.cur_1si_G, weight='weight')
        current_flow_betweenness_centrality_coef_var_of_classes = find_coef_of_var_for_nodes(current_flow_betweenness_centrality_nodes, self.svc_to_pod)
        load_centrality_nodes = nx.load_centrality(self.cur_1si_G, weight='weight')
        load_centrality_coef_var_of_classes = find_coef_of_var_for_nodes(load_centrality_nodes, self.svc_to_pod)
        harmonic_centrality_nodes = nx.harmonic_centrality(self.cur_1si_G, distance='weight')
        harmonic_centrality_coef_var_of_classes = find_coef_of_var_for_nodes(harmonic_centrality_nodes, self.svc_to_pod)

        for service in self.svc_to_pod.keys():
            self.graph_feature_dict['eigenvector_centrality_coef_of_var_' + str(service)] = eigenvector_centrality_coef_var_of_classes[service]
            #self.graph_feature_dict['information_centrality_coef_of_var_' + str(service)] = information_centrality_coef_var_of_classes[service]
            self.graph_feature_dict['betweeness_centrality_coef_of_var_' + str(service)] = betweeness_centrality_coef_var_of_classes[service]
            self.graph_feature_dict['current_flow_betweeness_centrality_coef_of_var_' + str(service)] = current_flow_betweenness_centrality_coef_var_of_classes[service]
            self.graph_feature_dict['load_centrality_coef_of_var_' + str(service)] = load_centrality_coef_var_of_classes[service]
            self.graph_feature_dict['harmonic_centrality_coef_of_var_' + str(service)] = harmonic_centrality_coef_var_of_classes[service]

        for svcOne_svcTwo_svcInQuestion, degree_coef_of_var in svc_triple_to_degree_coef_of_var.iteritems():
            svc_one = svcOne_svcTwo_svcInQuestion[0]
            svc_two = svcOne_svcTwo_svcInQuestion[1]
            svc_the_metric_is_for = svcOne_svcTwo_svcInQuestion[2]
            self.graph_feature_dict[str(svc_the_metric_is_for) + "_degree_coef_of_var_in_biparite_graph_of_" + \
                                    str(svc_one) + '_' + str(svc_two)] = degree_coef_of_var


        adjacency_matri = nx.to_pandas_adjacency(self.cur_1si_G, nodelist=self.current_total_node_list)
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


    def load_metrics(self):
        print "metrics_file", self.metrics_file
        with open(self.metrics_file, mode='rb') as f:
            #reader = csv.DictReader(f, self.graph_feature_dict_keys)
            #print "row_in_load_metrics_reader", [rows for rows in reader]
            #self.graph_feature_dict = {rows[0]: rows[1] for rows in reader}

            #self.graph_feature_dict = ast.literal_eval(f.read())
            dict_contents = f.read()
            self.graph_feature_dict = pickle.loads(dict_contents)

    def _create_class_level_graph(self):
        self.cur_class_G = prepare_graph(self.cur_1si_G, self.svcs, 'class', self.is_swarm, self.counter, self.injected_graph_loc,
                                    self.ms_s, self.container_to_ip, self.infra_service)

class set_of_injected_graphs():
    def __init__(self, time_granularity, window_size, raw_edgefile_names,
                svcs, is_swarm, ms_s, container_to_ip, infra_service, synthetic_exfil_paths, initiator_info_for_paths,
                attacks_to_times, collected_metrics_location, current_set_of_graphs_loc,
                 avg_exfil_per_min, exfil_per_min_variance, avg_pkt_size, pkt_size_variance):#, out_q):

        self.list_of_injected_graphs_loc = []
        self.time_granularity = time_granularity
        self.window_size = window_size
        self.raw_edgefile_names = raw_edgefile_names
        self.svcs = svcs
        self.is_swarm = is_swarm
        self.ms_s= ms_s
        self.container_to_ip = container_to_ip
        self.infra_service =infra_service
        self.synthetic_exfil_paths = synthetic_exfil_paths
        self.initiator_info_for_paths =initiator_info_for_paths
        self.attacks_to_times = attacks_to_times
        self.time_interval= time_granularity
        self.collected_metrics_location = collected_metrics_location
        self.current_set_of_graphs_loc = current_set_of_graphs_loc
        #self.out_q = out_q

        self.calculated_values = {}
        self.calculated_values_keys = None

        self.list_of_concrete_container_exfil_paths = []
        self.list_of_exfil_amts = []

        self.avg_exfil_per_min = avg_exfil_per_min
        self.exfil_per_min_variance =  exfil_per_min_variance
        self.avg_pkt_size  = avg_pkt_size
        self.pkt_size_variance =  pkt_size_variance
        self.list_of_amt_of_out_traffic_bytes = []
        self.list_of_amt_of_out_traffic_pkts = []

    def save(self):
        #with open(self.current_set_of_graphs_loc, 'wb') as output:  # Overwrites any existing file.
        #    pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

        with open(self.current_set_of_graphs_loc, 'wb') as f:  # Just use 'w' mode in 3.x
            f.write(pickle.dumps(self))

    def calc_serialize_metrics(self):
        adjacency_matrixes = []
        dns_in_metric_dicts = []

        list_of_svc_pair_to_density = []
        list_of_svc_pair_to_reciprocity = []
        list_of_svc_pair_to_coef_of_var = []
        fraction_pod_comm_but_not_VIP_comms_no_abs = []
        pod_comm_but_not_VIP_comms_no_abs = []
        dns_outside_inside_ratios = []
        dns_list_outside = []
        dns_list_inside = []
        pod_1si_density_list = []
        into_dns_from_outside_list = []
        into_dns_ratio = []

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

            for value_name, value_value in current_graph_feature_dict.iteritems():
                if 'adjacency_matrix' in value_name or 'weight_into_dns_dict' in value_name:
                    # these are handled seperately b/c we gotta get all the data then do some postprocessing
                    continue

                if value_name not in self.calculated_values.keys():
                    self.calculated_values[value_name] = []
                try:
                    self.calculated_values[value_name].append(value_value)
                except:
                    self.calculated_values[value_name].append(0.0)

            '''
            list_of_svc_pair_to_density.append(current_graph_feature_dict['list_of_svc_pair_to_density'])
            list_of_svc_pair_to_reciprocity.append(current_graph_feature_dict['list_of_svc_pair_to_reciprocity'])
            list_of_svc_pair_to_coef_of_var.append(current_graph_feature_dict['list_of_svc_pair_to_coef_of_var'])
            fraction_pod_comm_but_not_VIP_comms_no_abs.append(current_graph_feature_dict['fraction_pod_comm_but_not_VIP_comms_no_abs'])
            pod_comm_but_not_VIP_comms_no_abs.append(current_graph_feature_dict['pod_comm_but_not_VIP_comms_no_abs'])
            dns_outside_inside_ratios.append(current_graph_feature_dict['dns_outside_inside_ratios'])
            dns_list_outside.append(current_graph_feature_dict['dns_list_outside'])
            dns_list_inside.append(current_graph_feature_dict['dns_list_inside'])
            pod_1si_density_list.append(current_graph_feature_dict['pod_1si_density_list'])
            into_dns_from_outside_list.append(current_graph_feature_dict['into_dns_from_outside_list'])
            into_dns_ratio.append(current_graph_feature_dict['into_dns_ratio'])
            '''

            total_edgelist_nodes = injected_graph.total_edgelist_nodes
            current_total_node_list = injected_graph.current_total_node_list


        #total_edgelist_nodes = self.list_of_injected_graphs[-1].total_edgelist_nodes
        #current_total_node_list = self.list_of_injected_graphs[-1].current_total_node_list
        ide_angles_results = ide_angles(adjacency_matrixes, 6, total_edgelist_nodes)
        into_dns_eigenval_angles = change_point_detection(dns_in_metric_dicts, self.window_size, current_total_node_list)


        ## need to store these new results into the format that I've been using this far...
        '''
        for service_pair in list_of_svc_pair_to_density[0].keys():
            self.calculated_values[service_pair[0] + '_' + service_pair[1] + '_density'] = []
            self.calculated_values[service_pair[0] + '_' + service_pair[1] + '_reciprocity'] = []
            self.calculated_values[service_pair[0] + '_' + service_pair[1] + '_coef_of_var'] = []
        for counter,svc_pair_to_density in enumerate(list_of_svc_pair_to_density):
            for service_pair in list_of_svc_pair_to_density[0].keys():
                try:
                    self.calculated_values[service_pair[0] + '_' + service_pair[1] + '_reciprocity'].append(
                        list_of_svc_pair_to_reciprocity[counter][service_pair])
                except:
                    self.calculated_values[service_pair[0] + '_' + service_pair[1] + '_reciprocity'].append(0.0)
                try:
                    self.calculated_values[service_pair[0] + '_' + service_pair[1] + '_density'].append(
                        svc_pair_to_density[service_pair])
                except:
                    self.calculated_values[service_pair[0] + '_' + service_pair[1] + '_density'].append(0.0)
                try:
                    self.calculated_values[service_pair[0] + '_' + service_pair[1] + '_coef_of_var'].append(
                        list_of_svc_pair_to_coef_of_var[counter][service_pair])
                except:
                    self.calculated_values[service_pair[0] + '_' + service_pair[1] + '_coef_of_var'].append(0.0)
        '''
        #self.calculated_values[
        #    'Fraction of Communication Between Pods not through VIPs (no abs)'] = fraction_pod_comm_but_not_VIP_comms_no_abs
        #self.calculated_values['Communication Between Pods not through VIPs (no abs)'] = pod_comm_but_not_VIP_comms_no_abs
        self.calculated_values['Fraction of Communication Between Pods not through VIPs (w abs)'] = \
            [abs(i) for i in self.calculated_values['fraction_pod_comm_but_not_VIP_comms']]
        self.calculated_values['Communication Between Pods not through VIPs (w abs)'] = \
            [abs(i) for i in self.calculated_values['pod_comm_but_not_VIP_comms_no_abs']]

        #self.calculated_values['DNS outside-to-inside ratio'] = dns_outside_inside_ratios
        #self.calculated_values['DNS outside'] = dns_list_outside
        #self.calculated_values['DNS inside'] = dns_list_inside
        #self.calculated_values['1-step-induced-pod density'] = pod_1si_density_list
        #self.calculated_values['into_dns_from_outside'] = into_dns_from_outside_list
        #self.calculated_values['into_dns_ratio'] = into_dns_ratio

        self.calculated_values['into_dns_eigenval_angles'] = into_dns_eigenval_angles
        self.calculated_values['ide_angles'] = ide_angles_results
        self.calculated_values['ide_angles (w abs)'] = [abs(i) for i in ide_angles_results]

        self.calculated_values_keys = self.calculated_values.keys()

        #with open(self.collected_metrics_location, 'wb') as f:  # Just use 'w' mode in 3.x
        #    w = csv.DictWriter(f, self.calculated_values.keys())
        #    w.writeheader()
        #    w.writerow(self.calculated_values)

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

    def load_serialized_metrics(self):
        #with open(self.collected_metrics_location, mode='r') as f:
        #    reader = csv.DictReader(f, self.calculated_values_keys)
        #    self.calculated_values = {rows[0]: rows[1] for rows in reader}

        with open(self.collected_metrics_location, mode='rb') as f:
            cur_contents = f.read()
            #print "cur_contents", cur_contents
            self.calculated_values = pickle.loads(cur_contents)

    def calcuated_single_step_metrics(self):
        print("self.list_of_injected_graphs_loc",self.list_of_injected_graphs_loc)
        for counter, injected_obj_loc in enumerate(self.list_of_injected_graphs_loc):
            print("counter",counter)
            gc.collect()

            with open(injected_obj_loc, 'r') as input_file:
                injected_graph_obj = pickle.load(input_file)

            injected_graph_obj.calc_single_step_metrics()

            injected_graph_obj.save()

    def generate_injected_edgefiles(self):
        current_total_node_list = []
        svc_to_pod = {}
        node_attack_mapping = {}
        class_attack_mapping = {}
        total_edgelist_nodes = []
        avg_dns_weight = 0
        avg_dns_pkts = 0

        svcs = self.svcs
        is_swarm = self.is_swarm
        ms_s = self.ms_s
        container_to_ip = self.container_to_ip
        infra_service = self.infra_service
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

        num_graphs_to_process_at_once = 40
        for counter in range(0, len(self.raw_edgefile_names), num_graphs_to_process_at_once):

            file_paths = self.raw_edgefile_names[counter: counter + num_graphs_to_process_at_once]
            args = [counter, file_paths, svcs, is_swarm, ms_s, container_to_ip, infra_service,
                    synthetic_exfil_paths, initiator_info_for_paths, attacks_to_times,
                    time_interval, total_edgelist_nodes, svc_to_pod, avg_dns_weight, avg_dns_pkts,
                    node_attack_mapping, out_q, current_total_node_list, name_of_dns_pod_node, last_attack_injected,
                    carryover, avg_exfil_per_min, exfil_per_min_variance, avg_pkt_size, pkt_size_variance ]
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
            p.join()

            ## okay, literally the code above should be wrapped in a function call...
            ## however, you'd probably wanna process like 40-50 of these on a single call...
            self.list_of_concrete_container_exfil_paths.extend(concrete_cont_node_path)
            self.list_of_exfil_amts.extend(pre_specified_data_attribs)
            self.list_of_injected_graphs_loc.extend(injected_graph_obj_loc)
            self.list_of_amt_of_out_traffic_bytes.extend(amt_of_out_traffic_bytes)
            self.list_of_amt_of_out_traffic_pkts.extend(amt_of_out_traffic_pkts)


def process_and_inject_single_graph(counter_starting, file_paths, svcs, is_swarm, ms_s, container_to_ip, infra_service,
                                    synthetic_exfil_paths, initiator_info_for_paths, attacks_to_times,
                    time_interval, total_edgelist_nodes, svc_to_pod, avg_dns_weight, avg_dns_pkts,
                    node_attack_mapping, out_q, current_total_node_list,name_of_dns_pod_node,attack_injected, carryover,
                    avg_exfil_per_min, exfil_per_min_variance, avg_pkt_size, pkt_size_variance ):

    concrete_cont_node_path_list = []
    pre_specified_data_attribs_list = []
    injected_graph_obj_loc_list = []
    amt_of_out_traffic_bytes = []
    amt_of_out_traffic_pkts = []

    for counter_add, file_path in enumerate(file_paths):
        counter = counter_starting + counter_add
        gc.collect()
        G = nx.DiGraph()
        print "path to file is ", file_path

        f = open(file_path, 'r')
        lines = f.readlines()
        nx.parse_edgelist(lines, delimiter=' ', create_using=G)

        potential_name_of_dns_pod_node = find_dns_node_name(G)
        if potential_name_of_dns_pod_node != None:
            name_of_dns_pod_node = potential_name_of_dns_pod_node
        logging.info("name_of_dns_pod_node, " + str(name_of_dns_pod_node))
        print "name_of_dns_pod_node", name_of_dns_pod_node

        # if no dns pod in current graph....
        if potential_name_of_dns_pod_node is None:
            # if we know the name of the dns pod...
            if name_of_dns_pod_node is not None:
                # then add in the dns pod and vip to the graph
                # (this is needed for the centrality measures)
                G.add_node(name_of_dns_pod_node)
                G.add_node('kube-dns_VIP')

        logging.info("straight_G_edges")
        #for edge in G.edges(data=True):
        #    logging.info(edge)
        logging.info("end straight_G_edges")

        # nx.read_edgelist(file_path,
        #                 create_using=G, delimiter=',', data=(('weight', float),))
        cur_1si_G = prepare_graph(G, svcs, 'app_only', is_swarm, counter, file_path, ms_s,
                                  container_to_ip, infra_service)

        into_outside_bytes, into_outside_pkts = find_amt_of_out_traffic(cur_1si_G)
        amt_of_out_traffic_bytes.append(into_outside_bytes)
        amt_of_out_traffic_pkts.append(into_outside_pkts)

        # let's save the processed version of the graph in a nested folder for easier comparison during the
        # debugging process... and some point I could even decouple creating/processing the edgefiles and
        # calculating the corresponding graph metrics
        edgefile_folder_path = "/".join(file_path.split('/')[:-1])
        experiment_info_path = "/".join(edgefile_folder_path.split('/')[:-1])
        name_of_file = file_path.split('/')[-1]
        #name_of_injected_file = str(fraction_of_edge_weights) + '_' + str(fraction_of_edge_pkts) + '_' + \
        #                        file_path.split('/')[-1]

        prefix_for_inject_params = 'avg_exfil_' + str(avg_exfil_per_min) + ':' + str(exfil_per_min_variance) + \
                                   '_avg_pkt_' + str(avg_pkt_size) + ':' + str(pkt_size_variance) + '_'

        name_of_injected_file =  prefix_for_inject_params +  file_path.split('/')[-1]

        edgefile_pruned_folder_path = edgefile_folder_path + '/pruned_edgefiles/'
        graph_obj_folder_path = experiment_info_path + '/graph_objs/'

        ## if the graph object folder directory doesn't currently exist, then we'd want to create it...
        ## using the technique from https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python
        try:
            os.makedirs(graph_obj_folder_path)
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

        logging.info("cur_1si_G edges")
        #for edge in cur_1si_G.edges(data=True):
        #    logging.info(edge)
        logging.info("end cur_1si_G edges")

        ### NOTE: I think this is where we'd want to inject the synthetic attacks...
        for node in cur_1si_G.nodes():
            if node not in current_total_node_list:
                current_total_node_list.append(node)

        # print "right after graph is prepared", level_of_processing, list(cur_G.nodes(data=True))
        logging.info("svcs, " + str(svcs))
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

        #print("svc_to_pod",svc_to_pod)
        #exit(988)

        pod_to_svc = reverse_svc_to_pod_dict(svc_to_pod)

        cur_1si_G, node_attack_mapping, pre_specified_data_attribs, concrete_cont_node_path,carryover,attack_injected = \
        inject_synthetic_attacks(cur_1si_G, synthetic_exfil_paths, attacks_to_times, counter, node_attack_mapping,
                                 name_of_dns_pod_node, carryover, attack_injected, time_interval, avg_exfil_per_min,
                                 exfil_per_min_variance, avg_pkt_size, pkt_size_variance)

        cur_class_G = prepare_graph(cur_1si_G, svcs, 'class', is_swarm, counter, file_path, ms_s, container_to_ip,
                                    infra_service)

        # let's save a copy of the edgefile for the graph w/ the injected attack b/c that'll help with debugging
        # the system...
        edgefile_injected_folder_path = edgefile_folder_path + '/injected_edgefiles/'
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
        injected_graph_obj_loc = graph_obj_folder_path + 'graph_obj_' + name_of_injected_file  # TODO:: QQQ need new file path...
        print("injected_graph_obj_loc", injected_graph_obj_loc)
        injected_graph_obj = injected_graph(name, edgefile_injected_folder_path + name_of_injected_file,
                                            edgefile_pruned_folder_path + name_of_file,
                                            concrete_cont_node_path,
                                            pre_specified_data_attribs, svc_to_pod, pod_to_svc,
                                            total_edgelist_nodes,
                                            injected_graph_obj_loc, counter, name_of_dns_pod_node,
                                            current_total_node_list,
                                            svcs, is_swarm, ms_s, container_to_ip, infra_service,
                                            edgefile_injected_folder_path + 'class_' + name_of_injected_file,
                                            name_of_injected_file,
                                            edgefile_injected_folder_path + 'with_nodeAttribs' + name_of_injected_file,
                                            edgefile_injected_folder_path + 'class_' + 'with_nodeAttribs' + name_of_injected_file)

        injected_graph_obj.save()
        # at 53: 4.04 GB

        concrete_cont_node_path_list.append(concrete_cont_node_path)
        pre_specified_data_attribs_list.append(pre_specified_data_attribs)
        injected_graph_obj_loc_list.append(injected_graph_obj_loc)

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

def find_amt_of_out_traffic(cur_1si_G):
    into_outside_bytes = 0
    into_outside_pkts = 0
    edges_into_outside = cur_1si_G.in_edges('outside',data=True)
    for (u,v,d) in edges_into_outside:
        into_outside_bytes += d['weight']
        into_outside_pkts +=  d['frames']
    return into_outside_bytes,into_outside_pkts

## we have the times and the theoretical attacks... we just have to modify the graph
## accordingly...
# (1) identify whether a synthetic attack is injected here
# (2) identify whether this is the first occurence of injection... if it was injected
## earlier, then we need to re-use the mappings...
# (3) add the weights...
def inject_synthetic_attacks(graph, synthetic_exfil_paths, attacks_to_times, graph_number, attack_number_to_mapping,
                            name_of_dns_pod_node,old_carryover, last_attack, time_gran, avg_exfil_per_min, exfil_per_min_variance,
                             avg_pkt_size, pkt_size_variance):

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
                concrete_node = abstract_to_concrete_mapping(node, graph, [])
                if 'dns' in node:
                    print "new_mapping!", node, concrete_node
                current_mapping[node] = concrete_node
        attack_number_to_mapping[attack_occuring] = current_mapping

    concrete_node_path = determine_concrete_node_path(synthetic_exfil_paths[attack_occuring], attack_number_to_mapping[attack_occuring])

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
        ### (1): first src initiates flow [pod (DST) and vip are same service]
        ### (2). dst initiates flow [pod (src) and vip are same service
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
def abstract_to_concrete_mapping(abstract_node, graph, excluded_list):
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
            abstract_node = 'POD_' + abstract_node
        elif '_vip' in abstract_node:
            abstract_node = abstract_node.replace('_vip', '')
            abstract_node = abstract_node.replace('_', '-')
            abstract_node = abstract_node + '_VIP'
        else:
            print "neither internet nor pod nor vip"
            exit(453)

    print "modified abstract_node", abstract_node
    matching_concrete_nodes = [node for node in graph.nodes() if abstract_node in node if node not in excluded_list]
    '''
    if len(matching_concrete_nodes) == 0:
        # if no matching concrete nodes, then we are adding a new node to the graph, which'll be a
        # VIP that was not actually used... we're already transforming it into the correct format,
        # so we can just add it the matching set (and another function will add it to the graph later...)
        if node_granularity != 'class':
            print "LEN OF MATCHING CONCRETE NODES IS ZERO"
            matching_concrete_nodes = [abstract_node]
    '''
    print "matching_concrete_nodes", matching_concrete_nodes
    try:
        concrete_node = random.choice(matching_concrete_nodes)
    except:
        concrete_node = abstract_node # must be a node that isn't present in the graph
    print "concrete_node", concrete_node, "abstract_node", abstract_node
    return concrete_node

def add_edge_weight_graph(graph, concrete_node_src, concrete_node_dst, fraction_of_weight_median,
                          fraction_of_pkt_median):
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
                subgraph = make_bipartite(subgraph, nodes_one_with_vip, nodes_two_with_vip)
                ## (c) calculate subgraph values [done]
                # bipartite_density = bipartite.density(subgraph, nodes_one_with_vip)
                bipartite_density = nx.density(subgraph)
                weighted_reciprocity, _, _ = network_weidge_weighted_reciprocity(subgraph)
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

                service_one_degrees = subgraph.degree(nbunch=orig_nodes_one)
                service_one_degrees_list = []
                for svc_one_node_node_degree_tuple in service_one_degrees:
                    svc_one_node = svc_one_node_node_degree_tuple[0]
                    node_degree = svc_one_node_node_degree_tuple[1]
                    service_one_degrees_list.append( node_degree )
                svc_triple_to_degree_coef_of_var[(svc_one, svc_two, svc_one)] = np.std(service_one_degrees_list) / np.mean(service_one_degrees_list)

                service_two_degrees = subgraph.degree(nbunch=orig_nodes_two)
                service_two_degrees_list = []
                for svc_two_node_node_degree_tuple in service_two_degrees:
                    svc_two_node = svc_two_node_node_degree_tuple[0]
                    node_degree = svc_two_node_node_degree_tuple[1]
                    service_two_degrees_list.append( node_degree )
                svc_triple_to_degree_coef_of_var[(svc_one, svc_two, svc_two)] = np.std(service_two_degrees_list) / np.mean(service_two_degrees_list)


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

                service_one_degrees = subgraph.degree(nbunch=orig_nodes_one)
                service_one_degrees_list = []
                for svc_one_node_node_degree_tuple in service_one_degrees:
                    svc_one_node = svc_one_node_node_degree_tuple[0]
                    node_degree = svc_one_node_node_degree_tuple[1]
                    service_one_degrees_list.append( node_degree )
                svc_triple_to_degree_coef_of_var[(svc_one, svc_two, svc_one)] = np.std(service_one_degrees_list) / np.mean(service_one_degrees_list)


    ###plt.show()

    return svc_pair_to_reciprocity, svc_pair_to_density, svc_pair_to_coef_of_var, svc_triple_to_degree_coef_of_var

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
        if u in start_nodes:
            edge_weights.append(weight)
    # now find the coef of var...
    # recall: is std_dev / mean
    stddev_of_edge_weights = np.std(edge_weights)
    mean_of_edge_weights = np.mean(edge_weights)
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
    for svc,list_of_pods in svc_to_pod:
        current_feature_list = []
        for pod in list_of_pods:
            current_feature_list.append(node_feature_val_dict[pod])

        # coef of var is (standard_deviation / mean)
        feature_stddev = np.std(current_feature_list)
        feature_mean = np.mean(current_feature_list)
        class_to_coef_var_dict[svc] = feature_stddev / feature_mean

    return class_to_coef_var_dict

def update_total_edgelist_nodes_if_needed(cur_1si_G, total_edgelist_nodes):
    for node_one in cur_1si_G.nodes():
        for node_two in cur_1si_G.nodes():
            if (node_one,node_two) not in total_edgelist_nodes:
                total_edgelist_nodes.append( (node_one,node_two) )
    return total_edgelist_nodes

