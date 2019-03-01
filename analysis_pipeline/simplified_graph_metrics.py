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

# okay, so things to be aware of:
# (a) we are assuming that if we cannot label the node and it is not loopback or in the '10.X.X.X' subnet, then it is outside

class injected_graph():
    def __init__(self, name, injected_graph_loc, non_injected_graph_loc, concrete_container_exfil_paths, exfil_amt,
                 svc_to_pod, pod_to_svc, total_edgelist_nodes, where_to_save_this_obj, counter, name_of_dns_pod_node,
                 current_total_node_list, fraction_of_edge_weights, fraction_of_edge_pkts,
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
        self.fraction_of_edge_weights = fraction_of_edge_weights
        self.fraction_of_edge_pkts = fraction_of_edge_pkts
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
        svc_pair_to_reciprocity, svc_pair_to_density, svc_pair_to_coef_of_var = pairwise_metrics(self.cur_1si_G,
                                                                                                 svc_to_pod_with_outside)
        self.graph_feature_dict['list_of_svc_pair_to_reciprocity'] = svc_pair_to_reciprocity
        self.graph_feature_dict['list_of_svc_pair_to_density'] = svc_pair_to_density
        self.graph_feature_dict['list_of_svc_pair_to_coef_of_var'] = svc_pair_to_coef_of_var

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
            ''' # we're going to use a different method...
                w = csv.DictWriter(f, self.graph_feature_dict.keys())
                w.writeheader()
                w.writerow(self.graph_feature_dict)
            '''
            f.write(pickle.dumps(self.graph_feature_dict))
            #f.write(str(self.graph_feature_dict))

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
    def __init__(self, fraction_of_edge_weights, fraction_of_edge_pkts, time_granularity, window_size, raw_edgefile_names,
                svcs, is_swarm, ms_s, container_to_ip, infra_service, synthetic_exfil_paths, initiator_info_for_paths,
                attacks_to_times, collected_metrics_location, current_set_of_graphs_loc):#, out_q):

        self.list_of_injected_graphs_loc = []
        self.fraction_of_edge_weights = fraction_of_edge_weights
        self.fraction_of_edge_pkts = fraction_of_edge_pkts
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
            total_edgelist_nodes = injected_graph.total_edgelist_nodes
            current_total_node_list = injected_graph.current_total_node_list


        #total_edgelist_nodes = self.list_of_injected_graphs[-1].total_edgelist_nodes
        #current_total_node_list = self.list_of_injected_graphs[-1].current_total_node_list
        ide_angles_results = ide_angles(adjacency_matrixes, 6, total_edgelist_nodes)
        into_dns_eigenval_angles = change_point_detection(dns_in_metric_dicts, self.window_size, current_total_node_list)


        ## need to store these new results into the format that I've been using this far...
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

        self.calculated_values[
            'Fraction of Communication Between Pods not through VIPs (no abs)'] = fraction_pod_comm_but_not_VIP_comms_no_abs
        self.calculated_values['Communication Between Pods not through VIPs (no abs)'] = pod_comm_but_not_VIP_comms_no_abs
        self.calculated_values[
            'Fraction of Communication Between Pods not through VIPs (w abs)'] = [abs(i) for i in fraction_pod_comm_but_not_VIP_comms_no_abs]
        self.calculated_values['Communication Between Pods not through VIPs (w abs)'] = [abs(i) for i in pod_comm_but_not_VIP_comms_no_abs]
        self.calculated_values['DNS outside-to-inside ratio'] = dns_outside_inside_ratios
        self.calculated_values['DNS outside'] = dns_list_outside
        self.calculated_values['DNS inside'] = dns_list_inside
        self.calculated_values['1-step-induced-pod density'] = pod_1si_density_list
        self.calculated_values['into_dns_from_outside'] = into_dns_from_outside_list
        self.calculated_values['into_dns_ratio'] = into_dns_ratio
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
        fraction_of_edge_weights = self.fraction_of_edge_weights
        fraction_of_edge_pkts = self.fraction_of_edge_pkts
        synthetic_exfil_paths = self.synthetic_exfil_paths
        initiator_info_for_paths = self.initiator_info_for_paths
        attacks_to_times = self.attacks_to_times
        time_interval = self.time_interval
        out_q = multiprocessing.Queue()
        name_of_dns_pod_node = None

        num_graphs_to_process_at_once = 40
        for counter in range(0, len(self.raw_edgefile_names), num_graphs_to_process_at_once):

            file_paths = self.raw_edgefile_names[counter: counter + num_graphs_to_process_at_once]
            args = [counter, file_paths, svcs, is_swarm, ms_s, container_to_ip, infra_service, fraction_of_edge_weights,
                    fraction_of_edge_pkts, synthetic_exfil_paths, initiator_info_for_paths, attacks_to_times,
                    time_interval, total_edgelist_nodes, svc_to_pod, avg_dns_weight, avg_dns_pkts, class_attack_mapping,
                    node_attack_mapping, out_q, current_total_node_list, name_of_dns_pod_node]
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
            class_attack_mapping = out_q.get()
            node_attack_mapping = out_q.get()
            current_total_node_list = out_q.get()
            name_of_dns_pod_node = out_q.get()
            p.join()

            ## okay, literally the code above should be wrapped in a function call...
            ## however, you'd probably wanna process like 40-50 of these on a single call...
            self.list_of_concrete_container_exfil_paths.extend(concrete_cont_node_path)
            self.list_of_exfil_amts.extend(pre_specified_data_attribs)
            self.list_of_injected_graphs_loc.extend(injected_graph_obj_loc)

def process_and_inject_single_graph(counter_starting, file_paths, svcs, is_swarm, ms_s, container_to_ip, infra_service, fraction_of_edge_weights,
                    fraction_of_edge_pkts, synthetic_exfil_paths, initiator_info_for_paths, attacks_to_times,
                    time_interval, total_edgelist_nodes, svc_to_pod, avg_dns_weight, avg_dns_pkts, class_attack_mapping,
                    node_attack_mapping, out_q, current_total_node_list,name_of_dns_pod_node):

    concrete_cont_node_path_list = []
    pre_specified_data_attribs_list = []
    injected_graph_obj_loc_list = []
    attack_injected = None
    carryover = 0
    for counter_add, file_path in enumerate(file_paths):
        counter = counter_starting + counter_add
        gc.collect()
        G = nx.DiGraph()
        print "path to file is ", file_path

        f = open(file_path, 'r')
        lines = f.readlines()
        nx.parse_edgelist(lines, delimiter=' ', create_using=G)

        logging.info("straight_G_edges")
        #for edge in G.edges(data=True):
        #    logging.info(edge)
        logging.info("end straight_G_edges")

        # nx.read_edgelist(file_path,
        #                 create_using=G, delimiter=',', data=(('weight', float),))
        cur_1si_G = prepare_graph(G, svcs, 'app_only', is_swarm, counter, file_path, ms_s,
                                  container_to_ip, infra_service)

        # let's save the processed version of the graph in a nested folder for easier comparison during the
        # debugging process... and some point I could even decouple creating/processing the edgefiles and
        # calculating the corresponding graph metrics
        edgefile_folder_path = "/".join(file_path.split('/')[:-1])
        experiment_info_path = "/".join(edgefile_folder_path.split('/')[:-1])
        name_of_file = file_path.split('/')[-1]
        name_of_injected_file = str(fraction_of_edge_weights) + '_' + str(fraction_of_edge_pkts) + '_' + \
                                file_path.split('/')[-1]
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

        cur_class_G = prepare_graph(G, svcs, 'class', is_swarm, counter, file_path, ms_s, container_to_ip,
                                    infra_service)

        logging.info("cur_1si_G edges")
        #for edge in cur_1si_G.edges(data=True):
        #    logging.info(edge)
        logging.info("end cur_1si_G edges")

        potential_name_of_dns_pod_node = find_dns_node_name(G)
        if potential_name_of_dns_pod_node != None:
            name_of_dns_pod_node = potential_name_of_dns_pod_node
        logging.info("name_of_dns_pod_node, " + str(name_of_dns_pod_node))
        print "name_of_dns_pod_node", name_of_dns_pod_node

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

        pod_to_svc = reverse_svc_to_pod_dict(svc_to_pod)

        pre_injection_weight_into_dns_dict, pre_injection_weight_outof_dns_dict, pre_inject_packets_into_dns_dict, \
        pre_inject_packets_outof_dns_dict = create_dict_for_dns_metric(cur_1si_G, name_of_dns_pod_node)
        cur_avg_dns_weight, cur_avg_dns_pkts = avg_behavior_into_dns_node(pre_injection_weight_into_dns_dict,
                                                                          pre_inject_packets_into_dns_dict,
                                                                          pod_to_svc)
        print("process_rep: " + str(counter) + ':' + " cur_avg_dns_weight: " + str(
            cur_avg_dns_weight) + ', cur_avg_dns_pkts:' + str(cur_avg_dns_pkts))
        logging.info("process_rep: " + str(counter) + ':' + " cur_avg_dns_weight: " + str(
            cur_avg_dns_weight) + ', cur_avg_dns_pkts:' + str(cur_avg_dns_pkts))
        if cur_avg_dns_weight != 0:
            if avg_dns_weight == 0:
                avg_dns_weight = cur_avg_dns_weight
                avg_dns_pkts = cur_avg_dns_pkts
            else:
                avg_dns_weight = avg_dns_weight / 2.0 + cur_avg_dns_weight / 2.0
                avg_dns_pkts = avg_dns_pkts / 2.0 + cur_avg_dns_pkts / 2.0

        logging.info("process_rep: " + str(counter) + ':' + " avg_dns_weight: " + str(
            avg_dns_weight) + ', avg_dns_pkts:' + str(avg_dns_pkts))

        cur_1si_G, node_attack_mapping, pre_specified_data_attribs, concrete_cont_node_path,carryover,attack_injected = \
            inject_synthetic_attacks(cur_1si_G, synthetic_exfil_paths, initiator_info_for_paths,
                    attacks_to_times, 'app_only', time_interval, counter, node_attack_mapping,
                    fraction_of_edge_weights, fraction_of_edge_pkts, None,
                    name_of_dns_pod_node, avg_dns_weight, avg_dns_pkts,carryover, attack_injected)

        # quick Q: why even bother with this? why don't we just
        cur_class_G, class_attack_mapping, _, concrete_class_node_path, _, _ = \
            inject_synthetic_attacks(cur_class_G, synthetic_exfil_paths, initiator_info_for_paths, attacks_to_times,
                                    'class', time_interval, counter, class_attack_mapping, fraction_of_edge_weights,
                                    fraction_of_edge_pkts, pre_specified_data_attribs, name_of_dns_pod_node,
                                    avg_dns_weight, avg_dns_pkts,carryover)

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
                                            fraction_of_edge_weights, fraction_of_edge_pkts,
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
    out_q.put(class_attack_mapping)
    out_q.put(node_attack_mapping)
    out_q.put(current_total_node_list)
    out_q.put(name_of_dns_pod_node)

## we have the times and the theoretical attacks... we just have to modify the graph
## accordingly...
# (1) identify whether a synthetic attack is injected here
# (2) identify whether this is the first occurence of injection... if it was injected
## earlier, then we need to re-use the mappings...
# (3) add the weights...
def inject_synthetic_attacks(graph, synthetic_exfil_paths, initiator_info_for_paths, attacks_to_times,
                             node_granularity, time_granularity,graph_number, attack_number_to_mapping,
                             fraction_of_edge_weights, fraction_of_edge_pkts, pre_specified_data_attribs,
                             name_of_dns_pod_node, avg_dns_weight, avg_dns_pkts,old_carryover, last_attack):

    concrete_node_path = []
    current_time = graph_number
    attack_occuring = None
    fraction_of_pkt_min = 0
    fraction_of_weight_min = 0
    carryover = 0

    # step (1) : identify whether an attack needs to be injected here
    #print "attacks_to_times", attacks_to_times, type(attacks_to_times), current_time, node_granularity
    for counter, attack_ranges in enumerate(attacks_to_times):
        if current_time >= attack_ranges[0] and current_time < attack_ranges[1]:
            # then the attack occurs during this interval....
            attack_occuring = counter % len(synthetic_exfil_paths)
            print "attack in range found!"
            break

    if attack_occuring == None:
        return graph, attack_number_to_mapping, {'weight': fraction_of_weight_min,
                                                 'frames': fraction_of_pkt_min}, concrete_node_path, carryover

    # no VIPs in class granularity graph for some reason
    if node_granularity == 'class':
        synthetic_exfil_paths = copy.deepcopy(synthetic_exfil_paths)
        remaining_node_path = []
        for node in synthetic_exfil_paths[attack_occuring]:
            if 'vip' not in node:
                remaining_node_path.append(node)
        synthetic_exfil_paths[attack_occuring] = remaining_node_path


    fraction_of_pkt_min, fraction_of_weight_min = \
        determine_exfiltration_amt(pre_specified_data_attribs, attack_occuring, attack_number_to_mapping,
        last_attack, synthetic_exfil_paths, name_of_dns_pod_node, node_granularity, avg_dns_weight,
        avg_dns_pkts, time_granularity, fraction_of_edge_pkts, fraction_of_edge_weights, old_carryover)

    ## Step (4): use the previously calculated exfiltration rate to actually make the appropriate modifications
    ## to the graph.

    node_one_loc = 0
    while node_one_loc < (len(synthetic_exfil_paths[attack_occuring]) - 1):
        abstract_node_pair = (synthetic_exfil_paths[attack_occuring][node_one_loc],
                              synthetic_exfil_paths[attack_occuring][node_one_loc+1])
        concrete_possible_dst =  attack_number_to_mapping[attack_occuring][abstract_node_pair[1]]
        print "abstract_node_pair", abstract_node_pair
        print "concrete_possible_dst", concrete_possible_dst
        #synthetic_exfil_paths[attack_occuring][node_one_loc + 2]

        ### there are actually two subcases of this first case. 1: first src initiates flow
        ### 2. dst initiates flow (that is the currently covered case)
        ### note: for 1: pod (DST) and vip are same service
        ### note: for 2: pod (src) and vip are same service
        # note:e I think we can tell which is which by looking at the abstrct node pairs...
        if 'VIP' in concrete_possible_dst:
            ## in this case, we need to compensate for the VIP re-direction that occurs
            ## in the Kubernetes VIP.
            concrete_node_src_one = attack_number_to_mapping[attack_occuring][abstract_node_pair[0]]
            concrete_node_src_two = attack_number_to_mapping[attack_occuring][abstract_node_pair[1]]
            abstract_node_dst = synthetic_exfil_paths[attack_occuring][node_one_loc + 2]
            concrete_node_dst = attack_number_to_mapping[attack_occuring][abstract_node_dst]
            print "vip_located_xx", concrete_node_src_one, concrete_node_src_two, concrete_node_dst
            print synthetic_exfil_paths[attack_occuring]
            print attack_number_to_mapping[attack_occuring]
            print "concrete_node_path", node_one_loc, concrete_node_path
            if abstract_node_pair_same_service_p(abstract_node_pair[0], abstract_node_pair[1]):
                graph,carryover = add_edge_weight_graph(graph, concrete_node_src_one, concrete_node_dst,
                                              fraction_of_weight_min, fraction_of_pkt_min)
                #if concrete_node_path == []:
                #    concrete_node_path.append(concrete_node_src_one)
                print "concrete_node_path", node_one_loc, concrete_node_path
                graph,carryover = add_edge_weight_graph(graph, concrete_node_src_two, concrete_node_dst,
                                              fraction_of_weight_min, fraction_of_pkt_min)
                node_one_loc += 1 # b/c we're modifying two edges here, we need to increment the counter one more time...
                concrete_node_path.append((concrete_node_src_one,concrete_node_dst))
                concrete_node_path.append((concrete_node_src_two,concrete_node_dst))
                print "concrete_node_path", node_one_loc, concrete_node_path
            elif abstract_node_pair_same_service_p(abstract_node_dst, abstract_node_pair[1]):
                graph,carryover = add_edge_weight_graph(graph, concrete_node_src_one, concrete_node_src_two,
                                              fraction_of_weight_min, fraction_of_pkt_min)
                print "concrete_node_path", concrete_node_src_one, concrete_node_src_two
                graph,carryover = add_edge_weight_graph(graph, concrete_node_src_one, concrete_node_dst,
                                              fraction_of_weight_min, fraction_of_pkt_min)
                node_one_loc += 1 # b/c we're modifying two edges here, we need to increment the counter one more time...
                concrete_node_path.append((concrete_node_src_one,concrete_node_src_two))
                concrete_node_path.append((concrete_node_src_one,concrete_node_dst))
            else:
                print "apparently a vip in the path doesn't belong to either service??"
                exit(544)
            if carryover == fraction_of_weight_min:
                fraction_of_weight_min = 0
                fraction_of_pkt_min = 0
        else:
            # this case does not involve any redirection via the kubernetes network model, so it is simple
            concrete_node_src = attack_number_to_mapping[attack_occuring][abstract_node_pair[0]]
            concrete_node_dst = attack_number_to_mapping[attack_occuring][abstract_node_pair[1]]

            #if concrete_node_path == []:
            #    concrete_node_path.append(concrete_node_src)
            concrete_node_path.append((concrete_node_src,concrete_node_dst))

            graph,carryover = add_edge_weight_graph(graph, concrete_node_src, concrete_node_dst,
                                          fraction_of_weight_min,
                                          fraction_of_pkt_min)
            if carryover == fraction_of_weight_min:
                fraction_of_weight_min = 0
                fraction_of_pkt_min = 0

            print "concrete_node_path", node_one_loc, concrete_node_path, concrete_node_src, concrete_node_dst
        node_one_loc += 1

    print "modifications_to_graph...", concrete_node_path, fraction_of_weight_min, fraction_of_pkt_min

    return graph, attack_number_to_mapping, {'weight':fraction_of_weight_min, 'frames': fraction_of_pkt_min}, \
           concrete_node_path,carryover, attack_occuring

# abstract_to_concrete_mapping: abstract_node graph -> concrete_node (in graph)
def abstract_to_concrete_mapping(abstract_node, graph, node_granularity):
    print "abstract_to_concrete_mapping", abstract_node, graph.nodes(),node_granularity
    ## okay, so there's a couple of things that I should do???
    if abstract_node == 'internet':
        abstract_node = 'outside'
    else:
        if node_granularity == 'class':
            abstract_node = abstract_node.replace('_pod', '').replace('_vip', '').replace('_','-')
        else:
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
    matching_concrete_nodes = [node for node in graph.nodes() if abstract_node in node]
    if len(matching_concrete_nodes) == 0:
        # if no matching concrete nodes, then we are adding a new node to the graph, which'll be a
        # VIP that was not actually used... we're already transforming it into the correct format,
        # so we can just add it the matching set (and another function will add it to the graph later...)
        if node_granularity != 'class':
            print "LEN OF MATCHING CONCRETE NODES IS ZERO"
            matching_concrete_nodes = [abstract_node]
    print "matching_concrete_nodes", matching_concrete_nodes
    concrete_node = random.choice(matching_concrete_nodes)
    print "concrete_node", concrete_node, "abstract_node", abstract_node
    return concrete_node

# find_equiv_edge: concrete_node_pair graph -> concrete_node_pair
def find_equiv_edge(concrete_node_pair, graph, node_granularity):
    ## Step (1): which node to modify???
    #### if outside present, modify that. else, choose node randomly
    ## step (2): what to modify that node to??
    #### choose randomly from edges incident on remaining nodee
    print "find_equiv_edge", concrete_node_pair, graph.edges(), node_granularity
    #equiv_node_src = None
    #equiv_node_dst = None
    '''
    if concrete_node_pair[0] == 'outside' or concrete_node_pair[1] == 'outside':
        # we'll change the node that is labeled 'outside'
        if concrete_node_pair[0] == 'outside' and concrete_node_pair[1] != 'outside':
            logging.info("find_equiv_edge case 1: concrete_node_pair[0] == 'outside'")
            # okay, let's modify the [0] node...
            edges_incident_on_non_outside_node = graph.edges([concrete_node_pair[1]])
            # need to convert the view to a list...
            edges_incident_on_non_outside_node_list = []
            for edge in edges_incident_on_non_outside_node:
                if 'dns' not in edge[0]:
                    edges_incident_on_non_outside_node_list.append(edge)
            # we'll just choose randomly, since it seems the easiest...
            equiv_edge = random.choice(edges_incident_on_non_outside_node)
            print "equiv_edge", equiv_edge
        elif concrete_node_pair[0] != 'outside' and concrete_node_pair[1] == 'outside':
            logging.info("find_equiv_edge case 2: concrete_node_pair[1] == 'outside'")
            edges_incident_on_non_outside_node = graph.edges([concrete_node_pair[0]])
            edges_incident_on_non_outside_node_list = []
            for edge in edges_incident_on_non_outside_node:
                if 'dns' not in edge[1]:
                    edges_incident_on_non_outside_node_list.append(edge)
            equiv_edge = random.choice(edges_incident_on_non_outside_node_list)
            print "equiv_edge", equiv_edge
        else:
            print "both nodes being outside should not be possible..."
            exit(343)
    elif 'dns_vip' in concrete_node_pair[1] or 'dns_VIP' in concrete_node_pair[1]:
            logging.info("find_equiv_edge case 3: 'dns_vip' in concrete_node_pair[1]'")
            edges_incident_on_dns_vip = graph.edges([concrete_node_pair[1]])
            edges_into_dns_server_pod = []
            for edge in edges_incident_on_dns_vip:
                edges_into_dns_server_pod.append(edge)
            if edges_into_dns_server_pod == []:
                equiv_edge = None
            else:
                equiv_edge = random.choice(edges_into_dns_server_pod)
            print "equiv_dns_edge", equiv_edge
    elif 'dns_vip' in concrete_node_pair[0] or 'dns_VIP' in concrete_node_pair[0]:
            logging.info("find_equiv_edge case 4: 'dns_vip' in concrete_node_pair[0]'")
            #edges_incident_on_dns_vip = graph.edges([concrete_node_pair[1]])
            # this'll register as dns_vip -> dns_pod and we want to find an edge incident on the DNS pod
            edges_into_dns_server_pod = []
            for (u,v,d) in graph.edgese():
                if concrete_node_pair[1] in v:
                    edges_into_dns_server_pod.append((u,v,d))
            #edges_into_dns_server_pod = []
            #for edge in edges_incident_on_dns_vip:
            #    edges_into_dns_server_pod.append(edge)
            if edges_into_dns_server_pod == []:
                equiv_edge = None
            else:
                equiv_edge = random.choice(edges_into_dns_server_pod)
            print "equiv_dns_edge", equiv_edge
    else:
        logging.info("find_equiv_edge case 5: default rando on src node")
        # we'll have to choose randomly which one to keep, I suppose... (or maybe the lower weight one...)
        # again, there's probably a better way to do this...
        #remaining_node = random.choice([concrete_node_pair[0], concrete_node_pair[1]])
        # NO: We'll keep the first (src) node
        print "NEED TO FIND EQUIVALENT EDGE", concrete_node_pair, node_granularity

        #
        #pos = graphviz_layout(graph)
        #for key in pos.keys():
        #    pos[key] = (pos[key][0] * 4, pos[key][1] * 4)  # too close otherwise
        #nx.draw_networkx(graph, pos, with_labels=True, arrows=True, font_size=8, font_color='b')
        #edge_labels = nx.get_edge_attributes(graph, 'weight')
        #nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7, label_pos=0.3)
        #plt.show()
        #

        remaining_node = concrete_node_pair[0]
        edges_incident_on_remaining_node = graph.edges([remaining_node])
        edges_incident_list = []
        for edge in edges_incident_on_remaining_node:
            edges_incident_list.append(edge)
        equiv_edge = random.choice(edges_incident_list)

        print "equiv_edge", equiv_edge    
        '''

    edges_into_dest = []
    #logging.info("concrete node pair: " + str(concrete_node_pair[0]) + ", " + str(concrete_node_pair[1]))
    #logging.info("looking for in-edges to:", concrete_node_pair[1])
    for (u, v, d) in graph.in_edges(concrete_node_pair[1], data=True):
        #if concrete_node_pair[1] in v:
        if 'internet' not in u and 'outside' not in u and "." not in u and 'dns' not in u and 'kubernetes_VIP' not in u:
            #logging.info("one_possible_equiv_edge: " + str(u) + " " + str(v))
            edges_into_dest.append((u, v, d))
    try:
        equiv_edge = random.choice(edges_into_dest)
    except:
        equiv_edge = None

    return equiv_edge

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

    ## TODO: problem here. We cannot add the weight if it is less than 40. It makes no sense. Let's make it
    ## at least like 20 more than 40. So if it's less than 60, that is a problem.

    # now that all the nodes exist, we can add the weights
    #print graph.nodes(), concrete_node_src, concrete_node_dst

    carryover = 0 # this indicates how much we are taking to the next time step with us.
    # if packet bytes is > 60, let' sgo ahead and add it
    if fraction_of_weight_median > 60:
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
    else:
        carryover = fraction_of_weight_median

    return graph, carryover

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

def determine_exfiltration_amt(pre_specified_data_attribs, attack_occuring, attack_number_to_mapping,
                               last_attack, synthetic_exfil_paths, name_of_dns_pod_node, node_granularity,
                               avg_dns_weight, avg_dns_pkts, time_granularity, fraction_of_edge_pkts,
                               fraction_of_edge_weights, old_carryover):

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
                concrete_node = abstract_to_concrete_mapping(node, graph, node_granularity)
                if 'dns' in node:
                    print "new_mapping!", node, concrete_node
                current_mapping[node] = concrete_node
        attack_number_to_mapping[attack_occuring] = current_mapping

    # step (3) : determine the appropriate weight change for the edges leading to exfiltration
    # note that if we are working at class node granularity, then we should just use the weights
    # determined for the corresponding pod granularity graph
    if node_granularity != 'class':
        all_weights_in_exfil_path = []
        all_pkts_in_exfil_path = []
        #abstract_node_pair = None
        #concrete_node_pair = None
        first_concrete_node_pair = None
        dns_exfil_path = False
        for node_one_loc in range(0, len(synthetic_exfil_paths[attack_occuring]) -1 ):
            abstract_node_pair = (synthetic_exfil_paths[attack_occuring][node_one_loc],
                                  synthetic_exfil_paths[attack_occuring][node_one_loc+1])

            if 'dns_vip' in abstract_node_pair[1]:
                logging.info("dns_vip found in abstract node pair")
                dns_exfil_path = True
                # then let's clear all the early values... since they're all giant probably...
                all_weights_in_exfil_path = []
                all_pkts_in_exfil_path = []

            concrete_node_src = attack_number_to_mapping[attack_occuring][abstract_node_pair[0]]
            concrete_node_dst = attack_number_to_mapping[attack_occuring][abstract_node_pair[1]]
            concrete_node_pair = (concrete_node_src,concrete_node_dst)

            if not first_concrete_node_pair:
                first_concrete_node_pair = concrete_node_pair

            # wanna determine the relevant weight and then add it. (so in actualuality, this is #3 from below)
            print "concrete_nodes", concrete_node_src,concrete_node_dst
            concrete_edge = graph.get_edge_data(concrete_node_src, concrete_node_dst)
            print concrete_edge
            cur_exfil_weight = None
            if concrete_edge: # will return None if edge doesn't exist...
                cur_exfil_weight = concrete_edge['weight']
                all_weights_in_exfil_path.append( cur_exfil_weight )
                all_pkts_in_exfil_path.append( concrete_edge['frames'] )
            else:
                logging.info("need to find an equivalent edge to: " + str(concrete_node_src) + " " + str(concrete_node_dst))
                equivalent_edge = find_equiv_edge(concrete_node_pair, graph, node_granularity)
                if equivalent_edge:
                    logging.info("this equivalent_edge is:" + str(equivalent_edge[0]) + ', ' + str(equivalent_edge[1]))
                    equiv_concrete_node_src = equivalent_edge[0]
                    equiv_concrete_node_dst = equivalent_edge[1]
                    cur_exfil_data  = graph.get_edge_data(equiv_concrete_node_src, equiv_concrete_node_dst)
                    cur_exfil_weight = cur_exfil_data['weight']
                    cur_exfil_frames = cur_exfil_data['frames']
                    all_weights_in_exfil_path.append( cur_exfil_weight )
                    all_pkts_in_exfil_path.append( cur_exfil_frames )
                else:
                    if 'dns' in concrete_node_dst or 'dns' in concrete_node_src:
                        logging.info("no equivalent edge found and dns is in the name... must be dns")
                        # equivalent_edge is only None when it's a dns_exfil path and no dns nodes are present in the current graph
                        # in which case, we'll have to rely on previous dns behavior
                        # (this previous behavior will be passed as a parameter to the function...)
                        cur_exfil_weight = avg_dns_weight
                        all_weights_in_exfil_path.append(avg_dns_weight)
                        all_pkts_in_exfil_path.append(avg_dns_pkts)

            logging.info("found this weight: " + str(cur_exfil_weight))
        logging.info("all_weights_in_exfil_path" + str(all_weights_in_exfil_path) + ", time_gran: " + str(time_granularity))
        # so let's choose the weight/packets... let's maybe go w/ some fraction of the median...
        pkt_np_array = np.array(all_pkts_in_exfil_path)
        weight_np_array = np.array(all_weights_in_exfil_path)
        pkt_min = np.min(pkt_np_array)
        weight_min =  np.min(weight_np_array)

        logging.info("weight_min: " + str(weight_min))

        if not dns_exfil_path:
            fraction_of_pkt_min = max(int(math.ceil(pkt_min * fraction_of_edge_pkts)),1)
            fraction_of_weight_min = int(weight_min * fraction_of_edge_weights)+ old_carryover
        else:
            # NOTE THE 10* HERE
            fraction_of_pkt_min = max(int(math.ceil(pkt_min * 10.0 * fraction_of_edge_pkts)),1)
            fraction_of_weight_min = int(weight_min * 10.0 * fraction_of_edge_weights)+ old_carryover

        logging.info("fraction_of_weight_min: " + str(fraction_of_weight_min) + ";; fraction_of_edge_weights: " + str(
            fraction_of_edge_weights))
        attack_occuring_str = " ".join([str(z) for z in synthetic_exfil_paths[attack_occuring]])
        print("attack_occuring", attack_occuring_str)
        logging.info("attack_occuring: " + attack_occuring_str)

    else:
        ###  we should store the corresponding attribs from the app_only granularity and then just
        # use that (b/c class gran. gives super huge).
        fraction_of_pkt_min = pre_specified_data_attribs['frames']
        fraction_of_weight_min = pre_specified_data_attribs['weight']

    return fraction_of_pkt_min, fraction_of_weight_min

def pairwise_metrics(G, svc_to_nodes):
    svc_pair_to_reciprocity = {}
    svc_pair_to_density = {}
    svc_pair_to_coef_of_var = {}
    for svc_one,nodes_one in svc_to_nodes.iteritems():
        for svc_two,nodes_two in svc_to_nodes.iteritems():
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

    ###plt.show()

    return svc_pair_to_reciprocity, svc_pair_to_density,svc_pair_to_coef_of_var

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

def update_total_edgelist_nodes_if_needed(cur_1si_G, total_edgelist_nodes):
    for node_one in cur_1si_G.nodes():
        for node_two in cur_1si_G.nodes():
            if (node_one,node_two) not in total_edgelist_nodes:
                total_edgelist_nodes.append( (node_one,node_two) )
    return total_edgelist_nodes

