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
    calc_dns_metric, calc_outside_inside_ratio_dns_metric, find_dns_node_name, sum_max_pod_to_dns_from_each_svc,reverse_svc_to_pod_dict,turn_into_list
from analysis_pipeline.src.analyze_edgefiles import calc_VIP_metric, change_point_detection, ide_angles
from analysis_pipeline.prepare_graph import prepare_graph, get_svc_equivalents
import random
import copy
import logging
import time
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import os,errno
from networkx.algorithms import bipartite
import copy
import process_control_chart
#plt.switch_backend('gtkagg')

# okay, so things to be aware of:
# (a) we are assuming that if we cannot label the node and it is not loopback or in the '10.X.X.X' subnet, then it is outside

## next steps: extract relevant parts of prepare_graph... it'll be messy. + How exactly to handle multiple
## exps + training/testing data...
## plus more metrics... (which'll probably be tomorrow-- but if I can make it non-nonsensical today + structure for a
## set of experiments + setup the lasso, we'll be in very good shape...)
## okay, so I've kinda setup the structure to be decently comprehensible. Next steps:
## (a) setup for multiple experiments, with an integration point for the LASSO
## (b) do I wanna re-arrange where the graph_metrics are calculated...

'''
def pipeline_subset_analysis_step(filenames, ms_s, time_interval, basegraph_name, calc_vals_p, window_size, container_to_ip,
                           is_swarm, make_net_graphs_p, infra_service, synthetic_exfil_paths, initiator_info_for_paths,
                                  attacks_to_times, fraction_of_edge_weights, fraction_of_edge_pkts):
    total_calculated_values = {}
    if is_swarm:
        svcs = get_svc_equivalents(is_swarm, container_to_ip)
    else:
        print "this is k8s, so using these sevices", ms_s
        svcs = ms_s

    # okay, so the whole idea here is that I actually only want a small subset of the values that I was calculating before
    total_calculated_values[(time_interval, '')] = calc_subset_graph_metrics(filenames, time_interval,
                                                                               basegraph_name + '_subset_',
                                                                               calc_vals_p, window_size,
                                                                               ms_s, container_to_ip, is_swarm, svcs,
                                                                               infra_service, synthetic_exfil_paths,
                                                                               initiator_info_for_paths,
                                                                               attacks_to_times,fraction_of_edge_weights,
                                                                                fraction_of_edge_pkts)
    return total_calculated_values
'''

def calc_subset_graph_metrics(filenames, time_interval, basegraph_name, calc_vals_p, window_size, ms_s, container_to_ip,
                              is_swarm, svcs, infra_service, synthetic_exfil_paths, initiator_info_for_paths, attacks_to_times,
                              fraction_of_edge_weights, fraction_of_edge_pkts, size_of_neighbor_training_window):#, out_q):
    if calc_vals_p:
        pod_comm_but_not_VIP_comms = []
        fraction_pod_comm_but_not_VIP_comms = []
        pod_comm_but_not_VIP_comms_no_abs = []
        fraction_pod_comm_but_not_VIP_comms_no_abs = []
        neighbor_dicts = []
        dns_in_metric_dicts = []
        dns_out_metric_dicts = []
        pod_1si_density_list = []
        list_of_concrete_container_exfil_paths = []
        list_of_exfil_amts = []
        list_of_svc_pair_to_reciprocity = []
        list_of_svc_pair_to_density = []
        list_of_svc_pair_to_coef_of_var = []
        list_of_max_ewma_control_chart_scores = []
        adjacency_matrixes = []
        #ide_angles = []

        current_total_node_list = []
        into_dns_from_outside_list = []
        svc_to_pod = {}

        avg_dns_weight = 0
        avg_dns_pkts = 0

        # for cur_G in G_list:
        node_attack_mapping = {}
        class_attack_mapping = {}
        name_of_dns_pod_node = None # defining out here so it's accessible across runs
        old_dict = {}
        total_edgelist_nodes = []
        for counter, file_path in enumerate(filenames):
            gc.collect()
            G = nx.DiGraph()
            print "path to file is ", file_path

            f = open(file_path, 'r')
            lines = f.readlines()
            nx.parse_edgelist(lines, delimiter=' ', create_using=G)

            logging.info("straight_G_edges")
            for edge in G.edges(data=True):
                logging.info(edge)
            logging.info("end straight_G_edges")

            #nx.read_edgelist(file_path,
            #                 create_using=G, delimiter=',', data=(('weight', float),))
            cur_1si_G = prepare_graph(G, svcs, 'app_only', is_swarm, counter, file_path, ms_s, container_to_ip,
                                  infra_service)

            # let's save the processed version of the graph in a nested folder for easier comparison during the
            # debugging process... and some point I could even decouple creating/processing the edgefiles and
            # calculating the corresponding graph metrics
            edgefile_folder_path = "/".join(file_path.split('/')[:-1])
            name_of_file = file_path.split('/')[-1]
            edgefile_pruned_folder_path = edgefile_folder_path + '/pruned_edgefiles/'
            ## if the pruned folder directory doesn't currently exist, then we'd want to create it...
            ## using the technique from https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python
            try:
                os.makedirs(edgefile_pruned_folder_path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            nx.write_edgelist(cur_1si_G, edgefile_pruned_folder_path+name_of_file, data=['frames', 'weight'])

            cur_class_G = prepare_graph(G, svcs, 'class', is_swarm, counter, file_path, ms_s, container_to_ip,
                                  infra_service)

            logging.info("cur_1si_G edges")
            for edge in cur_1si_G.edges(data=True):
                logging.info(edge)
            logging.info("end cur_1si_G edges")

            potential_name_of_dns_pod_node = find_dns_node_name(G)
            if potential_name_of_dns_pod_node != None:
                name_of_dns_pod_node = potential_name_of_dns_pod_node
            logging.info("name_of_dns_pod_node, " + str(name_of_dns_pod_node))
            print "name_of_dns_pod_node", name_of_dns_pod_node

            ### NOTE: I think this is where we'd want to inject the synthetic attacks...
            pre_injection_weight_into_dns_dict, pre_injection_weight_outof_dns_dict, pre_inject_packets_into_dns_dict, \
                pre_inject_packets_outof_dns_dict = create_dict_for_dns_metric(cur_1si_G, name_of_dns_pod_node)
            cur_avg_dns_weight, cur_avg_dns_pkts = avg_behavior_into_dns_node(pre_injection_weight_into_dns_dict, pre_inject_packets_into_dns_dict)
            if cur_avg_dns_weight != 0:
                if avg_dns_weight == 0:
                    avg_dns_weight = cur_avg_dns_weight
                    avg_dns_pkts = cur_avg_dns_pkts
                else:
                    avg_dns_weight = avg_dns_weight / 2.0 + cur_avg_dns_weight / 2.0
                    avg_dns_pkts = avg_dns_pkts / 2.0 + cur_avg_dns_pkts / 2.0
            cur_1si_G, node_attack_mapping,pre_specified_data_attribs, concrete_cont_node_path = inject_synthetic_attacks(cur_1si_G, synthetic_exfil_paths,initiator_info_for_paths,
                                                 attacks_to_times,'app_only',time_interval,counter,node_attack_mapping,
                                                                      fraction_of_edge_weights, fraction_of_edge_pkts, None,
                                                                    name_of_dns_pod_node, avg_dns_weight, avg_dns_pkts)
            list_of_concrete_container_exfil_paths.append(concrete_cont_node_path)
            list_of_exfil_amts.append(pre_specified_data_attribs)
            cur_class_G, class_attack_mapping,_,concrete_class_node_path = inject_synthetic_attacks(cur_class_G, synthetic_exfil_paths,initiator_info_for_paths,
                                                 attacks_to_times,'class',time_interval,counter,class_attack_mapping,
                                                                        fraction_of_edge_weights, fraction_of_edge_pkts,
                                                                        pre_specified_data_attribs, name_of_dns_pod_node,
                                                                           avg_dns_weight, avg_dns_pkts)

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
            nx.write_edgelist(cur_1si_G, edgefile_injected_folder_path+name_of_file, data=['frames', 'weight'])

            ##continue ### <<<----- TODO: remove!
            #exit() #### <----- TODO: remove!!

            '''
            plt.figure(figsize=(12,12))  # todo: turn back to (27, 16)
            plt.title('after processing')
            pos = graphviz_layout(cur_1si_G)
            for key in pos.keys():
                pos[key] = (pos[key][0] * 4, pos[key][1] * 4)  # too close otherwise
            nx.draw_networkx(cur_1si_G, pos, with_labels=True, arrows=True, font_size=8, font_color='b')
            edge_labels = nx.get_edge_attributes(cur_1si_G, 'weight')
            nx.draw_networkx_edge_labels(cur_1si_G, pos, edge_labels=edge_labels, font_size=7, label_pos=0.3)

            plt.figure(figsize=(12,12))  # todo: turn back to (27, 16)
            plt.title('before processing')
            pos = graphviz_layout(G)
            for key in pos.keys():
                pos[key] = (pos[key][0] * 4, pos[key][1] * 4)  # too close otherwise
            nx.draw_networkx(G, pos, with_labels=True, arrows=True, font_size=8, font_color='b')
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, label_pos=0.3)
            plt.show()
            '''

            for node in cur_1si_G.nodes():
                if node not in current_total_node_list:
                    current_total_node_list.append(node)

            # print "right after graph is prepared", level_of_processing, list(cur_G.nodes(data=True))
            logging.info("svcs, " + str(svcs))
            for thing in cur_1si_G.nodes(data=True):
                logging.info(thing)
                try:
                    #print thing[1]['svc']
                    cur_svc = thing[1]['svc']
                    if 'VIP' not in thing[0] and cur_svc not in svc_to_pod:
                        svc_to_pod[cur_svc] = [thing[0]]
                    else:
                        if 'VIP' not in thing[0] and thing[0] not in svc_to_pod[cur_svc]:
                            svc_to_pod[cur_svc].append(thing[0])
                except:
                    #print "there was a svc error"
                    pass
            #print "svc_to_pod", svc_to_pod
            #time.sleep(50)

            '''
            ### this is where I need to implement the edge-correlation stuff...
            max_anom_score, old_dict = process_control_chart.ewma_control_chart_max_val(G, old_dict)
            list_of_max_ewma_control_chart_scores.append(max_anom_score)

            #'''
            '''
            density = nx.density(cur_1si_G)
            #print "cur_class_G",cur_class_G.nodes()
            #print "cur_1si_G", cur_1si_G.nodes()
            pod_1si_density_list.append(density)
            neighbor_dicts.append(generate_neig_dict(cur_class_G))
            weight_into_dns_dict, weight_outof_dns_dict, _,_ = create_dict_for_dns_metric(cur_1si_G, name_of_dns_pod_node)
            dns_in_metric_dicts.append(weight_into_dns_dict)
            try:
                into_dns_from_outside_list.append(weight_into_dns_dict['outside'])
            except:
                into_dns_from_outside_list.append(0.0)
            logging.info("weight_into_dns_dict, " + str(weight_into_dns_dict))
            dns_out_metric_dicts.append(weight_outof_dns_dict)
            logging.info("weight_outof_dns_dict, " + str(weight_outof_dns_dict))

            # print "right before calc_VIP_metric", level_of_processing
            pod_comm_but_not_VIP_comm, fraction_pod_comm_but_not_VIP_comm = calc_VIP_metric(cur_1si_G, True)
            pod_comm_but_not_VIP_comms.append(pod_comm_but_not_VIP_comm)
            fraction_pod_comm_but_not_VIP_comms.append(fraction_pod_comm_but_not_VIP_comm)
            pod_comm_but_not_VIP_comm_no_abs, fraction_pod_comm_but_not_VIP_comm_no_abs = calc_VIP_metric(cur_1si_G, False)
            pod_comm_but_not_VIP_comms_no_abs.append(pod_comm_but_not_VIP_comm_no_abs)
            fraction_pod_comm_but_not_VIP_comms_no_abs.append(fraction_pod_comm_but_not_VIP_comm_no_abs)

            ### TODO: this is where I want to implement the rest of my (new) graph metrics...
            ### okay, need to put the new g
            print cur_1si_G.nodes(data=True)
            print "svc_to_pod",svc_to_pod
            svc_to_pod_with_outside = copy.deepcopy(svc_to_pod)
            svc_to_pod_with_outside['outside'] = ['outside']
            svc_pair_to_reciprocity, svc_pair_to_density,svc_pair_to_coef_of_var = pairwise_metrics(cur_1si_G, svc_to_pod_with_outside)
            ## okay, so it appears like we already having a mapping... that's fun...
            list_of_svc_pair_to_reciprocity.append(svc_pair_to_reciprocity)
            list_of_svc_pair_to_density.append(svc_pair_to_density)
            list_of_svc_pair_to_coef_of_var.append(svc_pair_to_coef_of_var)
            ##
            print "list_of_svc_pair_to_reciprocity", list_of_svc_pair_to_reciprocity
            print "list_of_svc_pair_to_density",list_of_svc_pair_to_density
            ###exit(322) ## TODO::::<---remove!!!
            ## TODO: the adjacency matrix is NOT IN THE CORRECT FORMAT
            ## okay, so the current total_node_list should be edges (present in the adjacency matrix) here... so
            ## also cannot feed it current_total_node_list, well we'll to maintain a seperate list for it...
            ## so it's probably might not be as a bad...
            #'''
            #'''
            total_edgelist_nodes = update_total_edgelist_nodes_if_needed(cur_1si_G, total_edgelist_nodes)
            #adjacency_matrixes.append( make_edgelist_dict(cur_1si_G, total_edgelist_nodes) )
            adjacency_matrixes.append( nx.to_pandas_adjacency(cur_1si_G,nodelist=current_total_node_list))
            #num_items_drop_from_list = max(0, len(adjacency_matrixes) - (window_size + 1))
            #adjacency_matrixes = adjacency_matrixes[num_items_drop_from_list:]
            print "len_adjacency_matrixes",len(adjacency_matrixes)
            #adjacency_matrixes_list = turn_into_list(adjacency_matrixes, total_edgelist_nodes)
            #'''
            ##################
            '''
            ## TODO: at some point VVVV
            #print "adjacency_matrixes_list",adjacency_matrixes_list
            print "adjacency_matrixes", adjacency_matrixes, adjacency_matrixes[0].keys()
            ide_angle = ide_angles(adjacency_matrixes, 50, total_edgelist_nodes)
            #ide_angle = 0
            ##################

            print "ide_angle", ide_angle
            try:
                ide_angles.append(ide_angle[-1])
            except:
                ide_angles.append(float('NaN'))
            '''

            # TODO: would probably be a good idea to store these vals somewhere safe or something (I think
            # the program is holding all of the graphs in memory, which is leading to massive memory bloat)
            del G
            del cur_1si_G
            del cur_class_G

        #######
        #return #exit() ### <<<<----- TODO : remove!
        #######
        # which_nodes can be ['all', 'outside', or 'kube-dns_VIP']
        # let's make the training time 5 minutes
        #size_of_neighbor_training_window = (5 * 60) / time_interval
        ## TODO: at some point VVVV
        # print "adjacency_matrixes_list",adjacency_matrixes_list
        #print "adjacency_matrixes", adjacency_matrixes, adjacency_matrixes[0].keys()
        ## ide_angles = change_point_detection(adjacency_matrixes, window_size, total_edgelist_nodes)
        ide_angles_results = ide_angles(adjacency_matrixes, 6, total_edgelist_nodes)
        # ide_angle = 0
        ##################
        '''
        num_new_neighbors_outside = calc_neighbor_metric(neighbor_dicts, size_of_neighbor_training_window, 'outside')
        num_new_neighbors_dns = calc_neighbor_metric(neighbor_dicts, size_of_neighbor_training_window, 'kube-dns_VIP')
        num_new_neighbors_all = calc_neighbor_metric(neighbor_dicts, size_of_neighbor_training_window, 'all')

        dns_angles = calc_dns_metric(dns_in_metric_dicts, current_total_node_list, window_size)
        dns_outside_inside_ratios,dns_list_outside,dns_list_inside = calc_outside_inside_ratio_dns_metric(dns_in_metric_dicts,
                                                                                                          dns_out_metric_dicts)

        into_dns_ratio, into_dns_from_outside,into_dns_from_indeside = calc_outside_inside_ratio_dns_metric(dns_in_metric_dicts,
                                                                                                            dns_in_metric_dicts)

        into_dns_eigenval_angles = change_point_detection(dns_in_metric_dicts, window_size, current_total_node_list)
        pod_to_svc = reverse_svc_to_pod_dict(svc_to_pod)
        print "pod_to_svc",pod_to_svc
        #time.sleep(100)
        sum_of_max_pod_to_dns_from_each_svc = sum_max_pod_to_dns_from_each_svc(dns_in_metric_dicts, pod_to_svc, svc_to_pod.keys())
        outside_to_sum_of_max_pod_to_dns_from_each_svc_ratio = [dns_out_metric_dicts[i]['outside']/sum_of_max_pod_to_dns_from_each_svc[i]
                                                                if sum_of_max_pod_to_dns_from_each_svc[i] else float('nan')
                                                                for i in range(0,len(sum_of_max_pod_to_dns_from_each_svc))]
        print "sum_of_max_pod_to_dns_from_each_svc",sum_of_max_pod_to_dns_from_each_svc
        print "outside_to_sum_of_max_pod_to_dns_from_each_svc_ratio",outside_to_sum_of_max_pod_to_dns_from_each_svc_ratio
        print "CHECK"
        #time.sleep(300) # todo: remove

        # let's mess w/ our stored values a little bit, sense we won't need them anymore soon...
        for counter, in_dict in enumerate(dns_in_metric_dicts):
            print dns_out_metric_dicts[counter]
            print in_dict
            print '------'
            in_dict['outside'] = dns_out_metric_dicts[counter]['outside']
        dns_eigenval_angles = change_point_detection(dns_in_metric_dicts, window_size, current_total_node_list)
        #dns_eigenval_angles12 = change_point_detection(dns_in_metric_dicts, window_size * 2, current_total_node_list)
        #dns_eigenval_angles12 = [float('nan') for i in range(0,len(dns_eigenval_angles) - len(dns_eigenval_angles12))] +\
        #                        dns_eigenval_angles12
        #dns_eigenval_angles12 = dns_eigenval_angles12[:len(dns_eigenval_angles)]

        into_dns_eigenval_angles12 = change_point_detection(dns_in_metric_dicts, window_size*2, current_total_node_list)
        into_dns_eigenval_angles12 = into_dns_eigenval_angles12[0:len(into_dns_eigenval_angles)]
        #'''
        calculated_values = {}
        #'''
        ### TODO::: REMOVE!!!
        ### exit(999)
        '''
        for service_pair in list_of_svc_pair_to_density[0].keys():
            calculated_values[service_pair[0] + '_' + service_pair[1] + '_density'] = []
            calculated_values[service_pair[0] + '_' + service_pair[1] + '_reciprocity'] = []
            calculated_values[service_pair[0] + '_' + service_pair[1] + '_coef_of_var'] = []
        for counter,svc_pair_to_density in enumerate(list_of_svc_pair_to_density):
            for service_pair in list_of_svc_pair_to_density[0].keys():
                try:
                    calculated_values[service_pair[0] + '_' + service_pair[1] + '_reciprocity'].append(
                        list_of_svc_pair_to_reciprocity[counter][service_pair])
                except:
                    calculated_values[service_pair[0] + '_' + service_pair[1] + '_reciprocity'].append(0.0)
                try:
                    calculated_values[service_pair[0] + '_' + service_pair[1] + '_density'].append(
                        svc_pair_to_density[service_pair])
                except:
                    calculated_values[service_pair[0] + '_' + service_pair[1] + '_density'].append(0.0)
                try:
                    calculated_values[service_pair[0] + '_' + service_pair[1] + '_coef_of_var'].append(
                        list_of_svc_pair_to_coef_of_var[counter][service_pair])
                except:
                    calculated_values[service_pair[0] + '_' + service_pair[1] + '_coef_of_var'].append(0.0)

        calculated_values['New Class-Class Edges'] = num_new_neighbors_all
        calculated_values['New Class-Class Edges with Outside'] = num_new_neighbors_outside
        calculated_values['New Class-Class Edges with DNS'] = num_new_neighbors_dns
        calculated_values['Angle of DNS edge weight vectors'] = dns_angles
        calculated_values[
            'Fraction of Communication Between Pods not through VIPs (no abs)'] = fraction_pod_comm_but_not_VIP_comms_no_abs
        calculated_values['Communication Between Pods not through VIPs (no abs)'] = pod_comm_but_not_VIP_comms_no_abs
        calculated_values['DNS outside-to-inside ratio'] = dns_outside_inside_ratios
        calculated_values['DNS outside'] = dns_list_outside
        calculated_values['DNS inside'] = dns_list_inside
        calculated_values['1-step-induced-pod density'] = pod_1si_density_list
        calculated_values['DNS_eigenval_angles'] = dns_eigenval_angles
        #calculated_values['DNS_eigenval_angles_DoubleWindowSize'] = dns_eigenval_angles12
        calculated_values['into_dns_ratio'] = into_dns_ratio
        calculated_values['into_dns_eigenval_angles'] = into_dns_eigenval_angles
        calculated_values['into_dns_from_outside'] = into_dns_from_outside_list
        calculated_values['into_dns_eigenval_angles12'] = into_dns_eigenval_angles12
        calculated_values['sum_of_max_pod_to_dns_from_each_svc'] = sum_of_max_pod_to_dns_from_each_svc
        calculated_values['outside_to_sum_of_max_pod_to_dns_from_each_svc_ratio'] = outside_to_sum_of_max_pod_to_dns_from_each_svc_ratio
        #'''
        #calculated_values['max_ewma_control_chart_scores'] = list_of_max_ewma_control_chart_scores
        #'''
        print "ide_angles", ide_angles_results
        calculated_values['ide_angles'] = ide_angles_results

        with open(basegraph_name + '_processed_vales_' + 'subset' + '_' + '%.2f' % (time_interval) + '.txt',
                  'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for value_name, value in calculated_values.iteritems():
                spamwriter.writerow([value_name, [i if not math.isnan(i) else (None) for i in value]])
    else:
        calculated_values = {}
        with open(basegraph_name + '_processed_vales_' + 'subset' + '_' + '%.2f' % (time_interval) + '.txt',
                  'r') as csvfile:
            csvread = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in csvread:
                print row, "---", row[0], "---", row[1]
                #print [i if i != (None) else float('nan') for i in ast.literal_eval(row[1])]
                row[1] = row[1].replace('inf', "2e308")
                print "after mod", row[1]
                calculated_values[row[0]] = [i if i != (None) else float('nan') for i in ast.literal_eval(row[1])]
                print row[0], calculated_values[row[0]]

    #out_q.put(calculated_values)
    #out_q.put(list_of_concrete_container_exfil_paths)
    #out_q.put(list_of_exfil_amts)
    return calculated_values, list_of_concrete_container_exfil_paths, list_of_exfil_amts


def inject_synthetic_attacks(graph, synthetic_exfil_paths, initiator_info_for_paths, attacks_to_times,
                             node_granularity, time_granularity,graph_number, attack_number_to_mapping,
                             fraction_of_edge_weights, fraction_of_edge_pkts, pre_specified_data_attribs,
                             name_of_dns_pod_node, avg_dns_weight, avg_dns_pkts):

    ## we have the times and the theoretical attacks... we just have to modify the graph
    ## accordingly...
    # (1) identify whether a synthetic attack is injected here
    # (2) identify whether this is the first occurence of injection... if it was injected
    ## earlier, then we need to re-use the mappings...
    # (3) add the weights...

    # first, perform (1)
    concrete_node_path = []
    current_time = graph_number #* time_granularity
    attack_occuring = None
    fraction_of_pkt_min = 0
    fraction_of_weight_min = 0
    print "attacks_to_times", attacks_to_times, type(attacks_to_times), current_time, node_granularity
    #print "time_granularity", time_granularity
    for counter, attack_ranges in enumerate(attacks_to_times):
        #print "SWAZG", counter, attacks_to_times, len(attacks_to_times)
        if current_time >= attack_ranges[0] and current_time < attack_ranges[1]:
            # then the attack occurs during this interval....
            attack_occuring = counter % len(synthetic_exfil_paths)
            print "attack in range found!"
            break
    if attack_occuring != None:
        if node_granularity == 'class':
            synthetic_exfil_paths = copy.deepcopy(synthetic_exfil_paths)
            remaining_node_path = []
            for node in synthetic_exfil_paths[attack_occuring]:
                if 'vip' not in node:
                    remaining_node_path.append(node)
            synthetic_exfil_paths[attack_occuring] = remaining_node_path

        # second, perform (2)
        if attack_occuring not in attack_number_to_mapping.keys():
            ## this whole function is about determining attack_number_to_mapping (for the current attack)
            ##so two things remaining in this function: (a) determine mapping (this func)
            ##                                          (b) determine equiv edges (later on)
            # once a mapping exists, then we need to determine the corresponding weight that should
            # be added onto the relevant edge... for existing edges, we can just add some fraction of the
            # existing edge. For new edges, probably wanna find an equivalent and then add a fraction of
            # that (should be easy enough b/c this only happens in a very limited number of scenarios...)
            # to be specific, we should choose, like, the lowest or mean weight edge and then like 10% of
            # that to all the edges in the path (b/c we're assuming that we're going straight out...)

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

        if node_granularity != 'class':
            all_weights_in_exfil_path = []
            all_pkts_in_exfil_path = []
            abstract_node_pair = None
            concrete_node_pair = None
            first_concrete_node_pair = None
            dns_exfil_path = False
            for node_one_loc in range(0, len(synthetic_exfil_paths[attack_occuring]) -1 ):
                abstract_node_pair = (synthetic_exfil_paths[attack_occuring][node_one_loc],
                                      synthetic_exfil_paths[attack_occuring][node_one_loc+1])

                if 'dns_vip' in abstract_node_pair[1]:
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
                if concrete_edge: # will return None if edge doesn't exist...
                    all_weights_in_exfil_path.append( concrete_edge['weight'] )
                    all_pkts_in_exfil_path.append( concrete_edge['frames'] )
                else:
                    equivalent_edge = find_equiv_edge(first_concrete_node_pair, graph, node_granularity)
                    if equivalent_edge:
                        equiv_concrete_node_src = equivalent_edge[0]
                        equiv_concrete_node_dst = equivalent_edge[1]
                        all_weights_in_exfil_path.append(graph.get_edge_data(equiv_concrete_node_src, equiv_concrete_node_dst)['weight'])
                        all_pkts_in_exfil_path.append( graph.get_edge_data(equiv_concrete_node_src, equiv_concrete_node_dst)['frames'] )
                    else:
                        # equivalent_edge is only None when it's a dns_exfil path and no dns nodes are present in the current graph
                        # in which case, we'll have to rely on previous dns behavior
                        # (this previous behavior will be passed as a parameter to the function...)
                        all_weights_in_exfil_path.append(avg_dns_weight)
                        all_pkts_in_exfil_path.append(avg_dns_pkts)

            # so let's choose the weight/packets... let's maybe go w/ some fraction of the median...
            pkt_np_array = np.array(all_pkts_in_exfil_path)
            weight_np_array = np.array(all_weights_in_exfil_path)
            pkt_min = np.min(pkt_np_array)
            weight_min =  np.min(weight_np_array)

            if not dns_exfil_path:
                fraction_of_pkt_min = int(pkt_min * fraction_of_edge_pkts)
                fraction_of_weight_min = int(weight_min * fraction_of_edge_weights)
            else:
                fraction_of_pkt_min = int(pkt_min * 3) ## TODO: might wanna parametrize...
                fraction_of_weight_min = int(weight_min * 3) ## TODO: might wanna parametrize...
        else:
            ###  we should store the corresponding attribs from the app_only granularity and then just
            # use that (b/c class gran. gives super huge).
            fraction_of_pkt_min = pre_specified_data_attribs['frames']
            fraction_of_weight_min = pre_specified_data_attribs['weight']

        #########
        #### TODO: maybe this is actually where'd I'd want to split the whole thing??? I need to split it at some point,
        #### so that I can loop over the actual injection strenths... well maybe I'd want to return up above, where I still
        ##########

        # recall that we'd need to add traffic going both ways... or would we??? acks would be v small... no
        # it's worth it. Let's just assume all acks. Then same # of packets, Let's assume smallest, so 40 bytes.
        ## two situations: (a) need to modify existing edge weight (or rather, nodes already exist)
        node_one_loc = 0
        #for node_one_loc in range(0, len(synthetic_exfil_paths[attack_occuring]) -1 ):
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
                    graph = add_edge_weight_graph(graph, concrete_node_src_one, concrete_node_dst,
                                                  fraction_of_weight_min, fraction_of_pkt_min)
                    #if concrete_node_path == []:
                    #    concrete_node_path.append(concrete_node_src_one)
                    print "concrete_node_path", node_one_loc, concrete_node_path
                    graph = add_edge_weight_graph(graph, concrete_node_src_two, concrete_node_dst,
                                                  fraction_of_weight_min, fraction_of_pkt_min)
                    node_one_loc += 1 # b/c we're modifying two edges here, we need to increment the counter one more time...
                    concrete_node_path.append((concrete_node_src_one,concrete_node_dst))
                    concrete_node_path.append((concrete_node_src_two,concrete_node_dst))
                    print "concrete_node_path", node_one_loc, concrete_node_path
                elif abstract_node_pair_same_service_p(abstract_node_dst, abstract_node_pair[1]):
                    graph = add_edge_weight_graph(graph, concrete_node_src_one, concrete_node_src_two,
                                                  fraction_of_weight_min, fraction_of_pkt_min)
                    print "concrete_node_path", concrete_node_src_one, concrete_node_src_two
                    graph = add_edge_weight_graph(graph, concrete_node_src_one, concrete_node_dst,
                                                  fraction_of_weight_min, fraction_of_pkt_min)
                    node_one_loc += 1 # b/c we're modifying two edges here, we need to increment the counter one more time...
                    concrete_node_path.append((concrete_node_src_one,concrete_node_src_two))
                    concrete_node_path.append((concrete_node_src_one,concrete_node_dst))
                else:
                    print "apparently a vip in the path doesn't belong to either service??"
                    exit(544)
            else:
                # this case does not involve any redirection via the kubernetes network model, so it is simple
                concrete_node_src = attack_number_to_mapping[attack_occuring][abstract_node_pair[0]]
                concrete_node_dst = attack_number_to_mapping[attack_occuring][abstract_node_pair[1]]

                #if concrete_node_path == []:
                #    concrete_node_path.append(concrete_node_src)
                concrete_node_path.append((concrete_node_src,concrete_node_dst))

                graph = add_edge_weight_graph(graph, concrete_node_src, concrete_node_dst,
                                              fraction_of_weight_min,
                                              fraction_of_pkt_min)
                print "concrete_node_path", node_one_loc, concrete_node_path, concrete_node_src, concrete_node_dst
            node_one_loc += 1

        print "modifications_to_graph...", concrete_node_path, fraction_of_weight_min, fraction_of_pkt_min

        ###exit(99)###TODO: remove!

    return graph, attack_number_to_mapping, {'weight':fraction_of_weight_min, 'frames': fraction_of_pkt_min}, concrete_node_path

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
    if concrete_node_pair[0] == 'outside' or concrete_node_pair[1] == 'outside':
        # we'll change the node that is labeled 'outside'
        if concrete_node_pair[0] == 'outside' and concrete_node_pair[1] != 'outside':
            # okay, let's modify the [0] node...
            edges_incident_on_non_outside_node = graph.edges([concrete_node_pair[1]])
            # need to convert the view to a list...
            edges_incident_on_non_outside_node_list = []
            for edge in edges_incident_on_non_outside_node:
                edges_incident_on_non_outside_node_list.append(edge)
            # we'll just choose randomly, since it seems the easiest...
            equiv_edge = random.choice(edges_incident_on_non_outside_node)
            print "equiv_edge", equiv_edge
        elif concrete_node_pair[0] != 'outside' and concrete_node_pair[1] == 'outside':
            edges_incident_on_non_outside_node = graph.edges([concrete_node_pair[0]])
            edges_incident_on_non_outside_node_list = []
            for edge in edges_incident_on_non_outside_node:
                edges_incident_on_non_outside_node_list.append(edge)
            equiv_edge = random.choice(edges_incident_on_non_outside_node_list)
            print "equiv_edge", equiv_edge
        else:
            print "both nodes being outside should not be possible..."
            exit(343)
    elif 'dns_vip' in concrete_node_pair[1]:
            edges_incident_on_dns_vip = graph.edges([concrete_node_pair[1]])
            edges_incident_on_dns_vip_list = []
            for edge in edges_incident_on_dns_vip:
                edges_incident_on_dns_vip_list.append(edge)
            if edges_incident_on_dns_vip_list == []:
                equiv_edge = None
            else:
                equiv_edge = random.choice(edges_incident_on_dns_vip_list)
            print "equiv_dns_edge", equiv_edge
    elif 'dns_vip' in concrete_node_pair[0]:
            edges_incident_on_dns_vip = graph.edges([concrete_node_pair[0]])
            edges_incident_on_dns_vip_list = []
            for edge in edges_incident_on_dns_vip:
                edges_incident_on_dns_vip_list.append(edge)
            if edges_incident_on_dns_vip_list == []:
                equiv_edge = None
            else:
                equiv_edge = random.choice(edges_incident_on_dns_vip_list)
            print "equiv_dns_edge", equiv_edge
    else:
        # we'll have to choose randomly which one to keep, I suppose... (or maybe the lower weight one...)
        # again, there's probably a better way to do this...
        #remaining_node = random.choice([concrete_node_pair[0], concrete_node_pair[1]])
        # NO: We'll keep the first (src) node
        print "NEED TO FIND EQUIVALENT EDGE", concrete_node_pair, node_granularity

        '''
        pos = graphviz_layout(graph)
        for key in pos.keys():
            pos[key] = (pos[key][0] * 4, pos[key][1] * 4)  # too close otherwise
        nx.draw_networkx(graph, pos, with_labels=True, arrows=True, font_size=8, font_color='b')
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7, label_pos=0.3)
        plt.show()
        '''

        remaining_node = concrete_node_pair[0]
        edges_incident_on_remaining_node = graph.edges([remaining_node])
        edges_incident_list = []
        for edge in edges_incident_on_remaining_node:
            edges_incident_list.append(edge)
        equiv_edge = random.choice(edges_incident_list)

        print "equiv_edge", equiv_edge

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

    # now that all the nodes exist, we can add the weights
    #print graph.nodes(), concrete_node_src, concrete_node_dst

    # if no edge exists, then we need to add one...
    if not graph.has_edge(concrete_node_src, concrete_node_dst):
        graph.add_edge(concrete_node_src, concrete_node_dst, weight=0, frames=0)
    graph[concrete_node_src][concrete_node_dst]['weight'] += fraction_of_weight_median
    graph[concrete_node_src][concrete_node_dst]['frames'] += fraction_of_pkt_median

    # now need to account for the acks...
    if not graph.has_edge(concrete_node_dst, concrete_node_src):
        graph.add_edge(concrete_node_dst, concrete_node_src, weight=0, frames=0)
    graph[concrete_node_dst][concrete_node_src]['weight'] += (fraction_of_weight_median * ack_packet_size)
    graph[concrete_node_dst][concrete_node_src]['frames'] += fraction_of_pkt_median

    return graph

def abstract_node_pair_same_service_p(abstract_node_one, abstract_node_two):
    ## okay, well, let's give this a shot
    abstract_node_one_core = abstract_node_one.replace('_pod', '').replace('_vip', '')
    abstract_node_two_core = abstract_node_two.replace('_pod', '').replace('_vip', '')
    return abstract_node_one_core == abstract_node_two_core

def avg_behavior_into_dns_node(pre_injection_weight_into_dns_dict, pre_inject_packets_into_dns_dict):
    avg_dns_weight = 0
    avg_dns_pkts = 0
    non_null_edges = 0
    for node in pre_injection_weight_into_dns_dict.keys():
        weight = pre_injection_weight_into_dns_dict[node]
        pkts = pre_inject_packets_into_dns_dict[node]
        if pkts != 0 or weight != 0:
            avg_dns_weight += weight
            avg_dns_pkts += pkts
            non_null_edges += 1
    avg_dns_weight = avg_dns_weight / non_null_edges
    avg_dns_pkts = avg_dns_pkts / non_null_edges
    return avg_dns_weight, avg_dns_pkts

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
                coef_of_var = find_coef_of_variation(subgraph, nodes_one_with_vip)
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

