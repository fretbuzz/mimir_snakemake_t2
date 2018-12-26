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
    calc_dns_metric, calc_outside_inside_ratio_dns_metric, find_dns_node_name, sum_max_pod_to_dns_from_each_svc,reverse_svc_to_pod_dict
from analysis_pipeline.src.analyze_edgefiles import prepare_graph, calc_VIP_metric, get_svc_equivalents,change_point_detection
import random
import copy

# okay, so things to be aware of:
# (a) we are assuming that if we cannot label the node and it is not loopback or in the '10.X.X.X' subnet, then it is outside

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

def calc_subset_graph_metrics(filenames, time_interval, basegraph_name, calc_vals_p, window_size, ms_s, container_to_ip,
                              is_swarm, svcs, infra_service, synthetic_exfil_paths, initiator_info_for_paths, attacks_to_times,
                              fraction_of_edge_weights, fraction_of_edge_pkts):
    if calc_vals_p:
        pod_comm_but_not_VIP_comms = []
        fraction_pod_comm_but_not_VIP_comms = []
        pod_comm_but_not_VIP_comms_no_abs = []
        fraction_pod_comm_but_not_VIP_comms_no_abs = []
        neighbor_dicts = []
        dns_in_metric_dicts = []
        dns_out_metric_dicts = []
        pod_1si_density_list = []

        current_total_node_list = []
        into_dns_from_outside_list = []
        svc_to_pod = {}

        # for cur_G in G_list:
        node_attack_mapping = {}
        class_attack_mapping = {}
        for counter, file_path in enumerate(filenames):
            gc.collect()
            G = nx.DiGraph()
            print "path to file is ", file_path

            f = open(file_path, 'r')
            lines = f.readlines()
            G = nx.parse_edgelist(lines, delimiter=' ')

            #nx.read_edgelist(file_path,
            #                 create_using=G, delimiter=',', data=(('weight', float),))
            cur_1si_G = prepare_graph(G, svcs, 'app_only', is_swarm, counter, file_path, ms_s, container_to_ip,
                                  infra_service)

            cur_class_G = prepare_graph(G, svcs, 'class', is_swarm, counter, file_path, ms_s, container_to_ip,
                                  infra_service)

            ### NOTE: I think this is where we'd want to inject the synthetic attacks...
            cur_1si_G, node_attack_mapping,pre_specified_data_attribs = inject_synthetic_attacks(cur_1si_G, synthetic_exfil_paths,initiator_info_for_paths,
                                                 attacks_to_times,'app_only',time_interval,counter,node_attack_mapping,
                                                                      fraction_of_edge_weights, fraction_of_edge_pkts, None)
            cur_class_G, class_attack_mapping,_ = inject_synthetic_attacks(cur_class_G, synthetic_exfil_paths,initiator_info_for_paths,
                                                 attacks_to_times,'class',time_interval,counter,class_attack_mapping,
                                                                        fraction_of_edge_weights, fraction_of_edge_pkts,
                                                                        pre_specified_data_attribs)
            continue ### <<<----- TODO: remove!
            exit() #### <----- TODO: remove!!

            name_of_dns_pod_node = find_dns_node_name(G)
            print "name_of_dns_pod_node",name_of_dns_pod_node

            # print "right after graph is prepared", level_of_processing, list(cur_G.nodes(data=True))
            print "svcs",svcs
            for thing in cur_1si_G.nodes(data=True):
                print thing
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

            for node in cur_1si_G.nodes():
                if node not in current_total_node_list:
                    current_total_node_list.append(node)

            density = nx.density(cur_1si_G)
            print "cur_class_G",cur_class_G.nodes()
            print "cur_1si_G", cur_1si_G.nodes()
            pod_1si_density_list.append(density)
            neighbor_dicts.append(generate_neig_dict(cur_class_G))
            weight_into_dns_dict, weight_outof_dns_dict = create_dict_for_dns_metric(cur_1si_G, name_of_dns_pod_node)
            dns_in_metric_dicts.append(weight_into_dns_dict)
            try:
                into_dns_from_outside_list.append(weight_into_dns_dict['outside'])
            except:
                into_dns_from_outside_list.append(0.0)
            print "weight_into_dns_dict",weight_into_dns_dict
            dns_out_metric_dicts.append(weight_outof_dns_dict)
            print "weight_outof_dns_dict",weight_outof_dns_dict

            # print "right before calc_VIP_metric", level_of_processing
            pod_comm_but_not_VIP_comm, fraction_pod_comm_but_not_VIP_comm = calc_VIP_metric(cur_1si_G, True)
            pod_comm_but_not_VIP_comms.append(pod_comm_but_not_VIP_comm)
            fraction_pod_comm_but_not_VIP_comms.append(fraction_pod_comm_but_not_VIP_comm)
            pod_comm_but_not_VIP_comm_no_abs, fraction_pod_comm_but_not_VIP_comm_no_abs = calc_VIP_metric(cur_1si_G, False)
            pod_comm_but_not_VIP_comms_no_abs.append(pod_comm_but_not_VIP_comm_no_abs)
            fraction_pod_comm_but_not_VIP_comms_no_abs.append(fraction_pod_comm_but_not_VIP_comm_no_abs)

            # TODO: would probably be a good idea to store these vals somewhere safe or something (I think
            # the program is holding all of the graphs in memory, which is leading to massive memory bloat)
            del G
            del cur_1si_G
            del cur_class_G

        #######
        return #exit() ### <<<<----- TODO : remove!
        #######
        # which_nodes can be ['all', 'outside', or 'kube-dns_VIP']
        # let's make the training time 5 minutes
        size_of_training_window = (5 * 60) / time_interval
        num_new_neighbors_outside = calc_neighbor_metric(neighbor_dicts, size_of_training_window, 'outside')
        num_new_neighbors_dns = calc_neighbor_metric(neighbor_dicts, size_of_training_window, 'kube-dns_VIP')
        num_new_neighbors_all = calc_neighbor_metric(neighbor_dicts, size_of_training_window, 'all')

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

        calculated_values = {}

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
        calculated_values['into_dns_from_outside_list'] = into_dns_from_outside_list
        calculated_values['into_dns_eigenval_angles12'] = into_dns_eigenval_angles12
        calculated_values['sum_of_max_pod_to_dns_from_each_svc'] = sum_of_max_pod_to_dns_from_each_svc
        calculated_values['outside_to_sum_of_max_pod_to_dns_from_each_svc_ratio'] = outside_to_sum_of_max_pod_to_dns_from_each_svc_ratio

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

    return calculated_values

def inject_synthetic_attacks(graph, synthetic_exfil_paths, initiator_info_for_paths, attacks_to_times,
                             node_granularity, time_granularity,graph_number, attack_number_to_mapping,
                             fraction_of_edge_weights, fraction_of_edge_pkts, pre_specified_data_attribs):

    ## we have the times and the theoretical attacks... we just have to modify the graph
    ## accordingly...
    # (1) identify whether a synthetic attack is injected here
    # (2) identify whether this is the first occurence of injection... if it was injected
    ## earlier, then we need to re-use the mappings...
    # (3) add the weights...

    # first, perform (1)
    current_time = graph_number #* time_granularity
    attack_occuring = None
    fraction_of_pkt_median = None
    fraction_of_weight_median = None
    print "attacks_to_times", attacks_to_times, type(attacks_to_times), current_time, node_granularity
    #print "time_granularity", time_granularity
    for counter, attack_ranges in enumerate(attacks_to_times):
        #print "SWAZG", counter, attacks_to_times, len(attacks_to_times)
        if current_time >= attack_ranges[0] and current_time < attack_ranges[1]:
            # then the attack occurs during this interval....
            attack_occuring = counter
            print "attack in range found!"
            break
    if attack_occuring:
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
                concrete_node = abstract_to_concrete_mapping(node, graph, node_granularity)
                current_mapping[node] = concrete_node
            attack_number_to_mapping[attack_occuring] = current_mapping

        if node_granularity != 'class':
            all_weights_in_exfil_path = []
            all_pkts_in_exfil_path = []
            abstract_node_pair = None
            concrete_node_pair = None
            for node_one_loc in range(0, len(synthetic_exfil_paths[attack_occuring]) -1 ):
                abstract_node_pair = (synthetic_exfil_paths[attack_occuring][node_one_loc],
                                      synthetic_exfil_paths[attack_occuring][node_one_loc+1])

                concrete_node_src = attack_number_to_mapping[attack_occuring][abstract_node_pair[0]]
                concrete_node_dst = attack_number_to_mapping[attack_occuring][abstract_node_pair[1]]
                concrete_node_pair = (concrete_node_src,concrete_node_dst)

                # wanna determine the relevant weight and then add it. (so in actualuality, this is #3 from below)
                print "concrete_nodes", concrete_node_src,concrete_node_dst
                concrete_edge = graph.get_edge_data(concrete_node_src, concrete_node_dst)
                print concrete_edge
                if concrete_edge: # will return None if edge doesn't exist...
                    all_weights_in_exfil_path.append( concrete_edge['weight'] )
                    all_pkts_in_exfil_path.append( concrete_edge['frames'] )
                else:
                    pass

            ## potential problem: all_weights_in_exfil_path and all_pkts_in_exfil_path could be size zero!
            ## what would we want to do in that scenario?? we'd need to find an equivalent edge to use the weigth of
            if len(all_weights_in_exfil_path) == 0:
                equivalent_edge = find_equiv_edge(concrete_node_pair, graph, node_granularity)
                equiv_concrete_node_src = equivalent_edge[0]
                equiv_concrete_node_dst = equivalent_edge[1]
                all_weights_in_exfil_path.append(graph.get_edge_data(equiv_concrete_node_src, equiv_concrete_node_dst)['weight'])
                all_pkts_in_exfil_path.append( graph.get_edge_data(equiv_concrete_node_src, equiv_concrete_node_dst)['frames'] )

            # so let's choose the weight/packets... let's maybe go w/ some fraction of the median...
            pkt_np_array = np.array(all_pkts_in_exfil_path)
            weight_np_array = np.array(all_weights_in_exfil_path)
            pkt_median = np.median(pkt_np_array)
            weight_median =  np.median(weight_np_array)

            fraction_of_pkt_median = int(pkt_median * fraction_of_edge_pkts)
            fraction_of_weight_median = int(weight_median * fraction_of_edge_weights)
        else:
            ###  we should store the corresponding attribs from the app_only granularity and then just
            # use that (b/c class gran. gives super huge).
            fraction_of_pkt_median = pre_specified_data_attribs['frames']
            fraction_of_weight_median = pre_specified_data_attribs['weight']

        # recall that we'd need to add traffic going both ways... or would we??? acks would be v small... no
        # it's worth it. Let's just assume all acks. Then same # of packets, Let's assume smallest, so 40 bytes.
        ## two situations: (a) need to modify existing edge weight (or rather, nodes already exist)
        concrete_node_path = []
        for node_one_loc in range(0, len(synthetic_exfil_paths[attack_occuring]) -1 ):
            abstract_node_pair = (synthetic_exfil_paths[attack_occuring][node_one_loc],
                                  synthetic_exfil_paths[attack_occuring][node_one_loc+1])
            concrete_node_src = attack_number_to_mapping[attack_occuring][abstract_node_pair[0]]
            concrete_node_dst = attack_number_to_mapping[attack_occuring][abstract_node_pair[1]]
            if concrete_node_path == []:
                concrete_node_path.append(concrete_node_src)
            concrete_node_path.append(concrete_node_dst)

            ack_packet_size = 40 # bytes
            if concrete_node_src in graph and concrete_node_dst in graph:
                pass # don't need to do anything b/c the nodes already exist...
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
            print graph.nodes(), concrete_node_src, concrete_node_dst

            # if no edge exists, then we need to add one...
            if not graph.has_edge(concrete_node_src, concrete_node_dst):
                graph.add_edge(concrete_node_src, concrete_node_dst, weight = 0, frames = 0)
            graph[concrete_node_src][concrete_node_dst]['weight'] += fraction_of_weight_median
            graph[concrete_node_src][concrete_node_dst]['frames'] += fraction_of_pkt_median

            # now need to account for the acks...
            if not graph.has_edge(concrete_node_dst, concrete_node_src):
                graph.add_edge(concrete_node_dst, concrete_node_src, weight = 0, frames = 0)
            graph[concrete_node_dst][concrete_node_src]['weight'] += (fraction_of_pkt_median * ack_packet_size)
            graph[concrete_node_dst][concrete_node_src]['frames'] += fraction_of_pkt_median

        print "modifications_to_graph...", concrete_node_path, fraction_of_weight_median, fraction_of_pkt_median

    return graph,attack_number_to_mapping,{'weight':fraction_of_weight_median, 'frames': fraction_of_pkt_median}

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
    else:
        # we'll have to choose randomly which one to keep, I suppose... (or maybe the lower weight one...)
        # again, there's probably a better way to do this...
        remaining_node = random.choice([concrete_node_pair[0], concrete_node_pair[1]])
        edges_incident_on_remaining_node = graph.edges([remaining_node])
        equiv_edge = random.choice(edges_incident_on_remaining_node)
        print "equiv_edge", equiv_edge

    return equiv_edge