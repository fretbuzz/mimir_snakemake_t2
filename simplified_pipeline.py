import networkx as nx
import seaborn as sns;

sns.set()
import time
import seaborn as sns;
sns.set()
import math
import csv
import ast
import gc
from next_gen_metrics import calc_neighbor_metric, generate_neig_dict, create_dict_for_dns_metric, \
    calc_dns_metric, calc_outside_inside_ratio_dns_metric, find_dns_node_name, sum_max_pod_to_dns_from_each_svc,reverse_svc_to_pod_dict
from analyze_edgefiles import prepare_graph, calc_VIP_metric, get_svc_equivalents,change_point_detection

# okay, so things to be aware of:
# (a) we are assuming that if we cannot label the node and it is not loopback or in the '10.X.X.X' subnet, then it is outside

def pipeline_subset_analysis_step(filenames, ms_s, time_interval, basegraph_name, calc_vals_p, window_size, container_to_ip,
                           is_swarm, make_net_graphs_p, infra_service):
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
                                                                               infra_service)
    return total_calculated_values

def calc_subset_graph_metrics(filenames, time_interval, basegraph_name, calc_vals_p, window_size, ms_s, container_to_ip,
                              is_swarm, svcs, infra_service):
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
        for counter, file_path in enumerate(filenames):
            gc.collect()
            G = nx.DiGraph()
            print "path to file is ", file_path
            nx.read_edgelist(file_path,
                             create_using=G, delimiter=',', data=(('weight', float),))
            cur_1si_G = prepare_graph(G, svcs, 'app_only', is_swarm, counter, file_path, ms_s, container_to_ip,
                                  infra_service)

            cur_class_G = prepare_graph(G, svcs, 'class', is_swarm, counter, file_path, ms_s, container_to_ip,
                                  infra_service)

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