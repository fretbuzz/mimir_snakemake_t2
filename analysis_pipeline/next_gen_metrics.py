import math
import time
from functools import partial

import numpy as np

from analysis_pipeline import alert_triggers


# takes a graph and returns a dictionary, where each key is the name of a node and each
# value is a list of the nodes that have eedges incident on the key node
def generate_neig_dict(G):
    neigh_dict = {}
    for node in G.nodes():
        neighs = []
        for key_val in G.neighbors(node):
            neighs.append(key_val)
        neigh_dict[node] = neighs
    return neigh_dict


# for each key, there is a corresponding list. Find the number of items
# in the new_neigh_dict vals that are not in the old_neigh_dict vals
def calc_num_new_neighs(old_neigh_dict, new_neigh_dict):
    new_neighs = 0
    for node, neighbors in new_neigh_dict.iteritems():
        if node in old_neigh_dict:
            cur_new_neighs = len(np.setdiff1d(neighbors, old_neigh_dict[node]))
            #print "node", node, ";union",np.setdiff1d(neighbors, old_neigh_dict[node])
            new_neighs += cur_new_neighs
    return new_neighs

# for each key, calculates the union of the two values (which'll
# be lists)
def psudeo_merge_neigh_dicts(neigh_dict_one, neigh_dict_two):
    merged_neigh_dict = {}
    for node in list(set( neigh_dict_one.keys() + neigh_dict_two.keys() )):
        if node in neigh_dict_one and node in neigh_dict_two:
            merged_neigh_dict[node] = np.union1d(neigh_dict_one[node], neigh_dict_two[node])
        elif node in neigh_dict_one:
            merged_neigh_dict[node] = neigh_dict_one[node]
        elif node in neigh_dict_two:
            merged_neigh_dict[node] = neigh_dict_two[node]
        else:
            print "what the heck happened?"
            exit(22)
    return merged_neigh_dict

# returns a dictionary with only a single key-value pair. This pair comes from
# another dictionary, which is passed in as a parameter.
def make_one_item_dict(dict, key):
    one_item_dict = {}
    try:
        one_item_dict[key] = dict[key]
    except:
        one_item_dict[key] = []
    return one_item_dict

# provides a single point of integration for the metric to be incorporated into analyze_edgefiles...
# okay, but what about just for the DNS and outside nodes??? ---> need some way to restrict this somehow
# which_nodes can be ['all', 'outside', or 'kube-dns_VIP']
def calc_neighbor_metric(list_of_neigh_dicts, size_of_training_window, which_nodes):
    training_window_neigh_dict = {}
    #print "list_of_neigh_dicts", list_of_neigh_dicts
    for i in range(0, size_of_training_window):
        training_window_neigh_dict = psudeo_merge_neigh_dicts(list_of_neigh_dicts[i], training_window_neigh_dict)
    list_of_new_neighbors = [float('nan') for i in range(0,size_of_training_window)]
    for i in range(size_of_training_window, len(list_of_neigh_dicts)):
        if which_nodes == 'all':
            #print list_of_neigh_dicts[i]
            #print training_window_neigh_dict
            cur_new_neighbors = calc_num_new_neighs(training_window_neigh_dict, list_of_neigh_dicts[i])
        elif which_nodes == 'outside':
            relevant_training_window_neigh_dict = make_one_item_dict(training_window_neigh_dict, 'outside')
            relevant_current_dict = make_one_item_dict(list_of_neigh_dicts[i], 'outside')
            cur_new_neighbors = calc_num_new_neighs(relevant_training_window_neigh_dict, relevant_current_dict)
        elif which_nodes == 'kube-dns_VIP':
            relevant_training_window_neigh_dict = make_one_item_dict(training_window_neigh_dict, 'kube-dns_VIP')
            relevant_current_dict = make_one_item_dict(list_of_neigh_dicts[i], 'kube-dns_VIP')
            cur_new_neighbors = calc_num_new_neighs(relevant_training_window_neigh_dict, relevant_current_dict)
        else:
            print "which_nodes name not recognized!!"
            exit(23)
        print "cur_new_neighbors", cur_new_neighbors
        list_of_new_neighbors.append(cur_new_neighbors)
    return list_of_new_neighbors

# this functions takes a list_of_dictionaries, where each dictionaries contains keys for every node in the graph.
# The corresponding value is the weight of the edge (if edge does not exist -> val must be zero, but key MUST
# be present)
def calc_dns_metric(list_of_dicts, node_list, window_size):
    list_of_edge_weights = turn_into_list(list_of_dicts, node_list)
    list_of_angles = find_angles(list_of_edge_weights, window_size)
    return list_of_angles

# this function calculates the ratio of traffic going outside the cluster to traffic going inside
# the cluster
def calc_outside_inside_ratio_dns_metric(dns_in_metric_dicts,dns_out_metric_dicts):
    list_of_outside_inside_ratios = []
    list_outside = []
    list_inside = []
    for counter,dict in enumerate(dns_in_metric_dicts):
        print "outside_inside_ratio_dict", dict
        #time.sleep(10)
        try:
            outside_weight = dns_out_metric_dicts[counter]['outside'] #dict['outside']
        except:
            outside_weight = 0
        inside_wieght = 0.0
        for key,val in dict.iteritems():
            if key != 'outside':
                inside_wieght += val
        list_inside.append(inside_wieght)
        list_outside.append(outside_weight)
        if inside_wieght:
            ratio = float(outside_weight) / inside_wieght
        else:
            if outside_weight:
                ratio = float('inf')
            else:
                ratio = float('nan')
        list_of_outside_inside_ratios.append( ratio )
    return list_of_outside_inside_ratios,list_outside,list_inside

def create_dict_for_dns_metric(G, name_of_of_dns_node):
    return_dict = {}
    return_dict_two = {}
    # okay, so the end-goal is to get a dict that can be aggreated into a list and fed into calc_dns_metric
    print "create_dict_for_dns_metric nodes", G.nodes()
    try:
        # note: this will give edges that start at DNS and go other places...
        # what I probably want the edges that arrive at the DNS node??
        #for node,edge_attribs in G[name_of_of_dns_node].iteritems():
        for edge_tuple in G.in_edges(name_of_of_dns_node, data=True):
            node = edge_tuple[0]
            edge_attribs = edge_tuple[2]
            return_dict[node] = edge_attribs['weight']

        for node,edge_attribs in G[name_of_of_dns_node].iteritems():
            return_dict_two[node] = edge_attribs['weight']
    except: # no activity with DNS node...
        pass
    # looks like you'd be done... but not yet! we gotta make sure that all nodes in the graph are represented
    # as keys (w/ vals of zero if no edge w/ the DNS node)
    for node in G.nodes():
        if node not in return_dict:
            return_dict[node] = 0
        if node not in return_dict_two:
            return_dict_two[node] = 0
    return return_dict,return_dict_two

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
                current_nodes.append(0.0) # note: may wanna switch back to the below value (or not)
                #current_nodes.append(float('nan'))  # the current dict must not have an entry for node -> zero val
        node_vals.append(np.array(current_nodes))
    return node_vals


def reverse_svc_to_pod_dict(svc_to_pod):
    pod_to_svc = {}
    for svc,list_of_pods in svc_to_pod.iteritems():
        for pod in list_of_pods:
            pod_to_svc[pod] = svc
    return pod_to_svc

def sum_max_pod_to_dns_from_each_svc(dns_in_metric_dicts, pod_to_svc, svcs):
    list_of_sums = []
    for into_dns_dict in dns_in_metric_dicts:
        svc_to_max = {}
        for svc in svcs:
            svc_to_max[svc] = 0
        for pod,weight in into_dns_dict.iteritems():
            try:
                cur_svc = pod_to_svc[pod]
                if weight > svc_to_max[cur_svc]:
                    print "old weight", svc_to_max[cur_svc], "new weight", weight, "svc", cur_svc
                    svc_to_max[cur_svc] = weight
            except:
                pass
        print "svc_to_max", svc_to_max
        sum = 0
        for svc,weight in svc_to_max.iteritems():
            sum += weight
        list_of_sums.append(sum)
    return list_of_sums

def find_angles(list_of_vectors, window_size):
    angles = []
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

    for i in range(0, window_size):
        angles.insert(0, float('nan'))

    print "angles", angles
    return angles

def find_unit_vector(vector):
    return vector / np.linalg.norm(vector)

def find_dns_node_name(G):
    for node in G.nodes():
        if 'kube-dns' in node and 'POD' in node:
            return node
    return "foobar" # doesn't matter because it is not present anyway

def alert_fuction(weights, features_to_use, bias, row_from_csv):
    alert_score = 0
    #print "row_from_csv",row_from_csv
    for feature in features_to_use:
        cur_contrib_to_alert_score = row_from_csv[feature] * weights[feature]
        #print feature, row_from_csv
        #print cur_contrib_to_alert_score
        if not math.isnan(cur_contrib_to_alert_score):
            alert_score += cur_contrib_to_alert_score
        else:
            alert_score += 0 # could also just pass
    return alert_score + bias

def next_gen_ROCS(df_with_anom_features, time_gran, alert_file, sub_path):
    '''
      0.0041 * Communication Between Pods not through VIPs (no abs)200_5__mod_z_score +
      0.0008 * DNS outside-to-inside ratio200_5__mod_z_score +
      0.0009 * New Class-Class Edges200_5__mod_z_score +
      0      * DNS_eigenval_angles200_5__mod_z_score +
     -0.0109
    '''
    features_to_use = ['New Class-Class Edges200_5__mod_z_score',
                       'Communication Between Pods not through VIPs (no abs)200_5__mod_z_score',
                       'DNS outside-to-inside ratio200_5__mod_z_score']
    weights = {'New Class-Class Edges200_5__mod_z_score': 0.0009,
               'Communication Between Pods not through VIPs (no abs)200_5__mod_z_score': 0.0041,
               'DNS outside-to-inside ratio200_5__mod_z_score': 0.0008}
    ROC_path = alert_file + sub_path + '_good_roc_'
    bias = -0.0109
    cur_alert_function = partial(alert_fuction, weights, features_to_use, bias)
    title = 'ROC Linear Combination of Features at ' + str(time_gran)
    plot_name = 'sub_roc_lin_comb_features_' + str(time_gran)
    alert_triggers.create_ROC_of_anom_score(df_with_anom_features, time_gran, ROC_path, cur_alert_function, title,
                                            plot_name)

    for feature in features_to_use:
        title = 'ROC ' + feature + ' at ' + str(time_gran)
        plot_name = 'sub_roc_' + feature + '_' + str(time_gran)
        cur_alert_function = partial(alert_fuction, weights, [feature], 0.0)
        alert_triggers.create_ROC_of_anom_score(df_with_anom_features, time_gran, ROC_path, cur_alert_function,
                                                title, plot_name)