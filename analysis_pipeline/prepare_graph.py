import copy
import re

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from itertools import chain


# TODO: at some point, what I probably want to do is pass in the hosting machine's IP addresses, b/c I'm justing using heuristics
# to identify it ATM
def is_private_ip(addr_bytes):
    # note: i am going to assume that if the ip is not loopback or in the  '10.X.X.X' subnet, then it is outside

    # private_subnet_one = ('10.0.0.0', '10.255.255.255') # so if 10.XX.XX.XX
    # private_subnet_two = ('172.16.0.0', '172.31.255.255') # so if
    # private_subnet_three = ('192.168.0.0', '192.168.255.255') # so if 192.168.XX.XX

    if addr_bytes[0] == '10':
        return True
    #elif addr_bytes[0] == '172' and int(addr_bytes[1]) >= 16 and int(addr_bytes[1]) <= 31:
    #    return True
    # NOTE: it looks like this range normally used by the VM, so it is NOT fair game
    # to be used in cluster, since it'd have a different meaning
    #elif  addr_bytes[0] == '192' and addr_bytes[1] == '168':
    #    return True
    elif addr_bytes[0] == '127' and addr_bytes[1] == '0' and addr_bytes[2] == '0' and addr_bytes[3] == '1':
        return True
    # todo: this is a heuristic way to identify the host machine's IP
    elif addr_bytes[0] == '192' and addr_bytes[1] == '168'  and addr_bytes[3] != '1':
       return True # assuming that the only 192.168.XX.1 addresses will be the hosting computer
    # actually 172.17.0.1 can effectively be treated as an outside entity, due to NAT-like behavior
    elif addr_bytes[0] == '172' and addr_bytes[1] == '17' and addr_bytes[2] == '0' and addr_bytes[3] != '1':
    #    # this corresponds to pods in the minikube cluster... tho if I change the experimental apparatus, that might change too
        return True
    else:
        return False

def map_nodes_to_svcs(G, svcs):
    if svcs == [] or not svcs:
        return {}
    containers_to_ms = {}
    for u in G.nodes():
        for svc in svcs:
            if svc in u:
                containers_to_ms[u] = svc
                break
    return containers_to_ms

def is_ip(node_str):
    if not node_str:
        pass
    addr_bytes = node_str.split('.')
    if len(addr_bytes) != 4:
        return None
    # if any actual ip, then should be composed of numbers
    for addr_byte in addr_bytes:
        try:
            float(addr_byte)
        except ValueError:
            return None
    return addr_bytes

# aggregate_outside_nodes: Graph -> Graph
# this function combines all the nodes that correspond to outside entities into a single
# node, with the name 'outside'
# it identifies outside nodes by assuming that all in-cluster nodes either labeled, loopback,
# or in the '10.X.X.X' subnet
def aggregate_outside_nodes(G):
    outside_nodes = []

    for node in G.nodes():
        # okay, now check if the node is in the outside
        #print "aggregate_outside_nodes", node
        addr_bytes = is_ip(node)
        if ( addr_bytes ):
            if (not is_private_ip(addr_bytes)):
                outside_nodes.append( node )
                # might wanna put below line back in...
                #print "new outside node!", node
    # might wanna put below line back in...
    #print "outside nodes", outside_nodes
    #try:
    #print("trying...")
    if len(outside_nodes) == 0: #probably already aggregated
        H = G.copy()
        return H

    first_node = outside_nodes[0]
    H = G.copy()
    for cur_node in outside_nodes[1:]:
        ## going to modify the src code for contracted_edges in networkx 1.10
        ##  ## https://networkx.github.io/documentation/networkx-1.10/_modules/networkx/algorithms/minors.html#contracted_nodes
        in_edges = ((w, first_node, d) for w, x, d in G.in_edges(cur_node, data=True))
        out_edges = ((first_node, w, d) for x, w, d in G.out_edges(cur_node, data=True))
        new_edges = chain(in_edges, out_edges)
        #print("new_edges_chain...", new_edges)
        for new_edge in new_edges:
            if H.has_edge(new_edge[0], new_edge[1]):
                #print("modify exist edge",G[new_edge[0]][new_edge[1]])
                #G[new_edge[0]][new_edge[1]]['weight']=  G[new_edge[0]][new_edge[1]]['weight'] + new_edge[2]['weight']
                #G[new_edge[0]][new_edge[1]]['frames'] =  G[new_edge[0]][new_edge[1]]['frames'] + new_edge[2]['frames']
                H[new_edge[0]][new_edge[1]]['weight'] = H[new_edge[0]][new_edge[1]]['weight'] + new_edge[2]['weight']
                H[new_edge[0]][new_edge[1]]['frames'] = H[new_edge[0]][new_edge[1]]['frames'] + new_edge[2]['frames']
            else:
                #print("newe_edge",new_edge)
                #G.add_edge(new_edge[0],new_edge[1], frames=new_edge[2]['frames'], weight=new_edge[2]['weight'])
                H.add_edge(new_edge[0], new_edge[1], frames=new_edge[2]['frames'], weight=new_edge[2]['weight'])

                #G.add_edge(*new_edge)
        #print("removing...", cur_node)
        H.remove_node(cur_node)

        #G = nx.contracted_nodes(G, first_node, cur_node, self_loops=False)
    mapping = {first_node: 'outside'}
    nx.relabel_nodes(H, mapping, copy=False)
    #except:
    #    pass
    return H

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
            # print node
            if ms in node:
                mapping[ms].append(node)
                mapping_node_to_ms[node] = ms
                break
    # note: might wanna re-enable the below line...
    # print mapping_node_to_ms
    for (u, v, data) in G.edges(data=True):
        # print (u,v,data)
        try:
            H.add_edge(mapping_node_to_ms[u], mapping_node_to_ms[v], weight=data['weight'],
                       frames=data['frames'])
        except:
            # might wanna put back in vvv
            # print "this edge did NOT show up in the map!", (u,v,data)

            if u in mapping_node_to_ms:
                u = mapping_node_to_ms[u]
            if v in mapping_node_to_ms:
                v = mapping_node_to_ms[v]
            H.add_edge(u, v, weight=data['weight'], frames=data['frames'])

    pos = graphviz_layout(H)
    nx.draw_networkx(H, pos, with_labels=True, arrows=True)
    # plt.show()

    # while we are at it, let's also return a simpler graph, which is just
    # the multigraph but with all the edges aggregated together
    M = nx.DiGraph()
    mapping_edge_to_weight = {}
    mapping_edge_to_frames = {}
    for node_one in H.nodes():
        for node_two in H.nodes():
            mapping_edge_to_weight[(node_one, node_two)] = 0
            mapping_edge_to_frames[(node_one, node_two)] = 0

    for (u, v, data) in H.edges.data(data=True):
        mapping_edge_to_weight[(u, v)] += data['weight']
        mapping_edge_to_frames[(u, v)] += data['frames']

    # might wanna put back in vvv
    # print "mapping_edge_to_weight", mapping_edge_to_weight

    for edges, weights in mapping_edge_to_weight.iteritems():
        if weights:
            M.add_edge(edges[0], edges[1], weight=weights, frames=mapping_edge_to_frames[edges])

    pos = graphviz_layout(M)
    nx.draw_networkx(M, pos, with_labels=True, arrows=True)
    # plt.show()

    return H, M

# okay, so G is already a network, read in from an edgefile
# level_of_processing is one of ('app_only', 'none', 'class')
# where none = no other processing except aggregating outside entries (container granularity)
# where app_only = 1-step induced subgraph of the application containers (so leaving out infrastructure)
# where class = aggregate all containers of the same class into a single node
# func returns a new graph (so doesn't modify the input graph)
def prepare_graph(G, svcs, level_of_processing, is_swarm, counter, file_path, ms_s, container_to_ip,
                  infra_instances, drop_infra_p):
    G = copy.deepcopy(G)
    G = aggregate_outside_nodes(G)
    if level_of_processing == 'none':
        unprocessed_G = G.copy()
        containers_to_ms = map_nodes_to_svcs(unprocessed_G, svcs)# + infra_service)
        #print "container to service mapping: ", containers_to_ms
        #print "graph before attribs", list(G.nodes(data=True))
        nx.set_node_attributes(unprocessed_G, containers_to_ms, 'svc')
        #print "graph after attribs", list(G.nodes(data=True))

        filename = file_path.replace('.txt', '') + 'unprocessed_network_graph_container.png'
        make_network_graph(unprocessed_G, edge_label_p=True, filename=filename, figsize=(54,32),
                           node_color_p=False, ms_s=ms_s)

        return unprocessed_G

    elif level_of_processing == 'app_only':
        if is_swarm:
            G, svcs = process_graph(G, is_swarm, container_to_ip, ms_s)
        else:
            G,_ = process_graph(G, is_swarm, container_to_ip, ms_s)

        containers_to_ms = map_nodes_to_svcs(G, svcs)
        infra_nodes = []
        for node in G.nodes():
            for infra_instance_name, ip_and_type in infra_instances.iteritems():
                infra_instance_name, infra_instance_PodSvc = infra_instance_name, ip_and_type[1]
                if infra_instance_PodSvc == 'pod':
                    if infra_instance_PodSvc == node:
                        infra_nodes.append(node)
                elif infra_instance_PodSvc == 'svc':
                    if match_name_to_pod(infra_instance_name, node):
                        infra_nodes.append(node)

        #print "services to map for", svcs
        #print "container to service mapping: ", containers_to_ms
        #print "graph before attribs", list(G.nodes(data=True))
        nx.set_node_attributes(G, containers_to_ms, 'svc')
        #print "graph after attribs", list(G.nodes(data=True))

        # okay, we also want to include nodes that are a single hop away
        application_nodes = list(set(containers_to_ms.keys()).difference(set(infra_nodes)))
        one_hop_away_nodes = []

        #I'm not sure if the whole 1-step induced thing is really necessary....
        # but changing it will break all types of things later on, so I guess we should keep it for now.
        # NOTE: should really encorporate finding the DNS thing later....
        '''
        for (u, v, data) in G.edges(data=True):
            if u in application_nodes:
                if v not in application_nodes and v not in one_hop_away_nodes:
                    one_hop_away_nodes.append(v)
            if v in application_nodes:
                if u not in application_nodes and u not in one_hop_away_nodes:
                    one_hop_away_nodes.append(u)
        #print "app nodes and one hop away", application_nodes + one_hop_away_nodes
        induced_graph = G.subgraph(G.subgraph(application_nodes+one_hop_away_nodes)).copy()
        '''
        if drop_infra_p:
            induced_graph = G.subgraph(application_nodes)
        else:
            induced_graph = G
        # might want to put back in vvv
        #print "graph after induced", list(induced_graph.nodes(data=True))

        # need to relabel again, so we can get the infrastructure services labeled
        # actually don't wanna b/c inter-system stuff seems to handle VIPs differently
        #containers_to_ms = map_nodes_to_svcs(G, svcs + infra_service)
        #nx.set_node_attributes(G, containers_to_ms, 'svc')

        if counter < 85:  # keep # of network graphs to a reasonable amount
            filename = file_path.replace('.txt', '') + '_app_only_network_graph_container.png'
            make_network_graph(induced_graph, edge_label_p=True, filename=filename, figsize=(54, 32), node_color_p=False,
                               ms_s=ms_s)
        return induced_graph
    elif level_of_processing == 'class':
        aggreg_multi_G, aggreg_simple_G = aggregate_graph(G, ms_s)# + infra_service)
        if counter < 85:
            filename = file_path.replace('.txt', '') + '_network_graph_class.png'
            make_network_graph(aggreg_simple_G, edge_label_p=True, filename=filename, figsize=(16, 10),
                               node_color_p=False,
                               ms_s=ms_s)
        return aggreg_simple_G
    else:
        print "that type of processing not recognized"
        exit(1)

# note: this does nothing if its not docker swarm (and it's not docker swarm ATM)
def process_graph(G, is_swarm, container_to_ip, ms_s):
    G = G.copy()
    svcs = None
    if is_swarm:
        # okay, here is the deal. I probably want to have an unprocessed and a processed version
        # of the graphs + metrics

        # note a few edge cases not covered in 49-64, but i'll handle those as they occur
        for container_list_and_network in container_to_ip.values():
            # print "containerzzz", container_list_and_class[0]
            for ms in ms_s:
                if ms in container_list_and_network[0]:
                    if 'VIP' not in container_list_and_network[0]:
                        if container_list_and_network[0] not in G and 'endpoint' not in container_list_and_network[0]:
                            G.add_node(container_list_and_network[0])
                            break
        for (u, v, data) in G.edges(data=True):
            if 'VIP' in v:
                # so this connects the services VIP to the endpoint of that service's network.
                # however, it breaks down when the service is in more than one network.
                # NOTE THIS SOLUTION MIGHT BREAK DOWN WHEN A LARGE NUMBER OF NETWORKS AND SERVICE
                # ARE PRESENT
                endpoints = []
                for container_ip, container_name_and_net_name in container_to_ip.iteritems():
                    if container_name_and_net_name[0] == v:
                        # need to check if there is an edge between a container instance of the
                        # service and the endpoint -- here's an alternative solution, let's just merge
                        # all of the possible endpoints together, since who knows?
                        print "container_name_and_net_name", container_name_and_net_name, v
                        endpoint = container_name_and_net_name[1] + '-endpoint'
                        # note: the code below is just for atsea shop exp3 v2, b/c I am too time
                        # constrained to write general-purpose code
                        # okay, should not hardcode this type of thing in, but i am going to do
                        # it just this once, b/c i have other stuff to do
                        # if container_name_and_net_name[0] == 'atsea_database_VIP' and 'back' in container_name_and_net_name[1]:
                        #    break

                print "endpoint", endpoint, '\n'
                if G.has_edge(v, endpoint):
                    G[v][endpoint]['weight'] += data['weight']
                else:
                    G.add_edge(v, endpoint, weight=data['weight'])
        # '''
        for (u, v, data) in G.edges(data=True):
            if 'VIP' in u and 'endpoint' in v:
                if not G.has_edge(v, u):
                    # if only goes in a single direction, we
                    print (u, v, data)
                    # str() added below b/c it was being converted to unicode
                    G = nx.contracted_nodes(G, v, u, self_loops=False)

        # not this only applies for docker swarm (well maybe k8s? not sure...)
        # need to merge 'ingress-endpoint' and 'gateway_ingress_sbox'
        # b/c these are just two sides of the same NAT
        for u in G.nodes():
            for v in G.nodes():
                if u != v:
                    # print u,v, 'ingress-endpoint' in u, 'gateway_ingress-sbox' in v
                    if 'ingress-endpoint' in u and 'gateway_ingress-sbox' in v:
                        if not G.has_edge(u, v) and not G.has_edge(u, v):
                            G = nx.contracted_nodes(G, v, u, self_loops=False)

        # this section below is now outdated
        # this is misleading for k8s b/c it talks to other outside IPs too
        # 192.168.99.1 is really just the generic 'outside' here
        '''
        mapping = {'192.168.99.1': 'outside'}
        try:
            nx.relabel_nodes(G, mapping, copy=False)
        except KeyError:
            pass  # maybe it's not in the graph?
        '''
        svcs = get_svc_equivalents(is_swarm, container_to_ip)
        print "these services were found:", svcs
        containers_to_ms = map_nodes_to_svcs(G, svcs)
        print "container to service mapping: ", containers_to_ms
        nx.set_node_attributes(G, containers_to_ms, 'svc')
    else:
        pass
        '''
        # todo: this is ugly, make it nicer
        # I'm going to merge all the traffic that is coming to/from the outside together
        for u in G.nodes():#[:int(G.number_of_nodes() / 2.0)]:
            for v in G.nodes():#[int(G.number_of_nodes() / 2.0):]:
                if u != v:
                    # print u,v, 'ingress-endpoint' in u, 'gateway_ingress-sbox' in v
                    if 'k8s' not in u and '10.0.2' not in u:
                        if 'k8s' not in v and '10.0.2' not in v:
                            try:
                                G = nx.contracted_nodes(G, v, u, self_loops=False)
                            except:
                                print u, "and", v, "have probably been merged already"
        # note: this is not necessarily the case, but it seems like should more or less
        # hold for all the minikube deployments that I'd do
        
        mapping = {'10.0.2.15': 'default-http-backend(NAT)'}
        try:
            nx.relabel_nodes(G, mapping, copy=False)
        except KeyError:
            pass  # maybe it's not in the graph?
        '''
    return G, svcs


def get_svc_equivalents(is_swarm, container_to_ip):
    # going to set the node attributes with the svc now
    # first, going to find all of the services. The services will be in th e
    # container_to_ip structure w/ _VIP appended
    svcs = []
    if is_swarm:
        for container_list_and_network in container_to_ip.values():
            # if container_list_and_network[0] != '':
            #    continue
            print "container_list_and_network", container_list_and_network
            if 'VIP' in container_list_and_network[0] or 'sbox' in container_list_and_network[0]:
                print container_list_and_network[0]
                svcs.append(container_list_and_network[0].replace("_VIP", ""))
            if 'endpoint' in container_list_and_network[0]:
                svcs.append(container_list_and_network[0])  # this is so that the load-balancers can be included...
        svcs.append('outside')
        # sort services by length, note referred to
        # https://stackoverflow.com/questions/2587402/sorting-python-list-based-on-the-length-of-the-string
        svcs.sort(key=len)
    #else:
    #    pass # todo
    return svcs

def make_network_graph(G, edge_label_p, filename, figsize, node_color_p, ms_s):
    plt.clf()
    color_map = []
    if node_color_p:
        color_map = generate_network_graph_colormap(color_map, ms_s, G)
    plt.figure(figsize=figsize)  # todo: turn back to (27, 16)
    pos = graphviz_layout(G)
    for key in pos.keys():
        pos[key] = (pos[key][0] * 4, pos[key][1] * 4)  # too close otherwise
    nx.draw_networkx(G, pos, with_labels=True, arrows=True, font_size=8, font_color='b')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    # might wanna put the below line back in...
    #print "edge_labels", edge_labels
    if edge_label_p:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, label_pos=0.3)
    plt.savefig(filename, format='png')


def generate_network_graph_colormap(color_map, ms_s, G):
    # okay, now I want to color the different classes different colors. I am going to make the assumption
    # that if an container's name has a '.' in it, then I can get the class name by splitting on the '.'
    # and taking the value to the left
    for node in G:
        j = None
        for i in range(0, len(ms_s)):
            if 'endpoint' in node or 'VIP' in node or 'sbox' in node:
                j = len(ms_s) + 1
                break
            if ms_s[i] in node:
                j = i
                break
        # print "j", j
        if j != None:
            # assign color to the node here
            if j == len(ms_s) + 1:
                color_map.append(0.0)
            else:
                color_map.append(float(len(ms_s) + 2) / j)
        else:
            # okay, either load balancer or other
            color_map.append((len(ms_s) * 2) / (len(ms_s)))  # float(len(ms_s) + 2) / (len(ms_s) + 2))

    print "color_map", color_map, len(color_map), len(G.nodes()), len(ms_s), np.array(color_map)
    print [i for i in G.nodes()]  # range(len(G.nodes()))
    return color_map


def match_name_to_pod(abstract_node_name, concrete_pod_name):
    # OLD VERISON
    #matching_concrete_nodes = [node for node in graph.nodes() if abstract_node in node if node not in excluded_list]
    if '_VIP' in abstract_node_name:
        return abstract_node_name in concrete_pod_name
    else:
        valid = re.compile('.*' + abstract_node_name + '-[0-9].*')
        match_status = valid.match(concrete_pod_name)
        if match_status:
            return True
        else:
            return False