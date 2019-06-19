import copy
import re
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from itertools import chain


def is_private_ip(addr_bytes, ip_gatway):

    # if the IP is loopback or in the docker network, then it is local. Otherwise, it is not.
    # (note that most entities should have names, so they'd never even get here)
    ## NOTE: SO THE IP THAT DNS TRIES TO COMMYUNICATE WITH WILL RETURN FALSE (b/c it is the IP address of a ethernet interface)
    if addr_bytes[0] == '127' and addr_bytes[1] == '0' and addr_bytes[2] == '0' and addr_bytes[3] == '1':
        return True # loopback is definitely private
    elif addr_bytes[0] == ip_gatway[0] and addr_bytes[1] == ip_gatway[1] and addr_bytes[2] == ip_gatway[2] and \
                    addr_bytes[3] != ip_gatway[3]:
        return True
    else:
        return False

def ip2container_to_container2ip(ip_to_container):
    container2ip = {}
    for ip,name_and_attribs in ip_to_container.iteritems():
        name = name_and_attribs[0]
        name_and_attribs = list(name_and_attribs)
        name_and_attribs[0] = ip
        container2ip[name] = tuple(name_and_attribs)
    return container2ip

def get_svcs_from_mapping(name2ip):
    svc_to_label = {}
    label_to_svc = {}
    for name,attribs in name2ip.iteritems():
        if len(attribs) <= 2:
            continue # would be using the old parse function...
        entity_type = attribs[3]
        if entity_type == 'svc':
            name = name.replace('_VIP', '')
            labels = attribs[4]
            svc_to_label[name] = labels

            if labels is not None:
                for label in labels:
                    if label in label_to_svc and 'k8s-app' not in label:
                        print "more than one service with the same label -- wierd!!!", label_to_svc[label], name# but NOT actually an error...
                        #exit(233)
                        ## this is kinda a wierd criteria... but I'm going to say that the shorter name
                        ## is the definitive name of the service...
                        if len(name) < len(label_to_svc[label]):
                            label_to_svc[label] = name
                    else:
                        label_to_svc[label] = name
            # note: I am going to make the assumption that each label is associated with only one service.
            # the kube-system stuff might not follow this, but I'm not calculating their average behavior anyway

            #if label not in label_to_svc :
            #    label_to_svc[label] = [name]
            #else:
            #    label_to_svc[label].append(name)

    return svc_to_label,label_to_svc

def map_nodes_to_svcs(G, svcs, ip_to_entity):
    name2ip = ip2container_to_container2ip(ip_to_entity)

    if (not svcs or svcs == []) and not ip_to_entity:
        return {}
    containers_to_ms = {}
    svc_to_label,label_to_svc = get_svcs_from_mapping(name2ip)
    svcs = svc_to_label.keys()

    for u in G.nodes():
        # if the labels identify the service, then we can just use that...
        if u in name2ip:
            node_attribs = name2ip[u]
            found_matching_svc = False
            #print "node_attribs", node_attribs
            if len(node_attribs) >= 5:
                if node_attribs[4] != None:
                    ## the services have labels, and so do the nodes, and you have to match them!
                    labels = node_attribs[4]
                    for label in labels:
                        try:
                            svc = label_to_svc[label]
                            if found_matching_svc:
                                if svc != containers_to_ms[u]:
                                    print "one node matches two services..."
                                    exit(244)
                            found_matching_svc = True
                            containers_to_ms[u] = svc
                        except:
                            pass
                            '''
                            if 'k8s-app' in label:
                                pass
                            else:
                                print label, node_attribs
                                print("there's a label not a associated with a service! that's weird! .... exiting now...")
                                exit(222)
                            '''
            if found_matching_svc:
                continue
            for svc in svcs:
                if u == "wwwppp-wordpress_VIP" : # this is for debugging... I can  take it out eventually...
                    pass
                if match_name_to_pod(svc, u) or match_name_to_pod(svc + '_VIP', u):
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
### TODO: should pass the default params in from the calling function... (can be passed in from my dump of the docker network + logs)
def aggregate_outside_nodes(G, gateway_ip = '172.17.0.1'): #, minikube_ip = '192.168.99.100'):
    outside_nodes = []

    #minikube_ip = is_ip(minikube_ip)
    #gateway_ip = is_ip(gateway_ip)

    for node in G.nodes():
        # okay, now check if the node is in the outside
        #print "aggregate_outside_nodes", node
        addr_bytes = is_ip(node)
        if ( addr_bytes ):
            #if ((not is_private_ip(addr_bytes) or addr_bytes == gateway_ip) and addr_bytes != minikube_ip):
            # for now, we are keeping the docker network gateway as a seperate node
            #if (not is_private_ip(addr_bytes, is_ip(gateway_ip)) and node != minikube_ip and node != gateway_ip):
            if (not is_private_ip(addr_bytes, is_ip(gateway_ip)) and node != gateway_ip):
                outside_nodes.append( node )
                # might wanna put below line back in...
                #print "new outside node!", node

    if len(outside_nodes) == 0: #probably already aggregated
        H = G.copy()
        return H

    H = G.copy()
    H = consolidate_nodes(H, outside_nodes)

    ## now here's the compensation mechanism here...
    # this exists b/c health checks (and other traffic from the kubernetes daemon) are routed through the
    # docker gateway, which functions as a load balancer. Therefore, all the containers tend to talk to
    # the gateway IP. however, traffic going outside the cluster also goes throught the gateway IP.
    # if we want to detect when a pod is really talking to the outside, we'll see it talk to both an outside
    # IP address AND the gateway IP address. Sounds easy? Except traffic from the outside to the pod ONLY
    # shows up from the gateway IP (draw it out if this doesn't make sense). So if pod <-> gateway IP exists,
    # then we need to check if pod -> outside exists, in which case we want to keep pod -> outside and
    # gateway -> ip (b/c we do not care about communication with the kubernetes daemon)
    for cur_node in H.nodes():
        #print "cur_node", cur_node
        sending_nodes = [u for u,v,d in H.in_edges(cur_node, data=True)]
        receiving_nodes = [v for u,v,d in H.out_edges(cur_node, data=True)]
        # if the docker network gateway sends no traffic to the pod in question, then we can safely ignore it
        if gateway_ip not in sending_nodes:
            continue
        # but if the docker network gatway does send traffic to the pod, then does the pod send traffic to the outside??
        if 'outside' not in receiving_nodes:
            # if it does not, then the only data exchanged is stuff from the kubelet, so we can just delete it and move on
            H.remove_edge(gateway_ip, cur_node)
            if cur_node != gateway_ip: # why would this happen
                H.remove_edge(cur_node, gateway_ip)
        else:
            # if it does, then we cannot tell whether the traffic from the gateway is from the actual outside or from the
            # kubelet. In this case, we should keep node->outside and gateway->node, but delete node->gateway
            H.remove_edge(cur_node, gateway_ip)

    # and then consolidate again
    outside_nodes = ['outside', gateway_ip]
    H = consolidate_nodes(H, outside_nodes)

    return H

def consolidate_nodes(H, nodes_to_consolidate):
    first_node = nodes_to_consolidate[0]
    for cur_node in nodes_to_consolidate[1:]:
        ## going to modify the exp_support_scripts code for contracted_edges in networkx 1.10
        ##  ## https://networkx.github.io/documentation/networkx-1.10/_modules/networkx/algorithms/minors.html#contracted_nodes
        in_edges = ((w, first_node, d) for w, x, d in H.in_edges(cur_node, data=True))
        out_edges = ((first_node, w, d) for x, w, d in H.out_edges(cur_node, data=True))
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
    return H

# aggregate all nodes of the same class into a single node
# let's use a multigraph, so we can keep all the edges as intact...
def aggregate_graph(G, containers_to_ms):
    containers_to_ms['outside'] = 'outside'
    # step 1: convert containers_to_ms to ms_to_container
    '''
    ms_to_container = {}
    for container,ms in containers_to_ms.iteritems():
        if ms not in ms_to_container:
            ms_to_container[ms] = []
        ms_to_container[ms].append( container )
    '''

    H = nx.MultiDiGraph()
    '''
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
    '''
    # note: might wanna re-enable the below line...
    # print mapping_node_to_ms
    for (u, v, data) in G.edges(data=True):
        # print (u,v,data)
        #try:
        #    H.add_edge(containers_to_ms[u], containers_to_ms[v], weight=data['weight'],
        #               frames=data['frames'])
        #except:
        # might wanna put back in vvv
        # print "this edge did NOT show up in the map!", (u,v,data)

        #if u in containers_to_ms:
        #    u = containers_to_ms[u]
        #if v in containers_to_ms:
        #    v = containers_to_ms[v]

        u = containers_to_ms[u]
        v = containers_to_ms[v]
        H.add_edge(u, v, weight=data['weight'], frames=data['frames'])

    try:
        pos = graphviz_layout(H)
        nx.draw_networkx(H, pos, with_labels=True, arrows=True)
        # plt.show()
    except:
        pass

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
    try:
        pos = graphviz_layout(M)
        nx.draw_networkx(M, pos, with_labels=True, arrows=True)
        # plt.show()
    except:
        pass

    return H, M

def find_infra_components_in_graph(G, infra_instances):
    infra_nodes = []
    #print "infra_instances", infra_instances
    for node in G.nodes():
        for infra_instance_name, ip_and_type in infra_instances.iteritems():
            infra_instance_name, infra_instance_PodSvc,svc = infra_instance_name, ip_and_type[1], ip_and_type[2]
            if 'heapster' in node and 'heapster' in infra_instance_name:
                # print infra_instance_PodSvc == node
                pass  # to be used as a debug point...
            if infra_instance_PodSvc == 'pod':
                if infra_instance_name == node:
                    infra_nodes.append(node)
            elif infra_instance_PodSvc == 'svc':
                if match_name_to_pod(infra_instance_name + '_VIP', node, svc=svc):
                    infra_nodes.append(node)
    return infra_nodes

def remove_infra_from_graph(G, infra_nodes):
    nodes_in_g = [node for node in G.nodes()]
    application_nodes = list(set(nodes_in_g).difference(set(infra_nodes)))
    application_nodes = [node for node in application_nodes if not is_ip(node)]
    induced_graph = G.subgraph(G.subgraph(application_nodes)).copy()
    return induced_graph

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
    containers_to_ms = map_nodes_to_svcs(G, None, container_to_ip)
    if level_of_processing == 'app_only':
        nx.set_node_attributes(G, containers_to_ms, 'svc')
        if drop_infra_p:
            infra_nodes = find_infra_components_in_graph(G, infra_instances)
            induced_graph = remove_infra_from_graph(G, infra_nodes)
        else:
            induced_graph = G

        if counter < 85:  # keep # of network graphs to a reasonable amount
            filename = file_path.replace('.txt', '') + '_app_only_network_graph_container.png'
            make_network_graph(induced_graph, edge_label_p=True, filename=filename, figsize=(54, 32), node_color_p=False,
                               ms_s=ms_s)
        return induced_graph
    elif level_of_processing == 'class':
        #print "containers_to_ms",containers_to_ms
        aggreg_multi_G, aggreg_simple_G = aggregate_graph(G, containers_to_ms)# + infra_service)
        if counter < 85:
            filename = file_path.replace('.txt', '') + '_network_graph_class.png'
            make_network_graph(aggreg_simple_G, edge_label_p=True, filename=filename, figsize=(16, 10),
                               node_color_p=False,
                               ms_s=ms_s)
        return aggreg_simple_G
    else:
        print "that type of processing not recognized"
        exit(1)

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
    try:
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
    except:
        pass

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


def match_name_to_pod(abstract_node_name, concrete_pod_name, svc=None):

    if svc and svc == abstract_node_name and '_VIP' not in concrete_pod_name:
        return True

    if '_VIP' in abstract_node_name:
        return abstract_node_name in concrete_pod_name
    else:
        valid = re.compile('.*' + abstract_node_name + '-[a-z0-9][a-z0-9][a-z0-9][a-z0-9][a-z0-9][a-z0-9][a-z0-9][a-z0-9][a-z0-9].*')
        print "valid_match_result?", valid, abstract_node_name
        match_status = valid.match(concrete_pod_name)
        valid_two = re.compile('.*' + abstract_node_name + '-[0-9].*')
        match_status_two = valid_two.match(concrete_pod_name)
        if match_status or match_status_two:
            return True
        else:
            valid = re.compile('.*' + abstract_node_name + '-$') # end line is also acceptable.
            match_status = valid.match(concrete_pod_name)
            if match_status:
                return True
            else:
                return False