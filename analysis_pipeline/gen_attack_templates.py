import docker
import networkx as nx
import os
import matplotlib.pyplot as plt
import matplotlib
from networkx.drawing.nx_agraph import graphviz_layout
matplotlib.use('Agg',warn=False, force=True)
import matplotlib
import matplotlib.pyplot as plt
import time
import os
import random
import itertools

def run_mulval():
    client = docker.from_env()
    client.containers.list()
    cwd = os.getcwd()
    print "cwd", cwd
    print "mulval_related_dir", cwd + '/mulval_inouts'
    # note: going to start container running everytime (and later on will detelete it...)
    container = client.containers.run("risksense/mulval", detach=True,
                                      volumes={cwd + '/mulval_inouts': {'bind': '/mnt/vol2', 'mode': 'rw'}})
    print "cwd", cwd

    commands = [ ['/root/mulval/utils/graph_gen.sh', '/mnt/vol2/test_mulval_input.P'],
                 #['mv', '/root/mulval/utils/xsb_log.txt', '/mnt/vol2/xsb_log.txt'], # not sure why this line exists...
                 #['mv', '/root/mulval/utils/AttackGraph.txt', '/mnt/vol2/AttackGraph.txt'], # not sure why this line exists...
                 ['mv', '/input/AttackGraph.txt', '/mnt/vol2/AttackGraph.txt'], # this is b/c it wasn't working on hipsterstore... might break others, tbh....
                 ['mv', '/input/xsb_log.txt', '/mnt/vol2/xsb_log.txt']]  # this is b/c it wasn't working on hipsterstore... might break other applications, tbh....
    for command in commands:
        out = container.exec_run(command, workdir='/root/mulval/utils/')
        print out #out.output, out

    print "mulval cmds done..."

    print client.containers.list()

    print container.stop()
    print container.remove()
    #print client.containers.prune(filters={'id': container.id})

## NOTE: there can be NO - chars in mulval input... it causes problems with the trace parser!!
def prepare_mulval_input(ms_s, mapping, sensitive_ms, netsec_policy, intersvc_vip_pairs):
    # this is going to need to take the k8s input files and then convert to the input format for mulval
    ### Stuff to include:
    ###    internet <-> anything it wants, anything it wants
    ###    pods  -> send anywhere, anything
    ###    vips -> get set of three rules, heavily dictating what they accept and what they don't
    ###    + each pod can get the same vulnerability (so 2 lines + shared)
    ###    + each vip gets its own vulnerability (so 3 lines...)

    ## so we can break this down into two steps...
    ## (1) acquire all the config-related information
    ## (2) convert it into the mulval-compatible form...

    ## okay, well let's obviously start w/ (1)... can we leverage existing parsing
    ## capabilities to get it done?? looks like kinda, but not completely, b/c it puts everything
    ## into the 'mapping' dict, instead of keeping everything seperate...
    ## just us the ms (list of microservices) to filter the mappings and that'll be fine...
    ## so next step is wiring it in... comment out other parts afterr

    # step 1a: get list of services + pods... wait no, that's wrong... we only care about ms_s...
    # the mapping to the concrete pods happens in postprocessing... so we can create the model
    # declarivitevly via the following logic...
    svcs = []
    for ms in ms_s:
        ms = ms.replace('-','_')
        svcs.append( (ms + '_pod', ms+ '_vip', ms) )

    # step 2. let's generate the corresponding test_mulval_input.P
    # step 2.a. the internet (can talk to/from anything)
    #lines = ['%autogenerated','hacl(internet, _, _, _).']
    lines = ['%autogenerated']
    lines.append('attackGoal(execCode(internet, _)).')
    lines.append('vulExists(internet, \'attacker_can_access_anything\', is_computer).')
    lines.append('networkServiceInfo(internet, is_computer, _, _, _).')

    # i think these only needed to be included once...
    lines.append('vulProperty(\'attacker_can_access_anything\', remoteExploit, privEscalation).')

    # step 2.b. the hacl's ; do simulteanously for VIPs and Pods
    # recall that pods can do whatever they want, vips are more limited (obvi)
    sensitive_node = None
    for svc in svcs:
        if sensitive_ms.replace('-','_') + '_pod' in svc[0]:
            sensitive_node = svc[0]
            print "sensitive_node", sensitive_node
            lines.append('attackerLocated(' + svc[0] + ').')

        lines.append('')
        print 'svc', svc
        port, proto = None,None
        print  "svcs",svcs
        print "netsec_policy",netsec_policy
        #exit(344)
        for svc_two in svcs + [('internet', 'internet', 'internet')]:
            ## check if this is in the allowed dict...
            if netsec_policy is not None:
                svc_convert_to_netsec_pol = svc[0].replace('_pod', '').replace('_vip', '').replace('_','-') ## TODO
                svc_two_convert_to_netsec_pol = svc_two[0].replace('_pod', '').replace('_vip', '').replace('_','-')  ## TODO
                try:
                    print "svc_convert_to_netsec_pol", svc_convert_to_netsec_pol
                    corresponding_netsec_policy =  netsec_policy[svc_convert_to_netsec_pol]
                except:
                    corresponding_netsec_policy = []
                print "netsec_policy_keys", netsec_policy.keys(), svc_convert_to_netsec_pol, svc_two_convert_to_netsec_pol, svc_two_convert_to_netsec_pol not in corresponding_netsec_policy
                print "corresponding_netsec_policy", corresponding_netsec_policy
                if svc_two_convert_to_netsec_pol not in corresponding_netsec_policy:
                    continue

            print mapping
            for ip,individ_mapping in mapping.iteritems():
                try:
                    print "individ_mapping_stuff", svc[2], individ_mapping
                    if svc[2] in individ_mapping[0].replace('-','_') and 'VIP' in individ_mapping[0]:
                        port,proto = individ_mapping[2][0], individ_mapping[3][0]
                except:
                    pass

            print port, proto
            if svc_two != svc:
                print "svc_two", svc_two[0], svc[1],proto,port
                print svc_two, svc, proto
                #print 'hacl(', svc_two[0], ',', svc[1], ', ', proto, ', ', port[0], ').'
                print 'hacl(', svc_two[0], ',', svc[1], ', ', '_', ', ', '_', ').'
                if 'dns_pod' not in svc[0]:
                    #lines.append(  'hacl(' + svc[0] +',' + svc_two[0] + ', ' + proto +', ' + port[0] + ').'  )
                    lines.append('hacl(' + svc[0] + ',' + svc_two[0] + ', ' + '_' + ', ' + '_' + ').')

        #exit(344)
        ## MODIFICATIONS END HERE!!!

        #print svc, svc[2], proto, port
        ## probably wanna fix the port,proto mapping stuff... or not... who cares...
        if not proto:
            proto = '_'
        if not port:
            port = ['_']

        # step 2.c. then do the vulneabiltiy stuff (should be very 'boilerplate')
        lines.append('vulExists(' + svc[0] + ', \'attacker_can_access_anything\', is_computer).')
        lines.append('networkServiceInfo(' + svc[0] + ', is_computer,  _, _ , _).')


    # might want to handle in another place... also this is a kinda nasty workaround...
    lines.append('hacl(' + 'kube_dns_pod' + ',' + 'internet' + ', ' + 'UDP' + ', ' + '53' + ').')


    with open('./mulval_inouts/test_mulval_input.P', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

    print 'MAPS MAPS MAPS'
    for maps,vals in mapping.iteritems():
        print maps,vals

    return sensitive_node # do I want to return anything??

def post_process_mulval_result(sensitive_node, max_number_of_paths, intersvc_vip_pairs, max_path_length, max_dns_porportion):
    # well, not in accordance with the architecture document, if we're putting orchestrator logic into the
    # mulval component, then we'd need to include the 'chaining' together of various steps
    if intersvc_vip_pairs is None:
        intersvc_vip_pairs = set()
        intersvc_vip_pairs.add(('ALL', 'kube_dns_vip'))
        #print "it is necessary to specify at least some parts of the netsec polciy file. Concretly, at least give vip behavior w.r.t. dns "
        #exit(992)

    graph = open('./mulval_inouts/AttackGraph.txt', 'r')
    lines = graph.readlines()
    hacls = []
    for line in lines:
        if 'hacl' in line:
            hacls.append(line.rstrip().lstrip())
    propogation_graph_edges = []
    for hacl in hacls:
        only_hacl =  hacl.split('-', 1)[1]
        only_relevant = only_hacl.split('(')[1].split(')')[0]
        #print only_hacl, only_relevant.split(',')#onfly_hacl.split('('), only_hacl.split(',')
        propogation_graph_edges.append(only_relevant.split(','))

    # okay, for now we should use the simplest visualization process possible... networkx it is then...
    G = nx.DiGraph() # this'll be the propogation graph...
    # okay, so it looks like it is going to be some kinda loop....
    # looks like G.add_edge(NODE1, NODE2, weight=WEIGHT) is the way to do this...
    # so all we gotta do is a construct a set of tuples from the hacls... piece of cake...
    for edge in propogation_graph_edges:
        weight = 1
        G.add_edge(edge[0], edge[1], proto = edge[2], port = edge[3], weight=weight)

    print "G_edges", G.edges()

    for edge in G.edges():
        print "edge", edge

    #nx.draw(G)
    #plt.draw()

    print "making graphviz_layout"

    plt.close('all') ## needed to avoid crashing due to sigsev when I added the eval portion...
    try:
        pos = graphviz_layout(G)
        nx.draw_networkx(G, pos, with_labels=True, arrows=True)
        #plt.show() ## remove!!! <---- <---- <----
        plt.draw()
        if os.path.isfile('./mulval_inouts/propogation_graph.png'):
            os.remove('./mulval_inouts/propogation_graph.png')  # Opt.: os.system("rm "+strFile)
        plt.savefig('./mulval_inouts/propogation_graph.png', format='png', dpi=1000)
        print os.getcwd()
        print G.number_of_nodes(), [i for i in G.nodes()]
        #time.sleep(34)
        #exit(344)
    except:
        pass # no time to debug ATM... and it's not actually important..

    print "networkx graph created/saved"

    # okay, so let's generate all the paths using this...
    paths=[]

    for path in nx.shortest_simple_paths(G, source=sensitive_node, target='internet', weight='weight'):
        #print "not_cleared", path, len(path)
        #print "cleared", [i for i in path if 'vip' not in i and 'dns' not in i], len([i for i in path if 'vip' not in i and 'dns' not in i])
        #print "path_thing", path
        # here's another nasty workaround
        if path_is_valid(path):

            paths_with_vip = add_appropriate_vips(path, intersvc_vip_pairs)
            # test length
            if len(paths_with_vip[0]) > max_path_length:
                break

            print "paths_with_vips", paths_with_vip
            num_free_paths = (max_number_of_paths - len(paths))
            if  num_free_paths < len(paths_with_vip):
                # adjust porportion of dns

                # note: this doesn't actually work... can fix it later if I want... or not...
                proportion_of_dns = calc_portion_of_dns_paths(paths)
                if proportion_of_dns > max_dns_porportion:
                    paths_to_remove = []
                    dns_diff_prop = (proportion_of_dns - max_dns_porportion)
                    dns_diff = int(dns_diff_prop * len(paths))
                    # now remove this number of paths
                    for candidate in reversed(paths):
                        if 'kube_dns_pod' in candidate:
                            paths_to_remove.append(candidate)
                    paths_to_remove = paths_to_remove[:dns_diff]
                    for candidate in paths_to_remove:
                        paths.remove(candidate)

                num_free_paths = (max_number_of_paths - len(paths))
                if num_free_paths < len(paths_with_vip): # if still too big
                    paths.extend( paths_with_vip[:int(num_free_paths)] )
                else:
                    paths.extend(paths_with_vip)
            else:
                paths.extend(paths_with_vip)


        if len(paths) >= max_number_of_paths:
            break

    #print "paths", paths
    #exit(344)

    print paths

    proportion_of_dns = calc_portion_of_dns_paths(paths)
    initiator_info = determine_initator(paths)
    print "initiator_info",initiator_info

    print "has the graph been drawn???? yes, yes they have"
    return paths, initiator_info

def calc_portion_of_dns_paths(paths):
    paths_with_dns = [p for p in paths if 'kube_dns_pod' in p]
    num_paths_with_dns = len(paths_with_dns)
    proportion_of_dns = float(num_paths_with_dns) / len(paths)
    return proportion_of_dns


def add_appropriate_vips(path, intersvc_vip_pairs):
    paths_with_vips = []
    dst_pod = None
    for i in range(0, len(path) - 1):
        src_pod = path[i]
        dst_pod = path[i + 1]
        src_vip = src_pod.replace('_pod', "_vip")
        dest_vip = dst_pod.replace('_pod', "_vip")
        paths_with_vips.append(src_pod)
        if 'internet' in dst_pod:
            pass
        elif ('ALL', dest_vip) in intersvc_vip_pairs:
            paths_with_vips.append(dest_vip)
        elif  (src_pod, dest_vip) in intersvc_vip_pairs:
            #paths_with_vips.append([dest_vip])
            paths_with_vips.append(dest_vip)
        elif (dst_pod, src_vip) in intersvc_vip_pairs:
            #paths_with_vips.append([src_vip])
            paths_with_vips.append(src_vip)
        else:
            paths_with_vips.append([src_vip, dest_vip])
    paths_with_vips.append(dst_pod)

    list_present = [type(i) == list for i in paths_with_vips]
    list_present = True in list_present

    paths_with_vips_expanded = generate_expanded_paths(paths_with_vips)

    #print "paths_with_vips", paths_with_vips

    ## okay, what do I do now...
    if not list_present:
        paths_with_vips_expanded = [paths_with_vips_expanded]


    return paths_with_vips_expanded

def generate_expanded_paths(path):
    copy_of_path = []
    #recursed = False
    for item_spot in range(0, len(path)):
        item = path[item_spot]
        if type(item) == list:
            path_copies = []
            for option in item:
                new_path = generate_expanded_paths(copy_of_path + [option] + path[(item_spot+1):])
                if type(new_path[0]) == list: # should either be all nested lists or not
                    path_copies.extend( new_path )
                else:
                    path_copies.append( new_path )
            new_path = generate_expanded_paths(copy_of_path + path[(item_spot + 1):])
            if type(new_path[0]) == list:  # should either be all nested lists or not
                path_copies.extend(new_path)
            else:
                path_copies.append(new_path)
            copy_of_path = path_copies
            #recursed=True
            break
        else:
            copy_of_path.append(item)

    #if not recursed:
    #    copy_of_path = [copy_of_path]

    return copy_of_path

def generate_synthetic_attack_templates(mapping, ms_s,sensitive_ms, max_number_of_paths, netsec_policy,
                                        intersvc_vip_pairs, max_path_length, dns_porportion):
    print "generate_synthetic_attack_templates", mapping, ms_s,sensitive_ms, max_number_of_paths, netsec_policy, \
                                        intersvc_vip_pairs, max_path_length, dns_porportion

    if not max_number_of_paths:
        return [],{}

    sensitive_node = prepare_mulval_input(ms_s, mapping, sensitive_ms, netsec_policy, intersvc_vip_pairs)
    run_mulval()
    paths, initiator_info = post_process_mulval_result(sensitive_node, max_number_of_paths, intersvc_vip_pairs, max_path_length, dns_porportion)
    return paths, initiator_info

def determine_initator(paths):
    initiator_info = {} # a dict mapping (first_node,second_node) -> (initiator, 'initiator'),
                        # where 'initiator' exists to reading this code later on more clear
    for path in paths:
        for i in range(0, len(path)-1):
            first,second=path[i],path[i+1]
            if (first,second) not in initiator_info:
                # this if exists b/c initiator info is a property of the names, so won't very if shows up
                # multiple times...

                # okay, so there are three rules that we need to observe here...
                # rule 1: if both are pods / internet, then either can be the initiator
                if 'vip' not in first and 'vip' not in second: #pod,pod
                    initiator_info[(first,second)] = ('?', 'initiator')
                elif 'vip' not in first and 'vip' in second: #pod,vip
                    first_basename = first.replace('_pod', '') # recall, could also be internet
                    second_basename = second.replace('_vip', '')
                    if first_basename == second_basename:
                        initiator_info[(first, second)] = (second, 'initiator')
                    else:
                        initiator_info[(first, second)] = (first, 'initiator')
                elif 'vip' in first and 'vip' not in second: #vip,pod
                    first_basename = first.replace('_vip', '')
                    second_basename = second.replace('_pod', '')
                    if first_basename == second_basename:
                        initiator_info[(first, second)] = (first, 'initiator')
                    else:
                        initiator_info[(first, second)] = (second, 'initiator')
                else:
                    exit(12) # ERROR, should not be possible, need to check the mulval component thoroughly...

    return initiator_info

def parse_netsec_policy(netsec_policy):
    if netsec_policy is None:
        return None,None

    allowed_dict = {}
    allow_all = []
    intersvc_vip_pairs = set()

    with open(netsec_policy, 'r') as f:
        lines = f.readlines()
        hit_divider=False
        for line in lines:
            if '----------'  in line:
                hit_divider = True

            if not hit_divider:
                if line[0] != '#':
                    line_split = line.split(' ')
                    #print "line_split", line_split
                    if line_split[1].rstrip().lstrip() == 'ALLOWED':
                        if line_split[2].rstrip().lstrip() == 'all':
                            #print "found the allowed all!"
                            allow_all.append(line_split[0].rstrip().lstrip())
                            continue
                        #print "did not find the allowed all"
                        if line_split[0].rstrip().lstrip() in allowed_dict:
                            allowed_dict[line_split[0].rstrip().lstrip()].append(line_split[2].rstrip().lstrip())
                        else:
                            allowed_dict[line_split[0].rstrip().lstrip()] = [line_split[2].rstrip().lstrip()]
                    print line
            else:
                if line[0] != '#':
                    try:
                        line_split = line.rstrip().lstrip().split(' ')
                        print "line_split", line_split
                        intersvc_vip_pairs.add((line_split[0], line_split[1]))
                    except:
                        pass
    allowed_dict_keys = allowed_dict.keys()
    allowed_dict_vals = list(itertools.chain(*allowed_dict.values()))
    allowed_dict_keys = list(set(allowed_dict_keys+allowed_dict_vals))
    print allow_all
    #exit(34)
    for allow_all_item in allow_all:
        allowed_dict[allow_all_item] = allowed_dict_keys
    for allowed_dict_key in allowed_dict_keys:
        if allowed_dict_key in allowed_dict:
            allowed_dict[allowed_dict_key].extend(allow_all)
        else:
            allowed_dict[allowed_dict_key] = allow_all

    print "allowed_dict", allowed_dict
    #exit(34)
    return allowed_dict,intersvc_vip_pairs

def path_is_valid(path):
    print '------'
    location_of_vips = [i for i,j in enumerate(path) if 'vip' in j]
    valid = True
    for location in location_of_vips:
        prev_node = path[location-1]
        next_node = path[location+1]
        prev_node_class = prev_node.replace('_pod','').replace('_vip','') #'_'.join(prev_node.split('_')[:-1])
        next_node_class = next_node.replace('_pod','').replace('_vip','')  #'_'.join(next_node.split('_')[:-1])
        vip_class = path[location].replace('_pod','').replace('_vip','')  #'_'.join(path[location].split('_')[:-1])
        print path, location_of_vips
        print "test", prev_node_class, vip_class, next_node_class, prev_node_class in vip_class,next_node_class in vip_class,location
        if prev_node_class == vip_class or next_node_class == vip_class:
            if 'db' in prev_node_class or 'db' in next_node_class:
                valid =  valid and 'db' in vip_class
            else:
                valid = valid and True
        else:
            valid = valid and False
    return valid

if __name__ == "__main__":
    generate_synthetic_attack_templates([], [])