## the purpose of this file is to take the collected network traces and use them to generate a cilium
## security policy (at the L3 level). it'll also be to check this security policy against the logical
## exfiltration paths that other components of MIMIR generates
from process_pcap import process_pcap_via_tshark, convert_tshark_stats_to_edgefile
import os,errno
import subprocess
from pcap_to_edgelists import update_mapping
import glob
import time
import ast
import prepare_graph
from prepare_graph import aggregate_outside_nodes, map_nodes_to_svcs, find_infra_components_in_graph, remove_infra_from_graph
import math
import networkx as nx

# generate_pcap_slice: int file_location -> file_location
# Takes the location of the full pcap file and creates a slice consisting of the first time_length seconds
# (note: time_length should be in seconds (and be a float))
def generate_pcap_slice(time_length, pcap_location, split_pcap_dir, make_edgefile_p, length_of_each_split=10):
    #split_pcap_loc = split_pcap_dir + '/first_' + str(time_length) + "_sec.pap"
    split_pcap_loc = split_pcap_dir + '/split_cap_'
    new_files = []

    number_splits_to_create = int(math.ceil(time_length / length_of_each_split))
    if make_edgefile_p: ## TODO: put back in!!!!
        # okay, this is stupid, but what I am going to need to do is monitor the creation of the files in the system
        # and then kill the editcap process once one of the files comes into existance...

        # need to delete existing files to get some stuff later on to work...
        files = glob.glob(split_pcap_dir + '/*')
        for f in files:
            os.remove(f)

        inital_files_in_split_dir = set([name for name in os.listdir(split_pcap_dir) if os.path.isfile(os.path.join(split_pcap_dir, name))])
        print "inital_files_in_split_dir", inital_files_in_split_dir
        print "splitting with editcap now..."
        #split_cmds = ["tshark", "-r", pcap_location, "-Y", "frame.time_relative <= " + str(time_length), "-w", split_pcap_loc]
        split_cmds = ["editcap", "-i " + str(length_of_each_split), pcap_location, split_pcap_loc]

        proc = subprocess.Popen(split_cmds)

        # now monitor directory for existance the split file and then...
        new_split_pcap_exists = False
        while not new_split_pcap_exists:
            time.sleep(20)
            files_in_split_dir = set([name for name in os.listdir(split_pcap_dir) if os.path.isfile(os.path.join(split_pcap_dir, name))])
            print "files_in_split_dir...", files_in_split_dir
            new_files = list(files_in_split_dir.difference(inital_files_in_split_dir))
            print "new_files", new_files
            if len(new_files) >= (number_splits_to_create + 1): # need + 1 b/c first file is created before being completely written too... leads to problems if stopped before writing is done...
                new_split_pcap_exists = True

        proc.terminate()
        proc.kill()

    new_files.sort()
    print "final_new_files", new_files
    split_pcap_locs = new_files[0:number_splits_to_create]
    print "generate_relevant_pcap_slice_out", split_pcap_locs
    return [split_pcap_dir + '/' + split_pcap_loc for split_pcap_loc in split_pcap_locs]

def host2_host_comm(edgefile):
    communicating_ips = set()
    ips = set()
    with open(edgefile, 'r') as f:
        cont = f.read()
    print "cont", cont
    cont = cont.split('\n')
    print "cont", cont
    for line in cont:
        if len(line) > 0:
            print "line", line
            words = line.split(' ')
            print "words",words
            src_ip = words[0]
            dst_ip = words[1]
            edge_attribDict = words[2]
            edge_attribDict = ast.literal_eval(edge_attribDict)
            if edge_attribDict['frames'] > 0:
                communicating_ips.add((src_ip, dst_ip))
                ips.add(src_ip)
                ips.add(dst_ip)

    return communicating_ips, ips

def cal_host2svc(hosts, svcs):
    host2svc = {}
    for host in hosts:
        for svc in svcs:
            if svc in host:
                host2svc[host] = svc
                break
    return host2svc

def calc_svc2svc_communcating(host2svc, communicating_hosts, vip_debugging, gateway_ip):
    communicatng_svcs = set()
    for comm_host_pair in communicating_hosts:
        if comm_host_pair[0] in host2svc and not ('VIP' in comm_host_pair[0] and vip_debugging):
            src_svc = host2svc[ comm_host_pair[0] ]
        else:
            src_svc = comm_host_pair[0]


        if comm_host_pair[1] in host2svc and not ('VIP' in comm_host_pair[1] and vip_debugging):
            dst_svc = host2svc[ comm_host_pair[1] ]
        else:
            dst_svc = comm_host_pair[1]

        communicatng_svcs.add( (src_svc, dst_svc) )

    # now filter communicatng_svcs so that only real services remain...
    svc_pairs_to_remove = []
    svc_pairs_to_append = []
    for comm_svc in communicatng_svcs:
        src_svc = comm_svc[0]
        dst_svc = comm_svc[1]
        src_svc_is_outside = False
        dst_svc_is_outside = False

        ''' ## this should no longer be needed...
        if prepare_graph.is_ip(src_svc):
            addr_bytes = prepare_graph.is_ip(src_svc)
            if not prepare_graph.is_private_ip(addr_bytes, gateway_ip):
                #src_svc = 'outside'
                src_svc_is_outside = True
        if prepare_graph.is_ip(dst_svc):
            addr_bytes = prepare_graph.is_ip(dst_svc)
            if not prepare_graph.is_private_ip(addr_bytes, gateway_ip):
                #dst_svc = 'outside'
                dst_svc_is_outside = True
        '''

        if not vip_debugging:
            if 'POD' in src_svc or 'VIP' in src_svc or prepare_graph.is_ip(src_svc):
                svc_pairs_to_remove.append((src_svc,dst_svc))
            elif 'POD' in dst_svc or 'VIP' in dst_svc or prepare_graph.is_ip(dst_svc):
                svc_pairs_to_remove.append((src_svc,dst_svc))

        if (not ('POD' in dst_svc or 'VIP' in dst_svc)) and ( src_svc_is_outside or dst_svc_is_outside ):
            if src_svc_is_outside and not prepare_graph.is_ip(dst_svc):
                svc_pairs_to_append.append( ('outside', dst_svc) )
            elif dst_svc_is_outside and not prepare_graph.is_ip(src_svc):
                svc_pairs_to_append.append( (src_svc, 'outside') )

    for svc_pair in svc_pairs_to_remove:
        communicatng_svcs.remove(svc_pair)

    for svc_pair in svc_pairs_to_append:
        communicatng_svcs.add(svc_pair)

    return communicatng_svcs

# might want to do this at some point... generate the relevant cilium config file
# play around with ATM, but I'll try to fill the whole thing out later...
def generate_cilium_policy(communicating_svc, basefilename):
    for comm_svc_pair in communicating_svc:
        pass

    pass

# this function coordinates the overall functionality of the cilium component of MIMIR
# for more information, please see comment at top of page
### TODO: pass in the gateway IP from the calling function...
def cilium_component(time_length, pcap_location, cilium_component_dir, make_edgefiles_p, svcs, mapping,
                     pod_creation_log, gateway_ip = '172.17.0.1'):
    make_edgefiles_p = True ## TODO PROBABLY WANT TO REMOVE AT SOME POINT
    vip_debugging = False # this function exists for debugging purposes. It makese the cur_cilium_comms
                         # also print the relevant VIPS and then quit right after. This is useful for
                         # setting up the netsec policy.

    # step (0) make sure the directory where we are going to store all the MIMIR cilium component files exist
    try:
        os.makedirs(cilium_component_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    print "okay, there should now be the cilium_component_dir: ", cilium_component_dir

    # step (1) generate relevant slice of pcap
    print "calling generate_pcap_slice now..."
    pcap_slice_locations = generate_pcap_slice(time_length, pcap_location, cilium_component_dir, make_edgefiles_p)


    infra_instances = {}
    communicatng_svcs = []
    # step (2) generate corresponding tshark stats file
    for pcap_slice_location in pcap_slice_locations:
        pcap_slice_location_components = pcap_slice_location.split('/')
        pcap_slice_path = '/'.join(pcap_slice_location_components[:-1]) + '/'
        pcap_slice_name = pcap_slice_location_components[-1]
        print "getting convo stats via tshark now..."
        time.sleep(5)

        tshark_stats_path, tshark_stats_file = process_pcap_via_tshark(pcap_slice_path, pcap_slice_name,
                                                                       cilium_component_dir , make_edgefiles_p)

        # step (3) generate edgefile (hostnames rather than ip addresses)
        edgefile_name = cilium_component_dir + '/edgefile_first_' + str(time_length) + '_sec.txt'
        mapping,infra_instances = update_mapping(mapping, pod_creation_log, time_length, 0, infra_instances)

        edgefile =  convert_tshark_stats_to_edgefile('', edgefile_name, tshark_stats_path, tshark_stats_file,
                                         make_edgefiles_p, mapping)

        print "edgefile", edgefile
        print 'remainder of cilium component is... TODO'
        # step (4) generate service-to-ip mapping
        #communicating_hosts, hosts = host2_host_comm(edgefile)

        # i think we can repurpose the existing code that I wrote for the other component...
        f = open(edgefile, 'r')
        lines = f.readlines()
        G = nx.DiGraph()
        nx.parse_edgelist(lines, delimiter=' ', create_using=G)
        G = aggregate_outside_nodes(G)
        containers_to_ms = map_nodes_to_svcs(G, None, mapping)
        containers_to_ms['outside'] = 'outside'
        #_, service_G = aggregate_graph(G, containers_to_ms)
        nx.set_node_attributes(G, containers_to_ms, 'svc')
        infra_nodes = find_infra_components_in_graph(G, infra_instances)
        G = remove_infra_from_graph(G, infra_nodes)

        name_to_svc = {}
        for node,data in G.nodes(data=True):
            try:
                name_to_svc[node] = data['svc']
            except:
                print node, data

        communicating_hosts = []
        for (u,v,d) in G.edges(data=True):
            communicating_hosts.append((u,v))

        communicatng_svcs.extend(calc_svc2svc_communcating(name_to_svc, communicating_hosts, vip_debugging, gateway_ip))

    communicatng_svcs = list(set(communicatng_svcs))
    output_file_name = './cilium_comp_inouts/cur_cilium_comms'
    netsecoutput_file_name = './cilium_comp_inouts/cur_cilium_netsec_'
    vips_present_lines = ''
    if vip_debugging:
        additional_output_file_name = output_file_name +  '_vip_debugging'
        vips_present = set()
        for comm_pair in communicatng_svcs:
            #if 'VIP' in comm_pair[0] or 'VIP' in comm_pair[1]:

            # w/ current tshark setup, 'VIP' being in either 0 or 1 is semantically equivalent.
            if 'VIP' in comm_pair[1]:
                vips_present.add(comm_pair)
            #if 'VIP' in comm_pair[1]:
            #    vips_present.add(comm_pair[1])
        with open(additional_output_file_name + '.txt', 'w') as f:
            for item in vips_present:
                src = item[0].lower().replace('-', '_')
                dst = item[1].lower().replace('-', '_')
                src += '_pod'
                f.write(src + " " + dst  + '\n')
                vips_present_lines +=  src + " " + dst  + '\n'

    try:
        os.makedirs(output_file_name + '.txt')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    with open(output_file_name + '.txt', 'w') as f:
        for comm_svc in communicatng_svcs:
            f.write(str(comm_svc) + '\n')

    if vip_debugging:
        print "existing b/c vip_debugging is true. can check output file safely now."
        exit(233)

    basefilename = None
    generate_cilium_policy(communicatng_svcs, basefilename)
    generate_mimir_netsec_policy(netsecoutput_file_name, communicatng_svcs, vips_present_lines)

    return communicatng_svcs

def generate_mimir_netsec_policy(netsecoutput_file_name, communicating_services, vips_present_lines):
    with open(netsecoutput_file_name, 'w') as f:
        for comm_svc in communicating_services:
            svc_one = comm_svc[0]
            svc_two = comm_svc[1]
            comm_str = svc_one + ' ALLOWED ' + str(svc_two)
            f.write(comm_str + '\n')
        f.write("---------------------\n")
        f.write(vips_present_lines)
        f.write("ALL kube_dns_vip")


