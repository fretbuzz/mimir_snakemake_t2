## the purpose of this file is to take the collected network traces and use them to generate a cilium
## security policy (at the L3 level). it'll also be to check this security policy against the logical
## exfiltration paths that other components of MIMIR generates
from process_pcap import process_pcap_via_tshark, convert_tshark_stats_to_edgefile
import os,errno
import subprocess
from simplified_graph_metrics import update_mapping
import glob
import time
import ast
import networkx as nx
import prepare_graph

# generate_pcap_slice: int file_location -> file_location
# Takes the location of the full pcap file and creates a slice consisting of the first time_length seconds
# (note: time_length should be in seconds (and be a float))
def generate_pcap_slice(time_length, pcap_location, split_pcap_dir, make_edgefile_p):
    #split_pcap_loc = split_pcap_dir + '/first_' + str(time_length) + "_sec.pap"
    split_pcap_loc = split_pcap_dir + '/split_cap_'
    new_files = []

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
        split_cmds = ["editcap", "-i " + str(time_length), pcap_location, split_pcap_loc]

        proc = subprocess.Popen(split_cmds)

        # now monitor directory for existance the split file and then...
        new_split_pcap_exists = False
        while not new_split_pcap_exists:
            time.sleep(5)
            files_in_split_dir = set([name for name in os.listdir(split_pcap_dir) if os.path.isfile(os.path.join(split_pcap_dir, name))])
            print "files_in_split_dir...", files_in_split_dir
            new_files = list(files_in_split_dir.difference(inital_files_in_split_dir))
            print "new_files", new_files
            if len(new_files) >= 1:
                new_split_pcap_exists = True

        proc.terminate()
        proc.kill()

    new_files.sort()
    print "final_new_files", new_files
    split_pcap_loc = new_files[0]
    print "generate_relevant_pcap_slice_out", split_pcap_loc
    return split_pcap_dir + '/' + split_pcap_loc

# TODO: probably wanna incorporate who is initiating the flows at some point..
def find_communicating_ips():
    # okay, there's actually two different possible situations here...
    # if it is over TCP than we can (theoretically) see who initiated the flows
    # else, we can't know
    pass

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

def calc_svc2svc_communcating(host2svc, communicating_hosts, vip_debugging):
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

        if prepare_graph.is_ip(src_svc):
            addr_bytes = prepare_graph.is_ip(src_svc)
            if not prepare_graph.is_private_ip(addr_bytes):
                #src_svc = 'outside'
                src_svc_is_outside = True
        if prepare_graph.is_ip(dst_svc):
            addr_bytes = prepare_graph.is_ip(dst_svc)
            if not prepare_graph.is_private_ip(addr_bytes):
                #dst_svc = 'outside'
                dst_svc_is_outside = True

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

# TODO:: this is NOT DONE ATM... it's mostly just some starter code to
# play around with ATM, but I'll try to fill the whole thing out later...
def generate_cilium_policy(communicating_svc, basefilename):
    for comm_svc_pair in communicating_svc:
        pass

    pass

# this function coordinates the overall functionality of the cilium component of MIMIR
# for more information, please see comment at top of page
def cilium_component(time_length, pcap_location, cilium_component_dir, make_edgefiles_p, svcs, inital_mapping,
                     pod_creation_log):
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
    pcap_slice_location = generate_pcap_slice(time_length, pcap_location, cilium_component_dir, make_edgefiles_p)

    # step (2) generate corresponding tshark stats file
    pcap_slice_location_components = pcap_slice_location.split('/')
    pcap_slice_path = '/'.join(pcap_slice_location_components[:-1]) + '/'
    pcap_slice_name = pcap_slice_location_components[-1]
    print "getting convo stats via tshark now..."
    tshark_stats_path, tshark_stats_file = process_pcap_via_tshark(pcap_slice_path, pcap_slice_name,
                                                                   cilium_component_dir , make_edgefiles_p)

    # step (3) generate edgefile (hostnames rather than ip addresses)
    edgefile_name = cilium_component_dir + '/edgefile_first_' + str(time_length) + '_sec.txt'
    mapping = update_mapping(inital_mapping, pod_creation_log, time_length, 0)
    edgefile =  convert_tshark_stats_to_edgefile('', edgefile_name, tshark_stats_path, tshark_stats_file,
                                     make_edgefiles_p, mapping)

    print "edgefile", edgefile
    print 'remainder of cilium component is... TODO'
    # step (4) generate service-to-ip mapping
    communicating_hosts, hosts = host2_host_comm(edgefile)
    ip_to_svc = cal_host2svc(hosts, svcs)
    communicatng_svcs = calc_svc2svc_communcating(ip_to_svc, communicating_hosts, vip_debugging)

    output_file_name = './cilium_comp_inouts/cur_cilium_comms'
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

    with open(output_file_name + '.txt', 'w') as f:
        for comm_svc in communicatng_svcs:
            f.write(str(comm_svc) + '\n')

    if vip_debugging:
        print "existing b/c vip_debugging is true. can check output file safely now."
        exit(233)

    basefilename = None # TODO TODO TODO
    generate_cilium_policy(communicatng_svcs, basefilename)

    return communicatng_svcs

    ## TODO: is there a function that already does this for me???
    ## update: kinda... map_nodes_to_svcs(G, svcs) in process_graph has the logic, but doesn't directly
    ## apply b/c we are not doing stuff with graphs here...
    # okay, so now what we'd want to do is look at the generated edgefile and the communicating entities....
    # (a) this is pretty simple for a parser...
    # just look @ line and split by spaces. Then if # of bytes != 0, they communicate and you're done!

    # (b) so then we'll have sets of communicating hostnames. Then just map to communicating svc using the logical
    # already above

    # step (5) find which services communicate

    # step (6) [[might not necessarily take place here]],, but make sure that directionality is taken into account...
    ## (i think initiator info might be in the specs generated from the attack template component??)
    ## NOTE: NOT DOING THIS ATM...

# this function takes a series of logical attack paths (i.e. at time 1, attack path 1, at time 10, attack path 5,
# etc.) and returns a series (of equivalent length) on whether the attack would/would_not be allowed.
# TODO: how should I test/evaluate if it interferes w/ normal application function.... correct. Ideally...
# it'd actually make more sense to pass in the list of the injected edgefiles and determine using that...
## well... it's something to think about, certainly...
## well... we could maybe test this later idea?? I mean, as long as it is easy
## to pass the injected edgefiles at service granularity, then we can just iterate
## over the edges (wait, I take it back, this might be hard)
## edgesfiles are injeceted
def calc_cilium_component_performnace(class_edgefiles, allowed_intersvc_comm):
    pass