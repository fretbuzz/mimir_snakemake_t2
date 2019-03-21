## the purpose of this file is to take the collected network traces and use them to generate a cilium
## security policy (at the L3 level). it'll also be to check this security policy against the logical
## exfiltration paths that other components of MIMIR generates
from process_pcap import process_pcap_via_tshark, convert_tshark_stats_to_edgefile
import os,errno
import subprocess
from simplified_graph_metrics import update_mapping
import glob
import time


# generate_pcap_slice: int file_location -> file_location
# Takes the location of the full pcap file and creates a slice consisting of the first time_length seconds
# (note: time_length should be in seconds (and be a float))
def generate_pcap_slice(time_length, pcap_location, split_pcap_dir, make_edgefile_p):
    #split_pcap_loc = split_pcap_dir + '/first_' + str(time_length) + "_sec.pap"
    split_pcap_loc = split_pcap_dir + '/split_cap_'
    new_files = []

    if True: #if make_edgefile_p: ## TODO: put back in!!!!
        # okay, this is stupid, but what I am going to need to do is monitor the creation of the files in the system
        # and then kill the editcap process once one of the files comes into existance...

        # need to delete existing files to get some stuff later on to work...
        files = glob.glob(split_pcap_dir + '/*')
        for f in files:
            os.remove(f)

        print "splitting with editcap now..."
        #split_cmds = ["tshark", "-r", pcap_location, "-Y", "frame.time_relative <= " + str(time_length), "-w", split_pcap_loc]
        split_cmds = ["editcap", "-i " + str(time_length), pcap_location, split_pcap_loc]

        proc = subprocess.Popen(split_cmds)

        # TODO: monitor directory for existance the split file and then...
        new_split_pcap_exists = False
        inital_files_in_split_dir = set([name for name in os.listdir(split_pcap_dir) if os.path.isfile(os.path.join(split_pcap_dir, name))])
        while not new_split_pcap_exists:
            time.sleep(10)
            files_in_split_dir = set([name for name in os.listdir(split_pcap_dir) if os.path.isfile(os.path.join(split_pcap_dir, name))])
            new_files = list(inital_files_in_split_dir.difference(files_in_split_dir))
            if len(new_files) >= 1:
                new_split_pcap_exists = True

        proc.terminate()
        proc.kill()

        print "generate_pcap_slice_out", out

    split_pcap_loc = new_files[1]
    return split_pcap_loc

# TODO: probably wanna incorporate who is initiating the flows at some point..
def find_communicating_ips():
    # okay, there's actually two different possible situations here...
    # if it is over TCP than we can (theoretically) see who initiated the flows
    # else, we can't know
    pass

# gen_service_to_ip_mapping file_path list_of_svcs :
#
# following logic from map_nodes_to_svcs(G, svcs) in process_graph.py
def gen_service_to_ip_mapping(edgefile, svc):
    # step (1
    pass
    '''
    for u in G.nodes():
        for svc in svcs:
            if svc in u:
                containers_to_ms[u] = svc
                break
    '''

# this function coordinates the overall functionality of the cilium component of MIMIR
# for more information, please see comment at top of page
def cilium_component(time_length, pcap_location, cilium_component_dir, make_edgefiles_p, svcs, inital_mapping,
                     pod_creation_log, make_edgefile_p):
    # step (0) make sure the directory where we are going to store all the MIMIR cilium component files exist
    try:
        os.makedirs(cilium_component_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    print "okay, there should now be the cilium_component_dir: ", cilium_component_dir

    # step (1) generate relevant slice of pcap
    print "calling generate_pcap_slice now..."
    pcap_slice_location = generate_pcap_slice(time_length, pcap_location, cilium_component_dir, make_edgefile_p)

    # step (2) generate corresponding tshark stats file
    pcap_slice_location_components = pcap_slice_location.split('/')
    pcap_slice_path = '/'.join(pcap_slice_location_components[:-1])
    pcap_slice_name = pcap_slice_location_components[-1]
    print "getting convo stats via tshark now..."
    tshark_stats_path, tshark_stats_file = process_pcap_via_tshark(pcap_slice_path, pcap_slice_name, cilium_component_dir, make_edgefiles_p)

    # step (3) generate edgefile (hostnames rather than ip addresses)
    edgefile_name = cilium_component_dir + '/edgefile_first_' + str(time_length) + '_sec.txt'
    mapping = update_mapping(inital_mapping, pod_creation_log, time_le, 0)
    edgefile =  convert_tshark_stats_to_edgefile(cilium_component_dir, edgefile_name, tshark_stats_path, tshark_stats_file,
                                     make_edgefiles_p, mapping)

    # step (4) generate service-to-ip mapping
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

