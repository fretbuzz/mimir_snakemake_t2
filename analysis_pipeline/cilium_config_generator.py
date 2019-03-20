## the purpose of this file is to take the collected network traces and use them to generate a cilium
## security policy (at the L3 level). it'll also be to check this security policy against the logical
## exfiltration paths that other components of MIMIR generates
from process_pcap import process_pcap_via_tshark, convert_tshark_stats_to_edgefile
import os,errno

# generate_pcap_slice: int file_location -> file_location
# Takes the location of the full pcap file and creates a slice consisting of the first time_length seconds
# hint: refer to split_pcap in process_pcap
def generate_pcap_slice(time_length, pcap_location, split_pcap_loc):
    ## TODO
    pass

# gen_service_to_ip_mapping file_path list_of_svcs :
#
# following logic from map_nodes_to_svcs(G, svcs) in process_graph.py
def gen_service_to_ip_mapping(edgefile, svc):
    # step (1)

    for u in G.nodes():
        for svc in svcs:
            if svc in u:
                containers_to_ms[u] = svc
                break

# this function coordinates the overall functionality of the cilium component of MIMIR
# for more information, please see comment at top of page
def cilium_component(time_length, pcap_location, cilium_component_dir, make_edgefiles_p, svcs):
    # step (0) make sure the directory where we are going to store all the MIMIR cilium component files exist
    try:
        os.makedirs(cilium_component_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # step (1) generate relevant slice of pcap
    pcap_slice_location = generate_pcap_slice(time_length, pcap_location, cilium_component_dir)

    # step (2) generate corresponding tshark stats file
    pcap_slice_location_components = pcap_slice_location.split('/')
    pcap_slice_path = '/'.join(pcap_slice_location_components[:-1])
    pcap_slice_name = pcap_slice_location_components[-1]
    tshark_stats_path, tshark_stats_file = process_pcap_via_tshark(pcap_slice_path, pcap_slice_name, cilium_component_dir, make_edgefiles_p)

    # step (3) generate edgefile (hostnames rather than ip addresses)
    edgefile_name = None ## TODO
    mapping = None ## TODO (note: this is actually quite a bit harder that it appears, probably...)
                    ## (or maybe not if I'm clever w/ where I hook it into the rest of the system...)
    convert_tshark_stats_to_edgefile(cilium_component_dir, edgefile_name, tshark_stats_path, tshark_stats_file,
                                     make_edgefiles_p, mapping)

    # step (4) generate service-to-ip mapping
    ## TODO: is there a function that already does this for me???
    ## update: kinda... map_nodes_to_svcs(G, svcs) in process_graph has the logic, but doesn't directly
    ## apply b/c we are not doing stuff with graphs here...


    # step (5) find which services communicate

    # step (6) [[might not necessarily take place here]],, but make sure that directionality is taken into account...
    ## (i think initiator info might be in the specs generated from the attack template component??)

