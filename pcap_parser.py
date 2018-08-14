import time
from scapy.all import *
import re
import pickle
import pandas as pd
import csv
import math
import yaml
import analyze_edgefiles
import gc
import ast

# parse_pcap : packet_array seconds_per_time_interval ip_to_container_and_network, basename_of_output pcap_start_time
#              shouldnt_delete_old_edgefiles_p  -> unidentified_IPs list_of_filenames endtime (+ filenames filled w/ edgelists)
# this file creates edgefiles passed on the packet array. each edgefile lasts for a certain length of time.
def parse_pcap(a, time_intervals, mapping, basefile_name, start_time, dont_delete_old_edgefiles):
    time_to_graphs = {}

    current_time_interval = 0
    time_to_graphs[current_time_interval] = {}

    for a_pkt in a:
        # I don't think the belwo code is needed b/c it'll break anyway once all of the packets are processed
        #if a_pkt.time > end_time:
        #    break

        if a_pkt.time - (start_time + current_time_interval * time_intervals) > time_intervals:
            current_time_interval += 1
            time_to_graphs[current_time_interval] = {}

        #a_pkt.show()
        #print len(a_pkt)
        #print "#####"

        src_dst = ()
        src_dst_ports = ()
        #print "TIME", a_pkt.time # this is the unix time when packet was recieved
        # so it is in seconds
        if 'IP' in a_pkt:
            src_dst = (a_pkt['IP'].src, a_pkt['IP'].dst)
        elif 'ARP' in a_pkt:
            #print "there is an ARP packet!"
            pass
        elif 'IPv6' in a_pkt:
            pass
        else:
            print "so this is not an IP/ARP packet..."
            print a_pkt.show()
            exit(105)
        if 'TCP' in a_pkt:
            src_dst_ports = (a_pkt['TCP'].sport, a_pkt['TCP'].dport)
        if src_dst == ():
            continue

        # NAT-ing is clearly happening, can turn on the line below and observe it if you want...
        #src_dst = (src_dst[0]+':'+str(src_dst_ports[0]), src_dst[1] +':'+ str(src_dst_ports[1]))

        if src_dst in time_to_graphs[current_time_interval]:
            if 'IP' in a_pkt:
                time_to_graphs[current_time_interval][src_dst] += a_pkt['IP'].len
        else:
            if 'IP' in a_pkt:
                time_to_graphs[current_time_interval][src_dst] = a_pkt['IP'].len
        #str_payload = ''.join(["".join(n) if n != '\x08' else '' for n in pkt.load])
        #print str_payload
    time_to_parsed_mapping = {}
    for time in time_to_graphs:
        time_to_parsed_mapping[time] = {}

    no_mapping_found = []
    for time,graph_dictionary in time_to_graphs.iteritems():
        for item, weight in graph_dictionary.iteritems():
            print item, weight
            src = item[0]
            dst = item[1]
            try:
                src_ms = mapping[src][0]
            except:
                #print "not_mapped_src", src
                src_ms = src
                no_mapping_found.append(src)
            try:
                dst_ms = mapping[dst][0]
            except:
                #print "not_mapped_dst", dst
                dst_ms = dst
                no_mapping_found.append(dst)
            print "index stuff", time, src_ms, dst_ms
            time_to_parsed_mapping[time][src_ms, dst_ms] = weight
        for item, weight in time_to_parsed_mapping[time].iteritems():
            print item, weight

    time_counter = 0
    filesnames = []
    for interval in range(0,current_time_interval):
        ending = '_' + '%.2f' % (time_counter) + '_' + '%.2f' % (time_intervals)
        filename = basefile_name + ending + '.txt'
        # first time through, want to delete the old edgefiles
        if not dont_delete_old_edgefiles:
            try:
                os.remove(filename)
            except:
                print filename, "   ", "does not exist"
        with open(filename, 'ab') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for item, weight in time_to_parsed_mapping[interval].iteritems():
                spamwriter.writerow([item[0], item[1], weight])
        time_counter += time_intervals
        filesnames.append(filename)

    print "unidentified IPs present", list(set(no_mapping_found))
    return list(set(no_mapping_found)), filesnames, time_counter

# swarm_container_ips : file_path -> dictionary mapping IPs to (container_name, network_name)
# parses file containing all docker network -v output to find out the container that each IP refers to
# (note: virtual IPs of services as well as the IPs of network gateways are now included as well)
def ips_on_docker_networks(path):
    file = open(path, "r")
    swarm_configs = file.read()
    swarm_config_groups = swarm_configs.split("\n]\n")
    #print swarm_config_groups
    g = 0
    container_to_ip = {}
    for config in swarm_config_groups[:-1]: #end is just delimiter so let's not process that
        config += "\n]\n" # split removes delimiter, so let's add it back in
        print g
        g += 1

        current_config = yaml.safe_load(config)
        network_name = current_config[0]["Name"]
        for container_id, container in current_config[0]["Containers"].iteritems():
            container_name = container["Name"]
            split_container_name = container_name.split('.')
            if (len(split_container_name) == 3):
                container_name = split_container_name[0] + '.' + split_container_name[1]

            container_ip = container["IPv4Address"].split('/',1)[0]
            container_to_ip[container_ip] = (container_name, network_name)

        try:
            network_gateway = current_config[0]["IPAM"]["Config"]["Gateway"]
            print network_name, " has the following gateway: ", network_gateway
            container_to_ip[network_gateway] = (network_name + "_gateway", network_name)
        except:
            print network_name, "has no gateway"

        try:
            for service_name, service_vals in current_config[0]["Services"].iteritems():
                print service_name, service_vals["VIP"]
                service_vip = service_vals["VIP"]
                container_to_ip[service_vip] = (service_name + "_VIP", network_name)
        except:
            print "no sevices in this network..."

    return container_to_ip

# parse_kubernetes_svc_info : path_to_kubernetes_svc_info -> mapping_of_ip_to_service
# returns the virtual IPs for the kubernetes services
def parse_kubernetes_svc_info(kubernetes_svc_info_path):
    mapping = {}
    with open(kubernetes_svc_info_path, 'r') as svc_f:
        line = svc_f.readlines()
        for l in line[1:]:
            l_pieces = l.split()
            print l_pieces[1], l_pieces[3]
            mapping[l_pieces[1]] = l_pieces[3]
    return mapping

# run_data_anaylsis_pipeline : runs the whole analysis pipeline (or a part of it)
# (1) creates edgefiles, (2) creates communication graphs from edgefiles, (3) calculates (and stores) graph metrics
# (4) makes graphs of the graph metrics
# Note: see run_analysis_pipeline_recipes for pre-configured sets of parameters (there are rather a lot)
def run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles_p, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time = None, end_time = None, calc_vals=True,
                               graph_p = True, kubernetes_svc_info=None):

    # First, get a mapping of IPs to (container_name, network_name)
    mapping = ips_on_docker_networks(container_info_path)
    if not is_swarm:
        # if it is kubernetes, then it is also necessary to read in that file with all the
        # info about the svc's, b/c kubernetes service VIPs don't show up in the docker configs
        kubernetes_service_VIPs = parse_kubernetes_svc_info(kubernetes_svc_info)
        mapping.update(kubernetes_service_VIPs)
    try:
        del mapping['<nil>'] # sometimes this nonsense value shows up b/c of the host Docker network (no IP = just noise)
    except:
        pass
    print "container to ip mapping", mapping

    if start_time==None or end_time==None or make_edgefiles_p:
        current_pcap = rdpcap(pcap_paths[0])
        fst_pkt = current_pcap[0]
        start_time = fst_pkt.time

    print "start_time: ", start_time,
    interval_to_filenames = {}

    if make_edgefiles_p:
        pcaps_processed_at_time_interval = 0
        for current_pcap_path in pcap_paths:
            if current_pcap_path != pcap_paths[0]: #already loaded the first one in order to perform time interval calcs
                current_pcap = rdpcap(current_pcap_path) # current_pcap = PcapReader(current_pcap_path)
                gc.collect()  # it seems like this helps keep RAM usage reasonable might keep RAM reasonable?
            for time_interval_length in time_interval_lengths:
                unmapped_ips, filenames, end_time = parse_pcap(current_pcap, time_interval_length, mapping,
                                                            basefile_name, start_time, pcaps_processed_at_time_interval)
                print "unmapped ips", unmapped_ips
                if current_pcap_path == pcap_paths[0]:
                    interval_to_filenames[time_interval_length] = filenames
            pcaps_processed_at_time_interval += 1
            # okay, it looks like at this point interval_to_filenames will be fully loaded. I'm going to want to
            # write this to a file, so I can refer to it later.
            with open(basefile_name + 'edgefile_dict.txt', "w+") as f:
                f.write(interval_to_filenames)
    else:
        # I guess I am going to need to store the filenames in another file or something? b/c if I
        with open(basefile_name + 'edgefile_dict.txt', "r") as f:
            a = f.read()
            interval_to_filenames = ast.literal_eval(a)

    total_calculated_vals = {}
    for time_interval_length in time_interval_lengths:
        print "analyzing edgefiles..."
        newly_calculated_values = analyze_edgefiles.pipeline_analysis_step(interval_to_filenames[time_interval_length], ms_s,
                                                                           time_interval_length, basegraph_name, calc_vals, window_size,
                                                                           mapping, is_swarm)

        total_calculated_vals.update(newly_calculated_values)
    if graph_p:
        # (time gran) -> (node gran) -> metrics -> vals
        analyze_edgefiles.create_graphs(total_calculated_vals, basegraph_name, window_size, colors, time_interval_lengths,
                                        exfil_start_time, exfil_end_time, wiggle_room)