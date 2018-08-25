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
import json

# parse_pcap : packet_array seconds_per_time_interval ip_to_container_and_network, basename_of_output pcap_start_time
#    shouldnt_delete_old_edgefiles_p  -> unidentified_IPs list_of_filenames endtime (+ filenames filled w/ edgelists)
# this file creates edgefiles passed on the packet array. each edgefile lasts for a certain length of time.
def parse_pcap(a, time_intervals, mapping, basefile_name, start_time, dont_delete_old_edgefiles):
    time_to_graphs = {}

    current_time_interval = 0
    time_to_graphs[current_time_interval] = {}
    time_to_packet_graphs[current_time_interval] = {}
    weird_timing_pkts = []
    unidentified_pkts = [] 

    for a_pkt in a:
        # I don't think the belwo code is needed b/c it'll break anyway once all of the packets are processed
        #if a_pkt.time > end_time:
        #    break

        #if a_pkt.time - (start_time + current_time_interval * time_intervals) > time_intervals:
        #    current_time_interval += 1
        #    time_to_graphs[current_time_interval] = {}
	if not a_pkt.haslayer(DNS):
		continue


        pkt_messed_up = False
        while(a_pkt.time - (start_time + current_time_interval * time_intervals) > time_intervals):
            if (a_pkt.time - (start_time + current_time_interval * time_intervals)) > 900:
                a_pkt.show()
                weird_timing_pkts.append(a_pkt)
                pkt_messed_up = True
                print "about to break", a_pkt.time, current_time_interval, start_time + current_time_interval * time_intervals
                break
            current_time_interval += 1
            time_to_graphs[current_time_interval] = {}
	    time_to_packet_graphs[current_time_interval] = {}
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
            #exit(105)
            unidentified_pkts.append(a_pkt)
        if 'TCP' in a_pkt:
            src_dst_ports = (a_pkt['TCP'].sport, a_pkt['TCP'].dport)
        if src_dst == ():
            continue

        # NAT-ing is clearly happening, can turn on the line below and observe it if you want...
        #src_dst = (src_dst[0]+':'+str(src_dst_ports[0]), src_dst[1] +':'+ str(src_dst_ports[1]))

        if src_dst in time_to_graphs[current_time_interval]:
            if 'IP' in a_pkt:
                time_to_graphs[current_time_interval][src_dst] += a_pkt['IP'].len
        	time_to_packet_graphs[current_time_interval][src_dst] += 1
	else:
            if 'IP' in a_pkt:
                time_to_graphs[current_time_interval][src_dst] = a_pkt['IP'].len
		time_to_packet_graphs[current_time_interval][src_dst] = 1
        #str_payload = ''.join(["".join(n) if n != '\x08' else '' for n in pkt.load])
        #print str_payload
    time_to_parsed_mapping = {}
    time_to_parsed_packet_mapping = {}
    for time in time_to_graphs:
        time_to_parsed_mapping[time] = {}
	time_to_parsed_packet_mapping[time] = {}

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
	    time_to_parsed_packet_mapping[time][src_ms, dst_ms] = time_to_packet_graphs[item]
        for item, weight in time_to_parsed_mapping[time].iteritems():
            print item, weight

    time_counter = 0
    filesnames = []
    filesnames_packets = []
    for interval in range(0,current_time_interval):
        ending = '_' + '%.2f' % (time_counter) + '_' + '%.2f' % (time_intervals)
        filename = basefile_name + ending + '.txt'
        filename_packets = basefile_name + '_packets' + ending + '.txt'
        # first time through, want to delete the old edgefiles
        if not dont_delete_old_edgefiles:
            try:
                os.remove(filename)
            except:
                print filename, "   ", "does not exist"
	    try:
		os.remove(filename_packets)
	    except:
		print filename_packets, "   ", "does not exist"
        with open(filename, 'ab') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for item, weight in time_to_parsed_mapping[interval].iteritems():
                spamwriter.writerow([item[0], item[1], weight])
        with open(filename_packets, 'ab') as csvfile:
	    spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
	    for item, weight in time_to_parsed_packet_mapping[interval].iteritems():
		spamwriter.writerow([item[0], item[1],weight])
	time_counter += time_intervals
        filesnames.append(filename)
	filesnames_packets.append(filename_packets)

    print "unidentified IPs present", list(set(no_mapping_found))
    return list(set(no_mapping_found)), filesnames, filesnames_packets,  time_counter, unidentified_pkts, weird_timing_pkts

# creates a file w/ some features that only exist in relation to the sensitive-DB
def parse_pcap_sensitive_db_only(a, time_intervals, mapping, basefile_name, start_time, dont_delete_old_edgefiles, exfil_start_time, exfil_end_time, wiggle_room):
    time_to_graphs = {}

    current_time_interval = 0
    current_time = 0
    #time_to_graphs[current_time_interval] = {}
    current_graph = {}
    unidentified_pkts = []
    weird_timing_pkts = []

    no_mapping_found = []
    ending = '_' + '%.6f' % (time_intervals)
    filename = basefile_name + ending + '.txt'
    filesnames = [filename]
    if not dont_delete_old_edgefiles:
        try:
            os.remove(filename)
        except:
            print filename, "   ", "does not exist"
    with open(filename, 'ab') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for a_pkt in a:
            # I don't think the belwo code is needed b/c it'll break anyway once all of the packets are processed
            #if a_pkt.time > end_time:
            #    break

            pkt_messed_up = False
            while(a_pkt.time - (start_time + current_time_interval * time_intervals) > time_intervals):
                if (a_pkt.time - (start_time + current_time_interval * time_intervals)) > 900:
                    a_pkt.show()
                    weird_timing_pkts.append(a_pkt)
                    pkt_messed_up = True
                    print "about to break", a_pkt.time, current_time_interval, start_time + current_time_interval * time_intervals
                    break
                print a_pkt.time, current_time_interval, start_time + current_time_interval * time_intervals
                # okay, so before clearing the current stuff, I should write the current values to file
                # NOTE: this stuff is specific to a specific experiment/deployment
                for item, weight in current_graph.iteritems():
                    attack_happening = 0
                    if exfil_start_time - wiggle_room < current_time and current_time < exfil_end_time + wiggle_room:
                        attack_happening = 1
                    if 'database' in item[0] and 'VIP' not in item[0] and 'endpoint' not in item[0]:
                        if 'appserver' not in item[1]:
                            attack_happening = 0
                        spamwriter.writerow([current_time, item[1], 'out', weight, attack_happening])
                    elif 'database' in item[1] and 'VIP' not in item[1] and 'endpoint' not in item[1]:
                        if 'appserver' not in item[0]:
                            attack_happening = 0
                        spamwriter.writerow([current_time, item[0], 'in', weight, attack_happening])
                current_time_interval += 1
                current_graph = {}
                current_time = current_time_interval * time_intervals
                #time_to_graphs[current_time_interval] = {}
            if pkt_messed_up:
                continue # move onto the next one
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
                unidentified_pkts.append(a_pkt)
                #exit(105)
            if 'TCP' in a_pkt:
                src_dst_ports = (a_pkt['TCP'].sport, a_pkt['TCP'].dport)
            if src_dst == ():
                continue

            try:
                src_ms = mapping[src_dst[0]][0]
            except:
                # print "not_mapped_src", src
                src_ms = src_dst[0]
                no_mapping_found.append(src_dst[0])
            try:
                dst_ms = mapping[src_dst[1]][0]
            except:
                # print "not_mapped_dst", dst
                dst_ms = src_dst[1]
                no_mapping_found.append(src_dst[1])
            print "index stuff", a_pkt.time, src_ms, dst_ms
            src_dst = (src_ms, dst_ms)

        # NAT-ing is clearly happening, can turn on the line below and observe it if you want...
        #src_dst = (src_dst[0]+':'+str(src_dst_ports[0]), src_dst[1] +':'+ str(src_dst_ports[1]))

            if src_dst in current_graph: #time_to_graphs[current_time_interval]:
                if 'IP' in a_pkt:
                    #time_to_graphs[current_time_interval][src_dst] += a_pkt['IP'].len
                    current_graph[src_dst] += a_pkt['IP'].len
            else:
                if 'IP' in a_pkt:
                    #time_to_graphs[current_time_interval][src_dst] = a_pkt['IP'].len
                    current_graph[src_dst] = a_pkt['IP'].len
        #str_payload = ''.join(["".join(n) if n != '\x08' else '' for n in pkt.load])
        #print str_payload
        #'''

    '''
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
    '''
    #time_counter = 0
    #filesnames = []
    '''
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
    '''
    '''
    ending = '_' + '%.2f' % (time_intervals)
    filename = basefile_name + ending + '.txt'
    with open(filename, 'ab') as csvfile:
        for interval in range(0,current_time_interval):
            # first time through, want to delete the old edgefiles
            if not dont_delete_old_edgefiles:
                try:
                    os.remove(filename)
                except:
                    print filename, "   ", "does not exist"
                spamwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for item, weight in time_to_parsed_mapping[interval].iteritems():
                attack_happening = 0
                if exfil_start_time - wiggle_room < time_counter and time_counter < exfil_end_time + wiggle_room:
                    attack_happening = 1
                if 'database' in item[0] and 'VIP' not in item[0] and 'endpoint' not in item[0]:
                    spamwriter.writerow([time_counter, item[0], 'out', weight, attack_happening])
                elif 'database' in item[1] and 'VIP' not in item[1] and 'endpoint' not in item[1]:
                    spamwriter.writerow([time_counter, item[1], 'in', weight, attack_happening])
            time_counter += time_intervals
            filesnames.append(filename)
    '''
    print "unidentified IPs present", list(set(no_mapping_found))
    return list(set(no_mapping_found)), filesnames, current_time, unidentified_pkts, weird_timing_pkts

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
            #mapping[l_pieces[1]] = l_pieces[3]
            mapping[l_pieces[3]] = (l_pieces[1] + '_VIP', 'svc')
    print "these service mappings were found", mapping
    return mapping

def parse_cilium(cilium_config_path):
    mapping = {}
    with open(cilium_config_path, 'r') as f:
        config = json.loads(f.read())
    for pod_config in config:
        pod_name = pod_config['status']['external-identifiers']['pod-name']
        ipv4_addr = pod_config['status']['networking']['addressing'][0]['ipv4']
        ipv6_addr = pod_config['status']['networking']['addressing'][0]['ipv6']
        mapping[ipv4_addr] = (pod_name, 'cilium')
        mapping[ipv6_addr] = (pod_name, 'cilium')
    return mapping

# run_data_anaylsis_pipeline : runs the whole analysis pipeline (or a part of it)
# (1) creates edgefiles, (2) creates communication graphs from edgefiles, (3) calculates (and stores) graph metrics
# (4) makes graphs of the graph metrics
# Note: see run_analysis_pipeline_recipes for pre-configured sets of parameters (there are rather a lot)
def run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               ms_s, make_edgefiles_p, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time = None, end_time = None, calc_vals=True,
                               graph_p = True, kubernetes_svc_info=None, make_net_graphs_p=False, cilium_config_path=None,
                               rdpcap_p=False):

    # First, get a mapping of IPs to (container_name, network_name)
    mapping = ips_on_docker_networks(container_info_path)
    if not is_swarm:
        # if it is kubernetes, then it is also necessary to read in that file with all the
        # info about the svc's, b/c kubernetes service VIPs don't show up in the docker configs
        #pass
        #'''
        kubernetes_service_VIPs = parse_kubernetes_svc_info(kubernetes_svc_info)
        mapping.update(kubernetes_service_VIPs)

        if cilium_config_path:
            cilium_mapping = parse_cilium(cilium_config_path)
            mapping.update(cilium_mapping)
    try:
        del mapping['<nil>'] # sometimes this nonsense value shows up b/c of the host Docker network (no IP = just noise)
    except:
        pass
    try:
        del mapping['']
    except:
        pass
    print "container to ip mapping", mapping

    if start_time==None: # or make_edgefiles_p:
        for pkt in PcapReader(pcap_paths[0]):
            start_time = pkt.time
            break

    print "start_time: ", start_time, "<- that is the start time"
    interval_to_filenames = {}
    interval_to_filenames_packets = {}

    total_unidentified_pkts = []
    total_weird_timing_pkts = []
    if make_edgefiles_p:
        pcaps_processed_at_time_interval = 0
        for current_pcap_path in pcap_paths:
            #if current_pcap_path != pcap_paths[0]: #already loaded the first one in order to perform time interval calcs
            #    current_pcap = rdpcap(current_pcap_path) # current_pcap = PcapReader(current_pcap_path)
            #    gc.collect()  # it seems like this helps keep RAM usage reasonable might keep RAM reasonable?
            unidentified_pkts = []
            weird_timing_pkts = []
            for time_interval_length in time_interval_lengths:
                if rdpcap_p:
                    current_pcap = rdpcap(current_pcap_path)
                else:
                    current_pcap = PcapReader(current_pcap_path)
                unmapped_ips, filenames, filenames_packets, end_time, unidentified_pkts, weird_timing_pkts = parse_pcap(current_pcap, time_interval_length, mapping,
                                                            basefile_name, start_time, pcaps_processed_at_time_interval)#,
                                                            #exfil_start_time, exfil_end_time, wiggle_room)
                print "unmapped ips", unmapped_ips
                if current_pcap_path == pcap_paths[0]:
                    interval_to_filenames[str(time_interval_length)] = filenames
		    interval_to_filenames_packets[str(time_interval_length)] = filenames_packets
            pcaps_processed_at_time_interval += 1
            total_unidentified_pkts.extend(unidentified_pkts)
            total_weird_timing_pkts.extend(weird_timing_pkts)
            # okay, it looks like at this point interval_to_filenames will be fully loaded. I'm going to want to
            # write this to a file, so I can refer to it later.
            with open(basefile_name + 'edgefile_dict.txt', "w+") as f:
                f.write(json.dumps(interval_to_filenames))
	    with open(basefile_name + 'packets_edgefile_dict.txt', "w+") as f:
		f.write(json.dumps(interval_to_filenames_packets))
        total_unidentified_pkts = list(set(total_unidentified_pkts))
        total_weird_timing_pkts = list(set(total_weird_timing_pkts))
        #with open(basefile_name + 'unidentified_pkts.txt', "w+") as f:
            #f.write(json.dumps(total_unidentified_pkts))
        wrpcap(basefile_name + 'unidentified_pkts.txt', total_unidentified_pkts)
        #with open(basefile_name + 'weird_timing_pkts.txt', "w+") as f:
        #    f.write(json.dumps(total_weird_timing_pkts))
        wrpcap(basefile_name + 'weird_timing_pkts.txt', total_weird_timing_pkts)
    else:
        # I guess I am going to need to store the filenames in another file or something? b/c if I
        with open(basefile_name + 'edgefile_dict.txt', "r") as f:
            a = f.read()
            interval_to_filenames = json.loads(a)
            #print "interval_to_filenames", interval_to_filenames

    total_calculated_vals = {}
    for time_interval_length in time_interval_lengths:
        print "analyzing edgefiles..."
        newly_calculated_values = analyze_edgefiles.pipeline_analysis_step(interval_to_filenames[str(time_interval_length)], ms_s,
                                                                           time_interval_length, basegraph_name, calc_vals, window_size,
                                                                           mapping, is_swarm, make_net_graphs_p)

        total_calculated_vals.update(newly_calculated_values)
    if graph_p:
        # (time gran) -> (node gran) -> metrics -> vals
        print "total_calculated_vals", total_calculated_vals
        #time.sleep(20)
        analyze_edgefiles.create_graphs(total_calculated_vals, basegraph_name, window_size, colors, time_interval_lengths,
                                        exfil_start_time, exfil_end_time, wiggle_room)
