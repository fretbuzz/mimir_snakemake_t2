# this file might be used to write a scapy program that produces a communication graoh based off of the pcap files
# though maybe this would work just as well https://github.com/mateuszk87/PcapViz

# though even if that works, I'd still probably have to proce

# okay, so I guess what I want to do is produce hte traffic matrices here?
# I mean, that is simple enough, but do we want additional features?
# well, what format is oddball's input supposed to be??

# so the output should be
# in SrcID DestID Weight (3 columns) format.
# in a text document.
# so no need to get fancy with pickling or anything liek that

# I think I can write the file pretty simply using something like https://docs.python.org/2/library/csv.html

# I think that in order to seperate the data by ms class in a relevant way I am going to want to use nodeneigh_1.py
# (from leman's code) and then just parse / combine the output and then analyze the resulting graph
# so I am taking the subgraph where
# I might need to just modify leman's code.
# So I don't want to take a subgraph, I just only want to calculate the values for certain nodes.
# Well, I actually want to calculate all the values for all nodes, but I only wanna compare certain particular values
# instead of comparing all of them.
# So, it looks like I should parse leman's output and then go from there (as  opposed to implementing anything here)
import time
from scapy.all import *
import re
import pickle
import pandas as pd
import csv
import math
import yaml
import analyze_edgefiles

#a = rdpcap("/Users/jseverin/Documents/Microservices/munnin_pcaps/no_exfil_sockshop_0_rep_0_0")
# /Users/jseverin/Documents/Microservices/munnin/sockshop_info/sockshop_docker0_default_5.pcap
# /Users/jseverin/Documents/Microservices/munnin/sockshop_info/sockshop_default_swarm_br0.pcap
# /Users/jseverin/Documents/Microservices/munnin/sockshop_info/sockshop_swarm_container_configs.txt
# /Users/jseverin/Documents/Microservices/munnin/sockshop_info/sockshop_default_swarm_br0.pcap

# these lists are only need for processing the k8s pod info
microservices_sockshop= ['carts-db','carts','catalogue-db','catalogue','front-end','orders-db','orders',
        'payment','queue-master','rabbitmq','session-db','shipping','user-db','user','load-test']
minikube_infrastructure = ['etcd', 'kube-addon-manager', 'kube-apiserver', 'kube-controller-manager',
                           'kube-dns', 'kube-proxy', 'kube-scheduler', 'kubernetes-dashboard', 'metrics-server',
                            'storage-provisioner']
microservices_wordpress = ['mariadb-master', 'mariadb-slave', 'wordpress']


# TODO: should probably write an algorithm to parse these automatically from the results
# of kubectl get deploy (--all-namespaces)

# returns a dictionary from IP to the corresponding microservice
# for kubernetes (minikube, in particular)
def map_ip_to_ms(file_path, microservices):
    mapping = {}

    pod_ip_file = open(file_path, "r") # result of kubernetes get po -o wide (--all-namespaces)
    if pod_ip_file.mode == 'r':
        contents = pod_ip_file.read()
        print contents
    print "length", len(contents.split('\n')[1:])
    for line in contents.split('\n')[1:]:
        #print line
        vals = line.split()
        if len(vals) == 0:
            continue
        # print vals, len(vals)
        # print vals[1], vals[6]
        matching_image = ''
        for ms in microservices:
            if ms in vals[1]:
                matching_image = ms
                break
        mapping[vals[6]] = (vals[1], matching_image)
    print mapping
    return mapping

# NOTE: time_intervals is how many seconds each time interval should be
# NOTE: only_one_time_interval indicates whether we should only make a single time interval
# (and thereby disregard whatever time_intervals says)
def parse_pcap(a, time_intervals, mapping, basefile_name, start_time, end_time):
    # okay, so what I want is a dictionary mapping time intervals to graph_dictionaries
    time_to_graphs = {}
    print "pcap start time", a[0].time, "pcap end time", a[-1].time
    print "start time using", start_time, "end time using", end_time
    time_elapsed = end_time - start_time
    print "elapsed time (that I'm using (i.e. not the pcap))", time_elapsed, " sec"
    num_time_intervals = int(math.ceil(time_elapsed / time_intervals))
    print "there should be ", num_time_intervals, " time intervals"
    for i in range(0, num_time_intervals):
        time_to_graphs[i] = {}


    current_time_interval = 0

    for a_pkt in a:
        if a_pkt.time > end_time:
            break

        if a_pkt.time - (start_time + current_time_interval * time_intervals) > time_intervals:
            current_time_interval += 1

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

    #print time_to_parsed_mapping[0]
    #'''
    time_counter = 0
    filesnames = []
    for interval in range(0,num_time_intervals):
        filename = write_to_file(time_to_parsed_mapping[interval], time_counter, time_intervals, basefile_name)
        time_counter += time_intervals
        filesnames.append(filename)
    #'''
    #print time_to_parsed_mapping[0]
    print list(set(no_mapping_found))
    #exit(55)

    return list(set(no_mapping_found)), filesnames

# this function writes the result of parsing the pcap to a file
# in the format that leman's code takes as an input
# NOTE: looks like oddball code requires that the src and dest are integers
# example: basefile_name = './atsea_info/seastore_swarm_br0_0'
def write_to_file(mapping, time, time_interval, basefile_name):
    #'''
    # make mapping of pods to integers
    counter = 1
    map_pod_to_integer = {}
    for src_dst,weight in mapping.iteritems():
        if src_dst[0] not in map_pod_to_integer.keys():
            map_pod_to_integer[src_dst[0]] = counter
            counter += 1
        if src_dst[1] not in map_pod_to_integer.keys():
            map_pod_to_integer[src_dst[1]] = counter
            counter += 1
    #print map_pod_to_integer

    # now apply the mapping
    converted_mapping = {}
    for src_dst, weight in mapping.iteritems():
        mapped_src = map_pod_to_integer[src_dst[0]]
        mapped_dst = map_pod_to_integer[src_dst[1]]
        converted_mapping[(mapped_src, mapped_dst)] = weight
    #'''

    #ending = '_' + str(int(time)) + '_' + str(int(time_interval))
    print time, type(time),  time_interval, type(time_interval)
    ending = '_' + '%.2f' % (time) + '_' + '%.2f' % (time_interval)
    print "file suffix ", ending

    ''' note: can reenable if I want to use the odball code...
    with open('./atsea_info/seastore_swarm_br0_0' + ending + '.txt', 'ab') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for item, weight in converted_mapping.iteritems():
            spamwriter.writerow([item[0], item[1], weight])
    '''

    with open(basefile_name + ending + '.txt', 'ab') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for item, weight in mapping.iteritems():
            spamwriter.writerow([item[0], item[1], weight])

    ''' # note: can re-enable if I want use oddball code... 
        # let's also save the mapping
        with open('./atsea_info/seastore_swarm_br0_0_p2i_mapping' + ending +'.txt', 'ab') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for pod,integer in map_pod_to_integer.iteritems():
                spamwriter.writerow([pod,integer])
    '''

    #print "counter", counter
#find_switch_ports(a,1)

# modifies egonet feature file produced by lemnan's code such that each egonet is focused on a particular
# microservice
# (also, let's have one where we remove infrastructure)
def parse_egonet(file, mapping, microservices):
    # open feature file
    features = open(file, "r") # result of kubernetes get po -o wide (--all-namespaces)
    if features.mode == 'r':
        features_contents = features.read()
        print features_contents
    feature_lines = features_contents.split('\n')

    # okay, so now we want to take some of these lines and print them into a new file
    # But which lines?
    # We need to determine a mapping from (microservice class) -> (line number)

    pod_ip_file = open(mapping, "r") # result of kubernetes get po -o wide (--all-namespaces)
    if pod_ip_file.mode == 'r':
        contents = pod_ip_file.read()
    for line in contents.split('\n'):
        print line
    microservice_to_lines = {}
    for ms in microservices:
        microservice_to_lines[ms] = []
    for line in contents.split('\n'):
        for ms in microservices:
            if ms in line:
                print line
                microservice_to_lines[ms].append(int(line.split(',')[1]) - 1) # subtract by 1 b/c file is going to be
                break
                # treated as zero-indexed
    print microservice_to_lines

    # okay, now we are ready to create the new files!
    for ms in microservices:
        out_file = open(file+'-'+ms, "w")
        for line_numbers in microservice_to_lines[ms]:
            out_file.write(feature_lines[line_numbers] +'\n')
        out_file.close()

    # now let's also create an infrastructure-free feature file
    out_file = open(file+'-'+'no_infrastructure', "w")
    for ms in microservices:
        for line_numbers in microservice_to_lines[ms]:
            out_file.write(feature_lines[line_numbers] +'\n')
    out_file.close()

# for atsea, this is going to be harder b/c we got a bunch of networks (handeled via network list-related functionality)
# each  container may have multiple IP's (different ones on each of the networks)
# path = /Users/jseverin/Documents/Microservices/munnin/sockshop_info/sockshop_swarm_fixed_containers_config.txt
# network_list = ["sockshop_default" ]
# for at sea:
# network_list = ["atsea_back-tier", "atsea_default", "atsea_front-tier", "atsea_payment" ]
# path = /Users/jseverin/Documents/Microservices/munnin/atsea_info/atsea_redux_docker_container_configs.txt
def swarm_container_ips(path, network_list):
    file = open(path, "r")
    swarm_configs = file.read()
    swarm_config_groups = swarm_configs.split("\n]\n")
    #print swarm_config_groups
    g = 0
    #configs = yaml.safe_load(swarm_configs)
    container_to_ip = {}
    not_matched = []
    for config in swarm_config_groups[:-1]: #end is just delimiter so let's not process that
        config += "\n]\n" # split removes delimiter, so let's add it back in
        print g
        g += 1
        #print config
        #if g == 87:
        #    print config
        current_config = yaml.safe_load(config)
        found_something = 0
        network_name = current_config[0]["Name"]
        for container_id, container in current_config[0]["Containers"].iteritems():
            #print "hi", container
            container_name = container["Name"]
            # todo: re-enable if I need to
            split_container_name = container_name.split('.')
            if (len(split_container_name) == 3):
                container_name = split_container_name[0] + '.' + split_container_name[1]

            # how to do this???? -> let's split the string and then stick the parts that we want back together
            container_ip = container["IPv4Address"].split('/',1)[0]
            container_to_ip[container_ip] = (container_name, network_name)

        try:
            for service_name, service_vals in current_config[0]["Services"].iteritems():
                print service_name, service_vals["VIP"]
                service_vip = service_vals["VIP"]
                container_to_ip[service_vip] = (service_name + "_VIP", network_name)
        except:
            print "no sevices in this network..."

    return container_to_ip

# in our current edge-list generation, we have some pieces that
# do not correspond to the operation of the microservice application
# let's remove those so that we are only focusin on the application's communication
# file_path = '/Users/jseverin/Documents/Microservices/munnin/sockshop_swarm_br0_1_pod_names.txt'
def parse_resulting_edgelist(file_path, microservices):
    edgelist = open(file_path, "r") # result of kubernetes get po -o wide (--all-namespaces)
    relevant_lines = []
    if edgelist.mode == 'r':
        contents = edgelist.read()
    for line in contents.split('\n'):
        line_components = line.split(',')
        if len(line_components) == 1:
            continue
        #print len(line_components), line_components
        one_ms = False
        two_ms = False
        for mservice in microservices:
            if mservice in line_components[0]:
                one_ms = True
            if mservice in line_components[1]:
                two_ms = True
            if one_ms and two_ms:
                print line
                relevant_lines.append([line_components[0], line_components[1], line_components[2]])
                break
    with open('sockshop_swarm_br0_1_pod_names_only_ms.txt', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        for item in relevant_lines:
            print item[0], item[1], item[2]
            spamwriter.writerow([item[0], item[1], item[2]])

    # note: idk that this function is ever used, nor does it do something
    # list of pcaps should be all the pcaps that need to be aggregated together (and then return)
    # first item should be the directory (I'm assuming all the files are in the same directory)
    # the rest of the list are file names which are appended to the directory (so include a trailing '/' in
    # the directory name!)
    # for atsea store:
    # atsea_pcaps_trial_1 = ['/Users/jseverin/Documents/Microservices/munnin/atsea_info/', 'atsea_backtier', 'atsea_default', 'atsea_front_tier', 'atsea_ingress', 'atsea_payment']
    # we want to parse each of the overlay nets seperately
def aggregate_pcaps(list_of_pcaps, network_list):
    pkt_list = []
    index = 0
    for pcap in list_of_pcaps[1:]:
        pkt_list.extend(rdpcap(list_of_pcaps[0] + pcap + '_3.pcap'))


        index += 1
    return pkt_list

# this should be only the function I need to mess with (can comment stuff out for partial functionality)
# basefile_name = name (including path) of the file to which the endings should be added and an edgelist stored there
# is_swarm = 1 iff swarm, = 0 otherwise
# time_interval_lengths = list ; pcap_paths = list
# network_or_microservice_list = network list if swarm, microservice (class) list if k8s
# TODO can ms_s take the part of network_or_microservice_list under the appropriate scenario??
def run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               network_or_microservice_list, ms_s, make_edgefiles_p, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time = None, end_time = None, calc_vals=True,
                               graph_p = True):
    if is_swarm:
        mapping = swarm_container_ips(container_info_path, network_or_microservice_list)
    else:
        mapping = map_ip_to_ms(container_info_path, network_or_microservice_list)

    print "container to ip mapping", mapping
    #time.sleep(120)
    #time.sleep(120)
    # okay, how to do this... going to want to (1) generate the filenames and then (2) delete them.
    # problem, don't know how long the
    # going to divide the time intervals here. Input the list such that the you are fine with the first pcap
    # file being used for this.
    if start_time==None or end_time==None or make_edgefiles_p:
        current_pcap = rdpcap(pcap_paths[0])
        fst_pkt = current_pcap[0]
        last_pkt = current_pcap[-1]
        start_time = fst_pkt.time
        end_time = last_pkt.time

    print "start_time: ", start_time, "end_time:", end_time
    #exit(12)
    interval_to_filenames = {}

    for time_interval_length in time_interval_lengths:
        filenames = []
        number_of_time_intervals = (end_time - start_time) / time_interval_length
        number_of_time_intervals = int( math.ceil(number_of_time_intervals) )
        time_counter = 0
        for interval in range(1, number_of_time_intervals+1):
            #ending = '_' + str(int(time_counter)) + '_' + str(int(time_interval_length))
            ending = '_' + '%.2f' % (time_counter) + '_' + '%.2f'% (time_interval_length)
            filenames.append(basefile_name + ending + '.txt')
            time_counter += time_interval_length
        interval_to_filenames[time_interval_length] = filenames

    if make_edgefiles_p:
        for time_interval_length in time_interval_lengths:
            number_of_time_intervals = (end_time - start_time) / time_interval_length
            print "float of time intervals", number_of_time_intervals, ", corresponding integer (ceiling) ", int( math.ceil(number_of_time_intervals))
            number_of_time_intervals = int( math.ceil(number_of_time_intervals) )
            time_counter = 0

            # delete existing files with particular names (going to append to them in the next section, in order to
            # handle multiple pcaps
            for interval in range(1, number_of_time_intervals+1):
                # starts at 1 b/c nothing happened at 0
                # ends with +1 b/c it was leaving the last value out
                #ending = '_' + str(int(time_counter)) + '_' + str(int(time_interval_length))
                ending = '_' + '%.2f' % (time_counter) + '_' + '%.2f' % (time_interval_length)
                filename = basefile_name + ending + '.txt'
                try:
                    os.remove(filename)
                except:
                    print filename, "   ", "does not exist"
                time_counter += time_interval_length


        for current_pcap_path in pcap_paths:
            for time_interval_length in time_interval_lengths:
                if current_pcap_path != pcap_paths[0]: # already loaded the first one in order to perform time interval calculations
                    current_pcap = rdpcap(current_pcap_path)
                unmapped_ips, _ = parse_pcap(current_pcap, time_interval_length, mapping, basefile_name,
                                                         start_time, end_time)
                print "unmapped ips", unmapped_ips

    total_calculated_vals = {}
    for time_interval_length in time_interval_lengths:
        print "analyzing edgefiles..."
        newly_calculated_values = analyze_edgefiles.pipeline_analysis_step(interval_to_filenames[time_interval_length], ms_s,
                                                                           time_interval_length, basegraph_name, calc_vals, window_size,
                                                                           mapping)

        total_calculated_vals.update(newly_calculated_values)
    if graph_p:
        # (time gran) -> (node gran) -> metrics -> vals
        analyze_edgefiles.create_graphs(total_calculated_vals, basegraph_name, window_size, colors, time_interval_lengths,
                                        exfil_start_time, exfil_end_time, wiggle_room)


# here are some 'recipes'
# comment out the ones you are not using
def run_analysis_pipeline_recipes():
    # atsea store recipe

    '''
    pcap_paths = ['/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/seastore_redux_back-tier_1.pcap',
                   '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/seastore_redux_front-tier_1.pcap']
    is_swarm = 1
    basefile_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/edgefiles/seastore_swarm'
    basegraph_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/graphs/seastore_swarm'
    container_info_path = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_redux_docker_container_configs.txt'
    time_interval_lengths = [10, 1, 0.1] # seconds # note: 100 used to be here too
    network_or_microservice_list = ["atsea_back-tier", "atsea_default", "atsea_front-tier", "atsea_payment"]
    ms_s = ['appserver', 'reverse_proxy', 'database']
    make_edgefiles = False
    start_time = 1529180898.56
    end_time = 1529181277.03
    exfil_start_time = 40
    exfil_end_time = 70
    calc_vals = False
    window_size = 6
    graph_p = True # should I make graphs?
    colors = ['b', 'r']
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               network_or_microservice_list, ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p)
    '''

    # sockshop recipe
    '''
    # note: still gotta do calc_vals again...
    pcap_paths = ["/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/sockshop_swarm_fixed_br0_0.pcap"]
    is_swarm = 1
    basefile_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/edgefiles/sockshop_swarm_pipeline_br0'
    basegraph_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/graphs/sockshop_swarm'
    container_info_path = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/sockshop_swarm_fixed_containers_config.txt'
    time_interval_lengths = [10, 1] # seconds
    network_or_microservice_list = ["sockshop_default"]
    ms_s = microservices_sockshop
    make_edgefiles = True
    calc_vals = True
    graph_p = True # should I make graphs?
    start_time = 1529527610.6
    end_time = 1529527979.54
    window_size = 6
    colors = ['b', 'r']
    # here are some example colors:
    # b: blue ;  g: green  ;  r: red   ;   c: cyan    ; m: magenta
    exfil_start_time = None
    exfil_end_time = None
    wiggle_room = ??
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               network_or_microservice_list, ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time, 
                               calc_vals = calc_vals, graph_p = graph_p)
    #'''
    # wordpress recipe (TODO)
    '''
    pcap_paths = ??
    is_swarm = 1
    basefile_name = ??
    container_info_path = ??
    time_interval_lengths = [??, ??, ??] # seconds
    network_or_microservice_list = ??
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               network_or_microservice_list)
    '''

    # sockshop exp 1 (rep 0)
    ''' # note: still gotta do calc_vals again...
    pcap_paths = ["/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/sockshop_one_sockshop_default_0.pcap"]
    is_swarm = 1
    basefile_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/edgefiles/sockshop_one_pipeline_br0'
    basegraph_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/graphs/sockshop_one'
    container_info_path = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/sockshop_one_docker_0_container_configs.txt'
    time_interval_lengths = [10, 1]# , .1] # seconds (note eventually the 0.1 gran should be done and can re-enable)
    network_or_microservice_list = ["sockshop_default"]
    ms_s = microservices_sockshop
    make_edgefiles = False
    calc_vals = True
    graph_p = True # should I make graphs?
    start_time = None
    end_time = None
    window_size = 6
    colors = ['b', 'r']
    exfil_start_time = 270
    exfil_end_time = 310
    wiggle_room = ??
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               network_or_microservice_list, ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p)
    #'''
    # sockshop exp 2 (rep 0)
    ''' # note: still gotta do calc_vals again...
    pcap_paths = ["/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/sockshop_two_sockshop_default_0.pcap"]
    is_swarm = 1
    basefile_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/edgefiles/sockshop_two_pipeline_br0'
    basegraph_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/graphs/sockshop_two'
    container_info_path = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/sockshop_two_docker_0_container_configs.txt'
    time_interval_lengths = [50, 10, 1]# , .1] # seconds (note eventually the 0.1 gran should be done and can re-enable)
    network_or_microservice_list = ["sockshop_default"]
    ms_s = microservices_sockshop
    make_edgefiles = True
    calc_vals = False
    graph_p = False # should I make graphs?
    start_time = None
    end_time = None
    window_size = 6
    colors = ['b', 'r']
    exfil_start_time = 270
    exfil_end_time = 330
    wiggle_room = ???
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               network_or_microservice_list, ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p)
    #'''
    '''
    # sockshop exp 3 (rep 0)
    pcap_paths = ["/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/sockshop_three_sockshop_default_0.pcap"]
    is_swarm = 1
    basefile_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/edgefiles/sockshop_three_0'
    basegraph_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/graphs/sockshop_three_0'
    container_info_path = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/sockshop_info/sockshop_three_docker_0_container_configs.txt'
    time_interval_lengths = [50, 10, 1]#, .1] # seconds
    network_or_microservice_list = ["sockshop_default"]
    ms_s = microservices_sockshop
    make_edgefiles = False
    calc_vals = False
    graph_p = True # should I make graphs?
    start_time = None
    end_time = None
    window_size = 6
    colors = ['b', 'r']
    exfil_start_time = 300
    exfil_end_time = 360
    wiggle_room = ??
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               network_or_microservice_list, ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p)

    #'''

    # atsea exp 2 (v2)
    '''
    pcap_paths = ['/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_two_v2__atsea_back-tier_0.pcap',
                   '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_two_v2__atsea_front-tier_0.pcap',
                  '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_two_v2__ingress_0.pcap']
    is_swarm = 1
    basefile_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/edgefiles/atsea_store_exp_two_v2_'
    basegraph_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/graphs/atsea_store_exp_two_v2_'
    container_info_path = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_two_v2__docker_0_network_configs.txt'
    time_interval_lengths = [50, 10]#50, , 1] #, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    network_or_microservice_list = ["atsea_back-tier", "atsea_default", "atsea_front-tier", "atsea_payment"]
    ms_s = ['appserver_VIP', 'reverse_proxy_VIP', 'database_VIP', 'appserver', 'reverse_proxy', 'database', 'back-tier', 'front-tier']
    make_edgefiles = True
    start_time = 1533310837.05
    end_time = 1533311351.12
    exfil_start_time = 270
    exfil_end_time = 330
    calc_vals = True
    window_size = 6
    graph_p = True # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               network_or_microservice_list, ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p)
    '''

    # atsea exp 2 (v7) [good]
    #'''
    pcap_paths = ['/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_two_v7__atsea_back-tier_0.pcap',
                   '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_two_v7__atsea_front-tier_0.pcap',
                  '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_two_v7__ingress_0.pcap',
                  '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_two_v7__bridge_0.pcap',
                  '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_two_v7__ingress_sbox_0.pcap']
    is_swarm = 1
    basefile_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/edgefiles/atsea_store_exp_two_v7_'
    basegraph_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/graphs/atsea_store_exp_two_v7_'
    container_info_path = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_two_v7__docker_0_network_configs.txt'
    time_interval_lengths = [50, 30, 10]#50, , 1] #, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    network_or_microservice_list = ["atsea_back-tier", "atsea_default", "atsea_front-tier", "atsea_payment"]
    ms_s = ['appserver_VIP', 'reverse_proxy_VIP', 'database_VIP', 'appserver', 'reverse_proxy', 'database', 'back-tier', 'front-tier'. 'visualizer']
    make_edgefiles = False
    start_time = 1533377817.89
    end_time = 1533378712.2
    exfil_start_time = 270
    exfil_end_time = 330
    calc_vals = True
    window_size = 6
    graph_p = True # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2 # the number of seconds to extend the start / end of exfil time (to account for imperfect synchronization)
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               network_or_microservice_list, ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p)
    #'''
    # atsea exp 3 (v2) [good]
    '''
    pcap_paths = ['/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_three_v2__atsea_back-tier_0.pcap',
                   '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_three_v2__atsea_front-tier_0.pcap',
                  '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_three_v2__ingress_0.pcap',
                  '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_three_v2__bridge_0.pcap',
                  '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_three_v2__ingress_sbox_0.pcap']
    is_swarm = 1
    basefile_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/edgefiles/atsea_store_exp_three_v2_'
    basegraph_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/graphs/atsea_store_exp_three_v2_'
    container_info_path = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_exp_three_v2__docker_0_network_configs.txt'
    time_interval_lengths = [50, 30, 10, 1]#50, , 1] #, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    network_or_microservice_list = ["atsea_back-tier", "atsea_default", "atsea_front-tier", "atsea_payment"]
    ms_s = ['appserver_VIP', 'reverse_proxy_VIP', 'database_VIP', 'appserver', 'reverse_proxy', 'database', 'back-tier', 'front-tier']
    make_edgefiles = False
    start_time = 1533381724.66 #None
    end_time = 1533382619.64 #None
    exfil_start_time = 300
    exfil_end_time = 360
    calc_vals = True
    window_size = 6
    graph_p = True # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = 2
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               network_or_microservice_list, ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p)
    #'''

    '''
    # atsea exp 3 (rep 0)
    pcap_paths = ['/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_three_atsea_front-tier_0.pcap']
    # ^^ only a single value in list b/c connected database to front-tier network (so backtier didn't do anything)
    is_swarm = 1
    basefile_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/edgefiles/atsea_store_three'
    basegraph_name = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/graphs/atsea_store_three'
    container_info_path = '/Users/jseverin/Documents/Microservices/munnin/experimental_data/atsea_info/atsea_store_three_docker_0_container_configs.txt'
    time_interval_lengths = [50] #, 10, 1, 0.5] # note: not doing 100 or 0.1 b/c 100 -> not enough data points; 0.1 -> too many (takes multiple days to run)
    network_or_microservice_list = ["atsea_back-tier", "atsea_default", "atsea_front-tier", "atsea_payment"]
    ms_s = ['appserver', 'reverse_proxy', 'database']
    make_edgefiles = True
    start_time = None
    end_time = None
    exfil_start_time = 300
    exfil_end_time = 360
    calc_vals = True
    window_size = 6
    graph_p = True # should I make graphs?
    colors = ['b', 'r']
    wiggle_room = ??
    run_data_anaylsis_pipeline(pcap_paths, is_swarm, basefile_name, container_info_path, time_interval_lengths,
                               network_or_microservice_list, ms_s, make_edgefiles, basegraph_name, window_size, colors,
                               exfil_start_time, exfil_end_time, wiggle_room, start_time=start_time, end_time=end_time,
                               calc_vals = calc_vals, graph_p = graph_p)
    '''

# TODO TODO TODO TODO TODO TODO TODO
# okay so this is what I gotta do Monday
# (1) keep working on this whole pipeline business until I get seashop graphs
    # i think the edgefiles check out, so can disable the part that involves having to import the pcaps and whatnot
    # 2 hours? (so aiming for before 10 a.m.)
    # start with the todo about edges_iter in analyze_edgefiles and keep cranking on that until I get some coherent
    # graphs. They should hopefully be more or less clear. I'll need to save them locally at some point, as well,
    # and I might need to add another loop to handle multiple runnings of the experiment (and I'll definitly need
    # to make some modifications to the actual graphs (can't distinguish between trials at the moment))
# (2) get sockshop graphs too
    # still swarm, so I'm hoping for 1 hour (so aiming for before 11 a.m.)
    # this should be almost entirely using the facilities developed in (1), but I'm reserving some time for debugging
# (3) reread prev meeting notes
    # 15 min, try to finish before 11:30 a.m.
    # meeting was v short, so shouldn't be that bad
# (4) docker collaborater people? (the syscall people)
    # 1 hour to read paper, another 1/2 hour to think about. Let's take lunch at 12:30, so let's try to wrap this up by 1:30
# (5) sequester? (i think it could work, eventually...)
    # 1.5 hours to read paper, another 1/2 to think about, so let's aim for 4 p.m. for this...
# (6) finalize meeting visualizations
    # should take about an hour, let's aim to finish this up by about 5 p.m.
# (7) make some modifications to the draft of my paper (if time)
    # if time, let's do 5:15 - 5:45 p.m. (might as well start rewriting the thing...)
# (so ideally, I should pretty much have everything I want to talk about in the meeting done by the end of the day)

# okay, so let's firm up the new plan
# (1) get a couple of more graph metrics (i think I'm eyeballing 3 more, so let's see how those work)
# (2) get the graph situation under control (i.e. proper axis/titles/savings)
    # I STILL HAVE TO DO THIS (boxplots are being stubbornly resistant)
# (3) try sockshop. But if it doesn't work within .5 hours, I'm going to have to postpone
# (4) start preparation of meeting documents
    # graphs (which I hopefully generated in (2))
    # formal specification / writeup for all the graph metrics
# (5) docker colloboration / sequester will be pushed to first thing tomorrow (hopefully can finish it by 11:00
# (6) (after the previous thing tomorrow) finish meeting visualizations / notes/ prepare-in-general (should have about 2 hours for this...)


if __name__=="__main__":
    print "RUNNING"
    run_analysis_pipeline_recipes()
