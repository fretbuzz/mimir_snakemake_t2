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
