import json
import yaml
import os

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
    list_of_svcs = []
    with open(kubernetes_svc_info_path, 'r') as svc_f:
        line = svc_f.readlines()
        for l in line[1:]:
            l_pieces = l.split()
            print 'l_pieces', l_pieces
            print 'l_pieces', l_pieces[1], l_pieces[3]
            #mapping[l_pieces[1]] = l_pieces[3]
            ## TODO: get ports + protos
            ports = []
            protos = []
            for port_proto in l_pieces[5].split(','):
                port, proto = port_proto.split('/')

                # need to consider the mapping case...
                port_in_out = port.split(':')
                if len(port_in_out) > 1:
                    port_in,port_out = port_in_out[0], port_in_out[1]
                else:
                    port_in, port_out = port_in_out[0], port_in_out[0]
                ports.append((port_in,port_out))

                protos.append(proto)
            mapping[l_pieces[3]] = (l_pieces[1] + '_VIP', 'svc', ports, protos)
            list_of_svcs.append( l_pieces[1] )

    print "these service mappings were found", mapping
    return mapping, list_of_svcs

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

def parse_kubernetes_pod_info(kubernetes_pod_info):
    pod_ip_info = {}
    with open(kubernetes_pod_info) as f:
        lines = f.readlines()
        for line in lines:
            split_line = line.split()
            print "split_line", split_line
            #print line.split()[1], line.split()[6]
            if split_line[6] in pod_ip_info:
                pod_ip_info[split_line[6]] = (pod_ip_info[split_line[6]][0] + ';' + split_line[1],'pod')
            else:
                pod_ip_info[split_line[6]] = (split_line[1], 'pod')
    return pod_ip_info

def create_mappings(cluster_creation_log):
    #First, get a mapping of IPs to(container_name, network_name)
    initial_ips = cluster_creation_log[0]
    mapping = {}
    infra_services = {}
    ms_s = set()
    #            container_to_ip[container_ip] = (container_name, network_name)

    for name, ip_info in initial_ips.iteritems():
        if ip_info[3] != 'svc':
            mapping[ip_info[0]] = (name, None, ip_info[2], ip_info[3])
        else:
            mapping[ip_info[0]] = (name+'_VIP', None, ip_info[2], ip_info[3])
            ms_s.add(name)

        if ip_info[3] == 'svc' and ip_info[2] == 'kube-system':
            infra_services[name] = ip_info[0]

    return mapping, infra_services, list(ms_s)

def old_create_mappings(is_swarm, container_info_path, kubernetes_svc_info, kubernetes_pod_info, cilium_config_path, ms_s):
    #First, get a mapping of IPs to(container_name, network_name)
    mapping = ips_on_docker_networks(container_info_path)
    list_of_infra_services = []
    kubernetes_services = None
    if not is_swarm:
        # if it is kubernetes, then it is also necessary to read in that file with all the
        # info about the svc's, b/c kubernetes service VIPs don't show up in the docker configs
        # pass
        kubernetes_service_VIPs, total_list_of_services = parse_kubernetes_svc_info(kubernetes_svc_info)
        print "kubernetes_service_VIPs", kubernetes_service_VIPs
        mapping.update(kubernetes_service_VIPs)
        list_of_infra_services = []
        for total_svc in total_list_of_services:
            if total_svc not in ms_s:
                list_of_infra_services.append(total_svc)
        print "list_of_infra_services", list_of_infra_services
        # print [i for i in kubernetes_service_VIPs.items() ]
        # kubernetes_services = [i[1][0].split('_')[0] for i in kubernetes_service_VIPs.items()]
        # print "kubernetes_services", kubernetes_services

        if kubernetes_pod_info:
            kubernetes_pod_VIPS = parse_kubernetes_pod_info(kubernetes_pod_info)
            for ip, name_and_network in kubernetes_pod_VIPS.iteritems():
                if ip not in mapping:
                    mapping[ip] = name_and_network
            print "mapping updated with kubernetes_pod_info", mapping

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

    print "mapping", mapping
    print "list_of_infra_services", list_of_infra_services
    print "ms_s",ms_s
    #exit()
    return mapping, list_of_infra_services
