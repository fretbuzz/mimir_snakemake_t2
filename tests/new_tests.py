import math
import unittest
import networkx as nx
import numpy as np
import pandas as pd
import os
from analysis_pipeline.generate_graphs import get_points_to_plot
from analysis_pipeline.next_gen_metrics import calc_VIP_metric
import analysis_pipeline.simplified_graph_metrics
import multiprocessing
from analysis_pipeline.pcap_to_edgelists import create_mappings, old_create_mappings


class testSyntheticAttackInjector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_injector(self):
        print "test_injector"
        file_paths = ['./old_test_data/wordpress_thirteen_t1_default_bridge_0any_split_00036_20190220141725_edges.txt']
        counter_starting = 0
        svcs = ["my-release-pxc", "wwwppp-wordpress"]
        is_swarm = 0
        ms_s = svcs

        current_total_node_list = []
        svc_to_pod = {}
        node_attack_mapping = {}
        total_edgelist_nodes = []
        avg_dns_weight = 0
        avg_dns_pkts = 0

        container_info_path = "./old_test_data/wordpress_thirteen_t1_docker_0_network_configs.txt"
        cilium_config_path = None  # does NOT use cilium on reps 2-4
        kubernetes_svc_info = './old_test_data/wordpress_thirteen_t1_svc_config_0.txt'
        kubernetes_pod_info = './old_test_data/wordpress_thirteen_t1_pod_config_0.txt'

        container_to_ip, infra_service = old_create_mappings(is_swarm, container_info_path, kubernetes_svc_info,
                                                          kubernetes_pod_info, cilium_config_path, ms_s)

        initiator_info_for_paths = None # not actually need so no big deal
        name_of_dns_pod_node = None
        injected_file_path = './old_test_data/injected_edgefiles/with_nodeAttribsavg_exfil_10000_var_0wordpress_thirteen_t1_default_bridge_0any_split_00036_20190220141725_edges.txt'
        pruned_without_injected = './old_test_data/pruned_edgefiles/wordpress_thirteen_t1_default_bridge_0any_split_00036_20190220141725_edges.txt'

        last_attack_injected = None
        carryover = 0

        synthetic_exfil_paths = [['my_release_pxc_pod', 'wwwppp_wordpress_vip', 'wwwppp_wordpress_pod', 'internet']]
        attacks_to_times = [(0,1)]
        time_interval = 30
        out_q = multiprocessing.Queue()

        avg_exfil_per_min = 10000
        exfil_per_min_variance = 0
        avg_pkt_size = 500
        pkt_size_variance = 0

        end_of_training= 200
        pod_creation_log = None
        analysis_pipeline.simplified_graph_metrics.process_and_inject_single_graph(counter_starting, file_paths, svcs,
                        is_swarm, ms_s, container_to_ip, infra_service, synthetic_exfil_paths, initiator_info_for_paths,
                        attacks_to_times, time_interval, total_edgelist_nodes, svc_to_pod, avg_dns_weight, avg_dns_pkts,
                        node_attack_mapping, out_q, current_total_node_list, name_of_dns_pod_node, last_attack_injected,
                        carryover, avg_exfil_per_min, exfil_per_min_variance, avg_pkt_size, pkt_size_variance,
                        end_of_training, pod_creation_log, './old_test_data/', False )

        '''
        
        process_and_inject_single_graph(counter_starting, file_paths, svcs, is_swarm, ms_s, container_to_ip, infra_instances,
                                    synthetic_exfil_paths, initiator_info_for_paths, attacks_to_times,
                    time_interval, total_edgelist_nodes, svc_to_pod, avg_dns_weight, avg_dns_pkts,
                    node_attack_mapping, out_q, current_total_node_list,name_of_dns_pod_node,attack_injected, carryover,
                    avg_exfil_per_min, exfil_per_min_variance, avg_pkt_size, pkt_size_variance, end_of_training,
                     pod_creation_log, injection_rate_exp_path, drop_infra_from_graph)
        
        '''

        # okay, now I actually need to see if it did the right thing...
        #G = nx.DiGraph()
        G= nx.read_gpickle( injected_file_path )

        print "ZZZZZZ"
        G_pruned_without_injected = nx.DiGraph()
        f = open(pruned_without_injected, 'r')
        lines = f.readlines()
        nx.parse_edgelist(lines, delimiter=' ', create_using=G_pruned_without_injected, data=[('frames',int),('weight', int)])

        edges_in_inject_but_not_pruned = []
        weight_differences = []
        different_edges = []
        for (u,v,d) in G.edges(data=True):
            #print (u,v,d), d['weight'], G[u][v]['weight']
            try:
                weight_difference =  d['weight'] - G_pruned_without_injected[u][v]['weight']
                if weight_difference != 0:
                    weight_differences.append(weight_difference)
                    different_edges.append((u,v,d))
            except:
                weight_differences.append(d['weight'])
                different_edges.append((u, v, d, 'wasnt_in_pruned'))

        print "weight_differences ", weight_differences
        print "different_edges",different_edges
        print "in_injected_but_not_pruned", edges_in_inject_but_not_pruned
        for edge in edges_in_inject_but_not_pruned:
            print "in_injected_but_not_pruned", edge

        #print G.nodes()

        self.assertEqual(len(weight_differences), 6)
        self.assertEqual(len([i for i in weight_differences if i == 400]), 3)
        self.assertEqual(len([i for i in weight_differences if i == 5000]), 3)


    def test_injector_not_doing_anything(self):
        print "test_injector"
        file_paths = [
            './old_test_data/wordpress_thirteen_t1_default_bridge_0any_split_00036_20190220141725_edges.txt']
        counter_starting = 2
        svcs = ["my-release-pxc", "wwwppp-wordpress"]
        is_swarm = 0
        ms_s = svcs

        current_total_node_list = []
        svc_to_pod = {}
        node_attack_mapping = {}
        total_edgelist_nodes = []
        avg_dns_weight = 0
        avg_dns_pkts = 0

        container_info_path = "./old_test_data/wordpress_thirteen_t1_docker_0_network_configs.txt"
        cilium_config_path = None  # does NOT use cilium on reps 2-4
        kubernetes_svc_info = './old_test_data/wordpress_thirteen_t1_svc_config_0.txt'
        kubernetes_pod_info = './old_test_data/wordpress_thirteen_t1_pod_config_0.txt'

        print "cur_cwd", os.getcwd()
        container_to_ip, infra_service = old_create_mappings(is_swarm, container_info_path, kubernetes_svc_info,
                                                         kubernetes_pod_info, cilium_config_path, ms_s)

        initiator_info_for_paths = None  # not actually need so no big deal
        name_of_dns_pod_node = None
        injected_file_path = './old_test_data/injected_edgefiles/with_nodeAttribsavg_exfil_10000_var_0wordpress_thirteen_t1_default_bridge_0any_split_00036_20190220141725_edges.txt'
        pruned_without_injected = './old_test_data/pruned_edgefiles/wordpress_thirteen_t1_default_bridge_0any_split_00036_20190220141725_edges.txt'

        last_attack_injected = None
        carryover = 0

        synthetic_exfil_paths = [['my_release_pxc_pod', 'wwwppp_wordpress_vip', 'wwwppp_wordpress_pod', 'internet']]
        attacks_to_times = [(0, 1)]
        time_interval = 30
        out_q = multiprocessing.Queue()

        avg_exfil_per_min = 10000
        exfil_per_min_variance = 0
        avg_pkt_size = 500
        pkt_size_variance = 0

        end_of_training = 200
        pod_creation_log = None # this is GOOD b/c there's no pod_creation_log

        analysis_pipeline.simplified_graph_metrics.process_and_inject_single_graph(counter_starting, file_paths,
                                                                                   svcs,
                                                                                   is_swarm, ms_s, container_to_ip,
                                                                                   infra_service,
                                                                                   synthetic_exfil_paths,
                                                                                   initiator_info_for_paths,
                                                                                   attacks_to_times, time_interval,
                                                                                   total_edgelist_nodes, svc_to_pod,
                                                                                   avg_dns_weight, avg_dns_pkts,
                                                                                   node_attack_mapping, out_q,
                                                                                   current_total_node_list,
                                                                                   name_of_dns_pod_node,
                                                                                   last_attack_injected,
                                                                                   carryover, avg_exfil_per_min,
                                                                                   exfil_per_min_variance,
                                                                                   avg_pkt_size, pkt_size_variance,
                                                                                   end_of_training,
                                                                                   pod_creation_log,
                                                                                   './old_test_data/', False)

        # okay, now I actually need to see if it did the right thing...
        # G = nx.DiGraph()
        G = nx.read_gpickle(injected_file_path)

        print "ZZZZZZ"
        G_pruned_without_injected = nx.DiGraph()
        f = open(pruned_without_injected, 'r')
        lines = f.readlines()
        nx.parse_edgelist(lines, delimiter=' ', create_using=G_pruned_without_injected,
                          data=[('frames', int), ('weight', int)])

        edges_in_inject_but_not_pruned = []
        weight_differences = []
        different_edges = []
        for (u, v, d) in G.edges(data=True):
            # print (u,v,d), d['weight'], G[u][v]['weight']
            try:
                weight_difference = d['weight'] - G_pruned_without_injected[u][v]['weight']
                if weight_difference != 0:
                    weight_differences.append(weight_difference)
                    different_edges.append((u, v, d))
            except:
                weight_differences.append(d['weight'])
                different_edges.append((u, v, d, 'wasnt_in_pruned'))

        print "weight_differences_nothing ", weight_differences
        print "different_edges", different_edges
        print "in_injected_but_not_pruned", edges_in_inject_but_not_pruned
        for edge in edges_in_inject_but_not_pruned:
            print "in_injected_but_not_pruned", edge

        # print G.nodes()

        self.assertEqual(len(weight_differences), 0)
        #self.assertEqual(len([i for i in weight_differences if i == 400]), 3)
        #self.assertEqual(len([i for i in weight_differences if i == 5000]), 3)


    def test_dns_injection(self):
        print "test_injector"
        file_paths = [
            './old_test_data/wordpress_thirteen_t1_default_bridge_0any_split_00036_20190220141725_edges.txt']
        counter_starting = 0
        svcs = ["my-release-pxc", "wwwppp-wordpress"]
        is_swarm = 0
        ms_s = svcs

        current_total_node_list = []
        svc_to_pod = {}
        node_attack_mapping = {}
        total_edgelist_nodes = []
        avg_dns_weight = 0
        avg_dns_pkts = 0

        container_info_path = "./old_test_data/wordpress_thirteen_t1_docker_0_network_configs.txt"
        cilium_config_path = None  # does NOT use cilium on reps 2-4
        kubernetes_svc_info = './old_test_data/wordpress_thirteen_t1_svc_config_0.txt'
        kubernetes_pod_info = './old_test_data/wordpress_thirteen_t1_pod_config_0.txt'

        container_to_ip, infra_service = old_create_mappings(is_swarm, container_info_path, kubernetes_svc_info,
                                                         kubernetes_pod_info, cilium_config_path, ms_s)

        initiator_info_for_paths = None  # not actually need so no big deal
        name_of_dns_pod_node = None
        injected_file_path = './old_test_data/injected_edgefiles/with_nodeAttribsavg_exfil_10000_var_0wordpress_thirteen_t1_default_bridge_0any_split_00036_20190220141725_edges.txt'
        pruned_without_injected = './old_test_data/pruned_edgefiles/wordpress_thirteen_t1_default_bridge_0any_split_00036_20190220141725_edges.txt'

        last_attack_injected = None
        carryover = 0
        synthetic_exfil_paths = [['my_release_pxc_pod', 'my_release_pxc_vip', 'wwwppp_wordpress_pod', 'kube_dns_vip', 'kube_dns_pod',
                   'internet']]
        attacks_to_times = [(0, 1)]
        time_interval = 30
        out_q = multiprocessing.Queue()

        avg_exfil_per_min = 10000
        exfil_per_min_variance = 0
        avg_pkt_size = 500
        pkt_size_variance = 0

        end_of_training = 200
        pod_creation_log = None
        analysis_pipeline.simplified_graph_metrics.process_and_inject_single_graph(counter_starting, file_paths,
                                                                                   svcs,
                                                                                   is_swarm, ms_s, container_to_ip,
                                                                                   infra_service,
                                                                                   synthetic_exfil_paths,
                                                                                   initiator_info_for_paths,
                                                                                   attacks_to_times, time_interval,
                                                                                   total_edgelist_nodes, svc_to_pod,
                                                                                   avg_dns_weight, avg_dns_pkts,
                                                                                   node_attack_mapping, out_q,
                                                                                   current_total_node_list,
                                                                                   name_of_dns_pod_node,
                                                                                   last_attack_injected,
                                                                                   carryover, avg_exfil_per_min,
                                                                                   exfil_per_min_variance,
                                                                                   avg_pkt_size, pkt_size_variance,
                                                                                   end_of_training, pod_creation_log,
                                                                                   './old_test_data/', False)

        # okay, now I actually need to see if it did the right thing...
        # G = nx.DiGraph()
        G = nx.read_gpickle(injected_file_path)

        print "ZZZZZZ"
        G_pruned_without_injected = nx.DiGraph()
        f = open(pruned_without_injected, 'r')
        lines = f.readlines()
        nx.parse_edgelist(lines, delimiter=' ', create_using=G_pruned_without_injected,
                          data=[('frames', int), ('weight', int)])

        edges_in_inject_but_not_pruned = []
        weight_differences = []
        different_edges = []
        for (u, v, d) in G.edges(data=True):
            # print (u,v,d), d['weight'], G[u][v]['weight']
            try:
                weight_difference = d['weight'] - G_pruned_without_injected[u][v]['weight']
                if weight_difference != 0:
                    weight_differences.append(weight_difference)
                    different_edges.append((u, v, d))
            except:
                weight_differences.append(d['weight'])
                different_edges.append((u, v, d, 'wasnt_in_pruned'))
                edges_in_inject_but_not_pruned.append((u, v, d))

        print "weight_differences_dns ", weight_differences
        print "different_edges", different_edges
        print "in_injected_but_not_pruned", edges_in_inject_but_not_pruned
        for edge in different_edges:
            print "different_edges_indiv", edge

        # print G.nodes()

        self.assertEqual(len(weight_differences), 10)
        self.assertEqual(len([i for i in weight_differences if i == 400]), 5)
        self.assertEqual(len([i for i in weight_differences if i == 5000]), 5)
        print "edges_in_inject_but_not_pruned",edges_in_inject_but_not_pruned
        self.assertEqual(len(edges_in_inject_but_not_pruned), 4)


if __name__ == "__main__":
    unittest.main()

    #singletest = unittest.TestSuite()
    #singletest.addTest(testSyntheticAttackInjector())
    #unittest.TextTestRunner().run(singletest)
    #singletest.addTest(testSyntheticAttackInjector())
