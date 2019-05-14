import cPickle as pickle
import copy
import gc
import logging
import math
import multiprocessing
from itertools import groupby
from operator import itemgetter
import errno, os
import pandas as pd
import gen_attack_templates, process_pcap, process_graph_metrics, simplified_graph_metrics
from pcap_to_edgelists import create_mappings,old_create_mappings
import random
from statistical_analysis import drop_useless_columns_aggreg_DF
import cilium_config_generator
import time

# Note: see run_analysis_pipeline_recipes for pre-configured sets of parameters (there are rather a lot)
class data_anylsis_pipline(object):
    def __init__(self, pcap_paths=None, basefile_name=None, time_interval_lengths=None, make_edgefiles_p=False,
                 basegraph_name=None, calc_vals=True, make_net_graphs_p=False, alert_file=None,
                 sec_between_exfil_pkts=1, time_of_synethic_exfil=None, netsec_policy=None, skip_graph_injection=False,
                 cluster_creation_log=None, sensitive_ms=None, exfil_StartEnd_times=[], physical_exfil_paths=[],
                 old_mulval_info=None, base_experiment_dir='', no_processing_at_all=False):

        print "basefile_name", basefile_name
        try:
            os.makedirs(basefile_name)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        try:
            os.makedirs(base_experiment_dir + 'graphs/')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        try:
            os.makedirs(base_experiment_dir + 'alerts/')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


        print "log file can be found at: " + str(basefile_name) + '_logfile.log'
        logging.basicConfig(filename=basefile_name + '_logfile.log', level=logging.INFO)
        logging.info('run_data_anaylsis_pipeline Started')

        gc.collect()

        print "starting pipeline..."
        self.sub_path = 'sub_'  # NOTE: make this an empty string if using the full pipeline (and not the subset)
        if cluster_creation_log is None:
            self.cluster_creation_log = None
        else:
            if not old_mulval_info:
                cluster_creation_log = cluster_creation_log[0]
            with open(cluster_creation_log, 'r') as f:
                cluster_creation_log = f.read()
            self.cluster_creation_log = pickle.loads(cluster_creation_log)

        if old_mulval_info:
            self.ms_s = old_mulval_info["ms_s"]
            if 'kube-dns' not in self.ms_s:
                self.ms_s.append('kube-dns')
            container_info_path, kubernetes_svc_info = old_mulval_info["container_info_path"], old_mulval_info["kubernetes_svc_info"]
            kubernetes_pod_info, cilium_config_path = old_mulval_info["kubernetes_pod_info"], old_mulval_info["cilium_config_path"]
            self.sensitive_ms = sensitive_ms
            self.mapping, self.infra_instances = old_create_mappings(0, container_info_path, kubernetes_svc_info,
                                                                     kubernetes_pod_info, cilium_config_path, self.ms_s)
        else:
            self.sensitive_ms = sensitive_ms[0]
            self.mapping, self.infra_instances, self.ms_s = create_mappings(self.cluster_creation_log)

        # NOTE: if you follow the whole path, self.list_of_infra_services isn't really used for anything atm...
        self.calc_zscore_p=False
        self.time_interval_lengths = time_interval_lengths
        self.basegraph_name = basegraph_name
        self.basefile_name = basefile_name
        self.experiment_folder_path = base_experiment_dir
        self.cilium_component_dir = self.experiment_folder_path + 'cilium_stuff'
        self.exp_name = basefile_name.split('/')[-1]
        self.base_exp_name = self.exp_name
        self.orig_exp_name = self.exp_name

        self.pcap_path = "/".join(pcap_paths[0].split('/')[:-1]) + '/'
        self.pcap_file = pcap_paths[0].split('/')[-1]  # NOTE: assuming only a single pcap file...
        self.make_edgefiles_p = make_edgefiles_p #and only_exp_info
        if netsec_policy == 'None' or netsec_policy == 'none':
            netsec_policy = None
        self.netsec_policy = netsec_policy
        self.make_edgefiles_p=make_edgefiles_p
        self.time_of_synethic_exfil = time_of_synethic_exfil
        self.make_net_graphs_p=make_net_graphs_p
        self.alert_file=alert_file
        self.sec_between_exfil_events=sec_between_exfil_pkts
        self.orig_alert_file = self.alert_file
        self.orig_basegraph_name = self.basegraph_name
        self.skip_graph_injection = skip_graph_injection
        self.cilium_component_time_length= None # will be assigned @ call time...

        self.synthetic_exfil_paths = None
        self.initiator_info_for_paths = None
        self.calc_vals = calc_vals
        self.cilium_allowed_svc_comm = None

        self.time_gran_to_feature_dataframe=None
        self.time_gran_to_attack_labels=None
        self.time_gran_to_synthetic_exfil_paths_series=None
        self.time_gran_to_list_of_concrete_exfil_paths  = None
        self.time_gran_to_list_of_exfil_amts=None
        self.time_gran_to_new_neighbors_outside=None
        self.time_gran_to_new_neighbors_dns=None
        self.time_gran_to_new_neighbors_all=None
        self.time_gran_to_list_of_amt_of_out_traffic_bytes = None
        self.time_gran_to_list_of_amt_of_out_traffic_pkts = None
        self.intersvc_vip_pairs = None

        ## TODO: these values need to be encorporated into the pipeline.
        #self.exfil_StartEnd_times = exfil_StartEnd_times
        #self.physical_exfil_paths = physical_exfil_paths

        if not no_processing_at_all:
            self.process_pcaps()

            if exfil_StartEnd_times is None or exfil_StartEnd_times == [[]]:
                min_time_gran = min([int(i) for i in self.interval_to_filenames.keys()])
                exp_length = len(self.interval_to_filenames[str(min_time_gran)]) * min_time_gran
                self.exfil_StartEnd_times = [[exp_length, exp_length]]
                self.physical_exfil_paths = [[]]
                #self.exfil_start_time = exp_length
                #self.exfil_end_time = exp_length
            else:
                #self.exfil_start_time = exfil_start_time
                #self.exfil_end_time = exfil_end_time
                self.exfil_StartEnd_times = exfil_StartEnd_times
                self.physical_exfil_paths = physical_exfil_paths


    def generate_synthetic_exfil_paths(self, max_number_of_paths, max_path_length, dns_porportion):
        self.netsec_policy,self.intersvc_vip_pairs = gen_attack_templates.parse_netsec_policy(self.netsec_policy)
        app_ms_s = [ms for ms in self.ms_s if ms not in self.infra_instances] + ['kube-dns']
        synthetic_exfil_paths, initiator_info_for_paths = \
            gen_attack_templates.generate_synthetic_attack_templates(self.mapping, app_ms_s, self.sensitive_ms,
                                                                     max_number_of_paths, self.netsec_policy,
                                                                     self.intersvc_vip_pairs, max_path_length, dns_porportion)
        self.synthetic_exfil_paths = synthetic_exfil_paths
        self.initiator_info_for_paths = initiator_info_for_paths
        return synthetic_exfil_paths, initiator_info_for_paths

    def process_pcaps(self):
        self.interval_to_filenames,self.mapping, self.infra_instances = process_pcap.process_pcap(self.experiment_folder_path, self.pcap_file, self.time_interval_lengths,
                                                                            self.exp_name, self.make_edgefiles_p, copy.deepcopy(self.mapping),
                                                                            self.cluster_creation_log, self.pcap_path, self.infra_instances)
        time_grans = [int(i) for i in self.interval_to_filenames.keys()]
        self.smallest_time_gran = min(time_grans)

    def get_exp_info(self):
        self.total_experiment_length = len(self.interval_to_filenames[str(self.smallest_time_gran)]) * self.smallest_time_gran
        print "about to return from only_exp_info section", self.total_experiment_length, self.exfil_StartEnd_times, \
            None, None
        #return total_experiment_length, self.exfil_start_time, self.exfil_end_time, self.system_startup_time
        return self.total_experiment_length, self.exfil_StartEnd_times

    ## i'm not sure if this function is needed/wanted anymore...
    def correct_attacks_labels_using_exfil_amts(self, time_gran_to_attack_labels, time_gran_to_list_of_exfil_amts):
        time_gran_to_new_attack_labels = {}
        for time_gran, attack_labels in time_gran_to_attack_labels.iteritems():
            new_attack_labels = []
            list_of_exfil_amts = time_gran_to_list_of_exfil_amts[time_gran]
            for counter in range(0, max(len(attack_labels), len(list_of_exfil_amts))):
                try:
                    label = attack_labels[counter]
                except:
                    label = 0

                print counter, label, len(attack_labels)
                if counter == 237:
                    print "take manual control here..."

                if counter < len(list_of_exfil_amts):
                    if list_of_exfil_amts[counter] == 0: # if it equals zero, then we know there isn't an actual attack
                        new_attack_labels.append(0)
                    else:
                        new_attack_labels.append(label) # otherwise go w/ existing
            time_gran_to_new_attack_labels[time_gran] = new_attack_labels

        return time_gran_to_new_attack_labels

    def calculate_values(self,end_of_training, synthetic_exfil_paths_train, synthetic_exfil_paths_test,
                         avg_exfil_per_min, exfil_per_min_variance, avg_pkt_size, pkt_size_variance,
                         calc_ide, ide_window_size, drop_infra_from_graph,
                         pretrained_min_pipeline=None, no_labeled_data=False):
        self.end_of_training = end_of_training
        if self.calc_vals or calc_ide:
            exp_length = len(self.interval_to_filenames[str(self.smallest_time_gran)]) * self.smallest_time_gran
            print "exp_length_ZZZ", exp_length, type(exp_length)
            time_gran_to_attack_labels = process_graph_metrics.generate_time_gran_to_attack_labels(
                                        self.time_interval_lengths, self.exfil_StartEnd_times, exp_length)

            # print "interval_to_filenames_ZZZ",interval_to_filenames
            for interval, filenames in self.interval_to_filenames.iteritems():
                print "interval_ZZZ", interval, len(filenames)
            for time_gran, attack_labels in time_gran_to_attack_labels.iteritems():
                print "time_gran_right_after_creation", time_gran, "len of attack labels", len(attack_labels)

            print self.interval_to_filenames, type(self.interval_to_filenames), 'stufff', self.interval_to_filenames.keys()

            # most of the parameters are kinda arbitrary ATM...
            print "INITIAL time_gran_to_attack_labels", time_gran_to_attack_labels
            ## okay, I'll probably wanna write tests for the below function, but it seems to be working pretty well on my
            # informal tests...
            end_of_training = end_of_training
            synthetic_exfil_paths = []
            for path in synthetic_exfil_paths_train + synthetic_exfil_paths_test:
                if path not in synthetic_exfil_paths:
                    synthetic_exfil_paths.append(path)

            print "synthetic_exfil_paths_train", synthetic_exfil_paths_train
            print "synthetic_exfil_paths_test", synthetic_exfil_paths_test
            print "synthetic_exfil_paths", synthetic_exfil_paths
            time_gran_to_attack_labels, time_gran_to_attack_ranges, time_gran_to_physical_attack_ranges = \
                determine_attacks_to_times(time_gran_to_attack_labels, synthetic_exfil_paths,
                                           time_of_synethic_exfil=self.time_of_synethic_exfil, end_of_train=end_of_training,
                                           synthetic_exfil_paths_train=synthetic_exfil_paths_train,
                                           synthetic_exfil_paths_test=synthetic_exfil_paths_test)
            print "time_gran_to_attack_labels", time_gran_to_attack_labels
            print "time_gran_to_attack_ranges", time_gran_to_attack_ranges
            # time.sleep(50)

            time_gran_to_exfil_paths_series = determine_time_gran_to_exfil_paths_series(
                time_gran_to_attack_ranges,
                synthetic_exfil_paths, self.interval_to_filenames,
                time_gran_to_physical_attack_ranges, self.physical_exfil_paths)

            print "time_gran_to_exfil_paths_series", time_gran_to_exfil_paths_series
            # time.sleep(50)

            # exit(200) ## TODO ::: <<<---- remove!!

            # NOTE: there might be a problem with the assignment of the labels.
            ### OKAY, this is where I'd need to add in the component that loops over the various injected exfil weights
            # OKAY, let's verify that this determine_attacks_to_times function is wokring before moving on to the next one...
            total_calculated_vals, time_gran_to_list_of_concrete_exfil_paths, time_gran_to_list_of_exfil_amts, \
            time_gran_to_new_neighbors_outside, time_gran_to_new_neighbors_dns, time_gran_to_new_neighbors_all, \
            time_gran_to_list_of_amt_of_out_traffic_bytes, time_gran_to_list_of_amt_of_out_traffic_pkts, time_gran_to_exfil_paths_series = \
                calculate_raw_graph_metrics(self.time_interval_lengths, self.interval_to_filenames, self.ms_s, self.basegraph_name,
                                            self.calc_vals, ide_window_size, self.mapping, self.make_net_graphs_p,
                                            self.infra_instances, synthetic_exfil_paths, self.initiator_info_for_paths,
                                            time_gran_to_attack_ranges, avg_exfil_per_min, exfil_per_min_variance,
                                            avg_pkt_size, pkt_size_variance, self.skip_graph_injection,
                                            self.end_of_training, self.cluster_creation_log, calc_ide,
                                            self.basefile_name, drop_infra_from_graph, self.exp_name,
                                            self.sensitive_ms, time_gran_to_exfil_paths_series, no_labeled_data)

            ## time_gran_to_attack_labels needs to be corrected using time_gran_to_list_of_concrete_exfil_paths
            ## because just because it was assigned, doesn't mean that it is necessarily going to be injected (might
            ## have to wait...)
            #time_gran_to_attack_labels = self.correct_attacks_labels_using_exfil_amts(time_gran_to_attack_labels,
            #                                                                          time_gran_to_list_of_exfil_amts)

            time_gran_to_feature_dataframe = process_graph_metrics.generate_feature_dfs(total_calculated_vals,
                                                                                        self.time_interval_lengths)

            for time_gran,feature_dataframe in time_gran_to_feature_dataframe.iteritems():
                if 'attack_labels' in feature_dataframe:
                    time_gran_to_attack_labels[time_gran] = feature_dataframe['attack_labels']

            process_graph_metrics.save_feature_datafames(time_gran_to_feature_dataframe, self.alert_file + self.sub_path,
                                                         time_gran_to_attack_labels,
                                                         time_gran_to_exfil_paths_series,
                                                         time_gran_to_list_of_concrete_exfil_paths,
                                                         time_gran_to_list_of_exfil_amts,
                                                         int(end_of_training), time_gran_to_new_neighbors_outside,
                                                         time_gran_to_new_neighbors_dns, time_gran_to_new_neighbors_all,
                                                         time_gran_to_list_of_amt_of_out_traffic_bytes,
                                                         time_gran_to_list_of_amt_of_out_traffic_pkts)

        #else:
        #if True: # due to the behavior of some later components, we actually wanna read the values from memory everytime
                 # (even if we literally have those values in memory) b/c writing/loading changes some values (in particular
                 # types) in a way that makes the latter part actually work (other option is extensive debugging, which I do
                 # not have time to do at the moment)
        time_gran_to_feature_dataframe = {}
        time_gran_to_attack_labels = {}
        time_gran_to_exfil_paths_series = {}
        time_gran_to_list_of_concrete_exfil_paths = {}
        time_gran_to_list_of_exfil_amts = {}
        time_gran_to_list_of_amt_of_out_traffic_bytes = {}
        time_gran_to_list_of_amt_of_out_traffic_pkts = {}
        time_gran_to_new_neighbors_outside, time_gran_to_new_neighbors_dns, time_gran_to_new_neighbors_all = {}, {}, {}
        min_interval = min(self.time_interval_lengths)

        for interval in self.time_interval_lengths:
            # if interval in time_interval_lengths:
            print "time_interval_lengths", self.time_interval_lengths, "interval", interval
            print "feature_df_path", self.alert_file + self.sub_path + str(interval) + '.csv'
            time_gran_to_feature_dataframe[interval] = pd.read_csv(self.alert_file + self.sub_path + str(interval) + '.csv',
                                                                   na_values='?')
            # time_gran_to_feature_dataframe[interval] = time_gran_to_feature_dataframe[interval].apply(lambda x: np.real(x))
            # this just extracts the various necessary components into seperate variables...
            print "dtypes_of_df", time_gran_to_feature_dataframe[interval].dtypes
            time_gran_to_attack_labels[interval] = time_gran_to_feature_dataframe[interval]['labels']

            time_gran_to_list_of_amt_of_out_traffic_bytes[interval] = time_gran_to_feature_dataframe[interval]['amt_of_out_traffic_bytes']
            time_gran_to_list_of_amt_of_out_traffic_pkts[interval] = time_gran_to_feature_dataframe[interval]['amt_of_out_traffic_pkts']

            try:
                time_gran_to_new_neighbors_outside[interval] = time_gran_to_feature_dataframe[interval][
                    'new_neighbors_outside']
                time_gran_to_new_neighbors_dns[interval] = time_gran_to_feature_dataframe[interval][
                    'new_neighbors_dns']
                time_gran_to_new_neighbors_all[interval] = time_gran_to_feature_dataframe[interval][
                    'new_neighbors_all']
            except:
                time_gran_to_new_neighbors_outside[interval] = [[] for i in
                                                                range(0, len(time_gran_to_attack_labels[interval]))]
                time_gran_to_new_neighbors_dns[interval] = [[] for i in
                                                            range(0, len(time_gran_to_attack_labels[interval]))]
                time_gran_to_new_neighbors_all[interval] = [[] for i in
                                                            range(0, len(time_gran_to_attack_labels[interval]))]

            time_gran_to_exfil_paths_series[interval] = time_gran_to_feature_dataframe[interval][
                'exfil_path']
            ##recover time_gran_to_list_of_concrete_exfil_paths, time_gran_to_list_of_exfil_amts
            time_gran_to_list_of_concrete_exfil_paths[interval] = time_gran_to_feature_dataframe[interval][
                'concrete_exfil_path']
            list_of_exfil_amts = []
            for counter in range(0, len(time_gran_to_feature_dataframe[interval]['exfil_weight'])):
                weight = time_gran_to_feature_dataframe[interval]['exfil_weight'][counter]
                pkts = time_gran_to_feature_dataframe[interval]['exfil_pkts'][counter]
                current_exfil_dict = {'weight': weight, 'frames': pkts}
                list_of_exfil_amts.append(current_exfil_dict)
            time_gran_to_list_of_exfil_amts[interval] = list_of_exfil_amts
            if min_interval == interval:
                print time_gran_to_feature_dataframe[interval]['is_test'], type(
                    time_gran_to_feature_dataframe[interval]['is_test'])
                self.end_of_training = time_gran_to_feature_dataframe[interval]['is_test'].tolist().index(
                    1) * min_interval

        print "about to calculate some alerts!"

        self.time_gran_to_feature_dataframe_copy = copy.deepcopy(time_gran_to_feature_dataframe)
        for time_gran, feature_dataframe in time_gran_to_feature_dataframe.iteritems():
            try:
                del feature_dataframe['exfil_path']
                del feature_dataframe['exfil_weight']
                del feature_dataframe['exfil_pkts']
                del feature_dataframe['concrete_exfil_path']
                del feature_dataframe['is_test']
            except:
                pass

            try:
                time_gran_to_feature_dataframe[time_gran] = time_gran_to_feature_dataframe[time_gran].drop(
                    columns=[u'new_neighbors_dns'])
            except:
                pass
            try:
                time_gran_to_feature_dataframe[time_gran] = time_gran_to_feature_dataframe[time_gran].drop(
                    columns=[u'new_neighbors_all '])
            except:
                pass
            try:
                time_gran_to_feature_dataframe[time_gran] = time_gran_to_feature_dataframe[time_gran].drop(
                    columns=[u'new_neighbors_outside'])
            except:
                pass
            print "feature_dataframe_columns", time_gran_to_feature_dataframe[time_gran].columns

        self.time_gran_to_feature_dataframe=time_gran_to_feature_dataframe
        self.time_gran_to_attack_labels=time_gran_to_attack_labels
        self.time_gran_to_synthetic_exfil_paths_series=time_gran_to_exfil_paths_series
        self.time_gran_to_list_of_concrete_exfil_paths  = time_gran_to_list_of_concrete_exfil_paths
        self.time_gran_to_list_of_exfil_amts=time_gran_to_list_of_exfil_amts
        self.time_gran_to_new_neighbors_outside=time_gran_to_new_neighbors_outside
        self.time_gran_to_new_neighbors_dns=time_gran_to_new_neighbors_dns
        self.time_gran_to_new_neighbors_all=time_gran_to_new_neighbors_all
        self.time_gran_to_list_of_amt_of_out_traffic_bytes = time_gran_to_list_of_amt_of_out_traffic_bytes
        self.time_gran_to_list_of_amt_of_out_traffic_pkts = time_gran_to_list_of_amt_of_out_traffic_pkts

        return self.calculate_z_scores_and_get_stat_vals(pretrained_min_pipeline)

    def calculate_z_scores_and_get_stat_vals(self, pretrained_min_pipeline):
        mod_z_score_df_basefile_name = self.alert_file + 'mod_z_score_' + self.sub_path

        for time_gran, feature_df in self.time_gran_to_feature_dataframe.iteritems():
            self.time_gran_to_feature_dataframe[time_gran] = \
                drop_useless_columns_aggreg_DF(  self.time_gran_to_feature_dataframe[time_gran]  )

        if self.calc_zscore_p:
            # note: it's not actually mod_z_score anymore, but I'm keeping the name for compatibility...
            time_gran_to_mod_zscore_df,timegran_to_transformer = process_graph_metrics.normalize_data_v2(self.time_gran_to_feature_dataframe,
                                                                                 self.time_gran_to_attack_labels,
                                                                                 self.end_of_training,
                                                                                pretrained_min_pipeline)

            # note: do NOT actually want to normalize the ide angles... or do I?? wait, is there a shifting problem??
            # or something thike that
            for time_gran, feauture_df in self.time_gran_to_feature_dataframe.iteritems():
                try:
                    time_gran_to_mod_zscore_df[time_gran]['real_ide_angles_'] = feauture_df['real_ide_angles_']
                except:
                    pass

            process_graph_metrics.save_feature_datafames(time_gran_to_mod_zscore_df, mod_z_score_df_basefile_name,
                                                         self.time_gran_to_attack_labels,
                                                         self.time_gran_to_synthetic_exfil_paths_series,
                                                         self.time_gran_to_list_of_concrete_exfil_paths,
                                                         self.time_gran_to_list_of_exfil_amts, self.end_of_training,
                                                         self.time_gran_to_new_neighbors_outside,
                                                         self.time_gran_to_new_neighbors_dns,
                                                         self.time_gran_to_new_neighbors_all,
                                                         self.time_gran_to_list_of_amt_of_out_traffic_bytes,
                                                         self.time_gran_to_list_of_amt_of_out_traffic_pkts)

            #with open(mod_z_score_df_basefile_name  + '_transformer.pickle', 'w') as g:
            #    g.write(pickle.dumps(timegran_to_transformer))
        else:
            time_gran_to_mod_zscore_df = {}
            for interval in self.time_gran_to_feature_dataframe.keys():
                time_gran_to_mod_zscore_df[interval] = pd.read_csv(
                    mod_z_score_df_basefile_name + str(interval) + '.csv', na_values='?')

            #with open(mod_z_score_df_basefile_name + '_transformer.pickle', 'r') as g:
            #    cont = g.read()
            #    timegran_to_transformer = pickle.loads(cont)

        print "analysis_pipeline about to return!"

        # no longer a thing.
        #for time_gran, mod_z_score_df in time_gran_to_mod_zscore_df.iteritems():
        #    time_gran_to_mod_zscore_df[time_gran] = mod_z_score_df.drop(mod_z_score_df.index[:self.minimum_training_window])

        # self.time_gran_to_feature_dataframe_copy, \
        return time_gran_to_mod_zscore_df, None, self.time_gran_to_feature_dataframe,\
               self.time_gran_to_synthetic_exfil_paths_series, self.end_of_training #, timegran_to_transformer

    def run_cilium_component(self, time_length, results_dir, interval_to_filename):
        #if self.cilium_component_time_lengthL
        #    time_length = self.cilium_component_time_length
        #else:
        ####self.cilium_component_time_length = time_length

        print "calling_cilium_component_now..."
        self.cilium_allowed_svc_comm = cilium_config_generator.cilium_component(time_length, self.pcap_path + self.pcap_file,
                                                                                self.cilium_component_dir,
                                                                                self.make_edgefiles_p, self.ms_s, self.mapping,
                                                                                self.cluster_creation_log,
                                                                                results_dir, interval_to_filename)
        return self.cilium_allowed_svc_comm

    def calc_cilium_performance(self, avg_exfil_per_min, exfil_var_per_min, avg_pkt_size, avg_pkt_var, cilium_allowed_svc_comm):
        time_gran_to_cil_alerts = {}
        for time_gran in self.time_interval_lengths:
            #prefix_for_inject_params = 'avg_exfil_' + str(avg_exfil_per_min) + ':' + str(
            #    exfil_var_per_min) + '_avg_pkt_' + str(avg_pkt_size) + ':' + str(
            #    avg_pkt_var) + '_'
            #self.basegraph_name = self.orig_basegraph_name + prefix_for_inject_params
            #current_set_of_graphs_loc = self.basegraph_name + 'set_of_graphs' + str(time_gran) + '.csv'
            processed_graph_loc = "/".join(self.basefile_name.split("/")[:-1]) + '/' + 'exfil_at_' + str(avg_exfil_per_min) + '/'
            current_set_of_graphs_loc = processed_graph_loc + 'set_of_graphs' + str(time_gran) + '.csv'

            with open(current_set_of_graphs_loc, mode='rb') as f:
                current_set_of_graphs_loc_contents = f.read()
                current_set_of_graphs = pickle.loads(current_set_of_graphs_loc_contents)

            cilium_alerts = current_set_of_graphs.calculate_cilium_performance(cilium_allowed_svc_comm)
            time_gran_to_cil_alerts[time_gran] = cilium_alerts
        return time_gran_to_cil_alerts

def process_one_set_of_graphs(time_interval_length, ide_window_size,
                                filenames, svcs, ms_s, mapping,  infra_instances,
                                synthetic_exfil_paths, initiator_info_for_paths, attacks_to_times,
                               collected_metrics_location, current_set_of_graphs_loc, calc_vals, out_q,
                              avg_exfil_per_min, exfil_per_min_variance, avg_pkt_size, pkt_size_variance,
                              skip_graph_injection, end_of_training, pod_creation_log, calc_ide,
                              processed_graph_loc, drop_infra_from_graph, sensitive_ms,
                              exfil_paths_series, no_labeled_data):
    print "process_one_set_of_graphs"
    #time.sleep(30)
    if calc_vals:
        if skip_graph_injection:
            with open(current_set_of_graphs_loc, mode='rb') as f:
                current_set_of_graphs_loc_contents = f.read()
                current_set_of_graphs = pickle.loads(current_set_of_graphs_loc_contents)
        else:
            current_set_of_graphs = \
                simplified_graph_metrics.set_of_injected_graphs(time_interval_length, filenames, svcs, ms_s, mapping,
                                                                infra_instances, synthetic_exfil_paths, initiator_info_for_paths,
                                                                attacks_to_times, collected_metrics_location, current_set_of_graphs_loc,
                                                                avg_exfil_per_min, exfil_per_min_variance, avg_pkt_size, pkt_size_variance,
                                                                end_of_training, pod_creation_log, processed_graph_loc,
                                                                drop_infra_from_graph, exfil_paths_series)
            current_set_of_graphs.generate_injected_edgefiles()
            current_set_of_graphs.save() # I think this is valid...

        current_set_of_graphs.calcuated_single_step_metrics(sensitive_ms)
        current_set_of_graphs.calc_serialize_metrics(no_labeled_data=no_labeled_data)

        # these relate to ide
        #''

        current_set_of_graphs.save()
        #print "hi"
    else:
        with open(current_set_of_graphs_loc, mode='rb') as f:
            current_set_of_graphs_loc_contents = f.read()
            current_set_of_graphs = pickle.loads(current_set_of_graphs_loc_contents)
            print "current_set_of_graphs.list_of_injected_graphs_loc", current_set_of_graphs.list_of_injected_graphs_loc
            print "time_granularity", current_set_of_graphs.time_granularity
            print "current_set_of_graphs.raw_edgefile_names",current_set_of_graphs.raw_edgefile_names

        current_set_of_graphs.load_serialized_metrics()

    if calc_ide:
        print "waiting for ide angles to finish...."
        current_set_of_graphs.generate_aggregate_csv()
        real_ide_angles = current_set_of_graphs.ide_calculations(True, ide_window_size)
        current_set_of_graphs.calculated_values['real_ide_angles'] = real_ide_angles

        # okay, now save all of the values coherently..
        current_set_of_graphs.calculated_values_keys = current_set_of_graphs.calculated_values.keys()
        with open(current_set_of_graphs.collected_metrics_location, 'wb') as f:  # Just use 'w' mode in 3.x
            f.write(pickle.dumps(current_set_of_graphs.calculated_values))

        # okay,now save the whole hting coherently...
        current_set_of_graphs.save()

    current_set_of_graphs.put_values_into_outq(out_q)


def calculate_raw_graph_metrics(time_interval_lengths, interval_to_filenames, ms_s, basegraph_name, calc_vals, ide_window_size,
                                mapping, make_net_graphs_p, infra_instances,synthetic_exfil_paths,
                                initiator_info_for_paths, time_gran_to_attacks_to_times,
                                avg_exfil_per_min,
                                exfil_per_min_variance, avg_pkt_size, pkt_size_variance,
                                skip_graph_injection, end_of_training, pod_creation_log, calc_ide,
                                edgefile_path, drop_infra_from_graph, exp_name, sensitive_ms,
                                time_gran_to_exfil_paths_series, no_labeled_data):
    total_calculated_vals = {}
    time_gran_to_list_of_concrete_exfil_paths = {}
    time_gran_to_list_of_exfil_amts = {}
    time_gran_to_exfil_paths_series_new = {}
    time_gran_to_new_neighbors_outside = {}
    time_gran_to_new_neighbors_dns = {}
    time_gran_to_new_neighbors_all = {}
    time_gran_to_list_of_amt_of_out_traffic_bytes = {}
    time_gran_to_list_of_amt_of_out_traffic_pkts = {}

    for time_interval_length in time_interval_lengths:
        print "analyzing edgefiles...", "timer_interval...", time_interval_length

        print "this is k8s, so using these sevices", ms_s
        svcs = ms_s
        out_q = multiprocessing.Queue()

        collected_metrics_location = basegraph_name + 'collected_metrics_time_gran_' + str(time_interval_length) + '.csv'

        processed_graph_loc = "/".join(edgefile_path.split("/")[:-1]) + '/' + 'exfil_at_' + str(avg_exfil_per_min) + '/'
        current_set_of_graphs_loc = processed_graph_loc + 'set_of_graphs' + str(time_interval_length) + '.csv'

        try:
            os.makedirs(processed_graph_loc)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        # skip_graph_injection

        args = [time_interval_length, ide_window_size,
                interval_to_filenames[str(time_interval_length)], svcs, ms_s, mapping,
                infra_instances, synthetic_exfil_paths,  initiator_info_for_paths,
                time_gran_to_attacks_to_times[time_interval_length], collected_metrics_location, current_set_of_graphs_loc,
                calc_vals, out_q, avg_exfil_per_min, exfil_per_min_variance, avg_pkt_size, pkt_size_variance,
                skip_graph_injection, end_of_training, pod_creation_log, calc_ide,
                processed_graph_loc, drop_infra_from_graph, sensitive_ms,
                time_gran_to_exfil_paths_series[time_interval_length], no_labeled_data]
        p = multiprocessing.Process(
            target=process_one_set_of_graphs,
            args=args)
        p.start()
        total_calculated_vals[(time_interval_length, '')] = out_q.get()
        list_of_concrete_container_exfil_paths = out_q.get()
        list_of_exfil_amts = out_q.get()
        new_neighbors_outside =  out_q.get()
        new_neighbors_dns =  out_q.get()
        new_neighbors_all = out_q.get()
        list_of_amt_of_out_traffic_bytes = out_q.get()
        list_of_amt_of_out_traffic_pkts = out_q.get()
        list_of_logical_exfil_paths = out_q.get()

        p.join()

        print "process returned!"
        time_gran_to_list_of_concrete_exfil_paths[time_interval_length] = list_of_concrete_container_exfil_paths
        time_gran_to_list_of_exfil_amts[time_interval_length] = list_of_exfil_amts
        time_gran_to_new_neighbors_outside[time_interval_length] = None # no longer used
        time_gran_to_new_neighbors_dns[time_interval_length] = None # no longer used
        time_gran_to_new_neighbors_all[time_interval_length] = None # no longer used
        time_gran_to_list_of_amt_of_out_traffic_bytes[time_interval_length] = list_of_amt_of_out_traffic_bytes
        time_gran_to_list_of_amt_of_out_traffic_pkts[time_interval_length] = list_of_amt_of_out_traffic_pkts
        time_gran_to_exfil_paths_series_new[time_interval_length] = list_of_logical_exfil_paths

        #total_calculated_vals.update(newly_calculated_values)
        gc.collect()
    return total_calculated_vals, time_gran_to_list_of_concrete_exfil_paths, time_gran_to_list_of_exfil_amts,\
        time_gran_to_new_neighbors_outside, time_gran_to_new_neighbors_dns, time_gran_to_new_neighbors_all,\
        time_gran_to_list_of_amt_of_out_traffic_bytes, time_gran_to_list_of_amt_of_out_traffic_pkts, \
           time_gran_to_exfil_paths_series_new

## NOTE: portion_for_training is the percentage to devote to using for the training period (b/c attacks will be injected
## into both the training period and the testing period)
def determine_attacks_to_times(time_gran_to_attack_labels, synthetic_exfil_paths, time_of_synethic_exfil,
                               end_of_train, synthetic_exfil_paths_train, synthetic_exfil_paths_test):
    time_grans = time_gran_to_attack_labels.keys()
    largest_time_gran = sorted(time_grans)[-1]
    print "LARGEST_TIME_GRAN", largest_time_gran
    print "time_of_synethic_exfil",time_of_synethic_exfil
    time_periods_attack = float(time_of_synethic_exfil) / float(largest_time_gran)
    time_gran_to_attack_ranges = {} # a list that'll correspond w/ the synthetic exfil paths
    for time_gran in time_gran_to_attack_labels.keys():
        time_gran_to_attack_ranges[time_gran] = []

    ## assign injected attacks to times here...
    ### (a) add to time_gran_to_attack_ranges... just put the existing ranges w/ 'injection' as the marker'
    time_gran_to_physical_attack_ranges = {}
    for time_gran in time_gran_to_attack_labels.keys():
        time_gran_to_physical_attack_ranges[time_gran] = determine_physical_attack_ranges(time_gran_to_attack_labels[time_gran])
        print "physical_attack_ranges", time_gran_to_physical_attack_ranges[time_gran]

    # first, let's assign for the training period...
    counter = 0
    time_gran_to_attack_labels, time_gran_to_attack_ranges = assign_attacks_to_first_available_spots(time_gran_to_attack_labels, largest_time_gran,
                                            time_periods_attack, counter, time_gran_to_attack_ranges, synthetic_exfil_paths, synthetic_exfil_paths_train,
                                             int(end_of_train/largest_time_gran))
    # second, let's assign for the testing period...
    print end_of_train, largest_time_gran
    counter = int(end_of_train/largest_time_gran) #int(math.ceil(end_of_train/largest_time_gran)) #int(math.ceil(len(time_gran_to_attack_labels[largest_time_gran]) * end_of_train - time_periods_startup))
    print "second_counter!!", counter, "attacks_to_assign",len(synthetic_exfil_paths_test), time_gran_to_attack_labels[time_gran][counter:],time_gran_to_attack_labels[time_gran][counter:].count(0)

    time_gran_to_attack_labels, time_gran_to_attack_ranges = assign_attacks_to_first_available_spots(time_gran_to_attack_labels, largest_time_gran,
                                            time_periods_attack, counter, time_gran_to_attack_ranges, synthetic_exfil_paths, synthetic_exfil_paths_test,
                                            len(time_gran_to_attack_labels[largest_time_gran]))

    # okay, so now we have the times selected for the largest time granularity... we have to make sure
    # that the other granularities agree...
    print "HIGHEST GRAN SYNTHETIC ATTACKS CHOSEN -- START MAPPING TO LOWER GRAN NOW!"
    for j in range(0, len(time_gran_to_attack_ranges[largest_time_gran])):
        for time_gran, attack_labels in time_gran_to_attack_labels.iteritems():
            if time_gran == largest_time_gran:
                continue
            attack_ranges_at_highest_gran = time_gran_to_attack_ranges[largest_time_gran]
            current_attack_range_at_highest_gran = attack_ranges_at_highest_gran[j]
            time_period_conversion_ratio = float(largest_time_gran) / float(time_gran)
            current_start_of_attack = int(current_attack_range_at_highest_gran[0] * time_period_conversion_ratio)
            current_end_of_attack = int(current_attack_range_at_highest_gran[1] * time_period_conversion_ratio)
            time_gran_to_attack_ranges[time_gran].append( (current_start_of_attack, current_end_of_attack) )
            # also, modify the attack_labels
            for z in range(current_start_of_attack, current_end_of_attack):
                # print "z",z
                attack_labels[z] = 1
    return time_gran_to_attack_labels, time_gran_to_attack_ranges, time_gran_to_physical_attack_ranges


def assign_attacks_to_first_available_spots(time_gran_to_attack_labels, largest_time_gran, time_periods_attack,
                                            counter, time_gran_to_attack_ranges, synthetic_exfil_paths,
                                            current_exfil_paths, end_time_interval_largestTimeGran):

    print 'zip'
    total_number_free_spots = time_gran_to_attack_labels[largest_time_gran][counter:end_time_interval_largestTimeGran].count(0)
    number_spots_needed = len(current_exfil_paths) * time_periods_attack +  \
                          time_gran_to_attack_labels[largest_time_gran][counter:end_time_interval_largestTimeGran].count(1)
    #extra_spots = int(total_number_free_spots - number_spots_needed - 1)
    extra_spots = total_number_free_spots - number_spots_needed - 1
    num_attacks_to_inject = float(len(current_exfil_paths))

    j = 0
    for synthetic_exfil_path in current_exfil_paths: # synthetic_exfil_paths:
        print synthetic_exfil_path, synthetic_exfil_path in current_exfil_paths

        if synthetic_exfil_path in synthetic_exfil_paths: #current_exfil_paths:
            attack_spot_found = False
            number_free_spots = time_gran_to_attack_labels[largest_time_gran][counter:].count(0)
            if number_free_spots < time_periods_attack:
                exit(1244) # should break now b/c infinite loop (note: we're not handling the case where it is fragmented...)

            random.seed(0)
            current_possible_steps = int(math.ceil(float(extra_spots) / (num_attacks_to_inject - j)))
            #current_possible_steps = int(extra_spots/10.0) #int(extra_spots)
            if current_possible_steps > extra_spots:
                current_possible_steps = extra_spots

            print "current_possible_steps", current_possible_steps, "extra_spots", extra_spots
            time_periods_between_attacks = random.randint(0, current_possible_steps)  # don't wan to bias it too much towards the end
            extra_spots -= time_periods_between_attacks
            counter += time_periods_between_attacks

            print "counter",counter,"extra_spots",extra_spots, "time_periods_between_attacks", time_periods_between_attacks
            inner_loop_counter = 0
            while not attack_spot_found:
                inner_loop_counter += 1
                potential_starting_point = int(counter)
                #print "potential_starting_point", potential_starting_point
                if potential_starting_point == 107:
                    print "take manual control..."

                attack_spot_found = exfil_time_valid(potential_starting_point, time_periods_attack,
                                                     time_gran_to_attack_labels[largest_time_gran])
                if attack_spot_found:
                    # if the time range is valid, we gotta store it...
                    time_gran_to_attack_ranges[largest_time_gran].append((int(potential_starting_point),
                                                                          int(potential_starting_point + time_periods_attack)))
                    for i in range(potential_starting_point, int(potential_starting_point + time_periods_attack)):
                        time_gran_to_attack_labels[largest_time_gran][i] = 1
                else:
                    extra_spots -= 1
                    print "spot_not_found!"

                counter += int(time_periods_attack)
        else:
            ### by making these two points the same, this value will be 'passed over' by the other functions...
            potential_starting_point = int( counter)
            time_gran_to_attack_ranges[largest_time_gran].append((potential_starting_point,potential_starting_point))
            counter += 1
        j += 1
    return time_gran_to_attack_labels, time_gran_to_attack_ranges


def determine_physical_attack_ranges(physical_attack_labels):
    ## determine the indexes of contiguous sets of 1's...
    # step 1: find indexes of all the ones (using list comprehension)
    indexes_of_attack_labels = [i for i,j in enumerate(physical_attack_labels) if j == 1]
    print "indexes_of_attack_labels", indexes_of_attack_labels
    # step 2: find contiguous size of contigous numbers
    ### a solution to this is in the docs, so let's just
    #### do it that way: https://docs.python.org/2.6/library/itertools.html#examples
    physical_attack_ranges = []
    for k, g in groupby(enumerate(indexes_of_attack_labels), lambda (i, x): i - x):
        attack_grp =  map(itemgetter(1), g) #groupby, itemgetter
        physical_attack_ranges.append((attack_grp[0], attack_grp[-1] + 1))
    #print "physical_attack_ranges", physical_attack_ranges
    return physical_attack_ranges


def determine_time_gran_to_exfil_paths_series(time_gran_to_attack_ranges, synthetic_exfil_paths,
                                              interval_to_filenames, time_gran_to_physical_attack_ranges,
                                              physical_exfil_paths):
    time_gran_to_synthetic_exfil_paths_series = {}
    for time_gran, attack_ranges in time_gran_to_attack_ranges.iteritems():
        print interval_to_filenames.keys()
        time_steps = len(interval_to_filenames[str(time_gran)])
        current_exfil_path_series = pd.Series([0 for i in range(0,time_steps)])
        print "time_gran_attack_ranges", time_gran, attack_ranges

        # first add the physical attacks
        physical_attack_ranges = time_gran_to_physical_attack_ranges[time_gran]
        for attack_counter, attack_range in enumerate(physical_attack_ranges):
            for i in range(attack_range[0], attack_range[1]):
                print "physical_exfil_path",physical_exfil_paths
                current_exfil_path_series[i] = ['physical:'] + physical_exfil_paths[attack_counter]

        # then add the injected attacks
        for attack_counter, attack_range in enumerate(attack_ranges):
            for i in range(attack_range[0], attack_range[1]):
                current_exfil_path_series[i] = synthetic_exfil_paths[attack_counter % len(synthetic_exfil_paths)]
        #current_exfil_path_series.index *= 10
        time_gran_to_synthetic_exfil_paths_series[time_gran] = current_exfil_path_series
    #print "time_gran_to_synthetic_exfil_paths_series", time_gran_to_synthetic_exfil_paths_series

    #time.sleep(60)
    return time_gran_to_synthetic_exfil_paths_series

# returns whether the range does not already have an attack at that location... so if an attack is found
# then the range is not valid (So you'd wanna return false)
def exfil_time_valid(potential_starting_point, time_slots_attack, attack_labels):
    attack_found = False
    # now check if there's not already an attack selected for that time...
    #print potential_starting_point, potential_starting_point + time_slots_attack
    for i in attack_labels[potential_starting_point:int(potential_starting_point + time_slots_attack)]:
        if i:  # ==1
            attack_found = True
            break
    return not attack_found