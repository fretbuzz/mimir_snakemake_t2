###
'''
The purpose of this file is to run data through the data exfiltration detection pipeline.
Namely, this'll be DECANTeR at first, but hopefully also DUMONT later one.

At first, make this a STANDALONG script. Later one, we'll make it an integral part of the system.
'''

''' ## This is what needs to happen first.
DLP Pipeline (DECANTeR)
(0) clone DECANTeR directory into dlp_stuff (if it does not already exist)
(1) Generate Bro log files
    this'll involve calling bro on the relevant pcap and then storing the resulting logs in a nice locaiton
(2) "Live" Analysis:
	python2 main.py --training test-data/malware/vm7_decanter.log --testing test-data/malware/exiltration_logs/URSNIF_386.pcap.decanter.log -o 0
For DUMONT, we'd need to write a function that interacts with their
	python functions (so can't handle it all via the terminal)

Okay, well DECANTER looks actually pretty easy... Steps (1):
(1) Create new directory for this stuff
	clone appropriate github repo
(2) Setup Docker container w/ the new directory as a shared folder
	this'll be just like MULVAL
	    ## maybe this one??
	        https://github.com/blacktop/docker-bro
	        
	## docker run blacktop/bro -r /Volumes/exM/experimental_data/sockshop_info/sockshop_one_auto_mk11long/sockshop_one_auto_mk11long_bridge_any.pcap  dlp_stuff/decanter/decanter_dump_input

    ## literally just mod this expression appropriately
    docker run --rm -v `pwd`:/pcap -v `pwd`/local.bro:/usr/local/share/bro/site/local.bro blacktop/bro -r heartbleed.pcap local "Site::local_nets += { 192.168.11.0/24 }"
(3) Generate bro logs
    [phrase above will do this]
(4) Generate DECANTER output
    [should be literally just a command...]
'''
import subprocess
import os
import docker
import time
import errno
import shutil
#import dlp_stuff.decanter.bro_parser as bro_parser
import time
import pickle
import json
import numpy as np
import pandas as pd

def dlp_pipeline():
    pass

def run_decanter(training_log, testing_log):
    out = subprocess.check_output(['python', 'dlp_stuff/decanter/main.py', '-t', training_log, '-T', testing_log])
    print out

def path_to_label(path):
    label = 'physical:/'
    for part in path:
        label += part + '/'
    return label[:-1]

def run_bro(pcap_path, exp_name, gen_log_p, cloudlab):

    ############################################ OLD
    # (I don't think that we use the code in this block anymore...)
    # (I first tried for a docker-based approach, but eventually decided against that...)
    '''
    client = docker.from_env()
    client.containers.list()
    cwd = os.getcwd()
    print "cwd", cwd

    ##### TODO: might wanna parametrize this
    volume_binding = {'/Volumes/exM/experimental_data/sockshop_info/sockshop_one_auto_mk11long/': {'bind': '/pcap', 'mode': 'rw'},
     '/Users/jseverin/Documents/Microservices/munnin_snakemake_t2/mimir_v2/analysis_pipeline/dlp_stuff/decanter/decanter_dump_input.bro':
         {'bind': '/usr/local/share/bro/site/decanter_dump_input.bro', 'mode': 'rw'}}

    commands = [ ["-r", "sockshop_one_auto_mk11long_bridge_any.pcap", "decanter_dump_input"] ]
    '''
    #for command in commands:
    #    out = client.containers.run('blacktop/bro', command,  volumes=volume_binding, detach=False, auto_remove=True)
    #    print "out", out #out.output, out
    #print client.containers.list()
    ############################################## END

    outfile_loc = 'dlp_stuff/' + exp_name + '/decanter.log'

    if gen_log_p:

        os.chdir('./dlp_stuff')

        try:
            shutil.rmtree(exp_name)
        except:
            print exp_name, "folder must not exist yet...."

        print "exp_name", exp_name

        try:
            os.makedirs('./' + exp_name)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        os.chdir('./' + exp_name)

        decanter_rules_file = '../decanter/decanter_dump_input.bro'

        print "get_cwd", os.getcwd()
        print "cloudlab", cloudlab

        if cloudlab:
            bro_cmds = ['/opt/bro/bin/bro', '-r',
                        '../../' + pcap_path,
                        decanter_rules_file, '-C']
        else:
            bro_cmds = ['bro', '-r',
                        pcap_path,
                        decanter_rules_file, '-C']
        print "bro_cmds", bro_cmds, os.getcwd()
        out = subprocess.check_output(bro_cmds)
        print "subprocess_complete..."
        print "out",out
        os.chdir('..')
        os.chdir('..')

    #print container.stop()
    #print container.remove()
    #print client.containers.prune(filters={'id': container.id})

    decanter_log_file = os.getcwd() + '/' + outfile_loc

    return decanter_log_file

'''
def find_svc_behavior(path_to_log):
    bp = bro_parser.BroParser()
    log_df = bp.parseFile(path_to_log)
    log_df = log_df.T
    log_dict = log_df.to_dict()

    #print "log_dict", log_dict
    return log_dict

def process_log_dict(log_dict):
    svcpair_to_normal_behavior = {}
    for counter,vals in log_dict.iteritems():
        print "vals", vals
        src_ip = vals['id.orig_h']
        dst_ip = vals['id.resp_h']

        # TODO: convert this a svc pair at some point...
        include_vals = ['header_values', 'request_body_len', 'proxied', 'uri', 'version', 'orig_mime_types', 'method', 'mac_orig']
        header_vals = ['connection', 'content-length', 'content-type', 'host', 'accept']

        ip_pair = (src_ip, dst_ip)
        if ip_pair not in svcpair_to_normal_behavior:
            svcpair_to_normal_behavior[ip_pair] = {}
            for included_val in include_vals[1:]:
                svcpair_to_normal_behavior[ip_pair][included_val] = set()
            svcpair_to_normal_behavior[ip_pair][include_vals[0]] = {}
            for header_val in header_vals:
                svcpair_to_normal_behavior[ip_pair][include_vals[0]][header_val] = set()

        ## TODO: next dictionaries should be given the relevant values...
        for included_val in include_vals[1:]:
            svcpair_to_normal_behavior[ip_pair][included_val].add( vals[included_val]  )
        for header_val in header_vals:
            #print svcpair_to_normal_behavior[ip_pair][include_vals[0]][header_val]
            try:
                cur_val = vals[include_vals[0]][header_val]
            except:
                cur_val = ''
            svcpair_to_normal_behavior[ip_pair][include_vals[0]][header_val].add( cur_val )

        # possible vals...
        #['header_values', <- YES (all of the subtypes)
        #'uid',          <- no
        #'request_body_len', <-YES
        #'id.orig_p', <- no
        #'id.resp_h', <- no
        #'proxied',   <- YEES
        #'ts',       <- no
        #'uri',      <- YESE
        #'id.orig_h', <- no
        #'id.resp_p', <-no
        #'version',  <- YES
        #'orig_mime_types',  <- YES
        #'method', <- YES
        #'mac_orig'<- YES
        #]
        #
    return svcpair_to_normal_behavior
'''

def run_decanter_module():
    pass
    '''
    1.) Dump fingerprints in csv files.
        python2 main.py --training test-data/user/log/riccardo_linux_training_16-01.log --testing test-data/user/log/riccardo_linux_testing_18-01.log -o 1
    2.) Analyze the fingerprints.
        python2 main.py --csv ./
    '''

def main(train_pcap_path, train_exp_name, train_gen_bro_log, test_pcap_path, test_exp_name, test_gen_bro_log,
         gen_fingerprints_p, cloudlab=True):
    decanter_output_log_train = run_bro(train_pcap_path, train_exp_name, train_gen_bro_log, cloudlab)
    decanter_output_log_test = run_bro(test_pcap_path, test_exp_name, test_gen_bro_log, cloudlab)

    if gen_fingerprints_p:

        decanter_fingerprint_cmds = ['python', './dlp_stuff/decanter/main.py', '--training', decanter_output_log_train,
           '--testing', decanter_output_log_test, '-o', str(1)]

        print "cwd", os.getcwd()
        print "decanter_fingerprint_cmds: ", decanter_fingerprint_cmds

        start_time = time.time()

        out = subprocess.check_output(decanter_fingerprint_cmds)
        print "decanter out:", out
        print "decanter fingerprinting took this long to run: " + str(time.time() - start_time)

    #return 0

    print "cwd", os.getcwd()
    decanter_alert_cmds = ['python', './dlp_stuff/decanter/main.py', '--csv', './'] # crashes here in debug...
    print "calculating decanter alerts... "
    out = subprocess.check_output(decanter_alert_cmds, cwd=os.getcwd())
    print "decanter alert outputs..\n", out

    print "-----"

    path_to_alerts = './'
    if not cloudlab:
        path_to_alerts = './dlp_stuff/decanter/'
    with open(path_to_alerts + 'timestamps_with_alerts_training.txt', 'r') as f:
        training_alert_timestamps = pickle.load(f)
    with open(path_to_alerts + 'timestamps_with_alerts_testing.txt', 'r') as f:
        testing_alert_timestamps = pickle.load(f)

    return training_alert_timestamps, testing_alert_timestamps

def compute_performance_metrics(experiment_json):
    ## UPDATE: honestly, I'll need to make paired-down versions of both of the functions below

    # TODO: these imports will need to be parsed from the config used in parse_experimental_config
    # (but will need to make a new function for this!!)

    pass

def parse_experimental_data_json_dlp(config_file, experimental_folder, experiment_name, make_edgefiles,
                                     time_interval_lengths, pcap_file_path, pod_creation_log_path):
    with open(config_file) as f:
        config_file = json.load(f)

        base_experiment_dir =  experimental_folder + experiment_name + '/'

        '''
        basefile_name = experimental_folder + experiment_name + '/edgefiles/' + experiment_name + '_'
        basegraph_name = experimental_folder + experiment_name + '/graphs/' + experiment_name + '_'
        alert_file = experimental_folder + experiment_name + '/alerts/' + experiment_name + '_'

        try:
            sec_between_exfil_pkts = config_file["exfiltration_info"]['sec_between_exfil_pkts']
        except:
            sec_between_exfil_pkts = 1.0
        pod_creation_log = [ pod_creation_log_path + config_file['pod_creation_log_name']]
        sensitive_ms = config_file["exfiltration_info"]['sensitive_ms']
        '''

        pcap_paths = [ pcap_file_path + config_file['pcap_file_name'] ]

        try:
            physical_exfil_p = config_file["exfiltration_info"]["physical_exfil_performed"]
        except:
            physical_exfil_p = False

        try:
            if physical_exfil_p:
                exfil_StartEnd_times = config_file["exfiltration_info"]['exfil_StartEnd_times']
            else:
                exfil_StartEnd_times = [[]]
        except:
            exfil_StartEnd_times = [[]]

        try:
            if physical_exfil_p:
                physical_exfil_paths = config_file["exfiltration_info"]['exfil_paths']
            else:
                physical_exfil_paths = [[]]
        except:
            physical_exfil_paths = [[]]

    return physical_exfil_p, exfil_StartEnd_times, physical_exfil_paths[:len(exfil_StartEnd_times)], pcap_paths, base_experiment_dir


def parse_experimental_config_dlp(experimental_config_file, add_dropInfo_to_name=True):
    with open(experimental_config_file) as f:
        config_file = json.load(f)

        if 'time_interval_lengths' in config_file:
            time_interval_lengths = config_file['time_interval_lengths']
        else:
            time_interval_lengths = [10, 60]

        if 'avg_exfil_per_min' in config_file:
            avg_exfil_per_min = config_file['avg_exfil_per_min']
        else:
            avg_exfil_per_min = [10.0, 1.0, 0.1, 0.05, 0.01]

        if 'exfil_per_min_variance' in config_file:
            exfil_per_min_variance = config_file['exfil_per_min_variance']
        else:
            exfil_per_min_variance = [0.3, 0.15, 0.025, 0.0125, 0.0025]

        if 'avg_pkt_size' in config_file:
            avg_pkt_size = config_file['avg_pkt_size']
        else:
            # note: this literally has no effect on the system, so don't worry about it
            avg_pkt_size = [500.0 for i in range(0,len(avg_exfil_per_min))]

        if 'pkt_size_variance' in config_file:
            pkt_size_variance = config_file['pkt_size_variance']
        else:
            # note: this literally has no effect on the system, so don't worry about it
            pkt_size_variance = [100 for i in range(0,len(avg_exfil_per_min))]

        BytesPerMegabyte = 1000000
        avg_exfil_per_min = [BytesPerMegabyte * i for i in avg_exfil_per_min]
        exfil_per_min_variance = [BytesPerMegabyte * i for i in exfil_per_min_variance]

        cur_experiment_name = config_file['cur_experiment_name']

        exp_config_file = config_file['exp_config_file']
        if 'experimental_folder' in config_file:
            experimental_folder = config_file['experimental_folder']
        else:
            experimental_folder = "/".join(exp_config_file.split('/')[:-1]) + "/"

        if 'base_output_location' in config_file:
            base_output_location = config_file['base_output_location']
        else:
            base_output_location = experimental_folder + 'results/'

        try:
            os.makedirs(base_output_location)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        if 'drop_infra_from_graph' in config_file:
            drop_infra_from_graph = config_file['drop_infra_from_graph']
        else:
            drop_infra_from_graph = True

        if drop_infra_from_graph:
            if add_dropInfo_to_name: # exists for compatibility reasons
                cur_experiment_name += 'dropInfra'

        base_output_location += cur_experiment_name

        make_edgefiles = config_file['make_edgefiles']

        if 'pcap_file_path' in config_file:
            pcap_file_path = config_file['pcap_file_path']
        else:
            pcap_file_path = experimental_folder

        if 'pod_creation_log_path' in config_file:
            pod_creation_log_path = config_file["pod_creation_log_path"]
        else:
            pod_creation_log_path = experimental_folder

        if 'time_interval_lengths' in config_file:
            time_interval_lengths = config_file['time_interval_lengths']
        else:
            time_interval_lengths = [10, 60]

    physical_exfil_p, exfil_StartEnd_times, physical_exfil_paths, pcap_paths, base_experiment_dir  = \
        parse_experimental_data_json_dlp(exp_config_file, experimental_folder, cur_experiment_name, make_edgefiles,
                                         time_interval_lengths, pcap_file_path, pod_creation_log_path)

    # assuming only one PCAP...
    return physical_exfil_p, exfil_StartEnd_times, physical_exfil_paths, pcap_paths[0], base_experiment_dir, \
           avg_exfil_per_min, cur_experiment_name, time_interval_lengths

def end_to_end_microservice(train_experimental_config, test_experimental_config, train_gen_bro_log, test_gen_bro_log,
                            gen_fingerprints_p):
    # okay, so what do I wan to do here... well, I want to (1) get the pcap files for the train/test data,
    # (2) get the alert times, (3) match the alert times with the exfil_startEnd_times
    # PROBLEM: there isn't a "granularity" specification... (it's going to need to be brought in as a param, I guess...)

    _,train_exfil_periods, training_exfil_paths, train_pcap_path,_,_,train_exp_name,alert_granularities = parse_experimental_config_dlp(train_experimental_config)

    _,test_exfil_periods, testing_exfil_paths, test_pcap_path,_,_,test_exp_name,alert_granularities = parse_experimental_config_dlp(test_experimental_config)

    training_alert_timestamps, testing_alert_timestamps = main(train_pcap_path, train_exp_name, train_gen_bro_log,
                                                               test_pcap_path, test_exp_name, test_gen_bro_log,
                                                               gen_fingerprints_p, cloudlab=True)

    # TODO: okay, it is now time to implement the missing component... actually translating the alerts into performance
    # figures
    start_time, end_time = get_pcap_start_and_end_times(train_pcap_path)
    training_performance = calculate_performance_metrics(training_alert_timestamps, train_exfil_periods, min(alert_granularities),
                                                         start_time, end_time, training_exfil_paths)

    start_time, end_time = get_pcap_start_and_end_times(test_pcap_path)

    testing_performance_vals = {}
    for alert_granularity in alert_granularities:
        testing_performance_vals[alert_granularity] = calculate_performance_metrics(testing_alert_timestamps, test_exfil_periods, alert_granularity,
                                                         start_time, end_time, testing_exfil_paths)


    print "testing_performance_vals", testing_performance_vals

    return testing_performance_vals

def calculate_performance_metrics(alert_timestamps, exfil_periods, alert_granularity, start_time, end_time, exfil_paths):
    # (1) convert alerts to the appropriate granularity
    #time_period_to_alerts = {} # INDEX IS FOR THE START!!!
    #current_time = start_time
    #while (current_time + alert_granularity) <= end_time:
    alert_granularity, start_time, end_time = float(alert_granularity), float(start_time), float(end_time)
    print "alert_timestamps", type(alert_timestamps), alert_timestamps
    #print "float(end_time - start_time) / alert_granularity)", float(end_time - start_time) / alert_granularity, \
    #    type(float(end_time - start_time) / alert_granularity)
    hist, bin_edges = np.histogram(alert_timestamps, bins= int(float(end_time - start_time) / alert_granularity))
    exp_time = end_time - start_time
    bin_edges = [bin_edge * exp_time for bin_edge in bin_edges]
    print "hist", hist, len(hist)
    print "bin_edges", bin_edges, len(bin_edges)
    print "exfil_periods", exfil_periods, len(exfil_periods)

    # (2) calculate the number of periods covered / missed
    results_df_columns = ('tn', 'fp', 'fn', 'tp', 'exfil_weights')
    results_df = pd.DataFrame({}, columns=results_df_columns, index=['No Attack'])
    print "orig_exfil_paths", exfil_paths
    for exfil_path in exfil_paths:
        index_for_row = [path_to_label(exfil_path)] #[tuple(exfil_path)]
        print 'index_for_row', index_for_row
        print 'index_for_row', index_for_row
        new_row = pd.DataFrame({}, columns=results_df_columns, index=index_for_row)
        results_df = results_df.append(new_row)

    results_df.fillna(0, inplace=True)
    print 'empty_results_df', results_df

    tp,fp,tn,fn = 0,0,0,0
    for index, bin_edge in enumerate(bin_edges[:-1]):
        if hist[index] > 0:
            # there was an alert!
            cur_exfil_path =  is_in_exfil_period(exfil_periods, bin_edge, alert_granularity, exfil_paths)
            if cur_exfil_path:
                # true positive!
                #tp += 1
                results_df['tp'][[path_to_label(exfil_path)]] += 1
            else:
                # false positive!
                #fp += 1
                results_df['fp']["No Attack"] += 1
        else:
            # there was not an alert!
            cur_exfil_path = is_in_exfil_period(exfil_periods, bin_edge, alert_granularity, exfil_paths)
            if cur_exfil_path:
                # false negative!
                #fn += 1
                results_df['fn'][path_to_label(cur_exfil_path)] += 1
            else:
                # true negative!
                tn += 1
                results_df['tn']['No Attack'] += 1

    # (3) use this is to calculate the rest of our performance metrics
    #### [at least at the moment, I do not think that this is needed]

    '''
        testing_performance_vals[alert_granularity] = pd.DataFrame(testing_performance, index=[0])
    testing_performance_vals[alert_granularity]['exfil_weights'] = [[]]
    '''

    results_dict = {}
    results_dict['tp'] = tp
    results_dict['fp'] = fp
    results_dict['tn'] = tn
    results_dict['fn'] = fn

    return results_dict

    # (4) [not exactly in this function] call this function from generate_paper_graphs and modify the config files appropriately
    # (5) [not here either] add any additional dependencies to the cloudlab setup scripts

def is_in_exfil_period(exfil_periods, bin_edge, alert_granularity, exfil_paths):
    for exfil_counter, exfil_period in enumerate(exfil_periods):
        if exfil_period == []:
            continue
        #print "cur_exfil_period", exfil_period
        exfil_start, exfil_end = exfil_period
        ''' # eh, this code block may or may not be correct, but it's certainly bad either way
        if bin_edge >= exfil_start and (bin_edge+alert_granularity) <= exfil_end:
            return True
        elif bin_edge >= exfil_start and (bin_edge + alert_granularity) > exfil_end:
            return True
        elif bin_edge < exfil_start and (bin_edge + alert_granularity) > exfil_start:
            return True
        elif bin_edge <= exfil_start and (bin_edge + alert_granularity) >= exfil_end:
            return True
        '''

        if exfil_start < (bin_edge + alert_granularity) and exfil_end > bin_edge:
            return exfil_paths[exfil_counter]

    return None

def get_pcap_start_and_end_times(pcap_path):
    print "here we go...."

    out = subprocess.check_output(['capinfos', pcap_path, '-S'])
    print out

    out = out.split('\n')
    relevant_lines = []
    for line in out:
        if 'time:' in line:
            relevant_lines.append( line )

    print "relevant_lines", relevant_lines

    start,end = None,None
    for line in relevant_lines:
        type_of_time,the_time = line.split('time:')
        if 'First' in type_of_time:
            start = the_time
        elif 'Last' in type_of_time:
            end = the_time

    print "start,end", start,end

    return start,end


if __name__=="__main__":
    print "DLP_PIPELINE RUNNING..."

    #get_pcap_start_and_end_times('/Users/jseverin/Documents/sockshop_four_100_physical_bridge_any.pcap')

    #'''
    train_gen_bro_log = True # generate bro logs for the training data?
    #train_exp_name = 'sockshop_four_100_physical'
    #train_pcap_path = '/Users/jseverin/Documents/sockshop_four_100_physical_bridge_any.pcap'
    train_exp_name = 'sockshop_four_100_mk2_physical'
    train_pcap_path = '/Users/jseverin/Documents/sockshop_four_100_mk2_physical_bridge_any.pcap'
    test_gen_bro_log =  True # generate bro logs for the testing data?
    test_pcap_path = train_pcap_path # just for testing...
    test_exp_name = train_exp_name # just for testing...
    gen_fingerprints_p = True
    alert_granularities = [5, 60]
    # TODO: put back in!
    #test_path_to_exp_dir = '/Volumes/exM/experimental_data/sockshop_info/sockshop_one_auto_mk13/'
    #test_exp_name = 'sockshop_one_auto_mk13'
    cloudlab=False

    # TODO: test this and maybe make a CLI or something...
    main(train_pcap_path, train_exp_name, train_gen_bro_log, test_pcap_path, test_exp_name,
         test_gen_bro_log, gen_fingerprints_p, cloudlab)
    #'''
    #find_svc_behavior('dlp_stuff/' + exp_name + '/decanter.log')
    # honestly ^^ not sure why this code exists....

    # TODO: I don't think that DECANTER is ever actually called here... that needs to be fixed!!!!
    # not exactly sure what the problem is... can't I just call the main decanter cmd-line call on
    # (it just needs train and test bro logs...) this should be easy enough... probably not worth
    # fully integrating this into the MIMIR pipeline... just find the relevant parts and do it...
    # PLAN: get some PCAPs and then this should be straight-foward to test (actually I already got them...)
    # once it is working locally, can then make script to add dependencies and add it to the