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

def dlp_pipeline():
    pass

def run_decanter(training_log, testing_log):
    out = subprocess.check_output(['python', 'dlp_stuff/decanter/main.py', '-t', training_log, '-T', testing_log])
    print out

def run_bro(path_to_exp_dir, exp_name, gen_log_p):

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
        bro_cmds = ['bro', '-r',
                                       path_to_exp_dir + exp_name + '_bridge_any.pcap',
                                       decanter_rules_file, '-C']
        print "bro_cmds", bro_cmds
        out = subprocess.check_output(bro_cmds)
        print "out",out
        os.chdir('..')
        os.chdir('..')

    #print container.stop()
    #print container.remove()
    #print client.containers.prune(filters={'id': container.id})

    decanter_log_file = os.getcwd() + '/' + outfile_loc

    return decanter_log_file

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

        ''' possible vals...
        ['header_values', <- YES (all of the subtypes)
        'uid',          <- no
        'request_body_len', <-YES
        'id.orig_p', <- no
        'id.resp_h', <- no
        'proxied',   <- YEES
        'ts',       <- no
        'uri',      <- YESE
        'id.orig_h', <- no
        'id.resp_p', <-no
        'version',  <- YES
        'orig_mime_types',  <- YES
        'method', <- YES
        'mac_orig'<- YES
        ]
        '''
    return svcpair_to_normal_behavior

def run_decanter_module():
    pass
    '''
    1.) Dump fingerprints in csv files.
        python2 main.py --training test-data/user/log/riccardo_linux_training_16-01.log --testing test-data/user/log/riccardo_linux_testing_18-01.log -o 1
    2.) Analyze the fingerprints.
        python2 main.py --csv ./
    '''

def main(train_path_to_exp_dir, train_exp_name, train_gen_bro_log, test_path_to_exp_dir, test_exp_name, test_gen_bro_log,
         gen_fingerprints_p):
    decanter_output_log_train = run_bro(train_path_to_exp_dir, train_exp_name, train_gen_bro_log)
    decanter_output_log_test = run_bro(test_path_to_exp_dir, test_exp_name, test_gen_bro_log)

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

    decanter_alert_cmds = ['python', './dlp_stuff/decanter/main.py', '--csv', './']
    print "calculating decanter alerts... "
    out = subprocess.check_output(decanter_alert_cmds)
    print "decanter alert outputs..\n", out

    print "-----"

    with open('./dlp_stuff/decanter/' + 'timestamps_with_alerts_training.txt') as f:
        training_alert_timestamps = pickle.load(f)
    with open('./dlp_stuff/decanter/' + 'timestamps_with_alerts_testing.txt') as f:
        testing_alert_timestamps = pickle.load(f)

    return training_alert_timestamps, testing_alert_timestamps

if __name__=="__main__":
    print "DLP_PIPELINE RUNNING..."

    train_gen_bro_log = False # generate bro logs for the training data?
    train_exp_name = 'sockshop_four_100_physical'
    train_path_to_exp_dir = '/Users/jseverin/Documents/'
    test_gen_bro_log =  False # generate bro logs for the testing data?
    test_path_to_exp_dir = train_path_to_exp_dir # just for testing...
    test_exp_name = train_exp_name # just for testing...
    gen_fingerprints_p = True
    # TODO: put back in!
    #test_path_to_exp_dir = '/Volumes/exM/experimental_data/sockshop_info/sockshop_one_auto_mk13/'
    #test_exp_name = 'sockshop_one_auto_mk13'

    '''  # TODO: test this and maybe make a CLI or something...
    '''
    main(train_path_to_exp_dir, train_exp_name, train_gen_bro_log, test_path_to_exp_dir, test_exp_name,
         test_gen_bro_log, gen_fingerprints_p)


    #find_svc_behavior('dlp_stuff/' + exp_name + '/decanter.log')
    # honestly ^^ not sure why this code exists....

    # TODO: I don't think that DECANTER is ever actually called here... that needs to be fixed!!!!
    # not exactly sure what the problem is... can't I just call the main decanter cmd-line call on
    # (it just needs train and test bro logs...) this should be easy enough... probably not worth
    # fully integrating this into the MIMIR pipeline... just find the relevant parts and do it...
    # PLAN: get some PCAPs and then this should be straight-foward to test (actually I already got them...)
    # once it is working locally, can then make script to add dependencies and add it to the