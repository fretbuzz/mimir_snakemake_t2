'''
This file differs from multi_experiment_looper.py in that multi_experiment_looper.py has a bunch of custom instructions
and stuff for running the experiments, while this file literally just calls the e2e scripts and recovers data, etc.

All of this could be done by hand fairly easily, but I'm automating it because it's easier that way
'''
import argparse
import json
from collections import OrderedDict
import pwnlib.tubes.ssh
from pwn import *
import time
import pysftp

def parse_config(config_file_pth):
    with open(config_file_pth, 'r') as f:
        config_file = json.loads(f.read(), object_pairs_hook=OrderedDict)

        machine_ip = config_file["machine_ip"]
        e2e_script_to_follow = config_file["e2e_script_to_follow"]
        corresponding_local_directory = config_file["corresponding_local_directory"]
        remote_server_key = config_file["remote_server_key"]
        user = config_file["user"]

    return machine_ip, e2e_script_to_follow, corresponding_local_directory, remote_server_key, user

def sendline_and_wait_responses(sh, cmd_str, timeout=5, extra_rec=False):
    sh.sendline(cmd_str)
    if extra_rec:
        sh.recvline()
    line_rec = 'start'
    while line_rec != '':
        line_rec = sh.recvline(timeout=timeout)
        print("recieved line", line_rec)

def upload_data_to_remote_machine(sh, s, sftp, local_directory):
    # okay, what do I want to do here?
    # just upload each file in the local directory to the remote device
    # (is going to be kinda hard to test at the moment, because the data isn't setup like this...)

    sendline_and_wait_responses(sh, 'ls')
    clear_dir_cmd = "sudo rm -rf /mydata/mimir_v2/experiment_coordinator/experimental_data/;"
    #0clear_dir_cmd = "rm -rf /mydata/mimir_v2/experiment_coordinator/experimental_data/;"
    print "clear_dir_cmd", clear_dir_cmd
    sendline_and_wait_responses(sh, clear_dir_cmd, timeout=5)

    create_dir_cmd = "mkdir /mydata/mimir_v2/experiment_coordinator/experimental_data/"
    print "create_dir_cmd", create_dir_cmd
    sendline_and_wait_responses(sh, create_dir_cmd)

    #exit(1)

    for subdir, dirs, files in os.walk(local_directory):
        print "subdir", subdir #, subdir[-1]
        cur_dir = "/mydata/mimir_v2/experiment_coordinator/experimental_data/" + subdir.split('/')[-1]
        create_dir_cmd = "mkdir " + cur_dir
        print "create_dir_cmd", create_dir_cmd
        sendline_and_wait_responses(sh, create_dir_cmd)

        for file in files:
            cur_file = os.path.join(subdir, file)
            print cur_file
            # we want to upload every file in this directory (but NOT the subdirectories!)
            print "cur_file (local)",cur_file, "cur remote file ", cur_dir + '/' + file
            #s.upload(cur_file, remote=cur_dir + '/' + file)

            #sftp_upload_cmd = 'put ' + cur_file + ' ' + cur_dir + '/' + file
            #sendline_and_wait_responses(sftp, sftp_upload_cmd)
            '''
            # TODO: let's add some code with zipping and unzipping the file in question...
            # first let's compress the file
            tar_cmds = ['tar', '-czvf', cur_file + '.gz', cur_file]
            print "tar-ing the file...", tar_cmds
            # if tar-ed file does NOT already exist, then create it...
            if not os.path.exists(cur_file + '.gz') and '.gz' not in cur_file:
                tar_out = subprocess.check_output(tar_cmds)
                print "tar_out", tar_out

            # then let's send the zipped file
            print "uploading tar-ed file..."
            with sftp.cd(cur_dir + '/'):  # temporarily chdir to public
                sftp.put(cur_file + '.gz')  # upload file to public/ on remote

            # finally, let's unzip the zipped file
            unzip_cmd = 'tar -xvf ' + cur_dir + '/' + cur_file.split('/')[-1] + '.gz'
            print "unzip_cmd", unzip_cmd
            sendline_and_wait_responses(sh, unzip_cmd)
            '''

    print "all done uploading files! (hopefully it worked, because I haven't actually tested this function!)"

def retrieve_relevant_files_from_cloud(sh, s, sftp, local_directory, data_was_uploaded=False, machine_ip=None):
    # this function needs to grab the relevant files that were generated on the remote device and download them
    # so that they are available locally...

    # okay, so the easiest way to do this is to grab everything in the blahblahblah/experimental_data folder
    # however, then I'm grabbing all kinds of things that I don't want/need...

    # okay, step (1): get the subdirectories...
    get_subdirs_cmd = 'ls -d /mydata/mimir_v2/experiment_coordinator/experimental_data/*'
    print "get_subdirs_cmd", get_subdirs_cmd
    sh.sendline(get_subdirs_cmd)
    subdirs = []
    line_rec = 'blahblahblah'
    while line_rec != '':
        line_rec = sh.recvline(timeout=2)
        #print "line_rec", line_rec
        listed_subdirs = line_rec.split(' ')[1:]
        listed_subdirs = [potential_subdir.rstrip().lstrip() for potential_subdir in listed_subdirs if potential_subdir != '' and \
                          potential_subdir != '/mydata/mimir_v2/experiment_coordinator/experimental_data/']
        subdirs.extend(listed_subdirs)

    # step (1.5): if local directory doesn't exist, then make it!
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)

    print "subdirs", subdirs

    # then, step (2): recover the relevant files from each subdirectory
    for subdir in subdirs:
        cur_subdir = subdir #"/mydata/mimir_v2/experiment_coordinator/experimental_data/" + subdir
        get_files_in_subdir = "ls -p " + cur_subdir + " | grep -v /"

        sh.sendline(get_files_in_subdir)
        files_in_subdir = []
        line_rec = 'blahblahblah'
        while line_rec != '':
            line_rec = sh.recvline(timeout=2)
            if line_rec != '':
                files_in_subdir.append(line_rec.replace('$','').strip())
        print "files_in_subdir", files_in_subdir

        # step (2.5): if local directory for the current experiment does not exist, make it!
        if cur_subdir[-1] != '/':
            cur_subdir += '/'
        if local_directory[-1] != '/':
            local_directory += '/'
        cur_local_subdir = local_directory + subdir.split('/')[-1]
        print "cur_local_subdir",cur_local_subdir
        if not os.path.exists(cur_local_subdir):
            os.makedirs(cur_local_subdir)

        # Step (3): grab the files generated during the processing
        # note: if data was uploaded, then we don't need to recover the pcap/config files
        if not data_was_uploaded:
            for file_in_subdir in files_in_subdir:
                cur_file = cur_subdir + file_in_subdir
                print "cur_file", cur_file
                cur_local_file = cur_local_subdir + '/' + file_in_subdir
                print "cur_local_file", cur_local_file
                s.download(file_or_directory=cur_file, local=cur_local_file)
                #sendline_and_wait_responses(sftp, "get " + cur_file + " " + cur_local_file)
        # we should always recover the actual results...
        # step (4): make sure there's a nested results subdirectory
        try:
            cur_subdir += 'results/'
            cur_local_subdir = cur_local_subdir + '/results/'
            if not os.path.exists(cur_local_subdir):
                os.makedirs(cur_local_subdir)
            # step (4.5): get a list of all the generated results files
            get_files_in_subdir = "ls -p " + cur_subdir + " | grep -v /"
            sh.sendline(get_files_in_subdir)
            files_in_subdir = []
            line_rec = 'blahblahblah'
            while line_rec != '':
                line_rec = sh.recvline(timeout=2)
                if line_rec != '':
                    files_in_subdir.append(line_rec.replace('$', '').strip())        # step (5): retrieve all the generated results
            for file_in_subdir in files_in_subdir:
                cur_file = cur_subdir + file_in_subdir
                s.download(file_or_directory=cur_file, local=cur_local_subdir + file_in_subdir)
                #sendline_and_wait_responses(sftp, "get " + cur_file + " " + cur_local_subdir + file_in_subdir)

        except:
            print "results are not present for " + subdir + ' on ' + str(machine_ip)

    # TODO: recover the graphs generated by generate_paper_graphs here...
    # (okay, we need to actually test this tho...)
    dir_with_exp_graphs = '/mydata/mimir_v2/analysis_pipeline/multilooper_outs/'
    s.download(file_or_directory=dir_with_exp_graphs, local=local_directory)


def run_experiment(config_file_pth, only_retrieve, upload_data, only_process):
    # step 1: parse the config file
    machine_ip, e2e_script_to_follow, corresponding_local_directory, remote_server_key, user = parse_config(config_file_pth)

    # step 2: create ssh session on the remote device
    s = None
    while s == None:
        try:
            s = pwnlib.tubes.ssh.ssh(host=machine_ip,
                keyfile=remote_server_key,
                user=user)
        except:
            time.sleep(60)
    sh = s.run('sh')
    sh_screen = s.run('nice -11 screen -U')
    print "shell on the remote device is started..."

    # step 3: call the preliminary commands that sets up the shell correctly
    prelim_commands = "cd /mydata; export MINIKUBE_HOME=/mydata; " \
    "sudo chown jsev /mydata; " \
    "git clone https://github.com/fretbuzz/mimir_v2.git; "\
    "cd ./mimir_v2/experiment_coordinator/; " \
    "PATH=$PATH:/opt/bro/bin/; " \
    "sudo chown -R $USER $MINIKUBE_HOME/.minikube; \
    sudo chown -R $USER $HOME/.config; ls"

    sendline_and_wait_responses(sh, prelim_commands, timeout=5)
    sendline_and_wait_responses(sh_screen, prelim_commands, timeout=5)

    sftp =  pysftp.Connection(machine_ip, username=user, private_key=remote_server_key)
    #sftp = None
    ''''
    sftp_command = ['sftp', '-i', '~/Dropbox/cloudlab.pem', user + '@' + machine_ip]
    print "sftp_command", sftp_command
    sftp = pwnlib.tubes.ssh.process(sftp_command)
    print "sftp.recvline()", sftp.recvline()
    print "sftp.recvline() (2nd)", sftp.recvline(timeout=5)
    '''

    if not only_retrieve:
        # step 4: call the actual e2e script
        # if necessary, bypass pcap/data collection in the e2e script
        e2e_script_start_cmd = ". ../configs_to_reproduce_results/e2e_repro_scripts/" + e2e_script_to_follow
        if upload_data and not only_process:
            print "uploading_data...."
            upload_data_to_remote_machine(sh, s, sftp, corresponding_local_directory)
        if upload_data or only_process:
            e2e_script_start_cmd += ' --skip_pcap'
            #sftp.write('exit')
        print "e2e_script_start_cmd", e2e_script_start_cmd
        e2e_script_start_cmd += '; exit'
        #exit(2)
        sendline_and_wait_responses(sh_screen, e2e_script_start_cmd, timeout=5400)

    #return ## TODO<--- remove this in the future!!!

    # Step 5: Pull the relevant data to store locally
    # NOTE: what should be pulled depends on what (if anything) was uploaded
    #print "start sftp..."
    #sftp = pwnlib.tubes.ssh.process(['sftp', '-i', '~/Dropbox/cloudlab.pem', 'jsev@c240g5-110215.wisc.cloudlab.us'])
    retrieve_relevant_files_from_cloud(sh, s, None, corresponding_local_directory,
                                       data_was_uploaded=(upload_data and not only_process), machine_ip=machine_ip)
    #sftp.write('exit')

    # Step 6: maybe run the log file checker to make sure everything is legit?
    # TODO (what it says above): do this at a later point in time (there's already a ticket on the kanban board)
if __name__=="__main__":
    print "RUNNING"

    parser = argparse.ArgumentParser(description='This can run multiple experiments in a row on MIMIR. Also makes graphs')
    parser.add_argument('--config_json', dest='config_json', default=None,
                        help='this is the configuration file used to run to loop through several experiments')
    parser.add_argument('--only_retrieve', dest='only_retrieve',
                        default=False, action='store_true',
                        help='Does no computing activites on the remote host-- only downloads files')
    parser.add_argument('--upload_data', dest='upload_data',
                        default=False, action='store_true',
                        help='Should it upload the pcaps instead of generating them')
    parser.add_argument('--only_process', dest='only_process',
                        default=False, action='store_true',
                        help='Do not generator or upload pcaps-- only process pcaps *already* on the device')
    args = parser.parse_args()

    if not args.config_json:
        #config_file_pth = "./remote_experiment_configs/sockshop_scale_trial_1.json"
        #config_file_pth = "./remote_experiment_configs/sockshop_scale_take1.json"
        #config_file_pth = "./remote_experiment_configs/hipsterStore_scale_take1.json"

        #config_file_pth = "./remote_experiment_configs/trials/sockshop_scale_trial_1_rep1.json"
        #config_file_pth = "./remote_experiment_configs/trials/sockshop_scale_trial_1_rep2.json"
        #config_file_pth = "./remote_experiment_configs/trials/sockshop_scale_trial_1_rep3.json"

        #config_file_pth = "./remote_experiment_configs/sockshop_scale_newRepro.json"

        config_file_pth = "./remote_experiment_configs/wordpress_scale_trail_1.json"
    else:
        config_file_pth = args.config_json

    run_experiment(config_file_pth, args.only_retrieve, args.upload_data, args.only_process)