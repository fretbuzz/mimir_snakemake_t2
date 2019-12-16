import errno
import json
import os
import subprocess
from pcap_to_edgelists import update_mapping


def convert_tshark_stats_to_edgefile(edgefile_path, edgefile_name, tshark_path, tshark_name, make_edgefiles_p, mapping):
    if make_edgefiles_p:
        # okay, so then like print this and it should be decent
        f = open(tshark_path + tshark_name, 'r')
        lines = f.readlines()
        pair_to_attribs = {}
        for line in lines:
            if '.' in line:
                split_line = line.split(' ')
                proc_split_line = [i for i in split_line if i != '']

                try:
                    src = mapping[proc_split_line[2]][0]
                except:
                    src = proc_split_line[2]
                try:
                    dst = mapping[proc_split_line[0]][0]
                except:
                    dst = proc_split_line[0]

                # print proc_split_line[2] + ' ' + proc_split_line[0] + ' {\'packets\': ' +  proc_split_line[3]+  ', \'bytes\': ' +  proc_split_line[4] + '}'
                if (src, dst) in pair_to_attribs:
                    pair_to_attribs[(src, dst)]['frames'] += int(proc_split_line[3])
                    pair_to_attribs[(src, dst)]['bytes'] += int(proc_split_line[4])
                else:
                    pair_to_attribs[(src, dst)] = {'frames': int(proc_split_line[3]), 'bytes': int(proc_split_line[4])}

                try:
                    src = mapping[proc_split_line[0]][0]
                except:
                    src = proc_split_line[0]
                try:
                    dst = mapping[proc_split_line[2]][0]
                except:
                    dst = proc_split_line[2]

                if (src, dst) in pair_to_attribs:
                    pair_to_attribs[(src, dst)]['frames'] += int(proc_split_line[5])
                    pair_to_attribs[(src, dst)]['bytes'] += int(proc_split_line[6])
                else:
                    pair_to_attribs[(src, dst)] = {'frames': int(proc_split_line[5]), 'bytes': int(proc_split_line[6])}

        # okay, now we gotta write this to a file...
        output_file = edgefile_path + edgefile_name
        with open(output_file, 'w') as f:
            for ip_pair, attribs in pair_to_attribs.iteritems():
                #print "pairings", ip_pair[0], type(ip_pair[0]), ip_pair[1], type(ip_pair[1])
                f.write(ip_pair[0] + ' ' + ip_pair[1] +  ' {\'frames\':' + str(attribs['frames']) + ',\'weight\':' +  str(attribs['bytes']) + '}\n')
                #f.write(ip_pair[0] + ',' + ip_pair[1] + ','+ str(attribs['frames']) + ',' + str(attribs['bytes']) + '\n')
    else:
        output_file = edgefile_path + edgefile_name

    return output_file


# pcap_file -> list_of_pcap_files
# (split)
def split_pcap(path, pcap_file, out_pcap_path, out_pcap_basename, interval):
    ## Notes (3/16): the cmd below is failing, which is breaking downstream functions...

    cmd_list = ["editcap", "-i " + str(interval), path + pcap_file, out_pcap_path + out_pcap_basename]
    print "split_pcap_cmd_list", cmd_list
    out = subprocess.check_output(cmd_list)
    print out
    print "done"

    out = subprocess.check_output(['ls', out_pcap_path])
    out = out.split('\n')
    #print out

    print "let's process the filenames"
    files = []
    print "outoutout", out
    for file in out:
        print file, out_pcap_basename, out_pcap_basename  in file
        if out_pcap_basename  in file:
            files.append(file)

    # the code below makes the last entry wierd... let's just get rid of it entirely...
    files = files[:-1] # there, now its everything but the last file...

    # need to get rid of any partial packets at end of trace b/c killing tcpdump
    ## NOTE: put path in theere
    # if problems persist, move this inside the loop above... but I don't understand WHY what is going on??
    ## NOTE: NEW IDEA:: try it with file inside of files[-1]... prob still the same result tho...
    '''
    cmd_list = ["editcap", out_pcap_path + files[-1], out_pcap_path + files[-1]]
    try:
        print "editcap command", cmd_list
        out = subprocess.check_output(cmd_list)
        print out
    except:
        pass # will trigger the split packet error, but will fix it regardless
    '''
    return files

def process_pcap_via_tshark(path, pcap_name, tshark_stats_dir_path, make_edgefiles_p):
    print "zzz Processing_pcap_via_tshark..."

    # tshark -r wordpress_eight_rep_2_default_bridge_0any_COPY.pcap -z conv,ip -Q > conv_ip_info_full.txt
    cmd_list = ["tshark", "-r" + path + pcap_name, "-zconv,ip", "-Q"]
    ""
    ""
    ""
    try:
        print "cmd_list", cmd_list
        out = subprocess.check_output(cmd_list)
        out = out.split('\n')

        #print out

        if make_edgefiles_p:
            tshark_stats_file = tshark_stats_dir_path + pcap_name
            with open(tshark_stats_file, 'w') as f:
                for line in out:
                    f.write(line)
                    f.write('\n')
    except:
        print "pcap was probably cut short in middle... let's keep going though..."

    return tshark_stats_dir_path, pcap_name

def process_pcap(experiment_folder_path, pcap_file, intervals, exp_name, make_edgefiles_p, mapping,
                 cluster_creation_log, pcap_path, infra_instances, skip_to_calc_zscore, future_feature_df, smallest_time_gran):

    # first off, gotta make this new folder
    print "starting to process the pcap!"
    path_to_split_pcap_dir = experiment_folder_path + 'split_pcaps/'
    path_to_edgefile_dir = experiment_folder_path + 'edgefiles/'
    path_to_tshark_dir = experiment_folder_path + 'tshark_stats/'
    interval_to_edgefile_path = path_to_edgefile_dir + exp_name + '_edgefile_dict.txt'
    interval_to_mapping_path = path_to_edgefile_dir + exp_name + 'mapping.txt'
    interval_to_infra_path = path_to_edgefile_dir + exp_name + 'infra.txt'

    try:
        os.makedirs(path_to_split_pcap_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    try:
        os.makedirs(path_to_edgefile_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    try:
        os.makedirs(path_to_tshark_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    interval_to_files = {}
    if make_edgefiles_p:
        # unzip the pcap if necessary
        unzipped_pcap = False
        if '.pcap.gz' in pcap_file:
            unzipped_pcap = True
            out = subprocess.check_output(['gzip', '-d', pcap_path + pcap_file])
            print "unzipping output: ", out
        if (not os.path.exists(pcap_path + pcap_file) and os.path.exists(pcap_path + pcap_file + '.gz')):
            unzipped_pcap = True
            unzipping_cmds = ['gzip', '-d', pcap_path + pcap_file + '.gz']
            print "unzipping_cmds", unzipping_cmds
            out = subprocess.check_output(unzipping_cmds)
            print "unzipping output: ", out

        for interval in intervals:
            print "CURRENT INTERVAL", interval
            # wanna take ethe pcap and split the file
            split_pcaps = split_pcap(pcap_path, pcap_file, path_to_split_pcap_dir, pcap_file[:-5] + '_split',
                                     interval)

            # for each file, create the edgefiles
            edgefiles = []
            for edgefile_counter, split_pcap_file in enumerate(split_pcaps):
                tshark_stats_path, tshark_stats_file = process_pcap_via_tshark(path_to_split_pcap_dir, split_pcap_file,
                                                                               path_to_tshark_dir, make_edgefiles_p)
                edgefile_path = path_to_edgefile_dir
                edgefile_name = tshark_stats_file + '_edges.txt'

                mapping,infra_instances = update_mapping(mapping, cluster_creation_log, interval,
                                                         edgefile_counter, infra_instances=infra_instances)

                edgefile = convert_tshark_stats_to_edgefile(edgefile_path, edgefile_name, tshark_stats_path, tshark_stats_file,
                                                            make_edgefiles_p,mapping)
                edgefiles.append(edgefile)

                # okay, now I wanna delete the split pcap b/c they take up a LOT of space in aggregate...
                out = subprocess.check_output(['rm', path_to_split_pcap_dir + split_pcap_file])
                print "result of deleting split pcap file ", split_pcap_file, "...", out

            # probably wanna return a mapping of granularity to filepaths+nanes, just like I do in the current system...
            interval_to_files[str(interval)] = edgefiles

        with open(interval_to_edgefile_path, 'w') as f:
            f.write(json.dumps(interval_to_files))
        with open(interval_to_mapping_path, 'w') as f:
            f.write(json.dumps(mapping))
        with open(interval_to_infra_path, 'w') as f:
            f.write(json.dumps(infra_instances))

        # rezip the pcap if necessary
        if unzipped_pcap:
            out = subprocess.check_output(['gzip', pcap_path + pcap_file])
            print "zipping output: ", out

            # if zipping the pcaps is necessary, it also means that we don't have space to leave all the split pcaps
            # just lying around... we'll need to remove all the pcaps in the split_pcaps/ directory
            # using method from here: https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder-in-python
            for the_file in os.listdir(path_to_split_pcap_dir):
                file_path = os.path.join(path_to_split_pcap_dir, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)

    else:
        if not skip_to_calc_zscore:
            print interval_to_edgefile_path
            with open(interval_to_edgefile_path, 'r') as f:
                #f.write(json.dumps(interval_to_files))
                interval_to_files = json.load(f)
                print interval_to_files, type(interval_to_files)
            #with open(interval_to_mapping_path, 'r') as f:
            #    mapping = json.load(f)
            #with open(interval_to_infra_path, 'r') as f:
            #    infra_instances = json.load(f)
        else:
            # things get wierd when re-calculating models for data that was processed on a remote machine

            number_of_times = len(future_feature_df.index)
            print "number_of_times", number_of_times
            interval_to_files = {}
            interval_to_files[str(smallest_time_gran)] = [0 for i in range(0, number_of_times)]


    return interval_to_files, mapping, infra_instances


