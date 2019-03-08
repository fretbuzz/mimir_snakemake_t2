import subprocess
import argparse
import sys
import os
import time
import pickle

def pod_logger(log_file_loc, sentinal_file_loc):
    timestep_counter = 0
    last_timestep_mapping = {}
    time_step_to_changes = {}
    while (not os.path.exists(sentinal_file_loc)):
        #print "current_loop: ", timestep_counter
        current_mapping = {}
        out = subprocess.check_output(['kubectl', 'get', 'po', '-o','wide', '--all-namespaces'])

        for line in out.split('\n')[1:]:
            if line != '':
                line = [i for i in line.split('   ') if i != '']
                name,ip = line[1],line[-2]
                if '<none>' not in ip:
                    current_mapping[name] = ip

        # now compare to old mapping
        changes_this_time_step = {}
        for cur_name, cur_ip in current_mapping.iteritems():
            if cur_name not in last_timestep_mapping:
                changes_this_time_step[cur_name] = cur_ip

        # wait a bit
        time.sleep(1)
        time_step_to_changes[timestep_counter] = changes_this_time_step
        timestep_counter += 1
        last_timestep_mapping = current_mapping

    # now write changes to file (we're doing it all at the end b/c that's easier...)
    with open(log_file_loc, 'wb') as f:  # Just use 'w' mode in 3.x
        f.write(pickle.dumps(time_step_to_changes))

if __name__=="__main__":
    if len(sys.argv) < 3:
        print "not enough CLAs!"
        exit(344)

    log_file_loc= sys.argv[1]
    sentinal_file_loc= sys.argv[2]

    pod_logger(log_file_loc, sentinal_file_loc)