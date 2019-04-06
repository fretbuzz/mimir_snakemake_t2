import subprocess
import argparse
import sys
import os
import time
import pickle
import pyximport; pyximport.install()

def pod_logger(log_file_loc, sentinal_file_loc):
    time_behind = 0.0 # records how much time the system is behind where it should be
    timestep_counter = 0
    last_timestep_mapping = {}
    time_step_to_changes = {}
    while (not os.path.exists(sentinal_file_loc)):
        #print "current_loop: ", timestep_counter
        loop_starttime = time.time()
        current_mapping = {}
        out = subprocess.check_output(['kubectl', 'get', 'po', '-o','wide', '--all-namespaces', '--show-labels'])

        for line in out.split('\n')[1:]:
            if line != '':
                line = [i for i in line.split('   ') if i != '']
                name = line[1].rstrip().lstrip()
                ip = line[3].rstrip().lstrip()
                namespace=line[0].rstrip().lstrip()
                labels = line[8].rstrip().lstrip()
                if '<none>' not in ip:
                    current_mapping[name] = (ip, namespace, 'pod', labels)

        out = subprocess.check_output(['kubectl', 'get', 'svc', '-o', 'wide', '--all-namespaces', '--show-labels'])
        for line in out.split('\n')[1:]:
            if line != '':
                line = [i for i in line.split('   ') if i != '']
                name = line[1].rstrip().lstrip()
                ip = line[3].rstrip().lstrip()
                namespace = line[0].rstrip().lstrip()
                labels = line[7].rstrip().lstrip()
                current_mapping[name] = (ip, namespace, 'svc', labels)

        # now compare to old mapping
        changes_this_time_step = {}
        for cur_name, cur_ip in current_mapping.iteritems():
            if cur_name not in last_timestep_mapping:
                changes_this_time_step[cur_name] = (cur_ip[0], '+', cur_ip[1], cur_ip[2], cur_ip[3])
        for last_name,last_ip_tup in last_timestep_mapping.iteritems():
            if last_name not in current_mapping:
                changes_this_time_step[last_name] = (last_ip_tup[0], '-', last_ip_tup[1], last_ip_tup[2], last_ip_tup[3])

        ## https://kubernetes.io/docs/concepts/services-networking/service/
        ## 'The set of Pods targeted by a Service is (usually) determined by a Label Selector
        ## (see below for why you might want a Service without a selector)."

        time_step_to_changes[timestep_counter] = changes_this_time_step

        with open(log_file_loc, 'wb') as f:  # Just use 'w' mode in 3.x
            f.write(pickle.dumps(time_step_to_changes))

        time_to_sleep = 1.0 - (time.time() - loop_starttime) - time_behind
        if time_to_sleep < 0.0:
            print "time_to_sleep", time_to_sleep
            time_behind -= time_to_sleep
        else:
            time_behind = 0.0

        if int(time_to_sleep) < 0.0:
            print "skipping: ", abs(int(time_to_sleep))
            # if we are falling behind, then we need to skip if necessary
            for i in range(1, abs(int(time_to_sleep))):
                time_step_to_changes[timestep_counter + i] = {}
            timestep_counter += abs(int(time_to_sleep))
            time_behind -= abs(int(time_to_sleep))
        else:
            time_to_sleep = max(0.0, time_to_sleep)
            time.sleep(time_to_sleep)
            timestep_counter += 1
        last_timestep_mapping = current_mapping

if __name__=="__main__":
    if len(sys.argv) < 3:
        print "not enough CLAs!"
        exit(344)

    log_file_loc= sys.argv[1]
    sentinal_file_loc= sys.argv[2]

    pod_logger(log_file_loc, sentinal_file_loc)