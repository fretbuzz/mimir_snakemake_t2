'''
USAGE: python run_experiment.py [y/n restart kubernetes cluster] [y/n setup sock shop application]
note: need to start docker daemon beforehand
'''

import subprocess
import time
import requests
import sys
import os
import signal
import parameters
from exfil_data_v2 import how_much_data

def main(restart_kube, setup_sock):
    if restart_kube == "y":
        restart_minikube()
    if setup_sock == "y":
        setup_sock_shop()
    run_experiment()

def restart_minikube():
    # no point checking, just trying stopping + deleteing
    print "Stopping minikube..."
    try:
        out = subprocess.check_output(["minikube", "stop"])
        print out
    except:
        print "Minikube was not running"
    print "Stopping minikube completed"
    print "Deleteing minikube..."
    try:
        out = subprocess.check_output(["minikube", "delete"])
        print out
    except:
        print "No minikube image to delete"
    print "Deleteing minikube completed"

    # then start minikube
    print "Starting minikube..."
    out = subprocess.check_output(["minikube", "start", "--memory=6144", "--cpus=3"])
    print out
    print "Starting minikube completed"


    # TODO check if istio already exists (or not, doesn't really matter)
    print "Cloning Istio..."
    #ps = subprocess.Popen(("curl", "-L", "https://git.io/getIstio"), stdout=subprocess.PIPE)
    ps = subprocess.Popen(("curl", "-L", "https://git.io/getLatestIstio"), stdout=subprocess.PIPE)
    output = subprocess.check_output(("sh", "-"), stdin=ps.stdout)
    ps.wait()
    print output
    print "Completed cloning Istio"

    # then install instio
    print "Starting to install Istio"
    # get istio folder
    istio_folder = get_istio_folder()
    print istio_folder
    out = ""
    #try:
    # I am going to do kinda a lazy fix here. If this gives additional problems see Github issue #29i
    try:
        out = subprocess.check_output(["kubectl","apply", "-f", istio_folder + "/install/kubernetes/istio.yaml"])
    except:
        print "That istio race condition is happening. Need to try install istio a second time!"
        time.sleep(60)
        out = subprocess.check_output(["kubectl","apply", "-f", istio_folder + "/install/kubernetes/istio.yaml"])
        ## that file doesn't actually exist
        #out = subprocess.check_output(["kubectl", "apply", "-f", istio_folder + "/install/kubernetes/istio-customresources.yaml"])
    print out
    print "Completed installing Istio"


def setup_sock_shop():
    # then deploy application
    print "Starting to deploy sock shop..."
    istio_folder = get_istio_folder()
    out = subprocess.check_output(["Bash", "start_with_istio.sh", istio_folder])
    print out
    print "Completed installing sock shop..."

    istio_folder = get_istio_folder()
    minikube = subprocess.check_output(["minikube", "ip"])
    # wait until istio pods are started
    print "Checking if Istio pods are ready..."
    pods_ready_p = False
    while not pods_ready_p:
        out = subprocess.check_output(["kubectl", "get", "pods", "-n", "istio-system"])
        print out
        statuses = parse_kubeclt_output(out, [1,2])
        print statuses
        pods_ready_p = check_if_pods_ready(statuses)
        print "Istio pods are ready: ", pods_ready_p
        time.sleep(10)
    print "Istio pods are ready!"

    # Install prometheus plug-in so we can get the metrics
    print "Installing prometheus..."
    out = subprocess.check_output(["kubectl", "apply", "-f", istio_folder + "/install/kubernetes/addons/prometheus.yaml"])
    print "Prometheus installation complete"

    # wait for prometheus container to start
    print "Checking if Prometheus pods are ready..."
    pods_ready_p = False
    while not pods_ready_p:
        prom_status = []
        while len(prom_status) < 2:
            out = subprocess.check_output(["kubectl", "get", "pods", "-n", "istio-system"])
            print out
            # okay, need to get rid of all non-prometheus lines
            statuses = parse_kubeclt_output(out, [1,2])
            prom_status = []
            prom_status.append(statuses[0])
            for status in statuses[1:]:
                if 'prometheus' in status[0]:
                    prom_status.append(status)
            print prom_status
            time.sleep(10)
        pods_ready_p = check_if_pods_ready(prom_status)
        print "Prometheus pod is ready: ", pods_ready_p
        time.sleep(10)
    print "Prometheus pods are ready!"

    ## Activate the custom metrics that we will use
    print "Installing custom metrics..."
    out = ""
    try:
        out = subprocess.check_output([istio_folder + "/bin/istioctl", "create", "-f", "new_telemetry.yaml"])
    except:
        print "new_telemetry already exists"
    print out
    try:
        out = subprocess.check_output([istio_folder + "/bin/istioctl", "create", "-f", "tcp_telemetry_orig_orig.yaml"])
    except:
        print "tcp_telemetry_orig_orig already exists"
    print out
    print "Custom metrics are ready!"

    # Deploy manifests_tcp_take_2 (to switch the service ports)
    print "Modifying service port names..."
    out = subprocess.check_output(["Kubectl", "apply", "-f", "./manifests_tcp_take_2"])
    print out
    print "Completed modifying service port names"

    # expose prometheus
    print "Exposing prometheus..."
    #try:
    # first, get name of the prometheus pod
    out = subprocess.check_output(["kubectl", "get", "pods", "-n", "istio-system"])
    # okay, need to get rid of all non-prometheus lines
    statuses = parse_kubeclt_output(out, [1,2])
    prom_cont_name = ""
    for status in statuses[1:]:
        if 'prometheus' in status[0]:
            prom_cont_name = status[0]
    print prom_cont_name
    # okay, now can expose
    #try:
    # cannot assign output to a variable because then wont run in 'background'
    subprocess.Popen(["kubectl", "-n", "istio-system", "port-forward", prom_cont_name, "9090:9090"])
    #except:
    #    print "Promtheus was already exposed"
    #print out
    print "Completed Exposing Prometheus"

    # verify that prometheus is active
    print "Verifying that prometheus is active..."
    r = None
    while not r:
        try:
            r = requests.get('http://127.0.0.1:9090/')
        except:
            r = None
        if r:
            print r.status_code
            if r.status_code == 200:
                print "Prometheus is active and accessible!"
            else:
                print "Prometheus is not accessible!"
        else:
            print "Couldn't access prometheus endpoint!"
        time.sleep(10)
    print "Completed verifying that prometheus is active"

    # verify that all containers are active
    print "Checking if application pods are ready..."
    pods_ready_p = False
    time_waited = 0
    while not pods_ready_p:
        out = subprocess.check_output(["kubectl", "get", "pods"])
        print out
        statuses = parse_kubeclt_output(out, [1,2])
        print statuses
        parsed_statuses = []
        # realistically rabbitmq is never going to work ;-)
        for status in statuses:
            if "rabbitmq" not in status[0]:
                parsed_statuses.append(status)
        pods_ready_p = check_if_pods_ready(parsed_statuses)
        print "Application pods are ready: ", pods_ready_p
        time.sleep(10)
        time_waited = time_waited + 1
        # sometimes generating some traffic makes the pods get into shape
        if time_waited % 24 == 0:
            # first get minikube ip
            minikube = subprocess.check_output(["minikube", "ip"]).rstrip()
            try:
                out = subprocess.check_output(["docker", "run", "--rm", "weaveworksdemos/load-test", "-d", "5", "-h", minikube+":32001", "-c", "2", "-r", "60"])
            except:
                print "cannot even run locust yet..."
    print "Application pods are ready!"

    # okay, now it is time to register a bunch of users
    # note: we need to register users to a bunch of different 'levels', see GitHub issue #25 for why
    minikube = subprocess.check_output(["minikube", "ip"]).rstrip()
    # make c larger if it takes too long (I think 1000 users takes about 5 min currently)
    #num_customer_records = parameters.number_customer_records * 4 # b/c 4 calls per record
    out = subprocess.check_output(["locust", "-f", "pop_db.py", "--host=http://"+minikube+":32001", "--no-web", "-c", "15", "-r", "1", "-n", parameters.number_full_customer_records])
    out = subprocess.check_output(["locust", "-f", "pop_db_reg_and_andr.py", "--host=http://"+minikube+":32001", "--no-web", "-c", "15", "-r", "1", "-n", parameters.number_half_customer_records])
    out = subprocess.check_output(["locust", "-f", "pop_db_reg.py", "--host=http://"+minikube+":32001", "--no-web", "-c", "15", "-r", "1", "-n", parameters.number_quarter_customer_records])
    #print out
    ###### TODO: verift that the above thing worked via a call to the customers api

def run_experiment():
    ## okay, this is where the experiment is actualy going to be implemented (the rest is all setup)
    ## 0th step: determine how much data each of the data exfiltration calls gets so we can plan the exfiltration
    ## step accordingly
    minikube = subprocess.check_output(["minikube", "ip"]).rstrip()
    amt_custs, amt_addr, amt_cards = how_much_data("http://"+minikube+":32001")
    print amt_custs, amt_addr, amt_cards
    time.sleep(5) # want to make sure that we're not going to mess with the recorded traffic vals

    # First, start the background traffic
    synch_with_prom()
    devnull = open(os.devnull, 'wb')  # disposing of stdout manualy
    proc = subprocess.Popen(["locust", "-f", "background_traffic.py", "--host=http://"+minikube+":32001", "--no-web", "-c", parameters.num_background_locusts, "-r", parameters.rate_spawn_background_locusts], stdout=devnull, stderr=devnull, preexec_fn=os.setsid)
    print os.getpgid(proc.pid)
    #start_time = time.time() # I think it needs to be down below so the graphs make more sense

    # Second, sync with prometheus scraping (see function below for explanation) and then start experimental recording script
    
    # the plus one is so that what it pulls includes the last frame (b/c always a little over the current sec)
    subprocess.Popen(["python", "pull_from_prom.py", "n", str( parameters.desired_stop_time + 1)])
    start_time = time.time()

    # Third, wait some period of time and then start the data exfiltration
    # this has been modified to support multiple exfiltrations during a single time period
    print "Ready to exfiltrate!"
    exfil_times_left = [c_time for c_time in parameters.exfils.iterkeys()]
    print "exfil_times_left", exfil_times_left
    while exfil_times_left:
        cur_time = time.time() - start_time
        for i in exfil_times_left:
            print i, cur_time, i<cur_time
            if i < cur_time:
                exfil_times_left.remove(i)
        exfil_times_left.sort()
        print "exfil_times_left", exfil_times_left, "cur time: ", cur_time
        if not exfil_times_left:
            break
        next_exfil = int(exfil_times_left[0])
        sleep_time = next_exfil - time.time()
        print "next exfil at: ", next_exfil, "will sleep for: ", sleep_time
        if sleep_time > 0:
            time.sleep(sleep_time)
            # going to use the updated version instead
            out = subprocess.check_output(["python", "exfil_data_v2.py", "http://"+minikube+":32001", str(parameters.exfils[next_exfil]), str(amt_custs), str(amt_addr), str(amt_cards)])
            print "Data exfiltrated", out
    print "all data exfiltration complete"

    # Fourth, wait for some period of time and then stop the experiment
    # NOTE: going to leave sock shop and everything up, only stopping the experimental
    # stuff, not the stuff that the experiment is run on
    wait_time = parameters.desired_stop_time - (time.time() - start_time)
    print "wait time is: ", wait_time
    if wait_time > 0:
        time.sleep(wait_time)
    print "just stopped waiting!"
    # this stops the background traffic process
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM) # should kill it

    # Fifth, call the function that analyzes the traffic matrices
    # (It should output potential times that the exfiltration may have occured)
    # (which it does not do yet)
    print "About to analyze traffic matrices...."
    out = subprocess.check_output(["python", "analyze_traffix_matrixes.py", './experimental_data/' + parameters.rec_matrix_location, './experimental_data/' + parameters.sent_matrix_location])
    print out

    # Sixth, what is the FP / FN / TP / TN ??
    #### TODO (prob need to write a function for this + need to fix analyze_traffic_matrixes)

    
    print "Experiment complete!!"

# kc_out is the result from a "kubectl get" command
# desired_chunks is a list of the non-zero chunks in each
# line that the caller wants
def parse_kubeclt_output(kc_out, desired_chunks):
    g = kc_out.split('\n')
    results = []
    for line in g:
        k = line.split("   ")
        chunks_from_line = []
        non_empty_chunk_counter = 0
        for chunk in k[:]:
            # we want 5th non-empty chunk
            if chunk: # checks if chunk is not the empty string
                non_empty_chunk_counter = non_empty_chunk_counter + 1
                if non_empty_chunk_counter in desired_chunks:
                    chunks_from_line.append(chunk.strip())
        if chunks_from_line:
            results.append(chunks_from_line)
    return results

def is_pod_ready_p(pod_status):
    #print pod_status
    ready_vs_total = pod_status.split('/')
    print ready_vs_total
    if len(ready_vs_total) > 1:
        #print  ready_vs_total[0] == ready_vs_total[1]
        return ready_vs_total[0] == ready_vs_total[1]
    return False

def check_if_pods_ready(statuses):
    are_pods_ready = True
    for status in statuses[1:]:
        if len(status) > 1:
            #print status
            is_ready =  is_pod_ready_p(status[1])
            print is_ready
            if not is_ready:
                are_pods_ready = False
    return are_pods_ready

def get_istio_folder():
    out = subprocess.check_output(["ls"])
    for line in out.split('\n'):
        if "istio-" in line:
            return line

# prometheus scrapes from mixer every 5 seconds
# we want to exfiltrate data right after
# that pull, to maximize our chances of doing
# it all within a single 5 sec time interval
def synch_with_prom():
    # first, we are going to pull from prometheus to extract the server's internal timestamp
    prometheus_response = requests.get('http://localhost:9090/api/v1/query?query=istio_mongo_received_bytes')
    #for thing in prometheus_response.json()['data']['result']:
    #    print thing['value'][0], thing['value'][1]
    time_stamp = float(prometheus_response.json()['data']['result'][1]['value'][0])
    print "server time: ", time_stamp

    # second, we are going to get a five second size of the server's values, so we can see when it scrapes
    prometheus_range = requests.get('http://localhost:9090/api/v1/query_range?query=istio_mongo_sent_bytes&start=' + str( time_stamp - 25) + '&end=' + str(time_stamp) + '&step=1s')
    one_metric =  prometheus_range.json()['data']['result'][1]
    #print one_metric
    #print one_metric[u'values'], type(one_metric[u'values'])
    #print list(reversed(one_metric[u'values']))
    vals_in_reverse_time = list(reversed(one_metric[u'values']))
    #print vals_in_reverse_time
    current_val = vals_in_reverse_time[0][1]
    #print current_val
    time_since_change = 0
    for i in range(0, len(vals_in_reverse_time)):
        #print vals_in_reverse_time[i][0], vals_in_reverse_time[i][1], i
        if vals_in_reverse_time[i][1] != current_val:
            time_since_change = i
            break
    #print time_since_change

    # third, we are going to sleep for the (number of seconds until next prom scrape)
    time_to_sync = 5 - time_since_change
    if time_to_sync > 5 or time_to_sync < 0:
        print "SOMETHING WENT HORRIBLY WRONG IN SYNC FUNCTION"
    print "time to sync: ", time_to_sync
    time.sleep(time_to_sync)

if __name__=="__main__":
    restart_kube = "n"
    setup_sock = "n"
#    print sys.argv[1]
    if len(sys.argv) > 1:
        print "triggered"
        print sys.argv
        restart_kube = sys.argv[1] 
    if len(sys.argv) > 2:
        print "triggered num two"
        print sys.argv
        setup_sock = sys.argv[2] 
    main(restart_kube, setup_sock)
