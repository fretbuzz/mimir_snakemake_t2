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

def main(restart_kube, setup_sock):
    if restart_kube == "y":
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
        out = subprocess.check_output(["kubectl","apply", "-f", istio_folder + "/install/kubernetes/istio.yaml"])
        print out
        print "Completed installing Istio"
    
        # then deploy application
        print "Starting to deploy sock shop..."
        out = subprocess.check_output(["Bash", "start_with_istio.sh", istio_folder])
        print out
        print "Completed installing sock shop..."

    if setup_sock == "y":
        
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
        minikube = subprocess.check_output(["minikube", "ip"]).rstrip()
        # note for now, I'm going to make 1000 customeres (i.e. 3000 / 3 = 1000), but this might need to be adjusted later
        # make c larger if it takes too long (I think 1000 users takes about 5 min currently)
        out = subprocess.check_output(["locust", "-f", "pop_db.py", "--host=http://"+minikube+":32001", "--no-web", "-c", "15", "-r", "1", "-n", "4000"])
        #print out
        ###### TODO: verift that the above thing worked via a call to the customers api

    ## okay, this is where the experiment is actualy going to be implemented (the rest is all setup)
    desired_exfil_time = 10 # NOTE: choosing 10 seconds purely for testing purposes (real would be longer)
    desired_stop_time = 20 # Choosing 20 secods purely for testing purposes (real would be longer)

    # First, start the background traffic
    # I think this does it- need to verify though
    minikube = subprocess.check_output(["minikube", "ip"]).rstrip()
    devnull = open(os.devnull, 'wb')  # disposing of stdout manualy
    proc = subprocess.Popen(["locust", "-f", "background_traffic.py", "--host=http://"+minikube+":32001", "--no-web", "-c", "12", "-r", "1"], stdout=devnull, stderr=devnull, preexec_fn=os.setsid)
    #print pid_background ## TODO: kill automatically at the end
    print os.getpgid(proc.pid)
    #os.killpg(os.getpgid(proc.pid), signal.SIGTERM) # should kill it
    start_time = time.time()

    # Second, start experimental recording script
    ##### TODO-- call pull_from_prom

    # Third, wait some period of time and then start the data exfiltration
    print "Ready to exfiltrate!"
    sleep_time = desired_exfil_time - (time.time() - start_time)
    if sleep_time > 0:
        time.sleep(sleep_time )
    subprocess.check_output(["locust", "-f", "./exfil_data.py", "--host=http://"+minikube+":32001", "--no-web", "-c", "1", "-r", "1", "-n", "3"])
    print "Data exfiltrated"

    # Fourth, wait for some period of time and then stop the experiment
    # NOTE: going to leave sock shop and everything up, only stopping the experimental
    # stuff, not the stuff that the experiment is run on
    wait_time = desired_stop_time - (time.time() - start_time)
    if wait_time > 0:
        time.sleep(wait_time) 
    # this stops the background traffic process
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM) # should kill it

    # Fifth, call the function that analyzes the traffic matrices
    # (It should output potential times that the exfiltration may have occured)
    #### TODO


    ''' This is old and should be deleted at some point
    # start experiment 
    ##### TODO: Generate more realistic traffic
    minikube = subprocess.check_output(["minikube", "ip"]).rstrip()
    out = subprocess.check_output(["docker", "run", "--rm", "weaveworksdemos/load-test", "-h", minikube+":32001", "-c", "2", "-r", "60"])
    '''

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
