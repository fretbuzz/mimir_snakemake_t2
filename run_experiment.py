'''
USAGE: python run_experiment.py [y/n restart kubernetes cluster]
note: need to start docker daemon beforehand
'''
import subprocess
import time
import requests
import sys

## TODO: This file should really be split into start minikube/istio and start application
## and then move starting the experiment into an entirely new file
'''
1. Start minikube
minikube start --memory=6144
2. Install istio
kubectl apply -f install/kubernetes/istio.yaml
 3. Deploy application 
 Bash start_with_istio.sh
 3.5 Wait until istio pods are started 
 kubectl get po,svc,deploy --all-namespaces
  4. Install prometheus plug-in so we can get the metrics
  kubectl apply -f install/kubernetes/addons/prometheus.yaml
  4.5 Wait for prometheus container to start
  kubectl get po,svc,deploy --all-namespaces
  5. 'Activate' the custom metrics that we'll use
  istioctl create -f new_telemetry.yaml
  istioctl create -f tcp_telemetry_orig_orig.yaml
  5.5 deploy manifests_tcp_take_2 (to modify services that aren't friendly w/ metrics collection)
  Kubectl apply -f ./manifests_tcp_take_2
  6. Expose prometheus 
  kubectl -n istio-system port-forward $(kubectl -n istio-system get pod -l app=prometheus -o jsonpath='{.items[0].metadata.name}') 9090:9090 &
   7. Check that prometheus is active via the following url
   http://127.0.0.1:9090
   8. Generate some traffic (note ip might be different, check minikube ip)
   docker run --rm weaveworksdemos/load-test -d 5 -h 192.168.99.113:32001 -c 2 -r 60
'''

def main(restart_kube):
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

    # start experimental recording script
    ##### TODO 


    # start experiment 
    ##### TODO: Generate more realistic traffic
    minikube = subprocess.check_output(["minikube", "ip"]).rstrip()
    out = subprocess.check_output(["docker", "run", "--rm", "weaveworksdemos/load-test", "-h", minikube+":32001", "-c", "2", "-r", "60"])
    
    ##### TODO: Perform data exfiltration

    # did it work without crashing
    ##### TODO

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
    print sys.argv[1]
    if len(sys.argv) > 0:
        restart_kube = sys.argv[1]       
    main(restart_kube)
