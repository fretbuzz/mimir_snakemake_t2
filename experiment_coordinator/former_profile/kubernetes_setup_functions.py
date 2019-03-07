import subprocess
import time

def wait_until_pods_done(namespace):
        # wait until  pods are started
        print "Checking if " + namespace + " pods are ready..."
        pods_ready_p = False
        while not pods_ready_p:
                print "command to be used ", "kubectl get pods -n " + namespace
                out = subprocess.check_output(["kubectl", "get", "pods", "-n", namespace])
                print out
                #out2 = subprocess.check_output(["kubectl", "get", "pods", "--all-namespaces"])
                #print out2
                statuses = parse_kubeclt_output(out, [1,2,3])
                print statuses
                pods_ready_p = check_if_pods_ready(statuses)
                print namespace + " pods are ready: ", pods_ready_p
                time.sleep(10)
        print namespace + " pods are ready!"

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

def check_if_pods_ready(statuses):
    are_pods_ready = True
    #print statuses
    for status in statuses[1:]:
        if len(status) > 1:
            #print "status", status
            is_ready =  is_pod_ready_p(status[1])
            #print is_ready
            if not is_ready:
                are_pods_ready = False
        for part in status:
                #print "part", part
                if part == "Completed":
                        are_pods_ready = True
                        break
        print is_ready, are_pods_ready
    return are_pods_ready

def is_pod_ready_p(pod_status):
    #print pod_status
    ready_vs_total = pod_status.split('/')
    print ready_vs_total
    if len(ready_vs_total) > 1:
        #print  ready_vs_total[0] == ready_vs_total[1]
        return ready_vs_total[0] == ready_vs_total[1]
    return False

# note: we are assuming that it is the default namespace
def get_svc_ip(name_of_svc, namespace=None):
	if namespace == None:
		out = subprocess.check_output(["kubectl", "get", "svc"])
	else:
		out = subprocess.check_output(["kubectl", "get", "svc", "--namespace=" + namespace])
	parsed_out = parse_kubeclt_output(out, [1,3])
	for out_vals in parsed_out:
		if out_vals[0] == name_of_svc:
			return out_vals[1]
	return None
