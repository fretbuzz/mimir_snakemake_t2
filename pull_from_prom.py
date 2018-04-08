import numpy as np
import requests
import subprocess
import pandas as pd
import time
import sys

'''
USAGE: python pull_from_prom.py [y/n actively detect] [# in seconds to record]

This program pulls from an exposed prometheus pod (at 127.0.0.1:9090) every 5 seconds, computes the differential traffic matrix, and stores it in a pickle file for access to later.

While this sounds very simple, immature tooling in the microservice space causes it to take a little work (e.g to get the required data from prometheus, need to curl it on a regular basis, parse the output, lookup the service that pod IP's correspond to, etc.(
'''

services = [
        'carts-db',
        'carts',
        'catalogue-db',
        'catalogue',
        'front-end',
        'orders-db',
        'orders',
        'payment',
        'queue-master',
        'rabbitmq',
        'session-db',
        'shipping',
        'user-db',
        'user',
        'load-test',
        '127.0.0.1', #it's always there, so I guess we should treat it like a normal thing
        '172.17.0.1' # also this one too
]

def main(actively_detect, watch_time):
    print "starting to pull data"
    cumul_received_matrix = pd.DataFrame() # an empty pandas dataframe
    cumul_sent_matrix = pd.DataFrame() # an empty pandas dataframe
    last_recieved_matrix = pd.DataFrame()
    last_sent_matrix = pd.DataFrame()
    absolute_start_time = time.time()
    while int(time.time() - absolute_start_time) < watch_time:
        start_time = time.time()
        recieved_matrix, sent_matrix = pull_from_prometheus()
        
        print "recieved matrix: "
        #print last_recieved_matrix
        differential_recieved_matrix, last_recieved_matrix = calc_differential_matrix(last_recieved_matrix, recieved_matrix, start_time, absolute_start_time)
        if not differential_recieved_matrix.empty:
            cumul_received_matrix = cumul_received_matrix.append(differential_recieved_matrix)
            cumul_received_matrix.to_pickle("./experimental_data/cumul_received_matrix.pickle")
        
        print "sent matrix: "
        differential_sent_matrix, last_sent_matrix = calc_differential_matrix(last_sent_matrix, sent_matrix, start_time, absolute_start_time)
        if not differential_sent_matrix.empty:
            cumul_sent_matrix = cumul_sent_matrix.append(differential_sent_matrix)
            cumul_sent_matrix.to_pickle("./experimental_data/cumul_sent_matrix.pickle")
            if actively_detect:
                print "Should analyze the matrices here"
        
        print "Run time: ", time.time() - start_time
        time_to_sleep = 5 - (time.time() - start_time)
        print "Should sleep for ", time_to_sleep, " seconds"
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)


def calc_differential_matrix(last_matrix, current_matrix, start_time, absolute_start_time):
    if not last_matrix.empty:
        differential_matrix = current_matrix - last_matrix
        last_matrix = current_matrix.copy()
        differential_matrix['time'] = start_time - absolute_start_time
        print differential_matrix
    else:
        differential_matrix = pd.DataFrame()
        last_matrix = current_matrix.copy()
        print "First recieved_matrix pulled (so cannot compute differential yet):"
        print differential_matrix
    return differential_matrix, last_matrix

def pull_from_prometheus():
    r = requests.get('http://127.0.0.1:9090/')

    print r
    #print r.text
    print r.status_code
    if r.status_code == 200:
        print "Prometheus is active and accessible!"
    else:
        print "There is a problem with Prometheus!"
    #print r.headers['content-type']

    prometheus_recieved_bytes = requests.get('http://localhost:9090/api/v1/query?query=istio_mongo_received_bytes')
    prometheus_sent_bytes = requests.get('http://localhost:9090/api/v1/query?query=istio_mongo_sent_bytes')
    ip_to_service = get_ip_to_service_mapping()

    return process_prometheus_pull(prometheus_recieved_bytes.json(), prometheus_sent_bytes.json(), ip_to_service)

# note: this function exists mostly because it is a good testing point
def process_prometheus_pull(prometheus_recieved_bytes, prometheus_sent_bytes, ip_to_service):
    print "About to parse recieved data!"
    parsed_recieved_data = parse_prometheus_response(prometheus_recieved_bytes, ip_to_service)
    recieved_matrix = pd.DataFrame(np.zeros((len(services), len(services))),index=services,columns=services)
    construct_matrix(parsed_recieved_data, recieved_matrix)

    print "About to parse sent data!"
    parsed_sent_data = parse_prometheus_response(prometheus_sent_bytes, ip_to_service)
    sent_matrix = pd.DataFrame(np.zeros((len(services), len(services))),index=services,columns=services)
    construct_matrix(parsed_sent_data, sent_matrix)

    return recieved_matrix, sent_matrix

def get_ip_to_service_mapping():
    out = subprocess.check_output(["kubectl", "get", "po", "-o", "wide","--all-namespaces"])
    #print out
    return parse_ip_to_service_mapping(out)

# note: this function exists because it is easy to test
def parse_ip_to_service_mapping(kubectl_get_output):
    out = kubectl_get_output
    g = out.split('\n')
    #print g
    ip_to_service = {}
    for line in g:
        k = line.split("   ")
        IP = ' hello world'
        non_empty_chunk_counter = 0
        for chunk in k[1:]:
            # we want 5th non-empty chunk
            if chunk: # checks if chunk is not the empty string
                non_empty_chunk_counter = non_empty_chunk_counter + 1
                if non_empty_chunk_counter == 6:
                    IP = chunk
                if non_empty_chunk_counter == 1:
                    pod = chunk
        for service in services:
            if service in pod:
                pod = service
                break
        ip_to_service[IP.strip()] = pod.rstrip().lstrip()
        #print "ip", IP, "pod", pod
    #print ip_to_service
    return ip_to_service

def parse_prometheus_response(prometheus_response, ip_to_service):
    data = []
    for thing in prometheus_response['data']['result']:
        try:
            source_service = ip_to_service[thing['metric']['source_ip'].encode('ascii','ignore')]
        except KeyError:
            source_service = thing['metric']['source_ip'].encode('ascii','ignore')
        dst_service = thing['metric']['destination_service']
        for service in services:
            if service in thing['metric']['destination_service']:
                dst_service = service
        data.append( [source_service, dst_service,  thing['value'][1]] )
        #print "FROM ", source_service, " TO ", dst_service, " : ", thing['value'][1], "\n"
    #print data
    return data


# going to construct the matrix in accordance with the order found in the services list
def construct_matrix(data, df):
    for datum in data: 
        df.set_value(datum[0],datum[1],datum[2])
        #if datum[0] == "172.17.0.1":
        #   print datum
        does_it_make_since = False
        for service in services:
            if service in datum[0]:
                does_it_make_since = True
        if not does_it_make_since:
            print "Here is anomalous data!:"
            print datum
            print datum[0]
    #print df
    return df

if __name__=="__main__":
    actively_detect = False
    watch_time = float("inf")
    #print sys.argv[1]
    if len(sys.argv) > 1:
        if sys.argv[1] == "y":
           actively_detect = True
           print "actively detect: ", actively_detect
        if len(sys.argv) > 2:
            watch_time = int(sys.argv[2])
            print "watch time: ", watch_time
    main(actively_detect, watch_time)
