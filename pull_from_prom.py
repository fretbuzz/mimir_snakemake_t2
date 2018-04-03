import numpy as np
import requests
import subprocess
import pandas as pd
import time
# okay, next step is to store a whole bunch. And then compute the differentials.
# probably want to do a time-based loop. And set it up so I can easily pickle the files
# the next part I'll presuppose the existance of the pickled data frames, so get it all ready
# for that. I've decided on control charts and PCA as the initial steps.
# maybe look into the cloudlab bootscript thing. Or just check that the thing is even worthwhile (e.g. enough ram)

services = [
        'carts',
        'carts-db',
        'catalogue',
        'catalogue-db',
        'front-end',
        'orders',
        'orders-db',
        'payment',
        'queue-master',
        'rabbitmq',
        'session-db',
        'shipping',
        'user',
        'user-db',
        'load-test'
]

def main():
    print "starting to pull data"
    cumul_received_matrix = pd.DataFrame() # an empty pandas dataframe
    cumul_sent_matrix = pd.DataFrame() # an empty pandas dataframe
    #while True:
    start_time = time.time()
    recieved_matrix, sent_matrix = pull_from_prometheus()
    print "recieved matrix: "
    print recieved_matrix
    print "sent matrix: "
    print sent_matrix
    print "Run time: ", time.time() - start_time
    ### Note: I also need to compute the differentials between the matrixes, since they are monotonically increasing
    # if cumul_received_matrix:
    # ## append
    # else assign
    # same with sent
    ## make sure to update the key values with time stamps
    # time_to_sleep = time.time() - start_time - 5
    # if time_to_sleep > 0:
    # time.sleep(

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

    services.append("127.0.0.1") #it's always there, so I guess we should treat it like a normal thing
    services.append("172.17.0.1") # this is also always theree

    prometheus_recieved_bytes = requests.get('http://localhost:9090/api/v1/query?query=istio_mongo_received_bytes')
    #print r.text
    ip_to_service = get_ip_to_service_mapping()
    print "About to parse recieved data!"
    parsed_recieved_data = parse_prometheus_response(prometheus_recieved_bytes, ip_to_service)
    recieved_matrix = construct_matrix(parsed_recieved_data)

    prometheus_sent_bytes = requests.get('http://localhost:9090/api/v1/query?query=istio_mongo_sent_bytes')
    #print r.text
    ip_to_service = get_ip_to_service_mapping()
    print "About to parse sent data!"
    parsed_sent_data = parse_prometheus_response(prometheus_sent_bytes, ip_to_service)
    sent_matrix = construct_matrix(parsed_sent_data)

    return recieved_matrix, sent_matrix

def get_ip_to_service_mapping():
    out = subprocess.check_output(["kubectl", "get", "po", "-o", "wide","--all-namespaces"])
    print out
    g = out.split('\n')
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
        ip_to_service[IP.strip()] = pod
    print ip_to_service
    return ip_to_service

def parse_prometheus_response(prometheus_response, ip_to_service):
    data = []
    for thing in prometheus_response.json()['data']['result']:
        try:
            source_service = ip_to_service[thing['metric']['source_ip'].encode('ascii','ignore')]
        except KeyError:
            source_service = thing['metric']['source_ip'].encode('ascii','ignore')
        dst_service = thing['metric']['destination_service']
        for service in services:
            if service in thing['metric']['destination_service']:
                dst_service = service
        data.append( [source_service, dst_service,  thing['value'][1]] )
        print "FROM ", source_service, " TO ", dst_service, " : ", thing['value'][1], "\n"
    print data
    return data


# going to construct the matrix in accordance with the order found in the services list
def construct_matrix(data):
    df = pd.DataFrame(np.zeros((len(services), len(services))),index=services,columns=services)
    for datum in data: 
        df.set_value(datum[0],datum[1],datum[2])
        if datum[0] == "172.17.0.1":
            print datum
        does_it_make_since = False
        for service in services:
            if service in datum[0]:
                does_it_make_since = True
        if not does_it_make_since:
            print "Here is anomalous data!:"
            print datum
            print datum[0]
    print df
    return df

if __name__=="__main__":
               main()
