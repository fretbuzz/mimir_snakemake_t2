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
        'load-test',
        '127.0.0.1', #it's always there, so I guess we should treat it like a normal thing
        '172.17.0.1' # also this one too
]

def main():
    print "starting to pull data"
    cumul_received_matrix = pd.DataFrame() # an empty pandas dataframe
    cumul_sent_matrix = pd.DataFrame() # an empty pandas dataframe
    last_recieved_matrix = pd.DataFrame()
    last_sent_matrix = pd.DataFrame()
    while True:
        start_time = time.time()
        recieved_matrix, sent_matrix = pull_from_prometheus()
        print "recieved matrix: "
        
        # we want the differential matrixes, not just the aggregates
        if not last_recieved_matrix.empty:
            differential_recieved_matrix = recieved_matrix - last_recieved_matrix
            last_recieved_matrix = recieved_matrix.copy()
            print differential_recieved_matrix
            #cumul_received_matrix.append(differential_recieved_matrix)
            #print cumul_received_matrix
            # let's try this
            cumul_received_matrix = pd.concat([cumul_received_matrix, differential_recieved_matrix])
            #print cumul_received_matrix
            cumul_received_matrix.to_pickle("./experimental_data/cumul_received_matrix.pickle")
        else: 
            last_recieved_matrix = recieved_matrix.copy()
            print "First recieved_matrix pulled (so cannot compute differential yet):"
            print last_recieved_matrix

        print "sent matrix: "
        if not last_sent_matrix.empty:
            print "last sent matrix is not empty"
            differential_sent_matrix = sent_matrix - last_sent_matrix
            last_sent_matrix = sent_matrix.copy()
            print differential_sent_matrix
            #cumul_sent_matrix.append(differential_sent_matrix)
            #print cumul_sent_matrix
            cumul_sent_matrix = pd.concat([cumul_sent_matrix, differential_recieved_matrix])
            cumul_sent_matrix.to_pickle("./experimental_data/cumul_sent_matrix.pickle")
        else:
            print "last sent matrix is empty"
            last_sent_matrix = sent_matrix
            print "First sent matrix pulled (so cannot compute differential yet):"
            print last_sent_matrix
        print "Run time: ", time.time() - start_time
        time_to_sleep = 5 - (time.time() - start_time)
        print "Should sleep for ", time_to_sleep, " seconds"
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

    ## TODO make sure to update the key values with time stamps

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
    #print r.text
    ip_to_service = get_ip_to_service_mapping()
    print "About to parse recieved data!"
    parsed_recieved_data = parse_prometheus_response(prometheus_recieved_bytes, ip_to_service)
    recieved_matrix = pd.DataFrame(np.zeros((len(services), len(services))),index=services,columns=services)
    construct_matrix(parsed_recieved_data, recieved_matrix)

    prometheus_sent_bytes = requests.get('http://localhost:9090/api/v1/query?query=istio_mongo_sent_bytes')
    #print r.text
    ip_to_service = get_ip_to_service_mapping()
    print "About to parse sent data!"
    parsed_sent_data = parse_prometheus_response(prometheus_sent_bytes, ip_to_service)
    sent_matrix = pd.DataFrame(np.zeros((len(services), len(services))),index=services,columns=services)
    construct_matrix(parsed_sent_data, sent_matrix)

    return recieved_matrix, sent_matrix

def get_ip_to_service_mapping():
    out = subprocess.check_output(["kubectl", "get", "po", "-o", "wide","--all-namespaces"])
    #print out
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
    #print ip_to_service
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
    #print data
    return data


# going to construct the matrix in accordance with the order found in the services list
def construct_matrix(data, df):
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
