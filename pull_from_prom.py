import numpy as np
import requests
import subprocess
import pandas as pd
#import time
# okay, next step is to store a whole bunch. And then compute the differentials.
# probably want to do a time-based loop. And set it up so I can easily pickle the files
# the next part I'll presuppose the existance of the pickled data frames, so get it all ready
# for that. I've decided on control charts and PCA as the initial steps.
# maybe look into the cloudlab bootscript thing. Or just check that the thing is even worthwhile (e.g. enough ram)

# also, set up a git and simplify the file-structure of this project. 2 identical sock shops. Really??
#r = requests.get('http://www.google.com')
r = requests.get('http://127.0.0.1:9090/')

print r
print r.text
print r.status_code
print r.headers['content-type']

#r = requests.get('http://localhost:9090/api/v1/query?query=istio_mongo_received_bytes')
#print r.text

r = requests.get('http://localhost:9090/api/v1/query?query=istio_mongo_sent_bytes')
#print r.text
print r.json()
print "\n\n\n"
print r.json()['data']['result']#['metric']
print "\n\n\n"
'''
for thing in r.json()['data']['result']:#['metric']:
#    print thing
#    print thing['metric'],"\n"
#    print thing['value'],"\n"
    print "FROM ",thing['metric']['source_ip']," TO ", thing['metric']['destination_service'], " : ", thing['value'][1], "\n"
#for thing in r.json()['data']['result']:#['metric']:
#        print thing
'''
#for key,val in r.json()['data'].iteritems():
#    print key,val


#subprocess.call(["ls", "-l"])

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

data = []
for thing in r.json()['data']['result']:#['metric']:
#    print "FROM ",thing['metric']['source_ip']," TO ", thing['metric']['destination_service'], " : ", thing['value'][1], "\n"
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

## Okay, Now let's put it in matrix form
'''
for service in services:
    hasi_this_src_service = []
    for datum in data:
        if service is datum[1]:
            has_this_src_service.append(datum)
    for datum in has_this_src_service:

for datum in data:
'''

#df2 = pd.DataFrame(np.random.randint(low=0, high=10, size=(5, 5)), columns=services)

services.append("127.0.0.1") #it's always there, so I guess we should treat it like a normal thing
# going to construct the matrix in accordance with the order found in the services list
#d = {'col1': [1, 2], 'col2': [3, 4]}
#df = pd.DataFrame(data=d)
dates = pd.date_range('20130101',periods=6)
#d = pd.DataFrame(np.zeros((5, 5)))
#print d
#test = np.zeros(len(services),len(services))
#print test
df = pd.DataFrame(np.zeros((len(services), len(services))),index=services,columns=services)
'''
for i in services:
    for j in services:
        #pd[i,j] = 0
        for datum in data:
            print datum[0], i, datum[0]==i, "    :    ", datum[1],j,datum[1]==j
            if datum[0] == i and  datum[1] == j:
                print "matched!"
                df.set_value(i, j, datum[2])
'''
for datum in data: 
    df.set_value(datum[0],datum[1],datum[2])
    if datum[0] == "172.17.0.1":
        print datum
print df
