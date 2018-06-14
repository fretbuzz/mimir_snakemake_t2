import requests
import random
import json
import string
import time

## see notes on deplyonig this for docker swarm in the "notes" application

# So it looks like I am mainly going to have to use the REST API here.
# https://github.com/dockersamples/atsea-sample-shop-app/blob/master/REST.md

# this is going to require:
# (1) a setup phase
# (2) an execution phase

# it doesn't look too bad


s = requests.Session()

r = s.get('https://192.168.99.107', verify=False)
print r.text

print "########"

r = s.get('https://192.168.99.107:443', verify=False)
print r.text

print "########"
succeeded = False
while not succeeded:
    r = s.get('https://192.168.99.107/api/product/', verify=False)
    try:
        for product in r.json():
            print product
        succeeded = True
    except:
        succeeded = False
        time.sleep(1)

print "########"

try:
    product = random.choice(r.json())
except:
    print "timeout"
#print product['productId']

r = s.get('https://192.168.99.107/api/product/' + str(product['productId']), verify=False)
print r.text

print "########"

username = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
print username

email_front = ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
email_back = ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
email = email_front + '@' + email_back + '.com'
print email

name = ''.join(random.choice(string.ascii_lowercase) for _ in range(5)) + ' ' + ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
print name

password = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
print password

new_costumer_data = {
    "name"       : name,
    "address"    : "144 Townsend, San Francisco 99999",
    "email"      : email,
    "phone"      : "513 222 5555",
    "username"   : username,
    "password"   : password,
    "enabled"    : "true",
    "role"       : "USER"
}
trys = 0

for i in range(1,2):
    succeeded = False
    while not succeeded:
        r = s.post('https://192.168.99.107/api/customer/', headers={'Content-type': 'application/json', 'Accept': 'application/json'},
                   data = json.dumps(new_costumer_data), verify=False)
        trys += 1
        print trys
        if r.status_code == 201:
            succeeded = True
        else:
            print r
        ## will it infinitely loop??

print r, r.text, r.status_code, "trys", trys
customerId = (new_costumer_data, r.json())
if r.status_code == 201:
    print customerId
    print customerId[1]["customerId"]
else:
    print "registering user did not succeed"
    exit(1)

## ^^ seems like maybe some spacing between requests is necessary

print "########"

login_data = {
    "username" : username,
    "password" : password
}

#time.sleep(5)
succeeded = False
trys = 0
start_time = time.time()
time.sleep(.5)
while not succeeded:
    r = s.post('https://192.168.99.107/login/', headers={'Content-type': 'application/json', 'Accept': 'application/json'},
               data = json.dumps(login_data))
    trys += 1
    if r.status_code == 200:
        succeeded = True
        end_time = time.time()

print "login!"
print r, r.text
print "login trys", trys, "time", end_time - start_time
token = ''
try:
    token = r.json()["token"]
    print token
except:
    print "login failed"

print "########"

order_data = {
    "orderId": 1,
    "orderDate" : "2017-02-28T19:52:39Z",
    "customerId" : customerId[1]["customerId"],
    "productsOrdered" : {product['productId']: random.randint(1,4) }
}
print "proudct", product['productId']
print "ordering data", order_data

'''
succeeded = False
trys = 0
while not succeeded:
    r = s.post('https://192.168.99.107/api/order/', headers={'Content-type': 'application/json', 'Accept': 'application/json'},
           data = json.dumps(order_data))
    trys += 1
    print trys
    #succeeded = True
    if r.status_code == 201:
        succeeded = True
    elif r.status_code == 409:
        order_data['orderId'] += 1
    else:
        print r, r.text
    time.sleep(0.5)
    if trys > 100:
        break

print r, r.text
'''

print "########"

'''
GET: /atsea/purchase/

Host: localhost:8080
Auth: 
Content-type: application/json
Accept: application/json
'''
print "purchasing..."

succeeded = False
trys = 0
while not succeeded:
    r = s.get('https://192.168.99.107/purchase/', headers={'Content-type': 'application/json', 'Accept': 'application/json',
                                                           'Authorization': "Bearer " + token},
              data=json.dumps(order_data))
    # NOTE: I am testing the data part (it worked before w/o it but what was it ordering??)
              #auth = (username, password))
    print r,r.text
    trys += 1
    print trys
    #succeeded = True
    print r.status_code != 500, r.status_code !=401
    if r.status_code != 500 and r.status_code !=401:
        succeeded = True


print "########"


'''


r = s.get('http://localhost:3000/pets')
print r.text

pets = r.json()
print pets

pet = random.choice(pets)

print "########"

r = s.get('http://localhost:3000/pets' + pet)
print r.text

print "########"
'''