import base64

from locust import HttpLocust, TaskSet, task
#from random import randint, choice
import random
import string
import cPickle as pickle
import json

users = []

def gen_customer_data():
    username = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
    #print username

    email_front = ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
    email_back = ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
    email = email_front + '@' + email_back + '.com'
    #print email

    name = ''.join(random.choice(string.ascii_lowercase) for _ in range(5)) + ' ' + ''.join(
        random.choice(string.ascii_lowercase) for _ in range(5))
    #print name

    password = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
    #print password

    new_costumer_data = {
        "name": name,
        "address": "144 Townsend, San Francisco 99999",
        "email": email,
        "phone": "513 222 5555",
        "username": username,
        "password": password,
        "enabled": "true",
        "role": "USER"
    }

    return new_costumer_data

class PopulateDatabase(TaskSet):
    # we don't need "long" wait times here to simulate user delay,
    # we just want stuff in the DB
    min_wait = 100
    max_wait = 101

    # first register a user than register a credit card for that user (after login)
    @task
    def populate_data(self):
        print "about to populate this database!"
        new_customer_info = gen_customer_data()

        trys = 0
        succeeded = False
        while not succeeded:
            r = self.client.post("/api/customer/",
                       headers={'Content-type': 'application/json', 'Accept': 'application/json'},
                       data=json.dumps(new_customer_info), verify=False)
            trys += 1
            print "trys", trys
            if r.status_code == 201:
                succeeded = True
                print "suceeded"
            else:
                print "failed"
                print r
                print r.text
                print "end failure message"

        print r, r.text, r.status_code, "trys", trys
        customerId = (new_customer_info, r.json())
        if r.status_code == 201:
            print customerId
            print customerId[1]["customerId"]
        else:
            print "registering user did not succeed"

        users.append(customerId)
        #already_users = pickle.load(open("seastore_users.pickle", "rb"))
        #print "already users", already_users, "new user", users
        #users.extend(already_users)
        pickle.dump( users, open( "seastore_users.pickle", "wb" ) )

class loadDB(HttpLocust):
    print "Can I see this??" # yes, yes I can
    task_set = PopulateDatabase
