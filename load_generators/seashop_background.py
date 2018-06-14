import base64

from locust import HttpLocust, TaskSet, task
from random import randint, choice
#import random
import string
import cPickle as pickle
import time
import json
import random

# this way we can simulate users using already-existing accounts
users = pickle.load( open( "seastore_users.pickle", "rb" ) )

class BackgroundTraffic(TaskSet):
    # the way I wrote this script, these are really the times between new users showing up
    min_wait = 2000
    max_wait = 4000

    # okay, so the goal here is to simulate an actual user
    # the user is going to browse for some randomized period of time
    # and then they are going to buy something with some probability
    @task
    def browse(self):
        min_wait = 2000
        max_wait = 4000
        r = self.client.get('/', verify=False)
        # login before browsing with some probability
        login_p = randint(0,10)
        # let's arbitrarily say that 50% of visitors log in
        print "login p: ", login_p
        if login_p >= 5:
            # so let's randomly choose some already-registered user
            user = choice(users)
            # NOTE: this borrows code from weave's load-test repo
            #base64string = base64.encodestring('%s:%s' % (user, user)).replace('\n', '')
            #self.client.get("/login", headers={"Authorization":"Basic %s" % base64string})
            login_data = {
                "username": user[0]['username'],
                "password": user[0]['password']
            }

            # time.sleep(5)
            succeeded = False
            trys = 0
            start_time = time.time()
            while not succeeded:
                time.sleep(.5)
                r = self.client.post("/login/",
                           headers={'Content-type': 'application/json', 'Accept': 'application/json'},
                           data=json.dumps(login_data), verify=False)
                trys += 1
                if r.status_code == 200:
                    succeeded = True
                    end_time = time.time()

            print "login!"
            print "login trys", trys, "time", end_time - start_time
            token = ''
            try:
                token = r.json()["token"]
                print token
            except:
                print "login failed"
            time.sleep(randint(min_wait,max_wait) / 1000.0) # going to wait a bit between events


        # browse through a certain number of items
        num_browsing = randint(1,21)
        r = self.client.get(':443', verify=False)
        catalogue = self.client.get('/api/product/', verify=False)
        for i in range(0, num_browsing):
            time.sleep(randint(min_wait,max_wait) / 1000.0) # going to wait a bit between events
            product = random.choice(catalogue.json())
            r = self.client.get('/api/product/' + str(product['productId']), verify=False)

        # buy something with some probability
        # let's arbitrarily say that 30% percent of visiters buy something
        # NOTE: This function is still TODO
        if login_p >= 7:
            time.sleep(randint(min_wait,max_wait) / 1000.0) # going to wait a bit between events
            order_data = {
                "orderId": 1,
                "orderDate": "2017-02-28T19:52:39Z",
                "customerId": user[1]["customerId"],
                "productsOrdered": {product['productId']: random.randint(1, 4)}
            }
            succeeded = False
            trys = 0
            while not succeeded:
                r = self.client.get('/purchase/',
                          headers={'Content-type': 'application/json', 'Accept': 'application/json',
                                   'Authorization': "Bearer " + token},
                          data=json.dumps(order_data))
                # NOTE: I am testing the data part (it worked before w/o it but what was it ordering??)
                # auth = (username, password))
                print r, r.text
                trys += 1
                print trys
                # succeeded = True
                print r.status_code != 500, r.status_code != 401
                if r.status_code != 500 and r.status_code != 401:
                    succeeded = True

class GenBackgroundTraffic(HttpLocust):
    print "Can I see this??" # yes, yes I can
    task_set = BackgroundTraffic
