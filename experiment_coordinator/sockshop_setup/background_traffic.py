import base64

from locust import HttpLocust, TaskSet, task
from random import randint, choice
#import random
import string
import cPickle as pickle
import time
from pop_db import get_random_num, gen_random
import os

# this way we can simulate users using already-existing accounts
with open( "users.pickle", "rb" ) as f:
    users = pickle.loads( f.read() )

if os.path.isfile('prob_distro_sock.pickle'):
    with open('prob_distro_sock.pickle', 'r') as f:
        prob_distr = pickle.loads(f.read())
else:
    ## here's a hypothetical default probability distribution...
    prob_distr = {'buy': 0.3, 'browse': 0.5, 'register': 0.2}

num_browsing = 21
min_wait = 2000
max_wait = 4000

class BackgroundTraffic(TaskSet):
    #'''
    # the way I wrote this script, these are really the times between new users showing up
    min_wait = 2000
    max_wait = 4000

    ## note: assuming visitor will only login if they mean to buy osmething...
    @task(prob_distr['buy'] * 100)
    def buy(self):
        self.client.get("/")
        # first login
        user = choice(users)
        # NOTE: this borrows code from weave's load-test repo
        base64string = base64.encodestring('%s:%s' % (user, user)).replace('\n', '')
        self.client.get("/login", headers={"Authorization": "Basic %s" % base64string})
        time.sleep(randint(min_wait, max_wait) / 1000.0)  # going to wait a bit between events

        catalogue = self.client.get("/catalogue").json()
        time.sleep(randint(min_wait,max_wait) / 1000.0) # going to wait a bit between events
        category_item = choice(catalogue)
        item_id = category_item["id"]
        self.client.get("/category.html")
        time.sleep(randint(min_wait,max_wait) / 1000.0) # going to wait a bit between events
        self.client.get("/detail.html?id={}".format(item_id))

        time.sleep(randint(min_wait, max_wait) / 1000.0)  # going to wait a bit between events
        # NOTE: this borrows code from weave's load-test repo
        self.client.delete("/cart")
        item_num = 1  # randint(1,4) # can't be more than 100 and they have something that is 99.99
        self.client.post("/cart", json={"id": item_id, "quantity": item_num})
        time.sleep(randint(min_wait, max_wait) / 1000.0)  # going to wait a bit between events
        self.client.get("/basket.html")
        order_post = self.client.post("/orders")
        print order_post
        print order_post.text

    @task(prob_distr['browse'] * 100)
    def browse(self):
        self.client.get("/")
        catalogue = self.client.get("/catalogue").json()
        for i in range(1, num_browsing):
            time.sleep(randint(min_wait,max_wait) / 1000.0) # going to wait a bit between events
            category_item = choice(catalogue)
            item_id = category_item["id"]
            self.client.get("/category.html")
            time.sleep(randint(min_wait,max_wait) / 1000.0) # going to wait a bit between events
            self.client.get("/detail.html?id={}".format(item_id))

    @task(prob_distr['register'] * 100)
    def register(self):
        #print "about to populate this database!"
        # first register
        username = gen_random()
        # print "username: ", username
        # let's make it the same just to simplify things
        password = username
        # print "password: ", password
        # just to keep things simple....
        firstname = username + "ZZ"
        lastname = username + "QQ"
        email = username + "@gmail.com"
        # print "email: ", email
        # now create the object that we will pass for registration
        registerObject = {"username": username, "password": password, firstname: "HowdyG", "lastName": lastname,
                          "email": email}
        #print registerObject
        userID = self.client.post("/register", json=registerObject).text
        # userID = self.client.post("/register", json=registerObject).text
        # tested to here! first part is working!
        # Let's test only the above part for now
        #print "userID: ", userID
        # then login
        # login(username, password)
        base64string = base64.encodestring('%s:%s' % (username, password)).replace('\n', '')
        self.client.get("/login", headers={"Authorization": "Basic %s" % base64string})
        # then register a credit card
        cc_num = get_random_num(16)
        expir_date = "11/2020"  # let's give everything the same expir_date b/c why not?
        ccv = get_random_num(3)
        creditCardObject = {"longNum": str(cc_num), "expires": str(expir_date), "ccv": str(ccv), "userID": userID}
        #print creditCardObject
        cc_req = self.client.post("/cards", json=creditCardObject)
        #print cc_req

        # in order to buy stuff, also need to have an address on file
        # in the interests of simplicity, I am simply going to use the same address for everyone
        # NOTE: this was one of the address records that came already-present in the sock shop
        addressObject = {"street": "Whitelees Road", "number": "246", "country": "United Kingdom", "city": "Glasgow",
                         "postcode": "G67 3DL", "id": userID}
        cAddr = self.client.post("/addresses", json=addressObject)

class GenBackgroundTraffic(HttpLocust):
    print "Can I see this??" # yes, yes I can
    task_set = BackgroundTraffic
