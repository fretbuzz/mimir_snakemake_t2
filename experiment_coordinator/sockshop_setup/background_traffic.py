import base64

from locust import HttpLocust, TaskSet, task
from random import randint, choice
#import random
import string
import cPickle as pickle
import time
from pop_db import get_random_num, gen_random

# this way we can simulate users using already-existing accounts
with open( "users.pickle", "rb" ) as f:
    users = pickle.loads( f.read() )

class BackgroundTraffic(TaskSet):
    # the way I wrote this script, these are really the times between new users showing up
    min_wait = 2000
    max_wait = 4000

    # okay, so the goal here is to simulate an actual user
    # the user is going to browse for some randomized period of time
    # and then they are going to buy something with some probability
    @task(8)
    def browse(self):
        min_wait = 2000
        max_wait = 4000
        # login before browsing with some probability
        login_p = randint(0,10)
        # let's arbitrarily say that 70% of visitors log in
        if login_p >= 3:
            # so let's randomly choose some already-registered user
            user = choice(users)
            # NOTE: this borrows code from weave's load-test repo
            base64string = base64.encodestring('%s:%s' % (user, user)).replace('\n', '')
            self.client.get("/login", headers={"Authorization":"Basic %s" % base64string})
            time.sleep(randint(min_wait,max_wait) / 1000.0) # going to wait a bit between events            

        # browse through the socks for a certain number of times
        num_browsing = randint(1,21)
        # NOTE: this borrows  code from weave's load-test repo 
        item_id = 0 ##
        catalogue = self.client.get("/catalogue").json()
        for i in range(0, num_browsing):
            time.sleep(randint(min_wait,max_wait) / 1000.0) # going to wait a bit between events
            category_item = choice(catalogue)
            item_id = category_item["id"]
            self.client.get("/category.html")
            time.sleep(randint(min_wait,max_wait) / 1000.0) # going to wait a bit between events
            self.client.get("/detail.html?id={}".format(item_id))

        # buy some socks with some probability
        # let's arbitrarily say that 30% percent of visiters buy some socks
        if login_p >= 7:
            time.sleep(randint(min_wait,max_wait) / 1000.0) # going to wait a bit between events
            # NOTE: this borrows code from weave's load-test repo
            self.client.delete("/cart")
            item_num = 1 # randint(1,4) # can't be more than 100 and they have something that is 99.99
            self.client.post("/cart", json={"id": item_id, "quantity": item_num})
            time.sleep(randint(min_wait,max_wait) / 1000.0) # going to wait a bit between events
            self.client.get("/basket.html")
            order_post = self.client.post("/orders")
            print order_post
            print order_post.text

    @task(1) # this'll make a new user (just to keep things interesting)
    def populate_data(self):
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
        # ''' Let's test only the above part for now
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

    ## COPY-PASTED FROM HERE: https://github.com/microservices-demo/load-test/blob/master/locustfile.py
    ## I DID NOT WRITE THIS ONE--- it's from the offical repo
    @task(1)
    def load(self):
        base64string = base64.encodestring('%s:%s' % ('user', 'password')).replace('\n', '')

        catalogue = self.client.get("/catalogue").json()
        category_item = choice(catalogue)
        item_id = category_item["id"]

        self.client.get("/")
        self.client.get("/login", headers={"Authorization":"Basic %s" % base64string})
        self.client.get("/category.html")
        self.client.get("/detail.html?id={}".format(item_id))
        self.client.delete("/cart")
        self.client.post("/cart", json={"id": item_id, "quantity": 1})
        self.client.get("/basket.html")
        self.client.post("/orders")


class GenBackgroundTraffic(HttpLocust):
    print "Can I see this??" # yes, yes I can
    task_set = BackgroundTraffic
