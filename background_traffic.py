import base64

from locust import HttpLocust, TaskSet, task
from random import randint, choice
#import random
import string
import cPickle as pickle
import time

# this way we can simulate users using already-existing accounts
users = pickle.load( open( "users.pickle", "rb" ) )

class BackgroundTraffic(TaskSet):
    # the way I wrote this script, these are really the times between new users showing up
    min_wait = 2000
    max_wait = 4000

    # okay, so the goal here is to simulate an actual user
    # the user is going to browse for some randomized period of time
    # and then they are going to buy something with some probability
    @task
    def browse(self):

        #login before browsing with some probability
        login_p = randint(0,10)
        # let's arbitrarily say that 50% of visitors log in
        if login_p >= 5:
            # so let's randomly choose some already-registered user
            user = choice(users)
            # NOTE: this borrows code from weave's load-test repo
            base64string = base64.encodestring('%s:%s' % (user, password)).replace('\n', '')
            self.client.get("/login", headers={"Authorization":"Basic %s" % base64string})
            time.sleep(randint(min_wait,max_wait) / 1000.0) # going to wait a bit between events            

        # browse through the socks for a certain number of times
        num_browsing = randint(1,21)
        # NOTE: this borrows  code from weave's load-test repo 
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
            item_num = randint(1,6)
            self.client.post("/cart", json={"id": item_id, "quantity": item_num})
            time.sleep(randint(min_wait,max_wait) / 1000.0) # going to wait a bit between events
            self.client.get("/basket.html")
            self.client.post("/orders")
        
class GenBackgroundTraffic(HttpLocust):
    print "Can I see this??" # yes, yes I can
    task_set = BackgroundTraffic
