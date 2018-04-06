# modified from the one provided by the weave sock shop

import base64

from locust import HttpLocust, TaskSet, task
from random import randint, choice
import string

class WebTasks(TaskSet):
    
    def login(user, password):
        base64string = base64.encodestring('%s:%s' % (user, password)).replace('\n', '')
        self.client.get("/login", headers={"Authorization":"Basic %s" % base64string})

    @task
    def load(self):
        catalogue = self.client.get("/catalogue").json()
        category_item = choice(catalogue)
        item_id = category_item["id"]

        login('user', 'password')
        self.client.get("/")
        self.client.get("/category.html")
        self.client.get("/detail.html?id={}".format(item_id))
        self.client.delete("/cart")
        self.client.post("/cart", json={"id": item_id, "quantity": 1})
        self.client.get("/basket.html")
        self.client.post("/orders")

    #    @task
    #def browsing(self):
    #    pass

class ExfiltrateData(TaskSet):
    
    # is it this easy??
    @task
    def exfil_data(self):
        user_data = self.client.get("/customers")
        print user_data

class PopulateDatabase(TaskSet):
    # we don't need "long" wait times here to simulate user delay,
    # we just want stuff in the DB
    min_wait = 100
    max_wait = 101

    # let's implement login using the steps in the given Locust code
    def login(user, password):
        base64string = base64.encodestring('%s:%s' % (user, password)).replace('\n', '')
        self.client.get("/login", headers={"Authorization":"Basic %s" % base64string})
    
    def gen_random():
        username = ''
        for i in range(0,10):
            username += random.choice(string.ascii_lowercase)
        return username 
           
    def get_random_num(num):
        cc_num = ''
        for i in range(0,num):
            cc_num += random.choice(string.digits)
        return cc_num

    # first register a user than register a credit card for that user (after login)
    # TODO randomize certain fields and then stick a big ol' loop on there
    @task
    def populate_data(self):
        # first register
        username = gen_random()
        # let's make it the same just to simplify things
        password = username
        # just to keep things simple....
        email = username + "@gmail.com"
        # now create the object that we will pass for registration
        registerObject = {"username": username, "password":password,"email":email}
        userID = self.client.post("/register", json=registerObject)
        # then login
        login(username, password)
        # then register a credit card
        cc_num =  get_random_num(16)
        expir_date = "11/2020" # let's give everything the same expir_date b/c why not?
        ccv = get_random_num(3)
        creditCardObject = {"longNum": "string", "expires": "string", "ccv": "string", "userID": userID}
        self.client.post("/cards", json=creditCardObject)

class Web(HttpLocust):
    task_set = WebTasks
    min_wait = 6000
    max_wait = 3000

class loadDB(HttpLocust):
    task_set = PopulateDatabase
