# modified from the one provided by the weave sock shop

import base64

from locust import HttpLocust, TaskSet, task
from random import randint, choice


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

    @task
    def browsing(self):
        pass

Class ExfiltrateData(TaskSet):
    
    # is it this easy??
    @task
    def exfil_data(self):
        user_data = self.client.get("/customers")
        print user_data

Class PopulateDatabase(TaskSet):

    def login(user, password):
        base64string = base64.encodestring('%s:%s' % (user, password)).replace('\n', '')
        self.client.get("/login", headers={"Authorization":"Basic %s" % base64string})

    # first register a user than register a credit card for that user (after login)
    # TODO randomize certain fields and then stick a big ol' loop on there
    @task
    def populate_data(self):
        # first register
        username = "string"
        password = "string"
        registerObject = {"username": username, "password":password,"email":"string"}
        userID = self.client.post("/register", json=registerObject)
        # then login
        login(username, password)
        # then register a credit card
        creditCardObject = {"longNum": "string", "expires": "string", "ccv": "string", "userID": userID}
        self.client.post("/cards", json=creditCardObject)

class Web(HttpLocust):
    task_set = WebTasks
    min_wait = 6000
    max_wait = 3000
