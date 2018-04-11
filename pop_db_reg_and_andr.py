import base64

from locust import HttpLocust, TaskSet, task
#from random import randint, choice
import random
import string
import cPickle as pickle

## please see GitHub issue #25 for why registering users to different 'levels'
## is necessary

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

class PopulateDatabaseRegAndAndr(TaskSet):
    # we don't need "long" wait times here to simulate user delay,
    # we just want stuff in the DB
    min_wait = 100
    max_wait = 101

    # let's implement login using the steps in the given Locust code
    def login(user, password):
        base64string = base64.encodestring('%s:%s' % (user, password)).replace('\n', '')
        self.client.get("/login", headers={"Authorization":"Basic %s" % base64string})

    # registers users and gives them an address
    @task
    def populate_data_reg_and_andr(self):
        print "about to populate this database!"
        # first register
        username = gen_random()
        #print "username: ", username
        # let's make it the same just to simplify things
        password = username
        #print "password: ", password
        # just to keep things simple....
        firstname = username + "ZZ" 
        lastname = username + "QQ"
        email = username + "@gmail.com"
        #print "email: ", email
        # now create the object that we will pass for registration
        registerObject = {"username": username, "password": password, firstname: "HowdyG", "lastName": lastname,"email":email}
        print registerObject
        userID = self.client.post("/register", json = registerObject).text
        #userID = self.client.post("/register", json=registerObject).text
        # tested to here! first part is working!
        #''' Let's test only the above part for now
        print "userID: ", userID
        # then login
        #login(username, password)
        base64string = base64.encodestring('%s:%s' % (username, password)).replace('\n', '')
        self.client.get("/login", headers={"Authorization":"Basic %s" % base64string})
        # in the interests of simplicity, I am simply going to use the same address for everyone
        # NOTE: this was one of the address records that came already-present in the sock shop
        addressObject = {"street":"Whitelees Road","number":"246","country":"United Kingdom","city":"Glasgow","postcode":"G67 3DL","id":userID}
        cAddr = self.client.post("/addresses", json=addressObject)
        print "Response from posting address: ", cAddr.text, " done"


class loadDBRegAndAndr(HttpLocust):
    print "Can I see this??" # yes, yes I can
    task_set = PopulateDatabaseRegAndAndr
