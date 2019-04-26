import base64

from locust import HttpLocust, TaskSet, task
from random import randint, choice
#import random
import string
import cPickle as pickle
import time
import csv
import json
import os


# this way we can simulate users using already-existing accounts
#users = pickle.load( open( "users.pickle", "rb" ) )
urls = []
#failures_list = [0]

with open('./wordpress_setup/failures_list.txt', 'r') as f:
    lines = f.readlines()
    failures_list = [int(i.rstrip()) for i in lines]
    failures_list = list(set(failures_list))

#with open('./load_generators/wordpress_users.csv', 'rb') as csvfile:
with open('./wordpress_setup/wordpress_users.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    i = 0
    for row in spamreader:
        if 'http' in row[0]:
            #row[0] = row[0].replace('http', "https")
            urls.append(row[0].strip(','))


# id's refer to posts
lowest_id = 1
highest_id = 750

# I'm just having the admin modify everything ATM
app_pass = None #"Somv XGRh FStO U0wq uACo x4s2"

dir_path = os.path.dirname(os.path.realpath(__file__))
print "dir_path", dir_path
with open('./wordpress_setup/wordpress_api_pwd.txt','r') as f:
    lines = f.readlines()
    app_pass = lines[0]
	
user = 'user'
token = base64.standard_b64encode(user + ':' + app_pass)
headers = {'Authorization': 'Basic ' + token}

if os.path.isfile('prob_distro_wp.pickle'):
    with open('prob_distro_wp.pickle', 'r') as f:
        prob_distr = pickle.loads(f.read())
else:
    ## here's a hypothetical default probability distribution...
    prob_distr = {'visitor': 0.4, 'user_post': 0.4, 'user_update': 0.2}

min_wait = 2000
max_wait = 4000
num_browsing_max = 21

class BackgroundTraffic(TaskSet):
    min_wait = 2000
    max_wait = 4000

    @task(int(prob_distr['visitor'] * 100))
    def visitor(self):
        num_browsing = randint(1,num_browsing_max)
        for i in range(0, num_browsing):
            random_url = choice(urls)
            print "rand url", random_url
            r = self.client.get(random_url.replace('http', 'https'), verify=False)
            print r
            time.sleep(randint(min_wait,max_wait) / 1000.0) # going to wait a bit between events

    @task(int(prob_distr['user_post'] * 100))
    def user_post(self):
        title = ''.join(choice(string.ascii_lowercase + ' ') for _ in range(randint(5, 20)))
        cont = ''.join(choice(string.ascii_lowercase + ' ' + '.') for _ in range(randint(20, 80)))
        user_num = str(1)  # str( random.randint(1, 100) )

        # okay, this one is going to be a bit tricky...
        # okay, this one is going to be a bit tricky...
        date_year = str(randint(2015, 2018))
        date_month = str(randint(1, 12))
        # print date_month, len(date_month), date_month[0]
        if len(date_month) < 2:
            date_month = '0' + date_month
        date_day = str(randint(1, 29))
        if len(date_day) < 2:
            date_day = '0' + date_day
        date_hour = str(randint(0, 23))
        if len(date_hour) < 2:
            date_hour = '0' + date_hour
        date_minute = str(randint(0, 60))
        if len(date_minute) < 2:
            date_minute = '0' + date_minute
        date_second = str(randint(1, 60))
        if len(date_second) < 2:
            date_second = '0' + date_second
        date = date_year + '-' + date_month + '-' + date_day + 'T' + date_hour + ':' + date_minute + ':' + date_second
        print "date", date

        # 5% chance this post contains a picture
        if randint(1, 20) > 19:
            random_image = choice(os.listdir("./random_images/"))
            print "rand image", random_image
            picture = {'file': open("./random_images/" + random_image, 'rb'),
                       'caption': 'A picture :)'}
            image = self.client.post('/wp-json/wp/v2' + '/media',
                                     headers=headers, files=picture, verify=False)
            print('Your image is published on ' + json.loads(image.content)['link'])
            image_link = json.loads(image.content)['source_url']
            cont = cont + ' ' + '<img exp_support_scripts=' + image_link + '>'

        post = {'date': date,  # '2017-06-19T20:00:35',
                'title': title,
                'status': 'publish',
                'content': cont,
                'author': user_num,
                'format': 'standard'
                }

        r = self.client.post('/wp-json/wp/v2' + '/posts', headers=headers, data=post, verify=False)
        print('Your post is published on ' + json.loads(r.content)['link'])
        postid = json.loads(r.content)['id']

        print r
        time.sleep(randint(min_wait, max_wait) / 1000.0)  # going to wait a bit between events

        # note: user will also browse a little too (b/c more realistic that way)
        num_browsing = randint(1, num_browsing_max)
        for i in range(0, num_browsing):
            random_url = choice(urls)
            print "rand url", random_url
            r = self.client.get(random_url.replace('http', 'https'), verify=False)
            print r
            time.sleep(randint(min_wait, max_wait) / 1000.0)  # going to wait a bit between events

    @task(int(prob_distr['user_update'] * 100))
    def user_update(self):
        update_succeeded = False
        trys = 0
        cont = ''.join(choice(string.ascii_lowercase + ' ' + '.') for _ in range(randint(20, 80)))
        while not update_succeeded:
            trys += 1
            random_post = 0
            while random_post in failures_list:
                random_post = randint(lowest_id, highest_id)
            # random_post = 994

            updatedpost = {'content': cont + ' ' + str(random_post)}
            # note: i am just having the admin update the page (otherwise would
            # need a list of passwords and stuff)
            update = self.client.post('/wp-json/wp/v2' + '/posts/'
                                      + str(random_post), headers=headers, json=updatedpost, verify=False)
            try:
                print('The updated post is published on ' + json.loads(update.content)['link'])
                update_succeeded = True
                print 'that took this many tries: ', str(trys)
            except:
                print 'sorry buddy, but that post # was invalid'
                failures_list.append(random_post)

        time.sleep(randint(min_wait, max_wait) / 1000.0)  # going to wait a bit between events

        # note: user will also browse a little too (b/c more realistic that way)
        num_browsing = randint(1, num_browsing_max)
        for i in range(0, num_browsing):
            random_url = choice(urls)
            print "rand url", random_url
            r = self.client.get(random_url.replace('http', 'https'), verify=False)
            print r
            time.sleep(randint(min_wait, max_wait) / 1000.0)  # going to wait a bit between events

    def teardown(self):
        with open('./wordpress_setup/failures_list.txt', 'w') as f:
            for item in failures_list:
                f.write(str(item) + '\n')

class GenBackgroundTraffic(HttpLocust):
    print "Can I see this??" # yes, yes I can
    print "len ", len(urls)
    task_set = BackgroundTraffic
