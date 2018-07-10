import requests
import random
import string
import time
import base64
import json
import csv
#from loremipsum import get_sentences
import os

print "starting!"

# from https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python
username = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
print username

email_front = ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
email_back = ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
email = email_front + '@' + email_back + '.com'
print email

password = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
print password

# generate appropriate random values
#user_info = {'username': username, 'name': '', 'first_name': '', 'last_name': '',\
#            'email': email, 'url': '', 'description': '', 'locale': '', 'nickname': '',\
#            'roles': '',  'password': password}
# Required: username, email, password

s = requests.Session()
minikube_ip = '192.168.99.103'
wp_port = '31758'
# https://192.168.99.103:31758

print "getting admin page..."
r = s.get('https://' + minikube_ip + ':' + wp_port + '/wp-admin', verify=False)
print r
#print r.text

print "########"

# followed this guide to learn how to post posts
#https://discussion.dreamhost.com/t/how-to-post-content-to-wordpress-using-python-and-rest-api/65166
app_pass = 'a1tA u1Rc 6j5h 2NKc 90Hu COeo'
user = 'user'

token = base64.standard_b64encode(user + ':' + app_pass)
headers = {'Authorization': 'Basic ' + token}
# ^^ just include this in any requests that require sign in

print "posting..."
title = ''.join(random.choice(string.ascii_lowercase + ' ') for _ in range(random.randint(5,20)))
cont = ''.join(random.choice(string.ascii_lowercase + ' ' + '.') for _ in range(random.randint(20,80)))
user_num = str(1) #str( random.randint(1, 100) )

# okay, this one is going to be a bit tricky...
date_year = str( random.randint(2015,2018) )
date_month = str( random.randint(1,12) )
#print date_month, len(date_month), date_month[0]
if len(date_month) < 2:
    date_month = '0' + date_month
date_day = str( random.randint(1,29) )
if len(date_day) < 2:
    date_day = '0' + date_day
date_hour = str( random.randint(0,23) )
if len(date_hour) < 2:
    date_hour = '0' + date_hour
date_minute = str( random.randint(0,60) )
if len(date_minute) < 2:
    date_minute = '0' + date_minute
date_second = str( random.randint(1,60) )
if len(date_second) < 2:
    date_second = '0' + date_second
date = date_year + '-' + date_month + '-' + date_day + 'T' + date_hour + ':' +date_minute + ':' + date_second
print "date", date

# [[note: just following the guide (i.e. the website)]]
random_image = random.choice(os.listdir("./random_images/"))
print "rand image", random_image
picture = {'file': open("./random_images/" + random_image, 'rb'),
           'caption': 'A picture :)'}
image = requests.post('https://' + minikube_ip + ':' + wp_port + '/wp-json/wp/v2' + '/media',
                      headers=headers, files=picture, verify=False)
print('Your image is published on ' + json.loads(image.content)['link'])
image_link = json.loads(image.content)['source_url']

cont = cont + ' ' + '<img src=' + image_link + '>'

print cont

post = {'date': date, #'2017-06-19T20:00:35',
        'title': title,
        'status': 'publish',
        'content': cont,
        'author': user_num,
        'format': 'standard'
        }

r = requests.post('https://' + minikube_ip + ':' + wp_port + '/wp-json/wp/v2'
                  + '/posts', headers=headers, data=post, verify=False)
print('Your post is published on ' + json.loads(r.content)['link'])
postid = json.loads(r.content)['id']

#r = s.post('https://' + minikube_ip + ':' + wp_port + '/wp-login.php',
#                  data={'log':'user', 'pwd':'PoQPQSzNLE',
#                  'redirect_to': 'http://192.168.99.106:31721/wp-admin/'},
#            verify=False)
print r
#print r.text

print "########"
# following the same guide as above, I am now going to update the post
cont = ''.join(random.choice(string.ascii_lowercase + ' ' + '.') for _ in range(random.randint(20,80)))
cont = cont + ' ' + '<img src=' + image_link + '>'
updatedpost = {'content' : cont}
update = requests.post('https://' + minikube_ip + ':' + wp_port + '/wp-json/wp/v2' + '/posts/'
                       + str(postid), headers=headers, json=updatedpost, verify=False)
print('The updated post is published on ' + json.loads(update.content)['link'])

# this code has no real point now
#print 'checking admin page again...'
#r = s.get('https://' + minikube_ip + ':' + wp_port + '/wp-admin', verify=False)
#print r.text
#print r

print "########"


# should change IP as appropriate
#print "trying out the api..."
#r = s.post('https://' + minikube_ip + ':' + wp_port + '/wp-json/wp/v2/users', json=user_info,
#           auth=('user', 'ENQ81MfUBl'))
#print r.text
#r = s.get('https://' + minikube_ip + ':' + wp_port + '/wp-json/wp/v2/users')
#print r.text

#r = s.get('https://' + minikube_ip + ':' + wp_port + '/wp-json/')
#print r.text

r = s.get('https://' + minikube_ip + ':' + wp_port, verify=False)
#print r.text

r = s.get('https://192.168.99.103:31758/wp-json/wp/v2/posts', verify=False)
#print r.text

# okay, so let's randomly request a post.
# first, let's get the ID of the post with the highest ID
# we know what the ID of the post w/ the lowest ID is (??)
# so let's just randomly select a value and try that


### Okay, either use fakepress to just generate a super large number of stuff or get the posting to work here.
### TODO: (1) get login to work (done)
#         (2) create large # of users (save login) (can do using the automated tool)
#         (3) create large # of posts (using previous logins) (can do using automated tool + do
#               keep doing it during my tests using a the script)
#         (4) find way to walk through a bunch of random posts
#               # use API to list some subset (or whole thing) and then
#               # pick one randomly and repeat


urls = []

with open('wordpress_users.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    i = 0
    for row in spamreader:
        if 'http' in row[0]:
            urls.append(row[0].strip(','))
print urls
random_url = random.choice(urls)
print "rand url", random_url

r = s.get(random_url.replace('http', 'https'), verify=False)
print r
#print r.text

# let's 'try' to update a random page
# check first ID to get lowest ID
# https://192.168.99.103:31758/wp-json/wp/v2/posts?orderby=id&order=asc
# check first ID to get last ID
# https://192.168.99.103:31758/wp-json/wp/v2/posts?orderby=id&order=desc
# after selecting an ID, try
#https://192.168.99.103:31758/wp-json/wp/v2/posts?include=958
# now I think that every ID in this range should be 'taken', so we
# just need to choose one. randomly, using the random.choice
# function.
# note: there are a lot of gaps actually but I can just try twice then ;)

lowest_id = 1
highest_id = 994
update_succeeded = False
trys = 0
while not update_succeeded:
    trys += 1
    random_post = random.randint(lowest_id, highest_id)
    #random_post = 994

    updatedpost = {'content' : cont + ' ' + str(random_post)}
    update = requests.post('https://' + minikube_ip + ':' + wp_port + '/wp-json/wp/v2' + '/posts/'
                           + str(random_post), headers=headers, json=updatedpost, verify=False)
    try:
        print('The updated post is published on ' + json.loads(update.content)['link'])
        update_succeeded=True
        print 'that took this many tries: ', str(trys)
    except:
        print 'sorry buddy, but that post # was invalid'

# (1) I changed my mind, I am going to use the 'export all urls' method instead.
# so read in the csv file + choose a random url and do a get request on that.
# (2) follow guide on site to make a function that updates existing posts
# going to use magic admin powers to handle it
# (3) randomize values in posting + updating
# (4) move the thing over to the Locust thing.
    # ^^ this is where I am at
# (5) setup tcpdump for minikube network
# (6) get some pcaps
# (7) preprocess the pcaps to make sure that they are fine.
# (8) write down any limitations my load generator has.
# (9) sleep
