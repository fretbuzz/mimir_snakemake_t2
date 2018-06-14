import requests
import random
import string

print "starting!"

s = requests.Session()

r = s.get('http://192.168.99.106:31721/wp-admin')
print r.text

# this actualy seems not to complicated... okay which components are here
# TODO: (1) figure out how to autoscale docker compose
            # forget about autoscaling, let's just specify it in docker compose up --scale flag
            # too complicated w/ just compose. Easier w/ swarm
            # b/c automatic load balancing but need to modify
            # yaml file to save stuff to a local registry
            # note: I modified the compose file
            # note I think I need to make the image refernce
            # the local registry and not just use a name b/c
            # it looks for it in the cloud if local is not
            # specified (See https://docs.docker.com/engine/swarm/stack-deploy/#create-the-example-application)
            # okay
#      (2) things to go to to stress site:
            # so I gotta test the code that I wrote below
# http://localhost:3000/, http://localhost:3000/pets, http://localhost:3000/vendors, http://localhost:3000/about
# http://localhost:3000/pets/crabbe (so take each of the results from pets and then go to their 'more info' page)
# also need to send some emails; looks like the way to go about doing this is to post to http://localhost:8080/mail/send
# with this being what I saw sent out over chrome's developer tools
# also need to send some comments

# note there is no setup required for this site, just background traffic

''' # related to send emails
Request URL: http://localhost:8080/mail/send
Request Method: POST
Status Code: 200 OK
Remote Address: [::1]:8080
Referrer Policy: no-referrer-when-downgrade
Access-Control-Allow-Credentials: true
Access-Control-Allow-Origin: http://localhost:3000
connection: close
Vary: Origin
Accept: */*
Accept-Encoding: gzip, deflate, br
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Length: 39
Content-Type: application/json
DNT: 1
Host: localhost:8080
Origin: http://localhost:3000
Referer: http://localhost:3000/pets/crabbe?email=
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36
{email: "bob@bob.com", slug: "crabbe"}
'''
''' unparsed:
Access-Control-Allow-Credentials: true
Access-Control-Allow-Origin: http://localhost:3000
connection: close
Vary: Origin
POST /mail/send HTTP/1.1
Host: localhost:8080
Connection: keep-alive
Content-Length: 39
Origin: http://localhost:3000
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36
Content-Type: application/json
Accept: */*
DNT: 1
Referer: http://localhost:3000/pets/crabbe?email=
Accept-Encoding: gzip, deflate, br
Accept-Language: en-US,en;q=0.9
{"email":"bob@bob.com","slug":"crabbe"}


'''
''' # same but for comments (there were two more packets sent but I think it might have been updating page live instead
# of actually posting the info
Request URL: http://localhost:8080/comment/crabbe
Request Method: POST
Status Code: 201 Created
Remote Address: [::1]:8080
Referrer Policy: no-referrer-when-downgrade
Access-Control-Allow-Credentials: true
Access-Control-Allow-Origin: http://localhost:3000
connection: close
Vary: Origin
Accept: */*
Accept-Encoding: gzip, deflate, br
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Length: 36
Content-Type: application/json
DNT: 1
Host: localhost:8080
Origin: http://localhost:3000
Referer: http://localhost:3000/pets/crabbe
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36
{poster: "bob", content: "bobbert"}
content
:
"bobbert"
poster
:
"bob"

'''
## okay, so what services are we being sure to exercise
## vendors (yes, b/c looking @ vendors) (yes_
## offers (yes, b/c on homepage) (yes)
## comments (yes, registering comments (see above) (yes)
## functions (yes, automatically) (yes)
## email (yes, register emails) (yes)
## pets (yes, looking @ pets) (yes)
## front-end (yes, automatically) (yes)
## storefront (yes, automatically) (yes)


# http://localhost:3000/, http://localhost:3000/pets, http://localhost:3000/vendors, http://localhost:3000/about


s = requests.Session()

r = s.get('http://localhost:3000/')
print r.text

print "########"


r = s.get('http://localhost:3000/about')
print r.text

print "########"

r = s.get('http://localhost:3000/vendors')
print r.text

print "########"

r = s.get('http://localhost:3000/pets')
print r.text

pets = r.json()
print pets

pet = random.choice(pets)

print "########"

r = s.get('http://localhost:3000/pets' + pet)
print r.text

print "########"

# now email

# where crabbe should be replaced
r = s.post('http://localhost:8080/mail/send', data={'email': "bob@bob.com", 'slug': pet},
           headers={'referer': 'http://localhost:3000/pets/' + pet +'?email='})

print r.text

print "########"

# now comments

r = s.post('http://localhost:8080/comment/' + pet, data={'poster': "bob", 'content': "bobbert"},
           headers={'referer': 'http://localhost:3000/pets/' + pet})

print r.text

# so the next thing to do would be to put this into a locust file

# neo4j, mongodb, redis, pets, vendors, offers, comments,
# mail, storefront, frontend