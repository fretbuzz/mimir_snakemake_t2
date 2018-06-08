import requests
import random
import string

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
user_info = {'username': username, 'name': '', 'first_name': '', 'last_name': '',\
            'email': email, 'url': '', 'description': '', 'locale': '', 'nickname': '',\
            'roles': '',  'password': password}
# Required: username, email, password

s = requests.Session()

r = s.get('http://192.168.99.106:31721/wp-admin')
print r.text


print "########"


r = s.post('http://192.168.99.106:31721/wp-login.php',
                  data={'log':'user', 'pwd':'ENQ81MfUBl',
                  'redirect_to': 'http://192.168.99.106:31721/wp-admin/'})
#print r.text

print "########"

r = s.get('http://192.168.99.106:31721/wp-admin')
print r.text

print "########"


# should change IP as appropriate
r = s.post('http://192.168.99.106:31721/wp-json/wp/v2/users', json=user_info, auth=('user', 'ENQ81MfUBl'))
print r.text
r = s.get('http://192.168.99.106:31721/wp-json/wp/v2/users')
print r.text


### Okay, either use fakepress to just generate a super large number of stuff or get the posting to work here.
### TODO: (1) get login to work
#         (2) create large # of users (save login)
#         (3) create large # of posts (using previous logins)
#         (4) find way to walk through a bunch of random posts
#               # use API to list some subset (or whole thing) and then
#               # pick one randomly and repeat