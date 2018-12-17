# doesn't do much ATM, but in like 2 hours I could automate the whole setup for wordpress...

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import sys
import time

if len(sys.argv) <= 2:
    print "needs an ip address, budy"

print sys.argv
ip_of_wp = sys.argv[1]
port_of_wp = sys.argv[2]

driver = webdriver.Firefox()
#driver.get("http://www.python.org")
driver.get('https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin')
#assert "Python" in driver.title
#elem = driver.find_element_by_name("q")
elem = driver.find_element_by_name("log")
elem.clear()
elem.send_keys("user")
elem = driver.find_element_by_name("pwd")
elem.clear()
elem.send_keys("Efhzu97VQe")
elem.send_keys(Keys.RETURN)
print elem
time.sleep(5)
driver.get('https://' + ip_of_wp + ':' + port_of_wp + '/wp-admin/plugin-install.php?tab=search&s=faker')
faker_press_install = driver.find_element_by_class_name('install-now button')
#for elem in login_form:
#    print 'element.text: {0}'.format(elem)
print faker_press_install
faker_press_install.click()

#elem.send_keys("pycon")
#elem.send_keys(Keys.RETURN)
#assert "No results found." not in driver.page_source
time.sleep(10)
driver.close()