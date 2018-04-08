import unittest
import json
from pull_from_prom import *

'''
def mathTestSuite():
    #suite = unittest.TestSuite()
    #suite.addTest(WidgetTestCase("runMath"))
    suite = unittest.makeSuite(sampleTestCase)
    return suite

#mathTestCase = sampleTestCase("runMath")

class sampleTestCase(unittest.TestCase):
    def setup(self):
        self.foo = 2 + 3
    def tearDown(self):
        # nothing to tear down ATM
        pass
    def runMath(self):
        assert self.foo == 5

#class sampleTest(sampleTestCaseSetup):
#    def runTest(self):
#        assert self.foo == 5

#runner = unittest.TextTestRunner()
#runner.run(mathTestSuite)
'''

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

# okie, let's do some testing!
# I don't got all day to write tests, so we are only going
# to check some of the bigger functionality
class TestPullFromPromMethods(unittest.TestCase):
    maxDiff = None
    
    @classmethod
    def setUpClass(cls):
        # first read in the stored prometheus data from the files
        # this simulates the function that does an http get request 
        # to prometheus
        first_recieved = open("./tests/mongo_received_bytes_response.json", "r").read()
        second_recieved = open("./tests/istio_mongo_recieved_bytes_second.json", "r").read()
        first_sent = open("./tests/mongo_sent_bytes_response.json", "r").read()
        second_sent = open("./tests/istio_mongo_sent_bytes_second.json", "r").read()

        first_rec_json = json.loads(first_recieved)
        second_rec_json = json.loads(second_recieved)
        first_sent_json = json.loads(first_sent) 
        second_sent_json = json.loads(second_sent) 
  
        #print first_recieved
            
        #ip_to_service = json.loads(ip_to_serv_str)
        ip_to_service = {'172.17.0.18': ' kubernetes-dashboard-77d8b98585-ppjv6', '172.17.0.19': 'istio-mixer-86f5df6997-5srkr', '172.17.0.10': 'load-test', '172.17.0.11': 'orders-db', '172.17.0.12': 'catalogue', '172.17.0.13': 'user', '172.17.0.14': 'orders', '172.17.0.15': 'front-end', '172.17.0.16': 'payment', '172.17.0.17': 'load-test', '172.17.0.8': 'rabbitmq', '172.17.0.9': 'shipping', '172.17.0.2': 'user-db', '172.17.0.3': 'carts-db', '172.17.0.6': 'catalogue-db', '172.17.0.7': 'session-db', '172.17.0.4': 'queue-master', '172.17.0.5': 'carts', 'hello world': 'load-test', 'IP': 'NAME', '172.17.0.25': 'istio-ingress-5bb556fcbf-b8b8q', '172.17.0.24': 'istio-ca-86f55cc46f-wvtnc', '192.168.99.108': ' storage-provisioner', '172.17.0.26': 'prometheus-cf8456855-4x4gw', '172.17.0.21': ' influxdb-grafana-rzqtl', '172.17.0.20': ' heapster-srjxt', '172.17.0.23': 'istio-pilot-67d6ddbdf6-cl6pm', '172.17.0.22': ' kube-dns-54cccfbdf8-57cxw'}

        #print ip_to_service
        
        cls.first_recieved_matrix, cls.first_sent_matrix = process_prometheus_pull(first_rec_json, first_sent_json, ip_to_service)
        cls.sec_recieved_matrix, cls.sec_sent_matrix = process_prometheus_pull(second_rec_json, second_sent_json, ip_to_service)


    def test_ip_to_service_map(self):
        #print "testing ip to service functionality..."
        kubectl_get_out = open("./tests/kubectl_get_output.txt", "r").read()
        #print kubectl_get_out
        mapping = parse_ip_to_service_mapping(kubectl_get_out)
        #print mapping
        #print "done printing mapping..."
        #for ip,serv in mapping.iteritems():
        #    print ip, serv
        #print "correct mapping..."
        correct_mapping = {'172.17.0.5' : 'carts',
                '172.17.0.3': 'carts-db',
                '172.17.0.12': 'catalogue',
                '172.17.0.6': 'catalogue-db',
                '172.17.0.15': 'front-end',
                '172.17.0.14': 'orders',
                '172.17.0.11': 'orders-db',
                '172.17.0.16': 'payment',
                '172.17.0.4': 'queue-master',
                '172.17.0.8': 'rabbitmq',
                '172.17.0.7': 'session-db',
                '172.17.0.9': 'shipping',
                '172.17.0.13': 'user',
                '172.17.0.2': 'user-db',
                '172.17.0.24': 'istio-ca-86f55cc46f-wvtnc',
                '172.17.0.25': 'istio-ingress-5bb556fcbf-b8b8q',
                '172.17.0.19': 'istio-mixer-86f5df6997-5srkr',
                '172.17.0.23': 'istio-pilot-67d6ddbdf6-cl6pm',
                '172.17.0.26': 'prometheus-cf8456855-4x4gw',
                '172.17.0.20': 'heapster-srjxt',
                '172.17.0.21': 'influxdb-grafana-rzqtl',
                '192.168.99.108': 'kube-addon-manager-minikube',
                '172.17.0.22' : 'kube-dns-54cccfbdf8-57cxw',
                '172.17.0.18' : 'kubernetes-dashboard-77d8b98585-ppjv6',
                '192.168.99.108' : 'storage-provisioner', 
                '172.17.0.10' : 'load-test',
                '172.17.0.17': 'load-test',
                'hello world': 'load-test',
                'IP': 'NAME'} 
        #print correct_mapping
        #print "Here are the mappings..."
        #for ip,serv in correct_mapping.iteritems():
        #    print "extracted: ", ip, mapping[ip]
        #    print "correct:   ", ip,serv
        #print "mappings are complete..."
        #print correct_mapping
        #print mapping
        #for ip,serv in mapping.iteritems():

        self.assertDictEqual(mapping, correct_mapping)

    def test_process_prometheus_pull(self):        
        ## OKAY, we have all the matrixes, now let's compare them with what they 'should' be!
        ## Let's just spot test them for now, at some point maybe we can make it more complete
        #print self.first_recieved_matrix
        self.assertEqual(self.first_recieved_matrix.get_value('carts', 'carts-db'), 33958468)
        self.assertEqual(self.first_recieved_matrix.get_value('orders', 'carts'), 2731256)
        self.assertEqual(self.first_recieved_matrix.get_value('front-end', 'catalogue'), 1601785)
        #print first_recieved_matrix
    
    def test_calc_differential_matrix(self):
        self.assertEqual('FOO', 'FOO')

if __name__ == "__main__":
    unittest.main()
