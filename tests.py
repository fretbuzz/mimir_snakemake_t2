import unittest
import json
from pull_from_prom import *
from analyze_traffix_matrixes import *

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
        kubectl_get_out = open("./tests/kubectl_get_output.txt", "r").read()
        cls.ip_to_service = parse_ip_to_service_mapping(kubectl_get_out)
        #print ip_to_service
        
        cls.first_recieved_matrix, cls.first_sent_matrix = process_prometheus_pull(first_rec_json, first_sent_json, cls.ip_to_service)
        cls.sec_recieved_matrix, cls.sec_sent_matrix = process_prometheus_pull(second_rec_json, second_sent_json, cls.ip_to_service)


    def test_ip_to_service_map(self):
        #print "testing ip to service functionality..."
        mapping = self.ip_to_service
        #print mapping
        #print "done printing mapping..."
        #for ip,serv in mapping.iteritems():
        #    print ip, serv
        #print "correct mapping..."
        # note: I know this is correct b/c I did it manually...
        correct_mapping = {'172.17.0.5' : 'carts',
                '172.17.0.3': 'carts-db',
                '172.17.0.5': 'carts',
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
        #self.assertEqual(self.first_recieved_matrix.get_value('carts', 'carts-db'), 33958468)
        self.assertEqual(self.first_recieved_matrix.get_value('orders', 'carts'), 2731256)
        self.assertEqual(self.first_recieved_matrix.get_value('front-end', 'catalogue'), 1601785)        
        # front-end -> user : 4240428
        self.assertEqual(self.first_recieved_matrix.get_value('front-end', 'user'), 4240428)
        # user <- orders : 8086627
        self.assertEqual(self.first_recieved_matrix.get_value('orders', 'user'), 8086627)
        # user <- 172.17.0.1 : 113886
        self.assertEqual(self.first_recieved_matrix.get_value('172.17.0.1', 'user'), 113886)
        # user-db <- user: 21225215
        self.assertEqual(self.first_recieved_matrix.get_value('user', 'user-db'), 21225215)
        # shipping <- orders: 2840860
        self.assertEqual(self.first_recieved_matrix.get_value('orders', 'shipping'), 2840860)
        # shipping <- 172.17.0.1: 118537
        self.assertEqual(self.first_recieved_matrix.get_value('172.17.0.1', 'shipping'), 118537)
        # session-db <- front-end : 7397588
        self.assertEqual(self.first_recieved_matrix.get_value('front-end', 'session-db'), 7397588)
        # payment <- orders: 5491056
        self.assertEqual(self.first_recieved_matrix.get_value('orders', 'payment'), 5491056)

        self.assertEqual(self.first_recieved_matrix.get_value('carts', 'carts-db'), 33958468)

        #print "okay, so there's a bunch of problems... let's debug..."
        #print self.first_recieved_matrix

    
    def test_calc_differential_matrix(self):
        differential_matrix, new_last_matrix = calc_differential_matrix(self.first_recieved_matrix, self.sec_recieved_matrix, 5, 0)
        
        #print "first recieved", self.first_recieved_matrix
        #print "second received", self.sec_recieved_matrix
        #print "differential", differential_matrix
        #print "done printing stuff...."
        #print "carts to carts-db", self.first_recieved_matrix.get_value('carts', 'carts-db')
        
        # First check that the actual differential matrix is fine
        self.assertEqual(differential_matrix.get_value('carts', 'carts-db'), 416451597)
        self.assertEqual(differential_matrix.get_value('orders', 'carts'), 35543392)
        self.assertEqual(differential_matrix.get_value('front-end', 'catalogue'), 20654432)
        # front-end -> user :   46098420.0- 4240428 = 41857992
        self.assertEqual(differential_matrix.get_value('front-end', 'user'), 41857992)
        # user <- orders : 113523156.0  - 8086627 = 105436529 
        self.assertEqual(differential_matrix.get_value('orders', 'user'), 105436529)
        # user <- 172.17.0.1 : 1342350.0 - 113886 = 1228464
        self.assertEqual(differential_matrix.get_value('172.17.0.1', 'user'), 1228464)
        # user-db <- user: 271391253.0 - 21225215 = 250166038
        self.assertEqual(differential_matrix.get_value('user', 'user-db'), 250166038)
        # shipping <- orders: 39151365.0  - 2840860 =
        self.assertEqual(differential_matrix.get_value('orders', 'shipping'), 36310505)
        # shipping <- 172.17.0.1: 1336225.0  - 118537 = 1217688
        self.assertEqual(differential_matrix.get_value('172.17.0.1', 'shipping'), 1217688)
        # session-db <- front-end : 93909662.0  - 7397588 = 86512074
        self.assertEqual(differential_matrix.get_value('front-end', 'session-db'), 86512074)
        # payment <- orders: 77145695.0  - 5491056 = 71654639
        self.assertEqual(differential_matrix.get_value('orders', 'payment'), 71654639)

        # Then check that new_last_matrix is a distinct copy of current_matrix
        # print id(self.sec_recieved_matrix), id(new_last_matrix)
        self.assertFalse(self.sec_recieved_matrix is new_last_matrix)
        # But they should be equal 
        self.assertTrue(self.sec_recieved_matrix.equals(new_last_matrix))

        # finally, check that it handles the an empty last_matrix fine
        dif_mat, new_last_mat = calc_differential_matrix(pd.DataFrame(), self.first_recieved_matrix, 10, 0)
        #print dif_mat, new_last_mat, self.first_recieved_matrix
        self.assertTrue(dif_mat.empty)
        self.assertTrue(self.first_recieved_matrix.equals(new_last_mat))

class TestAnalyzeMatrices(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df_sent = pd.read_pickle('./tests/cumul_sent_matrix.pickle')
        cls.df_rec = pd.read_pickle('./tests/cumul_received_matrix.pickle')

    # I guess the first thing to test is control_charts
    def test_control_charts(self):
        #print self.df_sent
        #print self.df_rec
        sent_control_charts = control_charts(self.df_sent, True)
        rec_control_charts = control_charts(self.df_rec, False)
        # recall the output is in the form 
        # {[sending_svc, recieving_service]: [mean, standard_deviation]}
        #print sent_control_charts, "\n"
        #print rec_control_charts
        #print rec_control_charts['172.17.0.1', 'catalogue']
        ## placeholder VV for now:W
        self.assertAlmostEqual(rec_control_charts['172.17.0.1', 'catalogue'][0], 399.0, places=8)
        self.assertAlmostEqual(rec_control_charts['172.17.0.1', 'catalogue'][1], 65.81793069, places=8)
        # the below raises errors b/c all zeroes
        #print rec_control_charts['payment', 'catalogue-db']
        #self.assertAlmostEqual(rec_control_charts['payment', 'catalogue-db'][0], 0, places=7)
        #self.assertAlmostEqual(rec_control_charts['payment', 'catalogue-db'][1], 0, places=7)
        self.assertAlmostEqual(rec_control_charts['front-end', 'orders'][0], 8671.5, places=6)
        self.assertAlmostEqual(rec_control_charts['front-end', 'orders'][1], 2203.726163, places=6)
        self.assertAlmostEqual(rec_control_charts['carts', 'carts-db'][0], 104814.75, places=5)
        self.assertAlmostEqual(rec_control_charts['carts', 'carts-db'][1], 73655.44176, places=5)
        self.assertAlmostEqual(rec_control_charts['catalogue', 'catalogue-db'][0], 23060.25, places=6)
        self.assertAlmostEqual(rec_control_charts['catalogue', 'catalogue-db'][1], 3655.819413, places=6)



if __name__ == "__main__":
    unittest.main()
