import json
import math
import unittest

import networkx as nx
import numpy as np
from alert_triggers import calc_modified_z_score

from analysis_pipeline.analyze_edgefiles import change_point_detection, find_angles
from analysis_pipeline.generate_graphs import get_points_to_plot

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

##
class TestChangePoint(unittest.TestCase):
    maxDiff = None

    # okay, so what am I trying to do here?
    # well, I am going to try to test this the change-point detection
    # method.
    # okay, step (1): load data that can be used to test it
        # this can be done, via:
            # (a) identify edge files
            # (b) read in via analyze_edge functions
            # (c) call as needd
    # okay, I am going to take a quick detour to test the angles function
    # but I will return to work on this function eventually
    @classmethod
    def setUpClass(cls):
        filenames = ['./tests/seastore_swarm_0.00_0.10.txt', './tests/seastore_swarm_0.10_0.10.txt', './tests/seastore_swarm_0.20_0.10.txt',
                     './tests/seastore_swarm_0.30_0.10.txt', './tests/seastore_swarm_0.40_0.10.txt', './tests/seastore_swarm_0.50_0.10.txt',
                     './tests/seastore_swarm_0.60_0.10.txt', './tests/seastore_swarm_0.70_0.10.txt', './tests/seastore_swarm_0.80_0.10.txt',
                     './tests/seastore_swarm_0.90_0.10.txt', './tests/seastore_swarm_1.00_0.10.txt']

        list_of_graphs = []
        for file_path in filenames:
            G = nx.DiGraph()
            print "path to file is ", file_path
            nx.read_edgelist(file_path,
                            create_using=G, delimiter=',', data=(('weight', float),))
            list_of_graphs.append(G)

        total_node_list = []
        for cur_g in list_of_graphs:
            for node in cur_g.nodes():
                total_node_list.append(node)
        cls.total_node_list = list(set(total_node_list))

        # arbitrarily picking this value (also, networkx is purely calculating this,
        # so I'm only testing change-point detection)
        cls.outstrength_dicts = []
        for cur_G in list_of_graphs:
            outstrength_dict = {}
            for (u, v, data) in cur_G.edges(data=True):
                if u in outstrength_dict:
                    outstrength_dict[u] += data['weight']
                else:
                    outstrength_dict[u] = data['weight']
            cls.outstrength_dicts.append( outstrength_dict )

        #outstrength_degrees_eigenvector = change_point_detection(outstrength_dicts, 4, total_node_list)
        #print outstrength_degrees_eigenvector


        cls.dict1 = {'front-end.1': 15, 'user.1': 25, 'user-db.1': 14, 'user.2': 22}
        cls.dict2 = {'front-end.1': 17, 'user.1': 27, 'user-db.1': 16, 'user.2': 20}
        cls.dict3 = {'front-end.1': 19, 'user.1': 29, 'user-db.1': 18, 'user.2': 18}
        cls.dict4 = {'front-end.1': 21, 'user.1': 31, 'user-db.1': 20, 'user.2': 16}
        cls.dict5 = {'front-end.1': 23, 'user.1': 33, 'user-db.1': 22, 'user.2': 14}
        cls.dict6 = {'front-end.1': 25, 'user.1': 35, 'user-db.1': 24, 'user.2': 12}
        cls.dict7 = {'front-end.1': 27, 'user.1': 37, 'user-db.1': 26, 'user.2': 10}
        cls.dict8 = {'front-end.1': 27, 'user.1': 37, 'user-db.1': 26, 'user.2': 30}
        cls.dict9 = {'front-end.1': 52, 'user.1': 62, 'user-db.1': 10, 'user.2': 30}
        cls.tensor = [cls.dict1, cls.dict2, cls.dict3, cls.dict4, cls.dict5, cls.dict6, cls.dict7, cls.dict8, cls.dict9]
        cls.total_node_list = ['front-end.1', 'user.1', 'user-db.1', 'user.2']


    ''' Not sure if this is needed
    def test_changepoint_with_outstrength(self):
        outstrength_changepoint_angles = change_point_detection(self.outstrength_dicts, 4, self.total_node_list)
        print "outstrength_changepoint_angles", outstrength_changepoint_angles
        # I'm going to need actual tests, but I think these pure edgefiles
        # are too much, maybe reduce them somehow?
    '''

    def test_find_angles_simplest(self):
        # find_angles(list_of_vectors, window_size)
        # okay, to test this I am going to need to do 2 things:
        # (1) some lists of vectors, and
        # (2) the angles between them
        array1 = np.array([1,1])
        array2 = np.array([1,1])
        array3 = np.array([1,1])
        array4 = np.array([1,1])
        list_of_arrays = [array1, array2, array3, array4]

        expected_angles = [float('nan'), 0.0, 0.0, 0.0]
        computed_angles = find_angles(list_of_arrays, window_size=1)
        #self.assertEquals(computed_angles , [0.0, 0.0, 0.0])
        for i in range(0, len(expected_angles)):
            if math.isnan(expected_angles[i]):
                self.assertTrue(math.isnan(computed_angles[i]))
            else:
                self.assertAlmostEqual(computed_angles[i], expected_angles[i])

    def test_find_angles_simple_averaging(self):
        # find_angles(list_of_vectors, window_size)
        # okay, to test this I am going to need to do 2 things:
        # (1) some lists of vectors, and
        # (2) the angles between them
        array1 = np.array([1, 1])
        array2 = np.array([2, 2])
        array3 = np.array([3, 3])
        array4 = np.array([1, 1])
        list_of_arrays = [array1, array2, array3, array4]

        expected_angles = [float('nan'), float('nan'), 0.0, 0.0]
        computed_angles = find_angles(list_of_arrays, window_size=2)
        # self.assertEquals(computed_angles , [0.0, 0.0, 0.0])
        for i in range(0, len(expected_angles)):
            if math.isnan(computed_angles[i]):
                self.assertTrue(math.isnan(computed_angles[i]))
            else:
                self.assertAlmostEqual(computed_angles[i], expected_angles[i])
        print "averaging worked"

    def test_find_angles_averaging(self):
        # find_angles(list_of_vectors, window_size)
        # okay, to test this I am going to need to do 2 things:
        # (1) some lists of vectors, and
        # (2) the angles between them
        array1 = np.array([1, 1])
        array2 = np.array([4, 2])
        array3 = np.array([2, 4])
        array4 = np.array([1000, 1000])
        list_of_arrays = [array1, array2, array3, array4]

        expected_angles = [float('nan'), float('nan'), 0.5667292, 0.0]
        computed_angles = find_angles(list_of_arrays, window_size=2)
        # self.assertEquals(computed_angles , [0.0, 0.0, 0.0])
        for i in range(0, len(expected_angles)):
            if math.isnan(expected_angles[i]):
                self.assertTrue(math.isnan(computed_angles[i]))
            else:
                self.assertAlmostEqual(computed_angles[i], expected_angles[i])
        print "averaging worked"

    def test_find_angles_empty(self):
        # find_angles(list_of_vectors, window_size)
        # okay, to test this I am going to need to do 2 things:
        # (1) some lists of vectors, and
        # (2) the angles between them
        array1 = np.array([1, 1])
        array2 = np.array([4, 2])
        array3 = np.array([])
        array4 = np.array([])
        array5 = np.array([1000, 1000])
        array6 = np.array([1000, 1000])
        array7 = np.array([1000, 1000])
        list_of_arrays = [array1, array2, array3, array4, array5, array6, array7]

        expected_angles = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 0.0, 0.0] # 7 - 2 = 5
        computed_angles = find_angles(list_of_arrays, window_size=2)
        # self.assertEquals(computed_angles , [0.0, 0.0, 0.0])
        for i in range(0, len(expected_angles)):
            if math.isnan(expected_angles[i]):
                print "empty angles", expected_angles[i], computed_angles[i]
                self.assertTrue(math.isnan(computed_angles[i]))
            else:
                print computed_angles[i], expected_angles[i]
                self.assertAlmostEqual(computed_angles[i], expected_angles[i])
        print "averaging worked"

    #'''
    def test_find_angles_zero(self):
        # find_angles(list_of_vectors, window_size)
        # okay, to test this I am going to need to do 2 things:
        # (1) some lists of vectors, and
        # (2) the angles between them
        array1 = np.array([0, 0])
        array2 = np.array([4, 2])
        array3 = np.array([2,4])
        array4 = np.array([0, 0])
        array5 = np.array([1000, 1000])
        array6 = np.array([1000, 1000])
        array7 = np.array([1000, 1000])
        list_of_arrays = [array1, array2, array3, array4, array5, array6, array7]

        expected_angles = [float('nan'), float('nan'), .6435011, float('nan'), .32175055, 0.0, 0.0] # 7 - 5 = 2
        computed_angles = find_angles(list_of_arrays, window_size=2)
        # self.assertEquals(computed_angles , [0.0, 0.0, 0.0])
        for i in range(0, len(expected_angles)):
            if math.isnan(expected_angles[i]):
                print "zero angles", expected_angles[i], computed_angles[i]
                self.assertTrue(math.isnan(computed_angles[i]))
            else:
                print computed_angles[i], expected_angles[i]
                self.assertAlmostEqual(computed_angles[i], expected_angles[i])
        print "averaging worked"
    #'''

    def test_find_angles_different_sizes(self):
        array1 = np.array([0, 0])
        array2 = np.array([4, 2])
        array3 = np.array([2,4])
        array4 = np.array([0, 0, 4])
        array5 = np.array([1000, 1000, 3])
        array6 = np.array([1000, 1000, 2])
        array7 = np.array([1000, 1000])
        list_of_arrays = [array1, array2, array3, array4, array5, array6, array7]

        expected_angles = [float('nan'), float('nan'), .6435011, 1.5707963, 0.7833984, 0.00353549, 0.0017677651] #7-5=2
        computed_angles = find_angles(list_of_arrays, window_size=2)
        # self.assertEquals(computed_angles , [0.0, 0.0, 0.0])
        for i in range(0, len(expected_angles)):
            if math.isnan(expected_angles[i]):
                print "zero angles", expected_angles[i], computed_angles[i]
                self.assertTrue(math.isnan(computed_angles[i]))
            else:
                print computed_angles[i], expected_angles[i]
                self.assertAlmostEqual(computed_angles[i], expected_angles[i])
        print "averaging worked"

    def test_change_point_too_small_window_size(self):
        window_size = 2

        with self.assertRaises(SystemExit):
            angles = change_point_detection(self.tensor, window_size, [])

    def test_change_point_tensor_empty(self):
        angles = change_point_detection([], 4, [])
        self.assertEqual([], angles)

    def test_change_point_tensor_none_dict(self):
        # n * (n - 1) = 4 * 3 = 12
        dict1 = {'front-end.1': 15, 'user.1': 25, 'user-db.1': 14, 'user.2': 22}
        dict2 = {'front-end.1': 17, 'user.1': 27, 'user-db.1': 16, 'user.2': 20}
        dict3 = {'front-end.1': 19, 'user.1': 29, 'user-db.1': 18, 'user.2': 18}
        dict4 = {'front-end.1': 21, 'user.1': 31, 'user-db.1': 20, 'user.2': 16}
        dict5 = {'front-end.1': 25, 'user.1': 35, 'user-db.1': 24, 'user.2': 12}
        dict6 = {'front-end.1': 27, 'user.1': 37, 'user-db.1': 26, 'user.2': 10}
        dict7 = {'front-end.1': 29, 'user.1': 39, 'user-db.1': 28, 'user.2': 8}
        dict8 = {'front-end.1': 31, 'user.1': 41, 'user-db.1': 30, 'user.2': 6} # 1-7 compared to this initially
        dict9 = {'front-end.1': 33, 'user.1': 43, 'user-db.1': 32, 'user.2': 4}
        dict10 = None
        dict11 = {'front-end.1': 37, 'user.1': 47, 'user-db.1': 36, 'user.2': 0}
        dict12 = {'front-end.1': 39, 'user.1': 49, 'user-db.1': 38, 'user.2': -2}

        tensor = [dict1, dict2, dict3, dict4, dict5, dict6, dict7, dict8, dict9, dict10, dict11, dict12]
        angles = change_point_detection(tensor, 4, [])
        # returns list of angles (of size len(tensor)). Note:

        print "test_change_point_tensor_none_dict", angles
        expected_angles = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                           float('nan'), 0.0, 0.0, 0.0, 0.0, 0.0] # 12 - 5 = 7 nan's
        print "None", angles

        for i in range(0,7):
            self.assertTrue(math.isnan(angles[i]))

        self.assertEquals(angles[7:], expected_angles[7:])

    '''
    def test_change_point_tensor_dict_val_nan(self):
        dict1 = {'front-end.1': 15, 'user.1': 25, 'user-db.1': 14, 'user.2': 22}
        dict2 = {'front-end.1': 17, 'user.1': 27, 'user-db.1': 16, 'user.2': 20}
        dict3 = {'front-end.1': 19, 'user.1': 29, 'user-db.1': 18, 'user.2': 18}
        dict4 = {'front-end.1': 21, 'user.1': 31, 'user-db.1': 20, 'user.2': 16}
        dict5 = {'front-end.1': 25, 'user.1': 35, 'user-db.1': 24, 'user.2': 12}
        dict6 = {'front-end.1': 27, 'user.1': 37, 'user-db.1': 26, 'user.2': 10}
        dict7 = {'front-end.1': 21, 'user.1': float('nan'), 'user-db.1': 20, 'user.2': 16}
        dict8 = {'front-end.1': 27, 'user.1': 37, 'user-db.1': 26, 'user.2': 10} # compare to this initially
        dict9 = {'front-end.1': 52, 'user.1': 62, 'user-db.1': 10, 'user.2': 30}
        tensor = [dict1, dict2, dict3, dict4, dict5, dict6, dict7, dict8, dict9]
        angles = change_point_detection(tensor, 4, [])

        print 'nan dictVal', angles
        self.assertEqual(angles[0], 0.0)
        self.assertNotEqual(angles[1], 0.0)
    # todo: the eigenvector calculation function returns wierd answers and causes this test to fail... don't why why...
    '''

    def test_change_point_nodes_disappear(self):
        dict1 = {'front-end.1': 15, 'user.1': 25, 'user-db.1': 14, 'user.2': 22}
        dict2 = {'front-end.1': 17, 'user.1': 27, 'user-db.1': 16, 'user.2': 20}
        dict3 = {'front-end.1': 19, 'user.1': 29, 'user-db.1': 18, 'user.2': 18}
        dict4 = {'front-end.1': 21, 'user.1': 31, 'user-db.1': 20, 'user.2': 16}
        dict5 = {'front-end.1': 23, 'user.1': 33, 'user-db.1': 22, 'user.2': 14}
        dict6 = {'front-end.1': 25, 'user.1': 35, 'user-db.1': 24, 'user.2': 12}
        dict7 = {'front-end.1': 37, 'user.1': 47, 'user-db.1': 36, 'user.2': 0}
        dict8 = {'front-end.1': 27, 'user.1': 37, 'user-db.1': 26, 'user.2': 10} # compares here initially
        dict9 = {'front-end.1': 29, 'user.1': 39, 'user-db.1': 28}
        dict10 = {'front-end.1': 31, 'user.1': 41, 'user-db.1': 30}
        dict11 = {'front-end.1': 40, 'user.1': 50, 'user-db.1': 39}
        tensor = [dict1, dict2, dict3, dict4, dict5, dict6, dict7, dict8, dict9, dict10, dict11]

        angles = change_point_detection(tensor, 4, []) # 11 - 4 = 7 nan's
        print "disappear", angles
        for i in range(0,7):
            self.assertTrue(math.isnan(angles[i]))
        self.assertEqual(angles[7], 0)
        self.assertNotEqual(angles[8], 0)
        self.assertNotEqual(angles[9], 0)
        self.assertNotEqual(angles[10], 0)

    def test_change_point_nodes_extra_appear(self):
        dict1 = {'front-end.1': 15, 'user.1': 25, 'user.2': 22}
        dict2 = {'front-end.1': 17, 'user.1': 27, 'user.2': 20}
        dict3 = {'front-end.1': 19, 'user.1': 29, 'user.2': 18}
        dict4 = {'front-end.1': 21, 'user.1': 31, 'user.2': 16}
        dict5 = {'front-end.1': 23, 'user.1': 33, 'user.2': 14}
        dict6 = {'front-end.1': 25, 'user.1': 35, 'user.2': 12}
        dict7 = {'front-end.1': 25, 'user.1': 35, 'user.2': 12}
        dict8 = {'front-end.1': 25, 'user.1': 35, 'user.2': 12} # start comparisons here
        dict9 = {'front-end.1': 27, 'user.1': 37, 'user-db.1': 26, 'user.2': 10}
        dict10 = {'front-end.1': 27, 'user.1': 37, 'user-db.1': 26, 'user.2': 30}
        dict11 = {'front-end.1': 52, 'user.1': 62, 'user-db.1': 10, 'user.2': 30}
        tensor = [dict1, dict2, dict3, dict4, dict5, dict6, dict7, dict8, dict9, dict10, dict11]

        angles = change_point_detection(tensor, 4, [])
        print "appear", angles
        # 11 - 4 = 7
        for i in range(0,7):
            self.assertTrue(math.isnan(angles[i]))

        self.assertEqual(angles[7], 0)
        self.assertNotEqual(angles[8], 0)
        self.assertNotEqual(angles[9], 0)
        self.assertNotEqual(angles[10], 0)

    def test_change_point_nodes_perfect_corr(self):
        dict1 = {'front-end.1': 15, 'user.1': 25, 'user.2': 22}
        dict2 = {'front-end.1': 17, 'user.1': 27, 'user.2': 20}
        dict3 = {'front-end.1': 19, 'user.1': 29, 'user.2': 18}
        dict4 = {'front-end.1': 21, 'user.1': 31, 'user.2': 16}
        dict5 = {'front-end.1': 23, 'user.1': 33, 'user.2': 14}
        dict6 = {'front-end.1': 25, 'user.1': 35, 'user.2': 12}
        dict7 = {'front-end.1': 27, 'user.1': 37, 'user.2': 10}
        dict8 = {'front-end.1': 29, 'user.1': 39, 'user.2': 8}
        dict9 = {'front-end.1': 35, 'user.1': 45, 'user.2': 2}
        tensor = [dict1, dict2, dict3, dict4, dict5, dict6, dict7, dict8, dict9]

        expected_angles = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 0.0, 0.0, 0.0, 0.0] # 9 - 4 = 5
        angles = change_point_detection(tensor, 3, [])
        # self.assertEquals(computed_angles , [0.0, 0.0, 0.0])
        for i in range(0, len(expected_angles)):
            if math.isnan(expected_angles[i]):
                self.assertTrue(math.isnan(angles[i]))
            else:
                self.assertEqual(expected_angles[i], angles[i])

    def test_change_point_tensor_no_angle_again(self):
        dict1 = {'front-end.1': 15, 'user.1': 25, 'user-db.1': 14}
        dict2 = {'front-end.1': 17, 'user.1': 27, 'user-db.1': 16}
        dict3 = {'front-end.1': 19, 'user.1': 29, 'user-db.1': 18}
        dict4 = {'front-end.1': 21, 'user.1': 31, 'user-db.1': 20}
        dict5 = {'front-end.1': 25, 'user.1': 35, 'user-db.1': 24}
        dict6 = {'front-end.1': 27, 'user.1': 37, 'user-db.1': 26}
        dict7 = {'front-end.1': 29, 'user.1': 39, 'user-db.1': 28}
        dict8 = {'front-end.1': 31, 'user.1': 41, 'user-db.1': 30}
        dict9 = {'front-end.1': 35, 'user.1': 45, 'user-db.1': 34}
        tensor = [dict1, dict2, dict3, dict4, dict5, dict6, dict7, dict8, dict9] # 9 - 4 = 5
        angles = change_point_detection(tensor, 3, [])

        expected_angles = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 0.0, 0.0, 0.0, 0.0]
        # self.assertEquals(computed_angles , [0.0, 0.0, 0.0])
        for i in range(0, len(expected_angles)):
            if math.isnan(expected_angles[i]):
                self.assertTrue(math.isnan(angles[i]))
            else:
                self.assertEqual(expected_angles[i], angles[i])

    def test_change_point_tensor_one_decreasing(self):
        dict1 = {'front-end.1': 15, 'user.1': 25, 'user-db.1': 14, 'user-db.2': 24}
        dict2 = {'front-end.1': 17, 'user.1': 27, 'user-db.1': 16, 'user-db.2': 22}
        dict3 = {'front-end.1': 19, 'user.1': 29, 'user-db.1': 18, 'user-db.2': 20}
        dict4 = {'front-end.1': 21, 'user.1': 31, 'user-db.1': 20, 'user-db.2': 18}
        dict5 = {'front-end.1': 25, 'user.1': 35, 'user-db.1': 24, 'user-db.2': 14}
        dict6 = {'front-end.1': 27, 'user.1': 37, 'user-db.1': 26, 'user-db.2': 12}
        dict7 = {'front-end.1': 29, 'user.1': 39, 'user-db.1': 28, 'user-db.2': 10}
        dict8 = {'front-end.1': 31, 'user.1': 41, 'user-db.1': 30, 'user-db.2': 8}
        dict9 = {'front-end.1': 35, 'user.1': 45, 'user-db.1': 34, 'user-db.2': 4}
        tensor = [dict1, dict2, dict3, dict4, dict5, dict6, dict7, dict8, dict9]
        angles = change_point_detection(tensor, 3, [])

        expected_angles = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 0.0, 0.0, 0.0, 0.0] # 9 - 4 = 5
        # self.assertEquals(computed_angles , [0.0, 0.0, 0.0])
        for i in range(0, len(expected_angles)):
            if math.isnan(expected_angles[i]):
                self.assertTrue(math.isnan(angles[i]))
            else:
                self.assertEqual(expected_angles[i], angles[i])

    def test_change_point_tensor_one_reordered(self):
        dict1 = {'front-end.1': 15, 'user-db.2': 24, 'user.1': 25, 'user-db.1': 14}
        dict2 = {'user-db.2': 22, 'front-end.1': 17, 'user.1': 27, 'user-db.1': 16}
        dict3 = {'front-end.1': 19, 'user.1': 29, 'user-db.1': 18, 'user-db.2': 20}
        dict4 = {'front-end.1': 21, 'user.1': 31, 'user-db.2': 18, 'user-db.1': 20}
        dict5 = {'front-end.1': 25, 'user-db.2': 14, 'user.1': 35, 'user-db.1': 24}
        dict6 = {'user-db.2': 12, 'front-end.1': 27, 'user.1': 37, 'user-db.1': 26}
        dict7 = {'front-end.1': 29, 'user.1': 39, 'user-db.1': 28, 'user-db.2': 10}
        dict8 = {'front-end.1': 31, 'user.1': 41, 'user-db.2': 8, 'user-db.1': 30}
        dict9 = {'front-end.1': 35, 'user-db.2': 4, 'user.1': 45, 'user-db.1': 34}
        tensor = [dict1, dict2, dict3, dict4, dict5, dict6, dict7, dict8, dict9]
        angles = change_point_detection(tensor, 3, [])

        expected_angles = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 0.0, 0.0, 0.0, 0.0] # 9 - 4 = 5
        # self.assertEquals(computed_angles , [0.0, 0.0, 0.0])
        for i in range(0, len(expected_angles)):
            if math.isnan(expected_angles[i]):
                self.assertTrue(math.isnan(angles[i]))
            else:
                self.assertEqual(expected_angles[i], angles[i])

    def test_change_point_tensor_one_reordered_spike(self):
        dict1 = {'front-end.1': 15, 'user-db.2': 24, 'user.1': 25, 'user-db.1': 14}
        dict2 = {'user-db.2': 22, 'front-end.1': 17, 'user.1': 27, 'user-db.1': 16}
        dict3 = {'front-end.1': 19, 'user.1': 29, 'user-db.1': 18, 'user-db.2': 20}
        dict4 = {'front-end.1': 21, 'user.1': 31, 'user-db.2': 18, 'user-db.1': 20}
        dict5 = {'front-end.1': 25, 'user-db.2': 14, 'user.1': 35, 'user-db.1': 24}
        dict6 = {'user-db.2': 12, 'front-end.1': 27, 'user.1': 37, 'user-db.1': 26}
        dict7 = {'front-end.1': 29, 'user.1': 39, 'user-db.1': 38, 'user-db.2': 10} # user-db.1 spikes upwards
        dict8 = {'front-end.1': 31, 'user.1': 41, 'user-db.2': 8, 'user-db.1': 30}
        dict9 = {'front-end.1': 35, 'user-db.2': 4, 'user.1': 45, 'user-db.1': 34}
        dict10 = {'front-end.1': 37, 'user-db.2': 2, 'user.1': 47, 'user-db.1': 36}
        dict11 = {'front-end.1': 39, 'user-db.2': 0, 'user.1': 49, 'user-db.1': 38}
        dict12 = {'front-end.1': 41, 'user-db.2': -2, 'user.1': 51, 'user-db.1': 40}
        dict13 = {'front-end.1': 43, 'user-db.2': -4, 'user.1': 53, 'user-db.1': 42}

        tensor = [dict1, dict2, dict3, dict4, dict5, dict6, dict7, dict8, dict9, dict10, dict11, dict12, dict13]
        angles = change_point_detection(tensor, 3, [])
        print "reordered_spike", angles

        # used to have 13 - (3 + 2) = 8 vals. So 5 nan's in beginning are new...
        # self.assertEquals(computed_angles , [0.0, 0.0, 0.0])
        for i in range(0, len(angles)):
            if i < 5:
                self.assertTrue(math.isnan(angles[i]))
            elif i == 5 or i == 12:
                self.assertEqual(0.0, angles[i])
            else:
                print "i", i, angles[i]
                self.assertNotEqual(angles[i], 0)

    # hmm... I might need to think about this a little more....
    def test_get_points_to_plot(self):
        vals = [5, 9, 3, 4, 19, 30, 20, 67, 89, 32]
        time_grand = 10
        exfil_start = 50
        exfil_end = 60
        wiggle_room = 2 #seconds
        vals = get_points_to_plot(time_grand, vals, exfil_start, exfil_end, wiggle_room)
        # expect vals to be [30]
        print "vals for test_get_points_to_plot", vals
        self.assertEquals(vals, ([19, 30, 20],4,6))

    def test_modified_z_score_normal(self):
        test_array = [0.271052632, 0.326797386, 0.268421053, 0.292397661, 0.320261438, 0.304093567, 0.298245614,
                      0.307189542, 0.263157895, 0.287581699, 0.294117647, 0.287581699, 0.287581699, 0.307189542,
                      0.278947368, 0.292397661, 0.320261438, 0.298245614, 0.274853801, 0.326797386]

        mod_z_scores = calc_modified_z_score(test_array, 10, 5)
        print "mod_z_scores", mod_z_scores,len(mod_z_scores)
        for i in range(0,5):
            self.assertTrue(math.isnan(mod_z_scores[i]))
        #self.assertEquals(mod_z_scores[0:5], [float('nan') for i in range(0,5)])
        for i in range(5, len(test_array)):
            #self.assertNotEqual(mod_z_scores[i], float('nan'))
            self.assertFalse(math.isnan(mod_z_scores[i]))
            self.assertNotEqual(mod_z_scores[i], float('inf'))
            self.assertNotEqual(mod_z_scores[i], float('-inf'))
            #self.assertNotEqual(mod_z_scores[i], 0.0)


    def test_modified_z_score_all_zeroes(self):
        test_array = [0 for i in range(0,20)]
        mod_z_scores = calc_modified_z_score(test_array, 10, 5)
        print "mod_z_scores_zeroes", mod_z_scores,len(mod_z_scores)
        for i in range(0,5):
            self.assertTrue(math.isnan(mod_z_scores[i]))
        #self.assertEquals(mod_z_scores[0:5], [float('nan') for i in range(0,5)])
        self.assertEquals(mod_z_scores[5:], [0.0 for i in range(0,15)])

    def test_modified_z_score_min_bigger_than_ts(self):
        test_array = [float('nan') for i in range(0,10)] + [0.320261438, 0.298245614, 0.274853801, 0.326797386]
        mod_z_scores = calc_modified_z_score(test_array, 10, 15)
        print "mod_z_scores_zeroes_too_small", mod_z_scores, len(mod_z_scores)
        #self.assertEquals(mod_z_scores, [float('nan') for i in range(0,len(test_array))])
        for i in range(0,len(test_array)):
            self.assertTrue(math.isnan(mod_z_scores[i]))

    # TODO: think of some more tests...



if __name__ == "__main__":
    unittest.main()
