import sys

testSuite = sys.argv[1]

import unittest
import logging as log
import numpy as np

class TestNN(unittest.TestCase):

    def setUp(self):
        self.nn = Network([2,3,2])
        self.nn.W[0] = np.array([[0.1, 0.2],[-0.1, -0.2], [-0.1, 0.3]])
        self.nn.W[1] = np.array([[0.3, -0.2, -0.1], [-0.1, 0.3, 0.1]])
        self.nn.b[0] = np.array([-0.1, 0.1, 0.3])
        self.nn.b[1] = np.array([-0.2, 0.1])
        self.X_train = np.array([[1.0, 2.0]])
        self.y_train = np.array([[1, 0]], dtype=int)

    def testForwardProp(self): 
        """
        test forward propagation 
        """
        self.nn.forward_prop(self.X_train[0])
        self.assertAlmostEqual(self.nn.a[-1][0], 0.45768803332214286)
        self.assertAlmostEqual(self.nn.a[-1][1], 0.55713001269439544)

    def testPredict(self): 
        """
        test forward propagation 
        """
        yhat = self.nn.predict(self.X_train)
        self.assertEqual(yhat.shape[0], self.X_train.shape[0])
        self.assertEqual(yhat.shape[1], self.y_train.shape[1])
        self.assertEqual(yhat[0][0], 0)
        self.assertEqual(yhat[0][1], 1)

    def testBackProp(self):
        """
        test back propagation 
        """
        self.nn.back_prop(self.X_train[0],self.y_train[0])

        self.assertAlmostEqual(self.nn.dW[0][0,0], -0.013004963108364378)
        self.assertAlmostEqual(self.nn.dW[0][0,1], -0.026009926216728756)
        self.assertAlmostEqual(self.nn.dW[0][1,0],  0.016376329584037203)
        self.assertAlmostEqual(self.nn.dW[0][1,1],  0.032752659168074405)
        self.assertAlmostEqual(self.nn.dW[0][2,0],  0.0058198669975548425)
        self.assertAlmostEqual(self.nn.dW[0][2,1],  0.011639733995109685)

        self.assertAlmostEqual(self.nn.dW[1][0,0],  -0.080587603259700741)
        self.assertAlmostEqual(self.nn.dW[1][0,1],  -0.054019485926944437)
        self.assertAlmostEqual(self.nn.dW[1][0,2],  -0.092875456517653776)
        self.assertAlmostEqual(self.nn.dW[1][1,0],   0.082298072874257167)
        self.assertAlmostEqual(self.nn.dW[1][1,1],   0.055166047997716461)
        self.assertAlmostEqual(self.nn.dW[1][1,2],   0.094846735472303406)

        self.assertAlmostEqual(self.nn.db[0][0],  -0.013004963108364378)
        self.assertAlmostEqual(self.nn.db[0][1],   0.016376329584037203)
        self.assertAlmostEqual(self.nn.db[0][2],   0.0058198669975548425)

        self.assertAlmostEqual(self.nn.db[1][0],  -0.13460708918664518)
        self.assertAlmostEqual(self.nn.db[1][1],   0.13746412087197363) 

    def testGradCheck(self):
        """
        test numerical gradient checking 
        """
        self.nn.train(self.X_train, self.y_train, num_epochs=5, eta=0.25, lam=0.0, isPrint=False)
        rel_errs = self.nn.gradient_checking(self.X_train, self.y_train)
        for re in rel_errs:
            self.assertLess(re, 1e-8)


    def testSGD(self):
        """
        test unregularized stochastic gradient descent 
        """
        self.nn.train(self.X_train,self.y_train, eta=0.25, lam=0.0, num_epochs=2, isPrint=False)

        self.assertAlmostEqual(self.nn.W[0][0,0],  0.10672555566225433)
        self.assertAlmostEqual(self.nn.W[0][0,1],  0.21345111132450867)
        self.assertAlmostEqual(self.nn.W[0][1,0], -0.1078460521612368)
        self.assertAlmostEqual(self.nn.W[0][1,1], -0.2156921043224736)
        self.assertAlmostEqual(self.nn.W[0][2,0], -0.10254291273787794)
        self.assertAlmostEqual(self.nn.W[0][2,1],  0.29491417452424407)

        self.assertAlmostEqual(self.nn.W[1][0,0],  0.3398926248349024)
        self.assertAlmostEqual(self.nn.W[1][0,1], -0.17355445664174357)
        self.assertAlmostEqual(self.nn.W[1][0,2], -0.05426225729891615)
        self.assertAlmostEqual(self.nn.W[1][1,0], -0.14079398398421336)
        self.assertAlmostEqual(self.nn.W[1][1,1],  0.27295734199412003)
        self.assertAlmostEqual(self.nn.W[1][1,2],  0.053229161962644732)

        self.assertAlmostEqual(self.nn.b[0][0], -0.093274444337745677)
        self.assertAlmostEqual(self.nn.b[0][1],  0.092153947838763212)
        self.assertAlmostEqual(self.nn.b[0][2],  0.29745708726212206)

        self.assertAlmostEqual(self.nn.b[1][0], -0.13362224659049363)
        self.assertAlmostEqual(self.nn.b[1][1],  0.032122823060367525)

    def testRegularizedSGD(self):
        """
        test regularized stochastic gradient descent 
        """
        self.nn.train(self.X_train,self.y_train, eta=0.25, lam=1.0, num_epochs=2, isPrint=False)

        self.assertAlmostEqual(self.nn.W[0][0,0],  0.061442714519434803)
        self.assertAlmostEqual(self.nn.W[0][0,1],  0.12288542903886961)
        self.assertAlmostEqual(self.nn.W[0][1,0], -0.062135336255117087)
        self.assertAlmostEqual(self.nn.W[0][1,1], -0.12427067251023417)
        self.assertAlmostEqual(self.nn.W[0][2,0], -0.058105802131058891)
        self.assertAlmostEqual(self.nn.W[0][2,1],  0.16503839573788223)

        self.assertAlmostEqual(self.nn.W[1][0,0],  0.20276038197618426)
        self.assertAlmostEqual(self.nn.W[1][0,1], -0.088333400593527747)
        self.assertAlmostEqual(self.nn.W[1][0,2], -0.017047738414153385)
        self.assertAlmostEqual(self.nn.W[1][1,0], -0.09070778645394395)
        self.assertAlmostEqual(self.nn.W[1][1,1],  0.14427432137252863)
        self.assertAlmostEqual(self.nn.W[1][1,2],  0.016532074033061488)

        self.assertAlmostEqual(self.nn.b[0][0],  -0.093994475286292442)
        self.assertAlmostEqual(self.nn.b[0][1],   0.0930911431458806)
        self.assertAlmostEqual(self.nn.b[0][2],   0.29778045618159393)

        self.assertAlmostEqual(self.nn.b[1][0],  -0.13336928784220636)
        self.assertAlmostEqual(self.nn.b[1][1],   0.032433966255943282)



if testSuite == "prob 1A":
    prob1A = unittest.TestSuite()
    for test in ["testForwardProp"]:
        prob1A.addTest(TestNN(test))
    runner = unittest.TextTestRunner(verbosity=2).run(prob1A)

if testSuite == "prob 1B":
    prob1B = unittest.TestSuite()
    for test in ["testPredict"]:
        prob1B.addTest(TestNN(test))
    runner = unittest.TextTestRunner(verbosity=2).run(prob1B)

if testSuite == "prob 1C":
    prob1C = unittest.TestSuite()
    for test in ["testBackProp"]:
        prob1C.addTest(TestNN(test))
    runner = unittest.TextTestRunner(verbosity=2).run(prob1C)

if testSuite == "prob 1D":
    prob1D = unittest.TestSuite()
    for test in ["testGradCheck"]:
        prob1D.addTest(TestNN(test))
    runner = unittest.TextTestRunner(verbosity=2).run(prob1D)

if testSuite == "prob 1E":
    prob1E = unittest.TestSuite()
    for test in ["testSGD"]:
        prob1E.addTest(TestNN(test))
    runner = unittest.TextTestRunner(verbosity=2).run(prob1E)

if testSuite == "prob 1F":
    prob1F = unittest.TestSuite()
    for test in ["testRegularizedSGD"]:
        prob1F.addTest(TestNN(test))
    runner = unittest.TextTestRunner(verbosity=2).run(prob1F)


