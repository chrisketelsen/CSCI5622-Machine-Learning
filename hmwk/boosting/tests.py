import sys

testSuite = sys.argv[1]

import unittest

X_train = np.array([[6,9.5],[4,8.5],[9,8.75],[8,8.0],[3,7],[1,6.5],[5,6.5],[1.5,2.5],[2,1],[9,2]])
y_train = np.array([1,1,-1,1,-1,1,-1,1,-1,-1])

class TestBoost(unittest.TestCase):
    def setUp(self):
        self.clf = AdaBoost(n_learners=3)
        self.clf.fit(X_train, y_train)

    def test_alphas(self):
        """
        alphas test 
        """
        alphas = self.clf.alpha
        self.assertAlmostEqual(alphas[0], 0.42364893019360172)
        self.assertAlmostEqual(alphas[1], 0.64964149206513044)
        self.assertAlmostEqual(alphas[2], 0.92291334524916524)

    def test_prediction(self):
        """
        prediction test 
        """
        pred = self.clf.predict(X_train)
        for (p, yi) in zip(pred, y_train):
            self.assertAlmostEqual(p, yi)

    def test_score(self):
        score = self.clf.score(X_train, y_train)
        self.assertAlmostEqual(score, 1.0)

    def test_staged_score(self):
        """
        stage_score test 
        """
        staged_score = self.clf.staged_score(X_train, y_train)
        self.assertAlmostEqual(staged_score[0], 0.7)
        self.assertAlmostEqual(staged_score[1], 0.7)
        self.assertAlmostEqual(staged_score[2], 1.0)


if testSuite == "part A":
    partA = unittest.TestSuite()
    for test in ["test_alphas"]:
        partA.addTest(TestBoost(test))
    runner = unittest.TextTestRunner(verbosity=2).run(partA)

if testSuite == "part B":
    partA = unittest.TestSuite()
    for test in ["test_prediction"]:
        partA.addTest(TestBoost(test))
    runner = unittest.TextTestRunner(verbosity=2).run(partA)

if testSuite == "part C":
    partB = unittest.TestSuite()
    for test in ["test_score"]:
        partB.addTest(TestBoost(test))
    runner = unittest.TextTestRunner(verbosity=2).run(partB)

if testSuite == "part D":
    partB = unittest.TestSuite()
    for test in ["test_staged_score"]:
        partB.addTest(TestBoost(test))
    runner = unittest.TextTestRunner(verbosity=2).run(partB)

