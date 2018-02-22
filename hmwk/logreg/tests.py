import sys

testSuite = sys.argv[1]

import unittest

kTOY_VOCAB = "BIAS_CONSTANT A B C D".split()
kPOS = Example(1, "A:4 B:3 C:1".split(), kTOY_VOCAB)
kNEG = Example(0, "B:2 C:3 D:1".split(), kTOY_VOCAB)

class TestLogReg(unittest.TestCase):
    def setUp(self):
        self.logreg_learnrate = LogReg(train_set=[kPOS, kNEG], test_set=[], lam=0.0, eta=0.5)
        self.logreg_unreg = LogReg(train_set=[kPOS, kNEG], test_set=[], lam=0.0, eta=1.0)
        self.logreg_reg = LogReg(train_set=[kPOS, kNEG], test_set=[], lam=0.25, eta=1.0)

    def test_unreg(self):
        w = self.logreg_unreg.sgd_update(kPOS, 0)
        self.assertAlmostEqual(self.logreg_unreg.w[0], .5)
        self.assertAlmostEqual(self.logreg_unreg.w[1], 2.0)
        self.assertAlmostEqual(self.logreg_unreg.w[2], 1.5)
        self.assertAlmostEqual(self.logreg_unreg.w[3], 0.5)
        self.assertAlmostEqual(self.logreg_unreg.w[4], 0.0)

        w = self.logreg_unreg.sgd_update(kNEG, 1)
        self.assertAlmostEqual(self.logreg_unreg.w[0], -0.49330714907571527)
        self.assertAlmostEqual(self.logreg_unreg.w[1], 2.0)
        self.assertAlmostEqual(self.logreg_unreg.w[2], -0.48661429815143054)
        self.assertAlmostEqual(self.logreg_unreg.w[3], -2.479921447227146)
        self.assertAlmostEqual(self.logreg_unreg.w[4], -0.99330714907571527)

    def test_learnrate(self):
        w = self.logreg_learnrate.sgd_update(kPOS, 0)
        self.assertAlmostEqual(self.logreg_learnrate.w[0], 0.25)
        self.assertAlmostEqual(self.logreg_learnrate.w[1], 1.00)
        self.assertAlmostEqual(self.logreg_learnrate.w[2], 0.75)
        self.assertAlmostEqual(self.logreg_learnrate.w[3], 0.25)
        self.assertAlmostEqual(self.logreg_learnrate.w[4], 0.0)

        w = self.logreg_learnrate.sgd_update(kNEG, 1)
        self.assertAlmostEqual(self.logreg_learnrate.w[0], -0.21207090998937828)
        self.assertAlmostEqual(self.logreg_learnrate.w[1], 1.0)
        self.assertAlmostEqual(self.logreg_learnrate.w[2], -0.17414181997875655)
        self.assertAlmostEqual(self.logreg_learnrate.w[3], -1.1362127299681348)
        self.assertAlmostEqual(self.logreg_learnrate.w[4], -0.46207090998937828)       

    def test_reg(self):
        w = self.logreg_reg.sgd_update(kPOS, 0)
        self.assertAlmostEqual(self.logreg_reg.w[0], .5)
        self.assertAlmostEqual(self.logreg_reg.w[1], 1.0)
        self.assertAlmostEqual(self.logreg_reg.w[2], 0.75)
        self.assertAlmostEqual(self.logreg_reg.w[3], 0.25)
        self.assertAlmostEqual(self.logreg_reg.w[4], 0.0)

        w = self.logreg_reg.sgd_update(kNEG, 1)
        self.assertAlmostEqual(self.logreg_reg.w[0], -0.43991334982599239)
        self.assertAlmostEqual(self.logreg_reg.w[1], 1.0)
        self.assertAlmostEqual(self.logreg_reg.w[2], -0.56491334982599239)
        self.assertAlmostEqual(self.logreg_reg.w[3], -1.2848700247389886)
        self.assertAlmostEqual(self.logreg_reg.w[4], -0.2349783374564981)



if testSuite == "part A":
    partA = unittest.TestSuite()
    for test in ["test_unreg", "test_learnrate"]:
        partA.addTest(TestLogReg(test))
    runner = unittest.TextTestRunner(verbosity=2).run(partA)

if testSuite == "part B":
    partB = unittest.TestSuite()
    for test in ["test_reg"]:
        partB.addTest(TestLogReg(test))
    runner = unittest.TextTestRunner(verbosity=2).run(partB)

