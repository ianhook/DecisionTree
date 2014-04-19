#!/usr/bin/env python

import unittest
import TestProbabilityCalculation
import TestEntropyCalculation
import TestBestFeatureCalculation
import TestDecisionTreeInduction
import TestDecisionTreeClassification

class DecisionTreeTestCase( unittest.TestCase ):
    def checkVersion(self):
        import DecisionTree

testSuites = [unittest.makeSuite(DecisionTreeTestCase, 'test')] 

for test_type in [
            TestProbabilityCalculation,
            TestEntropyCalculation,
            TestBestFeatureCalculation,
            TestDecisionTreeInduction,
            TestDecisionTreeClassification
    ]:
    testSuites.append(test_type.getTestSuites('test'))


def getTestDirectory():
    try:
        return os.path.abspath(os.path.dirname(__file__))
    except:
        return '.'

import os
os.chdir(getTestDirectory())

runner = unittest.TextTestRunner()
runner.run(unittest.TestSuite(testSuites))
