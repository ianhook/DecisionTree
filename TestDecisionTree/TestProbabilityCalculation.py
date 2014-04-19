import DecisionTree
import unittest

training_datafile = "training.dat"


class TestProbabilityCalculation(unittest.TestCase):

    def setUp(self):
        print("Testing probability calculation on sample training file")
        self.dt = DecisionTree.DecisionTree(training_datafile = training_datafile)
        self.dt.get_training_data()
        self.dt.calculate_first_order_probabilities()
        self.dt.calculate_class_priors()

    def test_prior_probability_calculation(self):
        prob = self.dt.prior_probability_for_class( 'benign' )
        self.assertTrue( abs(prob - 0.7) < 0.01)
        prob = self.dt.prior_probability_for_class( 'malignant' )
        self.assertTrue( abs(prob - 0.3) < 0.01)

    def test_feature_value_probability_calculation(self):
        prob = self.dt.probability_of_feature_value( 'smoking', 'heavy')
        self.assertTrue( abs(prob - 0.5) < 0.01)

    def test_for_sequence_of_features_and_values(self):
        prob = self.dt.probability_of_a_sequence_of_features_and_values_or_thresholds(\
          ['smoking=heavy', 'exercising=regularly', 'fatIntake=heavy'])    
        self.assertTrue( abs(prob - 0.09) < 0.01)

    def test_for_feature_value_given_class(self):
        prob = self.dt.probability_of_feature_value_given_class('smoking', 'heavy', 'malignant')
        self.assertTrue( abs(prob - 0.733) < 0.01)


def getTestSuites(type):
    return unittest.TestSuite([
            unittest.makeSuite(TestProbabilityCalculation, type)
                             ])                    
if __name__ == '__main__':
    unittest.main()

