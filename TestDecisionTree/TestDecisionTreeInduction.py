import DecisionTree
import unittest

training_datafile = "training.dat"


class TestDecisionTreeInduction(unittest.TestCase):

    def setUp(self):
        print("Testing decision-tree induction on sample training file")
        self.dt = DecisionTree.DecisionTree(training_datafile = training_datafile)
        self.dt.get_training_data()
        self.dt.calculate_first_order_probabilities()
        self.dt.calculate_class_priors()
        self.root_node = self.dt.construct_decision_tree_classifier()

    def test_decision_tree_induction(self):
        num_of_nodes = self.root_node.how_many_nodes()
        self.assertEqual(num_of_nodes, 13)

def getTestSuites(type):
    return unittest.TestSuite([
            unittest.makeSuite(TestDecisionTreeInduction, type)
                             ])                    
if __name__ == '__main__':
    unittest.main()

