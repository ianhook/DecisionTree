#!/usr/bin/env python

## construct_dt_and_classify_one_sample_case1.py

##  This script shows the traditional usage of the
##  DecisionTree module.  By traditional I mean the usage up
##  to version 1.7.1.  The training data supplied through
##  the file `training.dat' is purely symbolic.  By the way,
##  this training data was produced by the script
##  generate_training_data_symbolic.py on the basis of the
##  parameters declared in the file `param_symbolic.txt'.

import DecisionTree
import sys

training_datafile = "training.dat"

dt = DecisionTree.DecisionTree( training_datafile = training_datafile,
                                entropy_threshold = 0.01,
                                max_depth_desired = 5,
                              )
dt.get_training_data()
#dt.calculate_first_order_probabilities_for_numeric_features()
dt.calculate_first_order_probabilities()
dt.calculate_class_priors()

#   UNCOMMENT THE FOLLOWING LINE if you would like to see the training
#   data that was read from the disk file:
#dt.show_training_data()

root_node = dt.construct_decision_tree_classifier()

#   UNCOMMENT THE FOLLOWING TWO LINES if you would like to see the decision
#   tree displayed in your terminal window:
print("\n\nThe Decision Tree:\n")
root_node.display_decision_tree("     ")

test_sample1 = [ 'exercising=never', 
                 'smoking=heavy', 
                 'fatIntake=heavy',
                 'videoAddiction=heavy']

test_sample2  = ['exercising=none', 
                'smoking=heavy', 
                'fatIntake=heavy', 
                'videoAddiction=none']  

# The rest of the script is for displaying the classification results:

classification = dt.classify(root_node, test_sample1)
solution_path = classification['solution_path']
del classification['solution_path']
which_classes = list( classification.keys() )
which_classes = sorted(which_classes, key=lambda x: classification[x], reverse=True)
print("\nClassification:\n")
print("     "  + str.ljust("class name", 30) + "probability")    
print("     ----------                    -----------")
for which_class in which_classes:
    if which_class is not 'solution_path':
        print("     "  + str.ljust(which_class, 30) +  str(classification[which_class]))

print("\nSolution path in the decision tree: " + str(solution_path))
print("\nNumber of nodes created: " + str(root_node.how_many_nodes()))

