#!/usr/bin/env python

##   construct_dt_and_classify_one_sample_case2.py

##  This script is for a mixture of symbolic and numeric
##  features.  Since we have numeric features, only a csv
##  file can be used for training.  Note how we tell the
##  module that the class label in each training record is
##  placed in the column indexed 2 and how the feature are
##  to be found in columns indexed 3, 4, 5, 6, 7, and 8.

##  Remember, the column indexing in the csv file is
##  zero-based.  That is, the first column is indexed 0.

##  The training file `stage3cancer.csv' was taken from the
##  RPART module by Terry Therneau and Elizabeth Atkinson.
##  This module is a part of the R based statistical package
##  for classification and regression by recursive
##  partitioning of data.

import DecisionTree
import sys

training_datafile = "stage3cancer.csv"

dt = DecisionTree.DecisionTree( training_datafile = training_datafile,
                                csv_class_column_index = 2,
                                csv_columns_for_features = [3,4,5,6,7,8],
                                entropy_threshold = 0.01,
                                max_depth_desired = 8,
                                symbolic_to_numeric_cardinality_threshold = 10,
                              )
dt.get_training_data()
dt.calculate_first_order_probabilities()
dt.calculate_class_priors()

#   UNCOMMENT THE FOLLOWING LINE if you would like to see the training
#   data that was read from the disk file:
#dt.show_training_data()

root_node = dt.construct_decision_tree_classifier()

#   UNCOMMENT THE FOLLOWING LINE if you would like to see the decision
#   tree displayed in your terminal window:
print("\n\nThe Decision Tree:\n")
root_node.display_decision_tree("     ")

test_sample  = ['g2 = 4.2',
                'grade = 2.3',
                'gleason = 4',
                'eet = 1.7',
                'age = 55.0',
                'ploidy = diploid']

# The result of the script is for displaying the results:

classification = dt.classify(root_node, test_sample)

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


