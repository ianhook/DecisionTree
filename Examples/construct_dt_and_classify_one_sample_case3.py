#!/usr/bin/env python

##  construct_dt_and_classify_one_sample_case3.py

##  This script does the same thing as the script
##  construct_dt_and_classify_one_sample_case2.py except
##  that it uses just two of the columns of the csv file for
##  DT construction and classification.  The two features
##  used are in columns indexed 3 and 5 of the csv file.

##  Remember that column indexing is zero-based in the csv
##  file.

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
                                csv_columns_for_features = [3,5],
                                entropy_threshold = 0.01,
                                max_depth_desired = 4,
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

test_sample = ['g2 = 20',
               'age = 80.0']

# The rest of the script is for displaying the classification results:

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


