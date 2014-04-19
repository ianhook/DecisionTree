#!/usr/bin/env python

## classify_by_asking_questions.py

import DecisionTree
import sys

#training_datafile = "training.dat"
training_datafile = "stage3cancer.csv"


dt = DecisionTree.DecisionTree( training_datafile = training_datafile,
                                csv_class_column_index = 2,
                                csv_columns_for_features = [3,4,5,6,7,8],
                                entropy_threshold = 0.01,
                                max_depth_desired = 8,
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
root_node.display_decision_tree("   ")

classification = dt.classify_by_asking_questions(root_node)

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
