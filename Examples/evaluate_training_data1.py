#!/usr/bin/env python

##   evaluate_training_data1.py


##  This script is for testing the class discriminatory
##  power of the training data contained in the file
##  `stage3cancer.csv'.

##  Through the class EvalTrainingData as shown below, this
##  script runs a 10-fold cross-validation test on the
##  training data.  This test divides all of the training
##  data into ten parts, with nine parts used for training a
##  decision tree and one part used for testing its ability
##  to classify correctly. This selection of nine parts for
##  training and one part for testing is carried out in all
##  of the ten different possible ways.

##  A script like this can also be used to test the
##  appropriateness of your choices for the constructor
##  parameters entropy_threshold, max_depth_desired, and
##  symbolic_to_numeric_cardinality_threshold.


import DecisionTree
import sys

training_datafile = "stage3cancer.csv"

eval_data = DecisionTree.EvalTrainingData( 
                               training_datafile = training_datafile,
                               csv_class_column_index = 2,
                               csv_columns_for_features = [3,4,5,6,7,8],
                               entropy_threshold = 0.01,
                               max_depth_desired = 5,
                               symbolic_to_numeric_cardinality_threshold = 10,
            )

eval_data.get_training_data()
eval_data.evaluate_training_data()
