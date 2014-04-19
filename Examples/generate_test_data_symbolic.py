#!/usr/bin/env python

import DecisionTree

parameter_file = "param_symbolic.txt"
output_test_datafile = "testdata.dat";
output_class_labels_file = "testdata_classlabels.dat";

test_data_gen = DecisionTree.TestDataGeneratorSymbolic(
                   output_test_datafile     = output_test_datafile,
                   output_class_labels_file = output_class_labels_file,
                   parameter_file           = parameter_file,
                   write_to_file            = 1,
                   number_of_test_samples   = 30,
#                   debug1                   = 1,
                )

test_data_gen.read_parameter_file()
test_data_gen.gen_test_data()
test_data_gen.write_test_data_to_file()

