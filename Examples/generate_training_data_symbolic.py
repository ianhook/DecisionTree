#!/usr/bin/env python

import DecisionTree

parameter_file = "param_symbolic.txt"
output_data_file = "training.dat";

training_data_gen = DecisionTree.TrainingDataGeneratorSymbolic( 
                              output_datafile   = output_data_file,
                              parameter_file    = parameter_file,
                              write_to_file     = 1,
                              number_of_training_samples = 100,
#                              debug1             = 1,
#                              debug2             = 1,
                    )

training_data_gen.read_parameter_file_symbolic()

training_data_gen.gen_symbolic_training_data()

training_data_gen.write_training_data_to_file()


