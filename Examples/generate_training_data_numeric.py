#!/usr/bin/env python

import DecisionTree

#parameter_file = "param_numeric.txt"
#parameter_file = "param_numeric_strongly_overlapping_classes.txt"
parameter_file = "param_numeric_extremely_overlapping_classes.txt"

#output_csv_file = "training.csv";
#output_csv_file = "training2.csv";
output_csv_file = "training3.csv";

training_data_gen = DecisionTree.TrainingDataGeneratorNumeric( 
                              output_csv_file   = output_csv_file,
                              parameter_file    = parameter_file,
                              number_of_samples_per_class = 50,
                              debug1             = 0,
                    )

training_data_gen.read_parameter_file_numeric()

training_data_gen.gen_numeric_training_data_and_write_to_csv()



