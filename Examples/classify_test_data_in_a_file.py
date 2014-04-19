#!/usr/bin/env python

import DecisionTree
import re
import sys

debug = 1

if len(sys.argv) != 4:
    sys.exit('''This script must be called with exactly three command-line '''
             '''arguments:\n''' 
             '''   1st arg: name of the training datafile\n'''
             '''   2nd arg: name of the test data file\n'''
             '''   3rd arg: the name of the output file to which class '''
             '''labels will be written\n''') 

training_datafile, test_datafile, outputfile = sys.argv[1],sys.argv[2],sys.argv[3]

dt = DecisionTree.DecisionTree(training_datafile = training_datafile)

dt.get_training_data()
dt.calculate_first_order_probabilities()
dt.calculate_class_priors()

#  UNCOMMENT THE NEXT LEFT-JUSTIFIED LINE if you would like to see the
#  training data that was read from the disk file:
#dt.show_training_data()

root_node = dt.construct_decision_tree_classifier()

#  UNCOMMENT THE NEXT LEFT-JUSTIFIED LINE if you would like to see the 
#  decision tree displayed in your terminal window:
#root_node.display_decision_tree("   ")

# NOW YOU ARE READY TO CLASSIFY TEST DATA IN A FILE:

TESTFILEHANDLE = open(test_datafile) 
OUTPUTHANDLE   = open(outputfile, 'w')

features = None
lineskip = r'^#'
testdata_header_pattern = r'^\s*Feature Order For Data:\s*(.+)'
features = []

while 1:
    line = TESTFILEHANDLE.readline()
    if line == '': break
    line = line.rstrip()
    if line == "": continue
    if re.search(lineskip, line): continue 
    if re.search(testdata_header_pattern, line, re.IGNORECASE):
        m = re.search(testdata_header_pattern, line, re.IGNORECASE)
        if m:
            features = m.group(1).split()
            if len(features) == 0:
                raise ValueError("Your feature labels are not listed")
            continue
        else:
            raise ValueError("Your testdata file does not list feature order")
    test_sample_entries = line.split()
    sample_name = test_sample_entries[0]
    del test_sample_entries[0]
    if len(features) != len(test_sample_entries):
        raise ValueError('''
         the number of features listed in the header of the testdata file
         does not match the number of values in the tabulated testdata 
         for at least one record
         ''')
    test_sample = []
    for i in range(0, len(features)):
        test_sample.append( \
               "".join([features[i], "=", test_sample_entries[i]]) )
    classification = dt.classify(root_node, test_sample)
    result = sample_name + ":  "
    which_classes = list( classification.keys() )
    which_classes.remove('solution_path')
    which_classes = sorted(which_classes, key=lambda x: classification[x], reverse=True)
    for which_class in which_classes:
        formatted = ":%.3f     " % float(classification[which_class])
        result += which_class + formatted
    if debug: print(result + "\n") 
    if sys.version_info[0] == 3:
        OUTPUTHANDLE.write(result + "\n\n")
    else:
        print>> OUTPUTHANDLE, result + "\n"
TESTFILEHANDLE.close()
OUTPUTHANDLE.close()

