#!/usr/bin/env python

### setup.py

from distutils.core import setup

setup(name='DecisionTree',
      version='2.2.1',
      author='Avinash Kak',
      author_email='kak@purdue.edu',
      maintainer='Avinash Kak',
      maintainer_email='kak@purdue.edu',
      url='https://engineering.purdue.edu/kak/distDT/DecisionTree-2.2.1.html',
      download_url='https://engineering.purdue.edu/kak/distDT/DecisionTree-2.2.1.tar.gz',
      description='A Python module for constructing a decision tree from multidimensional training data and for using the decision tree for classifying new data',
      long_description=''' 

**Version 2.2.1:** The changes made are all in the part of
the module that is used for evaluating the quality of
training data through a 10-fold cross validation test.  The
previous version used the default values for the constructor
parameters when constructing the decision trees in each
iteration of the test. The new version correctly uses the
user-supplied values.

**Version 2.2** fixes a bug discovered in the best feature
calculator function. This bug was triggered by certain
conditions related to the distribution of values for the
features in a training data file.  Additionally, and very
importantly, Version 2.2 allows you to test the quality of
your training data by running a 10-fold cross-validation
test on the data. This testing functionality in Version 2.2
can also be used to find the best values to use for the
constructor parameters for constructing a decision tree.

**Version 2.0** is a major rewrite of the DecisionTree
module.  This revision was prompted by a number of users
wanting to see numeric features incorporated in the
construction of decision trees.  So here it is!  This
version allows you to use either purely symbolic features,
or purely numeric features, or a mixture of the two. (A
feature is numeric if it can take any floating-point value
over an interval.)

With regard to the purpose of the module, assuming you have
placed your training data in a CSV file, all you have to do
is to supply the name of the file to this module and it does
the rest for you without much effort on your part for
classifying a new data sample.  A decision tree classifier
consists of feature tests that are arranged in the form of a
tree. The feature test associated with the root node is one
that can be expected to maximally disambiguate the different
possible class labels for a new data record.  From the root
node hangs a child node for each possible outcome of the
feature test at the root. This maximal class-label
disambiguation rule is applied at the child nodes
recursively until you reach the leaf nodes.  A leaf node may
correspond either to the maximum depth desired for the
decision tree or to the case when there is nothing further
to gain by a feature test at the node.

Typical usage syntax:

::

      training_datafile = "stage3cancer.csv"

      dt = DecisionTree.DecisionTree( 
                      training_datafile = training_datafile,
                      csv_class_column_index = 2,
                      csv_columns_for_features = [3,4,5,6,7,8],
                      entropy_threshold = 0.01,
                      max_depth_desired = 8,
                      symbolic_to_numeric_cardinality_threshold = 10,
           )
        dt.get_training_data()
        dt.calculate_first_order_probabilities()
        dt.calculate_class_priors()
        dt.show_training_data()

        root_node = dt.construct_decision_tree_classifier()
        root_node.display_decision_tree("   ")

        test_sample  = ['g2 = 4.2',
                        'grade = 2.3',
                        'gleason = 4',
                        'eet = 1.7',
                        'age = 55.0',
                        'ploidy = diploid']

        classification = dt.classify(root_node, test_sample)
        print "Classification: ", classification

          ''',

      license='Python Software Foundation License',
      keywords='data classification, decision trees, information analysis',
      platforms='All platforms',
      classifiers=['Topic :: Scientific/Engineering :: Information Analysis', 'Programming Language :: Python :: 2.7', 'Programming Language :: Python :: 3.2'],
      packages=['']
)
