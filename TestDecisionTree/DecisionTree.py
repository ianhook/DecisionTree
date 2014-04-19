#!/usr/bin/env python

__version__ = '2.2.1'
__author__  = "Avinash Kak (kak@purdue.edu)"
__date__    = '2013-September-5'
__url__     = 'https://engineering.purdue.edu/kak/distDT/DecisionTree-2.2.1.html'
__copyright__ = "(C) 2013 Avinash Kak. Python Software Foundation."

__doc__ = '''

DecisionTree.py

Version: ''' + __version__ + '''
   
Author: Avinash Kak (kak@purdue.edu)

Date: ''' + __date__ + '''


@title
CHANGES:

  Version 2.2.1:

    The changes made are all in the part of the module that is used for
    evaluating the quality of training data through a 10-fold cross
    validation test.  The previous version used the default values for the
    constructor parameters when constructing the decision trees in each
    iteration of the test. The new version correctly uses the user-supplied
    values.

  Version 2.2:

    This version fixes a bug discovered in the best feature calculator
    function. This bug was triggered by certain conditions related to the
    distribution of values for the features in a training data file.
    Additionally, and VERY IMPORTANTLY, Version 2.2 allows you to test the
    quality of your training data by running a 10-fold cross-validation
    test on the data.  This test divides all of the training data into ten
    parts, with nine parts used for training a decision tree and one part
    used for testing its ability to classify correctly. This selection of
    nine parts for training and one part for testing is carried out in all
    of the ten different possible ways.  This testing functionality in
    Version 2.2 can also be used to find the best values to use for the
    constructor parameters entropy_threshold, max_depth_desired, and
    symbolic_to_numeric_cardinality_threshold.

  Version 2.1:

    This is a cleaned up version of v. 2.0 of the module. Should run more
    efficiently for large training data files that contain both numeric and
    symbolic features.

  Version 2.0:

    This is a major rewrite of the DecisionTree module.  This revision was
    prompted by a number of users wanting to see numeric features
    incorporated in the construction of decision trees.  So here it is!
    This version allows you to use either purely symbolic features, or
    purely numeric features, or a mixture of the two. (A feature is numeric
    if it can take any floating-point value over an interval.)

  Version 1.7.1:

    This version includes a fix for a bug that was triggered by certain
    comment words in a training data file.  This version also includes
    additional safety checks that are useful for catching errors and
    inconsistencies in large training data files that do not lend
    themselves to manual checking for correctness.  As an example, the new
    version makes sure that the number of values you declare in each sample
    record matches the number of features declared at the beginning of the
    training data file.

  Version 1.7:

    This version includes safety checks on the consistency of the data you
    place in your training data file.  When a training data file contains
    thousands of records, it is difficult to manually check that you used
    the same class names in your sample records that you declared at the
    top of your training file or that the values you have for your features
    are legal vis-a-vis the earlier declarations regarding such values in
    the training file.  Another safety feature incorporated in this version
    is the non-consideration of classes that are declared at the top of the
    training file but that have no sample records in the file.

  Version 1.6.1:

    Fixed a bug in the method that generates synthetic test data.

  Version 1.6:

    This version includes several upgrades: The module now includes code
    for generating synthetic training and test data for experimenting with
    the DecisionTree classifier.  Another upgrade in the new version is
    that, after training, a decision tree can now be used in an interactive
    mode in which the user is asked to supply answers for the feature tests
    at the nodes as the classification process descends down the tree.

  Version 1.5:

    This is a Python 3.x compliant version of the DecisionTree module.
    This version should work with both Python 2.x and Python 3.x.

  Version 1.0:

    This is a Python implementation of the author's Perl module
    Algorithm::DecisionTree, Version 1.41.  The Python version should work
    faster for large decision trees since it uses probability and entropy
    caching much more extensively than Version 1.41 of the Perl module.
    (Note: I expect my next release of the Perl module to catch up with
    this Python version in terms of performance.)


@title
USAGE:

    If your training data includes numeric features (a feature is numeric
    if it can take any floating point value over an interval), you are
    expected to supply your training data through a CSV file and your call
    for constructing a decision tree will look like:

        training_datafile = "stage3cancer.csv"

        dt = DecisionTree.DecisionTree( 
                                training_datafile = training_datafile,
                                csv_class_column_index = 2,
                                csv_columns_for_features = [3,4,5,6,7,8],
                                entropy_threshold = 0.01,
                                max_depth_desired = 8,
                                symbolic_to_numeric_cardinality_threshold = 10,
             )

    The constructor option `csv_class_column_index' informs the module as
    to which column of your CSV file contains the class label.  THE COLUMN
    INDEXING IS ZERO BASED.  The constructor option
    `csv_columns_for_features' specifies which columns are to be used for
    feature values.  The first row of the CSV file must specify the names
    of the features.  See examples of CSV files in the `examples'
    subdirectory.

    The option `symbolic_to_numeric_cardinality_threshold' is also
    important.  For the example shown above, if an ostensibly numeric
    feature takes on only 10 or fewer different values in your training
    datafile, it will be treated like a symbolic feature.  The option
    `entropy_threshold' determines the granularity with which the entropies
    are sampled for the purpose of calculating entropy gain with a
    particular choice of decision threshold for a numeric feature or a
    feature value for a symbolic feature.

    After you have constructed an instance of the DecisionTree module, you
    read in the training data file and initialize the probability cache by
    calling:

        dt.get_training_data()
        dt.calculate_first_order_probabilities()
        dt.calculate_class_priors()

    Now you are ready to construct a decision tree for your training data
    by calling:

        root_node = dt.construct_decision_tree_classifier()

    where root_node is an instance of DTNode class that is also defined in
    the module file.  Now you are ready to classify a data record.  Let's
    say that your data record looks like:

        test_sample  = ['g2 = 4.2',
                        'grade = 2.3',
                        'gleason = 4',
                        'eet = 1.7',
                        'age = 55.0',
                        'ploidy = diploid']

    You can classify it by calling:

        classification = dt.classify(root_node, test_sample)

    The call to `classify()' returns a reference to a hash whose keys are
    the class names and the values the associated classification
    probabilities.  This hash also includes another key-value pair for the
    solution path from the root node to the leaf node at which the final
    classification was carried out.

    If your features are purely symbolic, you can continue to use the same
    constructor syntax that was used in the older versions of this module.
    However, your old `.dat' training files will not work with the new
    version.  The good news is that with just a small fix, you can continue
    to use them.  The fix and why it was needed is described in the file
    README_for_dat_files in the `examples' directory.  If you are going to
    use a `.dat' file for supplying the training data, your constructor
    syntax is likely to look like:

        training_datafile = "training.dat"

        dt = DecisionTree.DecisionTree( 
                                training_datafile = training_datafile,
                                entropy_threshold = 0.01,
                                max_depth_desired = 5,
             )

    You'd still need to make the following calls for reading in the
    training data, for initializing the probability cache, and for
    constructing the decision tree:

        dt.get_training_data()
        dt.calculate_first_order_probabilities()
        dt.calculate_class_priors()
        root_node = dt.construct_decision_tree_classifier()

    Now your test sample is likely to look like:

        test_sample = ['exercising=never', 
                       'smoking=heavy', 
                       'fatIntake=heavy', 
                       'videoAddiction=heavy']

    You'd now call the calssifier as before: 

        classification = dt.classify(root_node, test_sample)

    A decision tree can quickly become much too large (and much too slow to
    construct and to yield classification results) if the total number of
    features is large and/or if the number of different possible values for
    the symbolic features is large.  You can control the size of the tree
    through the constructor options `entropy_threshold' and
    `max_depth_desired'. The latter option sets the maximum depth of your
    decision tree to max_depth_desired value.  The parameter
    `entropy_threshold' sets the granularity with which the entropies are
    sampled.  Its default value is 0.001.  The larger the value you choose
    for entropy_threshold, the smaller the tree.


@title
INTRODUCTION:

    DecisionTree is a Python module for constructing a decision tree from a
    training data file containing multidimensional data in the form of a
    table. In one form or another, decision trees have been around for over
    fifty years. From a statistical perspective, they are closely related
    to classification and regression by recursive partitioning of
    multidimensional data. Early work that demonstrated the usefulness of
    such partitioning for classification and regression can be traced, in
    the statistics community, to the work of Terry Therneau in the early
    1980's and, in the machine learning community, to the work of Ross
    Quinlan in the mid 1990's.

    For those not familiar with decision tree ideas, the traditional way to
    classify multidimensional data is to start with a feature space whose
    dimensionality is the same as that of the data.  Each feature measures
    a specific attribute of an entity.  You use the training data to carve
    up the feature space into different regions, each corresponding to a
    different class.  Subsequently, when you try to classify a new data
    sample, you locate it in the feature space and find the class label of
    the region to which it belongs.  One can also give the data point the
    same class label as that of the nearest training sample. This is
    referred to as the nearest neighbor classification. There exist
    hundreds of variations of varying power on this basic approach to the
    classification of multidimensional data.

    A decision tree classifier works differently.  When you construct a
    decision tree, you select for the root node a feature test that
    partitions the training data in a way that causes maximal
    disambiguation of the class labels associated with the data.  In terms
    of information content as measured by entropy, such a feature test
    would cause maximum reduction in class entropy in going from all of the
    training data taken together to the data as partitioned by the feature
    test.  You then drop from the root node a set of child nodes, one for
    each partition of the training data created by the feature test at the
    root node. When your features are purely symbolic, you'll have one
    child node for each value of the feature chosen for the feature test at
    the root.  When the test at the root involves a numeric feature, you
    find the decision threshold for the feature that best bipartitions the
    data and you drop from the root node two child nodes, one for each
    partition.  Now at each child node you pose the same question that you
    posed when you found the best feature to use at the root: Which feature
    at the child node in question would maximally disambiguate the class
    labels associated with the training data corresponding to that child
    node?

    As the reader would expect, the two key steps in any approach to
    decision-tree based classification are the construction of the decision
    tree itself from a file containing the training data, and then using
    the decision tree thus obtained for classifying the data.

    What is cool about decision tree classification is that it gives you
    soft classification, meaning it may associate more than one class label
    with a given data vector.  When this happens, it may mean that your
    classes are indeed overlapping in the underlying feature space.  It
    could also mean that you simply have not supplied sufficient training
    data to the decision tree classifier.  For a tutorial introduction to
    how a decision tree is constructed and used, see

    https://engineering.purdue.edu/kak/Tutorials/DecisionTreeClassifiers.pdf


@title
WHAT PRACTICAL PROBLEM IS SOLVED BY THIS MODULE?

    If you are new to the concept of a decision tree, their practical
    utility is best understood with an example that only involves symbolic
    features.  However, as mentioned earlier, versions 2.0 and higher of
    this module handle both symbolic and numeric features.

    Consider the following scenario: Let's say you are running a small
    investment company that employs a team of stockbrokers who make
    buy/sell decisions for the customers of your company.  Assume that your
    company has asked the traders to make each investment decision on the
    basis of the following five criteria:

            price_to_earnings_ratio   (P_to_E)

            price_to_sales_ratio      (P_to_S)

            return_on_equity          (R_on_E)

            market_share              (M_S)

            sentiment                 (S)

    Since you are the boss, you keep track of the buy/sell decisions made
    by the individual traders.  But one unfortunate day, all of your
    traders decide to quit because you did not pay them enough.  So what
    are you to do?  If you had a module like the one here, you could still
    run your company and do so in such a way that your company would, on
    the average, perform better than any of the individual traders who
    worked for you previously.  This is what you would need to do:: You
    would pool together the individual trader buy/sell decisions you
    accumulated during the last one year.  This pooled information is
    likely to look like:


      example      buy/sell     P_to_E     P_to_S     R_on_E     M_S     S
      ====================================================================

      example_1     buy          high       low        medium    low    high
      example_2     buy          medium     medium     low       low    medium
      example_3     sell         low        medium     low       high   low
      ....
      ....

    This data would constitute your training file. Assuming that this training
    file is called 'training.dat', you would need to feed this file
    into the module by calling:

        dt = DecisionTree( training_datafile = "training.dat" )

        dt.get_training_data()

        dt.calculate_first_order_probabilities_for_numeric_features()

        dt.calculate_class_priors()

    Subsequently, you would construct a decision tree by calling:

        root_node = dt.construct_decision_tree_classifier()

    Now you and your company (with practically no employees) are ready to
    service the customers again. Suppose your computer needs to make a
    buy/sell decision about an investment prospect that is best described
    by:

        price_to_earnings_ratio   =  low
        price_to_sales_ratio      =  very_low
        return_on_equity          =  none
        market_share              =  medium    
        sentiment                 =  low

    All that your computer would need to do would be to construct a data
    vector like

        test_case = [ 'P_to_E=low', 
                      'P_to_S=very_low', 
                      'R_on_E=none',
                      'M_S=medium',
                      'S=low'  ]

    and call the decision tree classifier you just constructed by

        classification = dt.classify(root_node, test_case)

        print "Classification: ", classification

    The answer returned will be 'buy' and 'sell', along with the associated
    probabilities.  So if the probability of 'buy' is considerably greater
    than the probability of 'sell', that's what you should instruct your
    computer to do.

    The chances are that, on the average, this approach would beat the
    performance of any of your individual traders who worked for you
    previously since the buy/sell decisions made by the computer would be
    based on the collective wisdom of all your previous traders.
    DISCLAIMER: There is obviously a lot more to good investing than what
    is captured by the silly little example here. However, it does
    convey the sense in which the current module can be used.


@title
SYMBOLIC FEATURES VERSUS NUMERIC FEATURES

    A feature is symbolic when its values are compared using string
    comparison operators.  By the same token, a feature is numeric when its
    values are compared using numeric comparison operators.  Having said
    that, features that take only a small number of numeric values in
    the training data can be treated symbolically provided you are careful
    about handling their values in the test data.  At the least, you have to
    set the test data value for such a feature to its closest value in the
    training data.  

    The constructor parameter symbolic_to_numeric_cardinality_threshold
    let's you tell the module when to consider an otherwise numeric feature
    symbolically. Suppose you set this parameter to 10, that means that all
    numeric looking features that take 10 or fewer different values in the
    training datafile will be considered to be symbolic features.
    
    See the tutorial at

    https://engineering.purdue.edu/kak/Tutorials/DecisionTreeClassifiers.pdf

    for further information on the implementation issues related to the
    symbolic and numeric features.


@title
TESTING THE QUALITY OF YOUR TRAINING DATA:

    Version 2.2 includes a new class named EvalTrainingData, derived from
    the main class DecisionTree, that runs a 10-fold cross-validation test
    on your training data to test its ability to discriminate between the
    classes mentioned in the training file.

    The 10-fold cross-validation test divides all of the training data into
    ten parts, with nine parts used for training a decision tree and one
    part used for testing its ability to classify correctly. This selection
    of nine parts for training and one part for testing is carried out in
    all of the ten different possible ways.  

    The following code fragment illustrates how you invoke the testing
    function of the EvalTrainingData class:

        training_datafile = "training3.csv"
        eval_data = DecisionTree.EvalTrainingData(
                                training_datafile = training_datafile,
                                csv_class_column_index = 1,
                                csv_columns_for_features = [2,3],
                                entropy_threshold = 0.01,
                                max_depth_desired = 3,
                                symbolic_to_numeric_cardinality_threshold = 10,
                    )

        eval_data.get_training_data()
        eval_data.evaluate_training_data()

    The last statement above prints out a Confusion Matrix and the value of
    Training Data Quality Index on a scale of 100, with 100 designating
    perfect training data.  The Confusion Matrix shows how the different
    classes were mis-identified in the 10-fold cross-validation test.

    This testing functionality can also be used to find the best values to
    use for the constructor parameters entropy_threshold,
    max_depth_desired, and symbolic_to_numeric_cardinality_threshold.

    The following two scripts in the Examples directory illustrate the use
    of the EvalTrainingData class for testing the quality of your data:

        evaluate_training_data1.py

        evaluate_training_data2.py


@title
HOW TO MAKE THE BEST CHOICES FOR THE CONSTRUCTOR PARAMETERS:

    Assuming your training data is good, the quality of the results you get
    from a decision tree would depend on the choices you make for the
    constructor parameters entropy_threshold, max_depth_desired, and
    symbolic_to_numeric_cardinality_threshold.  You can optimize your
    choices for these parameters by running the 10-fold cross-validation
    test that is made available in Version 2.2 through the new class
    EvalTrainingData that is included in the module file.  A description of
    how to run this test is in the section titled "TESTING THE QUALITY OF
    YOUR TRAINING DATA" of this document.

    
@title
METHODS:

    The module provides the following methods for constructing a decision
    tree from training data in a disk file, and for data classification with
    the decision tree.


@title
Constructing a decision tree:

        dt = DecisionTree( training_datafile = training_datafile,
                           csv_class_column_index = 2,
                           csv_columns_for_features = [3,4,5,6,7,8],
                           entropy_threshold = 0.01,
                           max_depth_desired = 8,
                           symbolic_to_numeric_cardinality_threshold = 10,
                         )

    This yields a new instance of the DecisionTree class.  For this call to
    make sense, the training data in the training datafile must be conform
    to a certain format.  For example, the first row must name the
    features.  It must begin with the empty string `""' as shown by the CSV
    files in the Examples subdirectory.  The first column for all
    subsequent rows must carry a unique integer identifier for each data
    record.  When your features are purely symbolic, you are also allowed
    to use the `.dat' files that were used in the previous versions of this
    module.

    The constructor option csv_class_column_index supplies to the module
    zero-based index of the column that contains the class label for the
    training data records. In the example shown above, the class labels are
    in the third column.  The option csv_columns_for_features tells the
    module which of the features are supposed to be used for decision tree
    construction.  The constructor option max_depth_desired sets the
    maximum depth of the decision tree. The parameter entropy_threshold
    sets the granularity with which the entropies are sampled.  The
    parameter symbolic_to_numeric_cardinality_threshold allows the module
    to treat an otherwise numeric feature symbolically if it only takes a
    small number of different values in the training data file.  For the
    constructor call shown above, if a feature takes on only 10 or fewer
    different values in the training data file, it will be treated like a
    symbolic feature.


@title
The constructor parameters:

    training_datafile:

        This parameter supplies the name of the file that contains the
        training data.  This must be a CSV file if your training data
        includes both numeric and symbolic features.  If your data is
        purely symbolic, you can use the old-style `.dat' file.

    csv_class_column_index:

        When using a CSV file for your training data, this parameter
        supplies the zero-based column index for the column that contains
        the class label for each data record in the training file.

    csv_columns_for_features:

        When using a CSV file for your training data, this parameter
        supplies a list of columns corresponding to the features you wish
        to use for decision tree construction.  Each column is specified by
        its zero-based index.

    entropy_threshold:

        This parameter sets the granularity with which the entropies are
        sampled by the module.  For example, a feature test at a node in
        the decision tree is acceptable if the entropy gain achieved by the
        test exceeds this threshold.  The larger the value you choose for
        this parameter, the smaller the tree.  Its default value is 0.001.

    max_depth_desired:

        This parameter sets the maximum depth of the decision tree.  For
        obvious reasons, the smaller the value you choose for this
        parameter, the smaller the tree.

    symbolic_to_numeric_cardinality_threshold:

        This parameter allows the module to treat an otherwise numeric
        feature symbolically if the number of different values the feature
        takes in the training data file does not exceed the value of this
        parameter.

    You can choose the best values to use for the last three constructor
    parameters by running a 10-fold cross-validation test on your training
    data through the embedded class EvalTrainingData that comes with
    Version 2.2 of this module.  See the section "TESTING THE QUALITY OF
    YOUR TRAINING DATA" of this document page.


@title
Reading in the training data:

    After you have constructed a new instance of the DecisionTree class,
    you must now read in the training data that is contained in the file
    named above.  This you do by:

        dt.get_training_data()

    IMPORTANT: The training data file must be in a format that makes sense
    to the decision tree constructor.  If you use numeric features, you
    must use a CSV file for supplying the training data.  The first row of
    such a file must name the features and it must begin with the empty
    string `""' as shown in the `stage3cancer.csv' file in the Examples
    subdirectory.  The first column for all subsequent rows must carry a
    unique integer identifier for each training record.


@title
Initializing the probability cache:

    After a call to the constructor and the get_training_data() method, you
    must call the following methods for initializing the probabilities:

        dt.calculate_first_order_probabilities()
        dt.calculate_class_priors()


@title
Displaying the training data:

    If you wish to see the training data that was just digested by the
    module, call

        dt.show_training_data() 


@title
Constructing a decision-tree classifier:

    After the training data is ingested, it is time to construct a decision
    tree classifier.  This you do by

        root_node = dt.construct_decision_tree_classifier()

    This call returns an instance of type DTNode.  The DTNode class is
    defined within the main package file, at its end.  So, don't forget,
    that root_node in the above example call will be instantiated to an
    instance of type DTNode.


@title
Displaying the decision tree:

    You display a decision tree by calling

        root_node.display_decision_tree("   ")

    This displays the decision tree in your terminal window by using a
    recursively determined offset for each node as the display routine
    descends down the tree.

    I have intentionally left the syntax fragment root_node in the above
    call to remind the reader that display_decision_tree() is NOT called on
    the instance of the DecisionTree we constructed earlier, but on the
    Node instance returned by the call to
    construct_decision_tree_classifier().


@title
Classifying new data:

    You classify new data by first constructing a new data vector:

        test_sample  = ['g2 = 4.2',
                        'grade = 2.3',
                        'gleason = 4',
                        'eet = 1.7',
                        'age = 55.0',
                        'ploidy = diploid']

    and calling the classify() method as follows:
 
        classification = dt.classify(root_node, test_sample)

    where, again, root_node is an instance of type Node that was returned
    by calling construct_decision_tree_classifier().  The variable
    classification is a dictionary whose keys are the class labels and
    whose values the associated probabilities.  You can print it out by

        print "Classification: ", classification


@title
Displaying the number of nodes created:

    You can print out the number of nodes in a decision tree by calling

        root_node.how_many_nodes()


@title
Using the decision tree interactively:

    Starting with Version 1.6 of the module, you can use the DecisionTree
    classifier in an interactive mode.  In this mode, after you have
    constructed the decision tree, the user is prompted for answers to the
    questions regarding the feature tests at the nodes of the tree.
    Depending on the answer supplied by the user at a node, the classifier
    takes a path corresponding to the answer to descend down the tree to
    the next node, and so on.  The following method makes this mode
    possible.  Obviously, you can call this method only after you have
    constructed the decision tree.

        dt.classify_by_asking_questions(root_node)


@title
Generating synthetic training data:

    To generate synthetic training data, you first construct an instance of
    the class TrainingDataGenerator that is incorporated in the
    DecisionTree module.  A call to the constructor of this class will look
    like:

        parameter_file = "param_numeric.txt"
        output_csv_file = "training.csv";
        training_data_gen = TrainingDataGeneratorNumeric(
                              output_csv_file   = output_csv_file,
                              parameter_file    = parameter_file,
                              number_of_samples_per_class = some_number,
                            )
        training_data_gen.read_parameter_file_numeric()
        training_data_gen.gen_numeric_training_data_and_write_to_csv()

    The training data that is generated is according to the specifications
    described in the parameter file.  The structure of this file must be as
    shown in the file `param_numeric.txt' for the numeric training data and
    as shown in `param_symbolic.txt' for the case of symbolic training
    data.  Both these example parameter files are in the 'Examples'
    subdirectory.  The parameter file names the classes, the features for
    the classes, and the possible values for the features.

    If you want to generate purely symbolic training data, here is the
    constructor call to make:

        parameter_file = "param_symbolic.txt"
        output_data_file = "training.dat";
        training_data_gen = TrainingDataGeneratorSymbolic(
                              output_datafile   = output_data_file,
                              parameter_file    = parameter_file,
                              write_to_file     = 1,
                              number_of_training_samples = some_number,
                            )
        training_data_gen.read_parameter_file_symbolic()
        training_data_gen.gen_symbolic_training_data()
        training_data_gen.write_training_data_to_file()


@title
Generating synthetic test data:

    To generate synthetic test data, you first construct an instance of the
    class TestDataGeneratorSymbolic that is incorporated in the
    DecisionTree module.  A call to the constructor of this class will look
    like:

        test_data_gen = TestDataGeneratorSymbolic(
                          output_test_datafile     = an_output_data_file,
                          output_class_labels_file = a_file_for_class_labels,
                          parameter_file           = a_parameter_file,
                          write_to_file            = 1,
                          number_of_test_samples = some_number,
                        )

    The main difference between the training data and the test data is that
    the class labels are NOT mentioned in the latter.  Instead, the class
    labels are placed in a separate file whose name is supplied through the
    constructor option `output_class_labels_file' shown above.  The test
    data that is generated is according to the specifications described in
    the parameter file.  In general, this parameter file would be the same
    that you used for generating the training data.
    
@title
HOW THE CLASSIFICATION RESULTS ARE DISPLAYED

    It depends on whether you apply the classifier at once to all the data
    samples in a file, or whether you feed one data vector at a time into
    the classifier.

    In general, the classifier returns soft classification for a test data
    vector.  What that means is that, in general, the classifier will list
    all the classes to which a given data vector could belong and the
    probability of each such class label for the data vector. Run the
    examples scripts in the Examples directory to see how the output of
    classification can be displayed.

    For large test datasets, you would obviously want to process an entire
    file of test data at a time.  For the case of purely symbolic data, the
    best way to do this is to follow my script

        classify_test_data_in_a_file.py

    in the 'Examples' directory.  This script requires three command-line
    arguments, the first argument names the training datafile, the second
    the test datafile, and the third in which the classification results
    will be deposited.  The test datafile must mention the order in which
    the features values are presented.  For an example, see the file
    'testdata.dat' in the 'Examples' directory.

    With regard to the soft classifications returned by this classifier, if
    the probability distributions for the different classes overlap in the
    underlying feature space, you would want the classifier to return all
    of the applicable class labels for a data vector along with the
    corresponding class probabilities.  Another reason for why the decision
    tree classifier may associate significant probabilities with multiple
    class labels is that you used inadequate number of training samples to
    induce the decision tree.  The good thing is that the classifier does
    not lie to you (unlike, say, a hard classification rule that would
    return a single class label corresponding to the partitioning of the
    underlying feature space).  The decision tree classifier give you the
    best classification that can be made given the training data you fed
    into it.


@title
THE EXAMPLES DIRECTORY:

    See the 'Examples' directory in the distribution for how to construct a
    decision tree, and how to then classify new data using the decision
    tree.  To become more familiar with the module, run the scripts

        construct_dt_and_classify_one_sample_case1.py
        construct_dt_and_classify_one_sample_case2.py
        construct_dt_and_classify_one_sample_case3.py
        construct_dt_and_classify_one_sample_case4.py

    The first script is for the purely symbolic case, the second for the
    case that involves both numeric and symbolic features, the third for
    the case of purely numeric features, and the last for the case when the
    training data is synthetically generated by the script
    generate_training_data_numeric.py

    Next run the script as it is

       classify_test_data_in_a_file.py   training.dat   testdata.dat   out.txt

    This call will first construct a decision tree using the training data
    in the file 'training.dat'.  It will then calculate the class label for
    each data record in the file 'testdata.dat'.  The estimated class
    labels will be written out to the file 'out.txt'.

    The following script in the 'Examples' directory 

        classify_by_asking_questions.py

    shows how you can use a decision-tree classifier interactively.  In
    this mode, you first construct the decision tree from the training data
    and then the user is prompted for answers to the feature tests at the
    nodes of the tree.

    The 'Examples' directory also contains the following scripts:

        generate_training_data_numeric.py
        generate_training_data_symbolic.py
        generate_test_data_symbolic.py

    that show how you can use the module to generate synthetic training and
    test data.  Synthetic training and test data are generated according to
    the specifications laid out in a parameter file.  There are constraints
    on how the information is laid out in the parameter file.  See the
    files `param_numeric.txt' and `param_symbolic.txt' in the 'Examples'
    directory for how to structure these files.

    The Examples directory of Version 2.2 of the DecisionTree module also
    contains the following two scripts:

       evaluate_training_data1.py
       evaluate_training_data2.py

    that illustrate how the Python class EvalTrainingData can be used to
    evaluate the quality of your training data (as long as it resides in a
    `.csv' file.)  This new class is a subclass of the DecisionTree class
    in the module file.  See the README in the Examples directory for
    further information regarding these two scripts.


@title  
INSTALLATION:

    The DecisionTree class was packaged using Distutils.  For installation,
    execute the following command-line in the source directory (this is the
    directory that contains the setup.py file after you have downloaded and
    uncompressed the package):
 
            python setup.py install

    You have to have root privileges for this to work.  On Linux
    distributions, this will install the module file at a location that
    looks like

             /usr/lib/python2.7/dist-packages/

    If you do not have root access, you have the option of working directly
    off the directory in which you downloaded the software by simply
    placing the following statements at the top of your scripts that use
    the DecisionTree class:

            import sys
            sys.path.append( "pathname_to_DecisionTree_directory" )

    To uninstall the module, simply delete the source directory, locate
    where the DecisionTree module was installed with "locate DecisionTree"
    and delete those files.  As mentioned above, the full pathname to the
    installed version is likely to look like
    /usr/lib/python2.7/dist-packages/DecisionTree*

    If you want to carry out a non-standard install of the DecisionTree
    module, look up the on-line information on Disutils by pointing your
    browser to

              http://docs.python.org/dist/dist.html


@title
BUGS:

    Please notify the author if you encounter any bugs.  When sending
    email, please place the string 'DecisionTree' in the subject line.


@title
ACKNOWLEDGMENTS:

    The importance of the 'sentiment' feature in the "What Practical Problem
    is Solved by this Module" section was mentioned to the author by John
    Gorup.  Thanks John.

@title
AUTHOR:

    Avinash Kak, kak@purdue.edu

    If you send email, please place the string "DecisionTree" in your
    subject line to get past my spam filter.

@title
COPYRIGHT:

    Python Software Foundation License

    Copyright 2013 Avinash Kak

'''

import math
import re
import sys
import functools 
import operator
import itertools


#-----------------------------------  Utility Functions  ------------------------------------

def sample_index(sample_name):
    '''
    When the training data is read from a CSV file, we assume that the first column
    of each data record contains a unique integer identifier for the record in that
    row. This training data is stored in a dictionary whose keys are the prefix
    'sample_' followed by the identifying integers.  For the data in the old-style
    `.dat' files, we assume that each record begins with the string `sample_xx' where
    `xx' is a unique integer.  In both cases, the purpose of this function is to
    return the identifying integer associated with a data record.
    '''
    m = re.search('_(.+)$', sample_name)
    return int(m.group(1))

def deep_copy_array(array_in):
    '''
    Meant only for an array of scalars (no nesting):
    '''
    array_out = []
    for i in range(len(array_in)):
        array_out.append( array_in[i] )
    return array_out

def minimum(arr):
    '''
    Returns simultaneously the minimum value and its positional index in an
    array. [Could also have used min() and index() defined for Python's
    sequence types.]
    '''
    minval,index = None,None
    for i in range(0, len(arr)):  
        if minval is None or arr[i] < minval:
            index = i
            minval = arr[i]
    return minval,index

def convert(value):
    try:
        answer = float(value)
        return answer
    except:
        return value

def closest_sampling_point(value, arr):
    compare = [abs(x - value) for x in arr]
    minval,index = minimum(compare)
    return arr[index]

#------------------------------- DecisionTree Class Definition --------------------------------

class DecisionTree(object):
    def __init__(self, *args, **kwargs ):
        if kwargs and args:
            raise ValueError(  
                   '''DecisionTree constructor can only be called with keyword
                      arguments for the following keywords: training_datafile, 
                      entropy_threshold, max_depth_desired, csv_class_column_index,
                      symbolic_to_numeric_cardinality_threshold,
                      csv_columns_for_features, debug1, debug2, and debug3''') 
        allowed_keys = 'training_datafile','entropy_threshold','max_depth_desired','csv_class_column_index',\
                       'symbolic_to_numeric_cardinality_threshold', 'csv_columns_for_features', 'debug1',\
                       'debug2', 'debug3'
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise ValueError(keyword + ":  Wrong keyword used --- check spelling") 
        training_datafile = entropy_threshold = max_depth_desired = csv_class_column_index = None
        symbolic_to_numeric_cardinality_threshold = csv_columns_for_features = debug1 = debug2 = debug3 = None
        if kwargs and not args:
            if 'training_datafile' in kwargs : training_datafile = kwargs.pop('training_datafile')
            if 'entropy_threshold' in kwargs : entropy_threshold = kwargs.pop('entropy_threshold')
            if 'max_depth_desired' in kwargs : max_depth_desired = kwargs.pop('max_depth_desired')
            if 'csv_class_column_index' in kwargs: csv_class_column_index = kwargs.pop('csv_class_column_index')
            if 'csv_columns_for_features' in kwargs: \
                                  csv_columns_for_features = kwargs.pop('csv_columns_for_features')
            if 'symbolic_to_numeric_cardinality_threshold' in kwargs: \
                           symbolic_to_numeric_cardinality_threshold = \
                                                         kwargs.pop('symbolic_to_numeric_cardinality_threshold')
            if 'debug1' in kwargs  :  debug1 = kwargs.pop('debug1')
            if 'debug2' in kwargs  :  debug2 = kwargs.pop('debug2')
            if 'debug3' in kwargs  :  debug3 = kwargs.pop('debug3')
        if not args and training_datafile:
            self._training_datafile = training_datafile
        elif not args and not training_datafile:
                raise ValueError('''You must specify a training datafile''')
        else:
            if args[0] != 'evalmode':
                raise ValueError("""When supplying non-keyword arg, it can only be 'evalmode'""")
        if entropy_threshold: 
            self._entropy_threshold                         =      entropy_threshold
        else:
            self._entropy_threshold                         =      0.001        
        if max_depth_desired:
            self._max_depth_desired                         =      max_depth_desired 
        else:
            self._max_depth_desired                         =      None
        if debug1:
            self._debug1                                    =      debug1
        else:
            self._debug1                                    =      0
        if debug2:
            self._debug2                                    =      debug2
        else:
            self._debug2                                    =      0
        if debug3:
            self._debug3                                    =      debug3
        else:
            self._debug3                                    =      0
        if csv_class_column_index:
            self._csv_class_column_index                    =      csv_class_column_index
        else:
            self._csv_class_column_index                    =      None
        if csv_columns_for_features:
            self._csv_columns_for_features                  =      csv_columns_for_features
        else: 
            self._csv_columns_for_features                  =      None            
        if symbolic_to_numeric_cardinality_threshold:
            self._symbolic_to_numeric_cardinality_threshold =      symbolic_to_numeric_cardinality_threshold
        else:
            self._symbolic_to_numeric_cardinality_threshold =      10
        self._root_node                                     =      None
        self._probability_cache                             =      {}
        self._entropy_cache                                 =      {}
        self._training_data_dict                            =      {}
        self._features_and_values_dict                      =      {}
        self._features_and_unique_values_dict               =      {}
        self._samples_class_label_dict                      =      {}  
        self._class_names                                   =      []
        self._class_priors_dict                             =      {}
        self._feature_names                                 =      []
        self._numeric_features_valuerange_dict              =      {}
        self._sampling_points_for_numeric_feature_dict      =      {}
        self._feature_values_how_many_uniques_dict          =      {}
        self._prob_distribution_numeric_features_dict       =      {}
        self._histogram_delta_dict                          =      {}
        self._num_of_histogram_bins_dict                    =      {}

    def get_training_data_from_csv(self):
        class_name_in_column = self._csv_class_column_index - 1  # subtract 1 because first col has record labels
        if not self._training_datafile.endswith('.csv'): 
            sys.exit("Aborted. get_training_data_csv() is only for CSV files")
        all_data = [line.rstrip().split(',') for line in open(self._training_datafile,"rU")]
        data_dict = {line[0] : line[1:] for line in all_data}
        if '""' not in data_dict:
            sys.exit('''Aborted. The first row of CSV file must begin '''
                     '''with "" and then list the feature names and the class names''')
        feature_names = [item.strip('"') for item in data_dict['""']]
        class_column_heading = feature_names[class_name_in_column]
        feature_names = [feature_names[i-1] for i in self._csv_columns_for_features]
        class_for_sample_dict = { "sample_" + key.strip('"') : \
               class_column_heading + "=" + data_dict[key][class_name_in_column] \
                                                   for key in data_dict if key != '""'}
        feature_values_for_samples_dict = {"sample_" + key.strip('"') : \
              list(map(operator.add, list(map(operator.add, feature_names, "=" * len(feature_names))),   \
                    [str(convert(data_dict[key][i-1].strip('"'))) for i in self._csv_columns_for_features])) \
                                                               for key in data_dict if key != '""'}
        features_and_values_dict = {data_dict['""'][i-1].strip('"') : \
                      [convert(data_dict[key][i-1].strip('"')) for key in data_dict if key != '""']  \
                            for i in self._csv_columns_for_features} 
        all_class_names = sorted(list(set(class_for_sample_dict.values())))
        if self._debug1: print("\n All class names: "+ str(all_class_names))
        numeric_features_valuerange_dict = {}
        feature_values_how_many_uniques_dict = {}
        features_and_unique_values_dict = {}
        for feature in features_and_values_dict:
            unique_values_for_feature = list(set(features_and_values_dict[feature]))
            unique_values_for_feature = sorted(list(filter(lambda x: x != 'NA', unique_values_for_feature)))
            feature_values_how_many_uniques_dict[feature] = len(unique_values_for_feature)
            if all(isinstance(x,float) for x in unique_values_for_feature):
                numeric_features_valuerange_dict[feature] = \
                              [min(unique_values_for_feature), max(unique_values_for_feature)]
                unique_values_for_feature.sort(key=float)
            features_and_unique_values_dict[feature] = sorted(unique_values_for_feature)
        if self._debug1:
            print("\nAll class names: " + str(all_class_names))
            print("\nEach sample data record:")
            for item in sorted(feature_values_for_samples_dict.items(), key = lambda x: sample_index(x[0]) ):
                print(item[0]  + "  =>  "  + str(item[1]))
            print("\nclass label for each data sample:")
            for item in sorted(class_for_sample_dict.items(), key=lambda x: sample_index(x[0])):
                print(item[0]  + "  =>  "  + str(item[1]))
            print("\nfeatures and the values taken by them:")
            for item in sorted(features_and_values_dict.items()):
                print(item[0]  + "  =>  "  + str(item[1]))
            print("\nnumeric features and their ranges:")
            for item in sorted(numeric_features_valuerange_dict.items()):
                print(item[0]  + "  =>  "  + str(item[1]))
            print("\nnumber of unique values in each feature:")
            for item in sorted(feature_values_how_many_uniques_dict.items()):
                print(item[0]  + "  =>  "  + str(item[1]))
        self._class_names = all_class_names
        self._feature_names = feature_names
        self._samples_class_label_dict    =  class_for_sample_dict
        self._training_data_dict          =  feature_values_for_samples_dict
        self._features_and_values_dict    =  features_and_values_dict
        self._features_and_unique_values_dict    =  features_and_unique_values_dict
        self._numeric_features_valuerange_dict = numeric_features_valuerange_dict
        self._feature_values_how_many_uniques_dict = feature_values_how_many_uniques_dict

    def get_training_data(self):
        '''
        If your training data is purely symbolic, as in Version 1.7.1, you might find
        it easier to create a `.dat' file.  For purely numeric data, or mixed
        symbolic and numeric data, you MUST use a `.csv' file.  See examples of these
        files in the `Examples' subdirectory.
        '''
        if self._training_datafile.endswith('.csv'):
            self.get_training_data_from_csv()
        elif self._training_datafile.endswith('.dat'):
            self.get_training_data_from_dat()
        else:
            sys.exit("Your training datafile must either be a csv file or a dat file")

    def get_training_data_from_dat(self):
        '''
        Meant for purely symbolic data (as in all versions up to v. 1.7.1)
        '''
        recording_features_flag = 0
        recording_training_data = 0
        table_header = None
        column_labels_dict = {}
        FILE = None
        try:
            FILE = open( self._training_datafile )
        except IOError:
            print("unable to open %s" % self._training_datafile)
            sys.exit(1)
        for line in FILE:
            line = line.rstrip()
            if line is '': continue            
            if line.startswith(r'#'): continue
            if line.startswith(r'======='): continue
            if re.search(r'\s*class', line, re.IGNORECASE) \
                       and not recording_training_data \
                       and not recording_features_flag:
                classpattern = r'^\s*class names:\s*([ \S]+)\s*'
                m = re.search(classpattern, line, re.IGNORECASE)
                if not m: 
                    raise ValueError('''No class names in training file''')
                self._class_names = m.group(1).split()
                continue
            elif re.search(r'\s*feature names and their values', \
                               line, re.IGNORECASE):
                recording_features_flag = 1
                continue
            elif re.search(r'^\s*training data:\s*$', line, re.IGNORECASE):
                recording_training_data = 1
                recording_features_flag = 0
                continue
            elif not recording_training_data and recording_features_flag:
                feature_name_value_pattern = r'^\s*(\S+)\s*=\s*(.+)'
                m = re.search(feature_name_value_pattern, line, re.IGNORECASE)
                feature_name = m.group(1)
                feature_values = m.group(2).split()
                self._features_and_values_dict[feature_name]  = feature_values
                self._features_and_unique_values_dict[feature_name]  = sorted(list(set(feature_values)))
            elif recording_training_data:
                if not table_header:
                    table_header = line.split()
                    for i in range(2, len(table_header)):
                        column_labels_dict[i] = table_header[i]
                    if (len(self._features_and_values_dict) != len(table_header)-2):
                        sys.exit('''Incorrect number of feature names in the header '''
                                 '''line that is just below the "Training Data:" line''')
                    continue
                record = line.split()
                if record[1] not in self._class_names:
                    print("Error in data record: " + record[0])
                    sys.exit('''For the data record named above, the class name '''
                             '''does not match the class names extracted earlier from '''
                             '''the file. You may have used commas or some other '''
                             '''punctuation to separate out the class names '''
                             '''earlier''' )                
                self._samples_class_label_dict[record[0]] = record[1]
                self._training_data_dict[record[0]] = []
                if (len(self._features_and_values_dict) != len(record) - 2):
                    print("Error in data record: " + record[0])
                    sys.exit("Number of values supplied for the above data record is wrong")
                for i in range(2, len(record)):
                    feature_name_for_i = column_labels_dict[i]
                    if record[i] not in self._features_and_values_dict[feature_name_for_i]:
                        print("Error in data record: " + record[0])
                        sys.exit('''For the data record named above, one of the feature '''
                                 '''values does not correspond to the different possible '''
                                 '''values declared at the top of the training file. You '''
                                 '''may have used commas or other punctuation marks to '''
                                 '''separate out the feature values in the data record''' )
                    self._training_data_dict[record[0]].append(
                          column_labels_dict[i] + "=" + record[i] )
        FILE.close()                        
        self._feature_names = list(self._features_and_values_dict.keys())
        empty_classes = []
        for classname in self._class_names:        
            if classname not in self._samples_class_label_dict.values():
                empty_classes.append( classname )
        if empty_classes and self._debug1:
            num_empty_classes = len(empty_classes)
            print('''\nDid you know you have %d empty classes?  The DecisionTree '''
                  '''module can ignore these classes for you.''' % (num_empty_classes))
            print("EMPTY CLASSES: " , empty_classes) 
            ans = None
            if sys.version_info[0] == 3:
                ans = input("\nIgnore empty classes and continue? Enter 'y' if yes:  ")
            else:
                ans = raw_input("\nIgnore empty classes and continue? Enter 'y' if yes:  ")
            ans = ans.strip()
            if ans != 'y':
                sys.exit(0)
        for classname in empty_classes:
            self._class_names.remove(classname)
        if self._debug1:
            print("Class names: ", self._class_names)
            print( "Feature names: ", self._feature_names)
            print("Features and values: ", self._features_and_values_dict.items())
        for feature in self._feature_names:
            values_for_feature = self._features_and_values_dict[feature]
            for value in values_for_feature:
                feature_and_value = "".join([feature, "=", value])
                self._probability_cache[feature_and_value] = self.probability_of_feature_value(feature, value)

    def calculate_first_order_probabilities(self):
        for feature in self._feature_names:
            self.probability_of_feature_value(feature,None)

            if self._debug2:
                if feature in self._prob_distribution_numeric_features_dict:
                    print("\nPresenting probability distribution for a feature considered to be numeric:")
                    for sampling_point in \
                         sorted(self._prob_distribution_numeric_features_dict[feature].keys()):
                        print(feature + '::' + str(sampling_point) + ' = ' +  "{0:.5f}".format(\
                               self._prob_distribution_numeric_features_dict[feature][sampling_point]))  
                else:
                    print("\nPresenting probabilities for the values of a feature considered to be symbolic:")
                    values_for_feature = self._features_and_unique_values_dict[feature]
                    for value in values_for_feature:
                        prob = self.probability_of_feature_value(feature,value) 
                        print(feature + '::' + str(value) + ' = ' +  "{0:.5f}".format(prob))

    def show_training_data(self):
        print("Class names: ", self._class_names)
        print("\n\nFeatures and Their Possible Values:\n\n")
        features = self._features_and_values_dict.keys()
        for feature in sorted(features):
            print("%s ---> %s" \
                  % (feature, self._features_and_values_dict[feature]))
        print("\n\nSamples vs. Class Labels:\n\n")
        for item in sorted(self._samples_class_label_dict.items(), \
                key = lambda x: sample_index(x[0]) ):
            print(item)
        print("\n\nTraining Samples:\n\n")
        for item in sorted(self._training_data_dict.items(), \
                key = lambda x: sample_index(x[0]) ):
            print(item)

#-----------------------------    Classify with Decision Tree  --------------------------------

    def classify(self, root_node, features_and_values):
        '''
        Classifies one test sample at a time using the decision tree constructed from
        your training file.  The data record for the test sample must be supplied as
        shown in the scripts in the `Examples' subdirectory.  See the scripts
        construct_dt_and_classify_one_sample_caseX.py in that subdirectory.
        '''
        if not self._check_names_used(features_and_values):
            raise ValueError("Error in the names you have used for features and/or values") 
        new_features_and_values = []
        pattern = r'(\S+)\s*=\s*(\S+)'
        for feature_and_value in features_and_values:
            m = re.search(pattern, feature_and_value)
            feature,value = m.group(1),m.group(2)
            value = convert(value)
            newvalue = value
            if ((feature not in self._prob_distribution_numeric_features_dict) and
                (all(isinstance(x,float) for x in self._features_and_unique_values_dict[feature]))):
                newvalue = closest_sampling_point(value, self._features_and_unique_values_dict[feature])
            new_features_and_values.append(feature + " = " + str(newvalue))
        features_and_values = new_features_and_values       
        if self._debug3: print("\nCL1 New feature and values: "+ str(features_and_values))
        answer = {class_name : None for class_name in self._class_names}
        answer['solution_path'] = []
        classification = self.recursive_descent_for_classification(root_node,features_and_values, answer)
        answer['solution_path'].reverse()
        if self._debug3: 
            print("\nCL2 The classification:")
            for class_name in self._class_names:
                print("    " + class_name + " with probability " + str(classification[class_name]))
        classification_for_display = {}
        for item in classification.keys():
            if isinstance(classification[item], float):
                classification_for_display[item] = "%0.3f" % classification[item]
            else:
                classification_for_display[item] =  ["NODE" + str(x) for x in classification[item]]
        return classification_for_display

    def recursive_descent_for_classification(self, node, feature_and_values, answer):
        children = node.get_children()
        if len(children) == 0:
            leaf_node_class_probabilities = node.get_class_probabilities()
            for i in range(len(self._class_names)):
                answer[self._class_names[i]] = leaf_node_class_probabilities[i]
            answer['solution_path'].append(node.get_serial_num())
            return answer
        feature_tested_at_node = node.get_feature()
        if self._debug3: print("\nCLRD1 Feature tested at node for classification: " + feature_tested_at_node)
        value_for_feature = None
        path_found = None
        pattern = r'(\S+)\s*=\s*(\S+)'
        for feature_and_value in feature_and_values:
            m = re.search(pattern, feature_and_value)
            feature,value = m.group(1),m.group(2)
            if feature == feature_tested_at_node:
                value_for_feature = convert(value)
        if feature_tested_at_node in self._prob_distribution_numeric_features_dict:
            if self._debug3: print( "\nCLRD2 In the truly numeric section")
            for child in children:
                branch_features_and_values = child.get_branch_features_and_values_or_thresholds()
                last_feature_and_value_on_branch = branch_features_and_values[-1] 
                pattern1 = r'(.+)<(.+)'
                pattern2 = r'(.+)>(.+)'
                if re.search(pattern1, last_feature_and_value_on_branch):
                    m = re.search(pattern1, last_feature_and_value_on_branch)
                    feature,threshold = m.group(1),m.group(2)
                    if value_for_feature <= float(threshold):
                        path_found = True
                        answer = self.recursive_descent_for_classification(child, feature_and_values, answer)
                        answer['solution_path'].append(node.get_serial_num())
                        break
                if re.search(pattern2, last_feature_and_value_on_branch):
                    m = re.search(pattern2, last_feature_and_value_on_branch)
                    feature,threshold = m.group(1),m.group(2)
                    if value_for_feature > float(threshold):
                        path_found = True
                        answer = self.recursive_descent_for_classification(child, feature_and_values, answer)
                        answer['solution_path'].append(node.get_serial_num())
                        break
            if path_found: return answer 
        else:
            feature_value_combo = "".join([feature_tested_at_node,"=", str(convert(value_for_feature))])
            if self._debug3: 
                print("\nCLRD3 In the symbolic section with feature_value_combo: " + feature_value_combo)
            for child in children:
                branch_features_and_values = child.get_branch_features_and_values_or_thresholds()
                if self._debug3: print("\nCLRD4 branch features and values: "+str(branch_features_and_values))
                last_feature_and_value_on_branch = branch_features_and_values[-1] 
                if last_feature_and_value_on_branch == feature_value_combo:
                    answer = self.recursive_descent_for_classification(child, feature_and_values, answer)
                    answer['solution_path'].append(node.get_serial_num())
                    path_found = True
                    break
            if path_found: return answer
        if not path_found:
            leaf_node_class_probabilities = node.get_class_probabilities()
            for i in range(0, len(self._class_names)):
                answer[self._class_names[i]] = leaf_node_class_probabilities[i]
            answer['solution_path'].append(node.get_serial_num())
        return answer

    def classify_by_asking_questions(self, root_node):
        '''
        If you want classification to be carried out by engaging a human user in a
        question-answer session, this is the method to use for that purpose.  See the
        script classify_by_asking_questions.py in the Examples subdirectory for an
        illustration of how to do that.
        '''
        answer = {class_name : None for class_name in self._class_names}
        answer['solution_path'] = []
        scratchpad_for_numeric_answers = {feature : None \
                                   for feature in self._prob_distribution_numeric_features_dict}
        classification = self.interactive_recursive_descent_for_classification(root_node, \
                                                                   answer, scratchpad_for_numeric_answers)
        classification['solution_path'].reverse()
        classification_for_display = {}
        for item in classification.keys():
            if isinstance(classification[item], float):
                classification_for_display[item] = "%0.3f" % classification[item]
            else:
                classification_for_display[item] =  ["NODE" + str(x) for x in classification[item]]
        return classification_for_display

    def interactive_recursive_descent_for_classification(self, node, answer, scratchpad_for_numerics):
        pattern1 = r'(.+)<(.+)'
        pattern2 = r'(.+)>(.+)'
        user_value_for_feature = None
        children = node.get_children()
        if len(children) == 0:
            leaf_node_class_probabilities = node.get_class_probabilities()
            for i in range(len(self._class_names)):
                answer[self._class_names[i]] = leaf_node_class_probabilities[i]
            answer['solution_path'].append(node.get_serial_num())
            return answer
        list_of_branch_attributes_to_children = []
        for child in children:
            branch_features_and_values = child.get_branch_features_and_values_or_thresholds()
            feature_and_value_on_branch = branch_features_and_values[-1] 
            list_of_branch_attributes_to_children.append(feature_and_value_on_branch)
        feature_tested_at_node = node.get_feature()
        feature_value_combo = None
        path_found = None
        if feature_tested_at_node in self._prob_distribution_numeric_features_dict:
            if scratchpad_for_numerics[feature_tested_at_node]:
                user_value_for_feature = scratchpad_for_numerics[feature_tested_at_node]
            else:
                valuerange =  self._numeric_features_valuerange_dict[feature_tested_at_node]
                while 1: 
                    if sys.version_info[0] == 3:
                        user_value_for_feature = \
                           input( "\nWhat is the value for the feature '" + \
                       feature_tested_at_node + "'?" + "\n" +    \
                       "Enter a value in the range: " + str(valuerange) + " => " )
                    else:
                        user_value_for_feature = \
                           raw_input( "\nWhat is the value for the feature '" + \
                       feature_tested_at_node + "'?" + "\n" +    \
                       "Enter a value in the range: " + str(valuerange) + " => " )
                    user_value_for_feature = convert(user_value_for_feature.strip())
                    answer_found = 0
                    if valuerange[0] <= user_value_for_feature <= valuerange[1]:
                        answer_found = 1
                        break    
                    if answer_found == 1: break
                    print("You entered illegal value. Let's try again")
                scratchpad_for_numerics[feature_tested_at_node] = user_value_for_feature
            for i in range(len(list_of_branch_attributes_to_children)):
                branch_attribute = list_of_branch_attributes_to_children[i]
                if re.search(pattern1, branch_attribute):
                    m = re.search(pattern1, branch_attribute)
                    feature,threshold = m.group(1),m.group(2)
                    if user_value_for_feature <= float(threshold):
                        answer = self.interactive_recursive_descent_for_classification(children[i], \
                                                                          answer, scratchpad_for_numerics)
                        path_found = True
                        answer['solution_path'].append(node.get_serial_num())
                        break
                if re.search(pattern2, branch_attribute):
                    m = re.search(pattern2, branch_attribute)
                    feature,threshold = m.group(1),m.group(2)
                    if user_value_for_feature > float(threshold):
                        answer = self.interactive_recursive_descent_for_classification(children[i], \
                                                                          answer, scratchpad_for_numerics)
                        answer['solution_path'].append(node.get_serial_num())
                        break
            if path_found: return answer
        else:
            possible_values_for_feature = self._features_and_unique_values_dict[feature_tested_at_node]
            while 1:
                if sys.version_info[0] == 3:
                    user_value_for_feature = \
                       input( "\nWhat is the value for the feature '" + feature_tested_at_node + \
                              "'?" + "\n" + "Enter one of: " + str(possible_values_for_feature) + " => " )
                else:
                    user_value_for_feature = \
                       raw_input( "\nWhat is the value for the feature '" + feature_tested_at_node + \
                                  "'?" + "\n" + "Enter one of: " + str(possible_values_for_feature) + " => " )
                user_value_for_feature = convert(user_value_for_feature.strip())
                answer_found = 0
                for value in possible_values_for_feature:
                    if value == user_value_for_feature: 
                        answer_found = 1
                        break
                if answer_found == 1: break
                print("You entered illegal value. Let's try again")
            feature_value_combo = "".join([feature_tested_at_node,"=",str(user_value_for_feature)])
            for i in range(len(list_of_branch_attributes_to_children)):
                branch_attribute = list_of_branch_attributes_to_children[i]
                if branch_attribute == feature_value_combo:
                    answer = self.interactive_recursive_descent_for_classification(children[i], \
                                                                          answer, scratchpad_for_numerics)
                    path_found = True
                    answer['solution_path'].append(node.get_serial_num())
                    break
            if path_found: return answer    
        if not path_found:
            leaf_node_class_probabilities = node.get_class_probabilities()
            for i in range(0, len(self._class_names)):
                answer[self._class_names[i]] = leaf_node_class_probabilities[i]
            answer['solution_path'].append(node.get_serial_num())

        return answer        

##-------------------------------  Construct Decision Tree  ------------------------------------

    def construct_decision_tree_classifier(self):
        '''
        At the root node, we find the best feature that yields the greatest reduction
        in class entropy from the entropy based on just class priors. The logic for
        finding this feature is different for symbolic features and for numeric
        features.  That logic is built into the best feature calculator.
        '''
        if self._debug3:        
            self.determine_data_condition() 
            print("\nStarting construction of the decision tree:\n") 
        class_probabilities = list(map(lambda x: self.prior_probability_for_class(x), \
                                                   self._class_names))
        if self._debug3:         
            print("\nPrior class probabilities: " + str(class_probabilities))
            print("\nClass names: " + str(self._class_names))
        entropy = self.class_entropy_on_priors()
        if self._debug3: print("\nClass entropy on priors: "+ str(entropy))
        root_node = DTNode(None, entropy, class_probabilities, [], 'root')

        root_node.set_class_names(self._class_names)
        self._root_node = root_node
        self.recursive_descent(root_node)
        return root_node        

    def recursive_descent(self, node):
        '''
        After the root node of the decision tree is constructed by the previous
        methods, we invoke this method recursively to create the rest of the tree.
        At each node, we find the feature that achieves the largest entropy reduction
        with regard to the partitioning of the training data samples that correspond
        to that node.
        '''
        if self._debug3:
            print("\n==================== ENTERING RECURSIVE DESCENT ==========================")
        node_serial_number = node.get_serial_num()
        features_and_values_or_thresholds_on_branch = node.get_branch_features_and_values_or_thresholds()
        existing_node_entropy = node.get_node_entropy()
        if self._debug3: 
            print("\nRD1 NODE SERIAL NUMBER: "+ str(node_serial_number))
            print("\nRD2 Existing Node Entropy: " + str(existing_node_entropy))
            print("\nRD3 features_and_values_or_thresholds_on_branch: " + \
                                                        str(features_and_values_or_thresholds_on_branch))
            class_probs = node.get_class_probabilities()
            print("\nRD4 Class probabilities: " + str(class_probs))
        if existing_node_entropy < self._entropy_threshold: 
            if self._debug3: print("\nRD5 returning because existing node entropy is below threshold")
            return
        copy_of_path_attributes = deep_copy_array(features_and_values_or_thresholds_on_branch)
        best_feature,best_feature_entropy,best_feature_val_entropies,decision_val = \
                       self.best_feature_calculator(copy_of_path_attributes, existing_node_entropy)
        node.set_feature(best_feature)
        if self._debug3: node.display_node() 
        if self._max_depth_desired is not None and \
         len(features_and_values_or_thresholds_on_branch) >= self._max_depth_desired:
            if self._debug3: print("\nRD6 REACHED LEAF NODE AT MAXIMUM DEPTH ALLOWED")
            return
        if best_feature is None: return

        if self._debug3:
            print("\nRD7 Existing entropy at node: " + str(existing_node_entropy))
            print("\nRD8 Calculated best feature is %s and its value %s" % (best_feature, decision_val))
            print("\nRD9 Best feature entropy: "+ str(best_feature_entropy))
            print("\nRD10 Calculated entropies for different values of best feature: " + \
                                                                           str(best_feature_val_entropies))
        entropy_gain = existing_node_entropy - best_feature_entropy
        if self._debug3: print("\nRD11 Expected entropy gain at this node: " + str(entropy_gain))
        if entropy_gain > self._entropy_threshold:
            if best_feature in self._numeric_features_valuerange_dict and \
                         self._feature_values_how_many_uniques_dict[best_feature] > \
                                             self._symbolic_to_numeric_cardinality_threshold:
                best_threshold = decision_val                 # as returned by best feature calculator
                best_entropy_for_less, best_entropy_for_greater = best_feature_val_entropies
                extended_branch_features_and_values_or_thresholds_for_lessthan_child = \
                                  deep_copy_array(features_and_values_or_thresholds_on_branch)
                extended_branch_features_and_values_or_thresholds_for_greaterthan_child  = \
                                  deep_copy_array(features_and_values_or_thresholds_on_branch)
                feature_threshold_combo_for_less_than = \
                                            "".join([best_feature,"<",str(convert(best_threshold))])
                feature_threshold_combo_for_greater_than = \
                                            "".join([best_feature,">",str(convert(best_threshold))])
                extended_branch_features_and_values_or_thresholds_for_lessthan_child.append( \
                                                              feature_threshold_combo_for_less_than)
                extended_branch_features_and_values_or_thresholds_for_greaterthan_child.append( \
                                                           feature_threshold_combo_for_greater_than)
                if self._debug3:
                    print("\nRD12 extended_branch_features_and_values_or_thresholds_for_lessthan_child: "+ \
                             str(extended_branch_features_and_values_or_thresholds_for_lessthan_child))
                    print("\nRD13 extended_branch_features_and_values_or_thresholds_for_greaterthan_child: " +\
                             str(extended_branch_features_and_values_or_thresholds_for_greaterthan_child))
                class_probabilities_for_lessthan_child_node = list(map(lambda x: \
                       self.probability_of_a_class_given_sequence_of_features_and_values_or_thresholds(\
                      x, extended_branch_features_and_values_or_thresholds_for_lessthan_child), \
                                                           self._class_names))
                class_probabilities_for_greaterthan_child_node = list(map(lambda x: \
                   self.probability_of_a_class_given_sequence_of_features_and_values_or_thresholds(\
                      x, extended_branch_features_and_values_or_thresholds_for_greaterthan_child), \
                                                           self._class_names))
                if self._debug3:
                    print("\nRD14 class entropy for going down lessthan child: " + str(best_entropy_for_less))
                    print("\nRD15 class_entropy_for_going_down_greaterthan_child: " + 
                                                                        str(best_entropy_for_greater))
                if best_entropy_for_less < existing_node_entropy - self._entropy_threshold:
                    left_child_node = DTNode(None, best_entropy_for_less, \
                                      class_probabilities_for_lessthan_child_node,\
                                      extended_branch_features_and_values_or_thresholds_for_lessthan_child)
                    node.add_child_link(left_child_node)
                    self.recursive_descent(left_child_node)
                if best_entropy_for_greater < existing_node_entropy - self._entropy_threshold:
                    right_child_node = DTNode(None, best_entropy_for_greater,
                                      class_probabilities_for_greaterthan_child_node, \
                                      extended_branch_features_and_values_or_thresholds_for_greaterthan_child)
                    node.add_child_link(right_child_node)
                    self.recursive_descent(right_child_node)
            else:
                if self._debug3:
                    print("\nRD16 RECURSIVE DESCENT: In section for symbolic features for creating children")
                values_for_feature = self._features_and_unique_values_dict[best_feature]
                if self._debug3:
                    print("\nRD17 Values for feature %s are %s" % (best_feature, str(values_for_feature)))
                feature_value_combos = \
                  map(lambda x: "".join([best_feature,"=",x]), map(str, map(convert, values_for_feature)))
                feature_value_combos = sorted(feature_value_combos)
                class_entropies_for_children = []
                for feature_and_value_index in range(len(feature_value_combos)):
                    if self._debug3:
                        print("\nRD18 Creating a child node for: " + \
                                                       str(feature_value_combos[feature_and_value_index])) 
                    extended_branch_features_and_values_or_thresholds = None
                    if features_and_values_or_thresholds_on_branch is None:
                        extended_branch_features_and_values_or_thresholds = \
                                                 [feature_value_combos[feature_and_value_index]]
                    else:
                        extended_branch_features_and_values_or_thresholds = \
                            deep_copy_array(features_and_values_or_thresholds_on_branch)
                        extended_branch_features_and_values_or_thresholds.append(\
                                                 feature_value_combos[feature_and_value_index])
                    class_probabilities = list(map(lambda x: \
                       self.probability_of_a_class_given_sequence_of_features_and_values_or_thresholds(\
                                 x, extended_branch_features_and_values_or_thresholds), self._class_names))
                    class_entropy_for_child = \
                          self.class_entropy_for_a_given_sequence_of_features_and_values_or_thresholds(\
                                                        extended_branch_features_and_values_or_thresholds)
                    if self._debug3: 
                        print("\nRD19 branch attributes: "+ \
                                                 str(extended_branch_features_and_values_or_thresholds))
                        print("\nRD20 class entropy for child: " + str(class_entropy_for_child)) 
                    if existing_node_entropy - class_entropy_for_child > self._entropy_threshold:
                        child_node = DTNode(None, class_entropy_for_child, \
                             class_probabilities, extended_branch_features_and_values_or_thresholds)
                        node.add_child_link( child_node )
                        self.recursive_descent(child_node)
                    elif self._debug3: print("\nRD21 This child will NOT result in a node")
        else:
            if self._debug3:
                print("\nRD22 REACHED LEAF NODE NATURALLY for: " + \
                                      str(features_and_values_or_thresholds_on_branch))
            return   

    def best_feature_calculator(self, features_and_values_or_thresholds_on_branch, existing_node_entropy):
        '''
        This is the heart of the decision tree constructor.  Its main job is to
        figure out the best feature to use for partitioning the training data samples
        that correspond to the current node.  The search for the best feature is
        carried out differently for symbolic features and for numeric features.  For
        a symbolic feature, the method estimates the entropy for each value of the
        feature and then averages out these entropies as a measure of the
        discriminatory power of that features.  For a numeric feature, on the other
        hand, it estimates the entropy reduction that can be achieved if we were to
        partition the set of training samples at each possible threshold for that
        numeric feature.  For a numeric feature, all possible sampling points
        relevant to the node in question are considered as candidates for thresholds.
        '''
        pattern1 = r'(.+)=(.+)'
        pattern2 = r'(.+)<(.+)'
        pattern3 = r'(.+)>(.+)'
        all_symbolic_features = []
        for feature_name in self._feature_names:
            if feature_name not in self._prob_distribution_numeric_features_dict:
                all_symbolic_features.append(feature_name)
        symbolic_features_already_used = []
        for feature_and_value_or_threshold in features_and_values_or_thresholds_on_branch:
            if re.search(pattern1, feature_and_value_or_threshold):
                m = re.search(pattern1, feature_and_value_or_threshold)
                feature = m.group(1)
                symbolic_features_already_used.append(feature)
        symbolic_features_not_yet_used = [x for x in all_symbolic_features \
                                                if x not in symbolic_features_already_used]
        true_numeric_types = []        
        symbolic_types = []
        true_numeric_types_feature_names = []
        symbolic_types_feature_names = []
        for item in features_and_values_or_thresholds_on_branch:
            if re.search(pattern2, item):
                true_numeric_types.append(item)
                m = re.search(pattern2, item)
                feature,value = m.group(1),m.group(2)
                true_numeric_types_feature_names.append(feature)
            elif re.search(pattern3, item): 
                true_numeric_types.append(item)
                m = re.search(pattern3, item)
                feature,value = m.group(1),m.group(2)
                true_numeric_types_feature_names.append(feature)
            else:
                symbolic_types.append(item) 
                m = re.search(pattern1, item)
                feature,value = m.group(1),m.group(2)
                symbolic_types_feature_names.append(feature)
        true_numeric_types_feature_names = list(set(true_numeric_types_feature_names))
        symbolic_types_feature_names = list(set(symbolic_types_feature_names))
        bounded_intervals_numeric_types = self.find_bounded_intervals_for_numeric_features(true_numeric_types)
        # Calculate the upper and the lower bounds to be used when searching for the best
        # threshold for each of the numeric features that are in play at the current node:
        upperbound = {feature : None for feature in true_numeric_types_feature_names}
        lowerbound = {feature : None for feature in true_numeric_types_feature_names}
        for item in bounded_intervals_numeric_types:
            if item[1] == '>':
                lowerbound[item[0]] = float(item[2])
            else:
                upperbound[item[0]] = float(item[2])
        entropy_values_for_different_features = {}
        partitioning_point_child_entropies_dict = {feature : {} for feature in self._feature_names}
        partitioning_point_threshold = {feature : None for feature in self._feature_names}
        entropies_for_different_values_of_symbolic_feature ={feature : [] for feature in self._feature_names}
        for i in range(len(self._feature_names)):
            feature_name = self._feature_names[i]
            if self._debug3: 
                print("\n\nBFC1          FEATURE BEING CONSIDERED: " + feature_name)
            if feature_name in symbolic_features_already_used:
                continue
            elif feature_name in self._numeric_features_valuerange_dict and \
                              self._feature_values_how_many_uniques_dict[feature_name] > \
                                                  self._symbolic_to_numeric_cardinality_threshold:
                values = self._sampling_points_for_numeric_feature_dict[feature_name]
                if self._debug3: print("\nBFC2 values for %s are %s                " % (feature_name, values))
                newvalues = []
                if feature_name in true_numeric_types_feature_names:
                    if upperbound[feature_name] and lowerbound[feature_name] and \
                                                 lowerbound[feature_name] >= upperbound[feature_name]:
                        continue
                    elif upperbound[feature_name] and lowerbound[feature_name] and \
                                                       lowerbound[feature_name] < upperbound[feature_name]:
                        newvalues = [x for x in values \
                                             if lowerbound[feature_name] < x <= upperbound[feature_name]]
                    elif upperbound[feature_name]:
                        newvalues = [x for x in values if x <= upperbound[feature_name]]
                    elif lowerbound[feature_name]:
                        newvalues = [x for x in values if x > lowerbound[feature_name]]
                    else:
                        sys.exit("Error in bound specifications in best feature calculator")
                else:
                    newvalues = values
                if len(newvalues) == 0:
                    continue
                partitioning_entropies = []
                for value in newvalues:
                    feature_and_less_than_value_string = "".join([feature_name,"<",str(convert(value))]) 
                    feature_and_greater_than_value_string = "".join([feature_name,">",str(convert(value))])
                    for_left_child = for_right_child = None
                    if features_and_values_or_thresholds_on_branch:
                        for_left_child = deep_copy_array(features_and_values_or_thresholds_on_branch)
                        for_left_child.append(feature_and_less_than_value_string)
                        for_right_child = deep_copy_array(features_and_values_or_thresholds_on_branch)
                        for_right_child.append(feature_and_greater_than_value_string)
                    else:
                        for_left_child = [feature_and_less_than_value_string]
                        for_right_child = [feature_and_greater_than_value_string]
                    entropy1 = self.class_entropy_for_less_than_threshold_for_feature(\
                                        features_and_values_or_thresholds_on_branch,feature_name,value)
                    entropy2 = self.class_entropy_for_greater_than_threshold_for_feature(\
                                        features_and_values_or_thresholds_on_branch,feature_name, value)
                    partitioning_entropy = entropy1 * \
                       self.probability_of_a_sequence_of_features_and_values_or_thresholds(for_left_child) \
                           + \
                                           entropy2 * \
                       self.probability_of_a_sequence_of_features_and_values_or_thresholds(for_right_child)
                    partitioning_entropies.append(partitioning_entropy)
                    partitioning_point_child_entropies_dict[feature_name][value] = (entropy1, entropy2)
                min_entropy,best_partition_point_index = minimum(partitioning_entropies)         
                if min_entropy < existing_node_entropy:
                    partitioning_point_threshold[feature_name] = newvalues[best_partition_point_index]
                    entropy_values_for_different_features[feature_name] = min_entropy
            else:          
                if self._debug3:
                    print("\nBFC3 Best feature calculator: Entering section reserved for symbolic features")
                    print("\nBFC4 Feature name: " + feature_name)
                values =  self._features_and_unique_values_dict[feature_name]
                values = sorted(list(set(filter(lambda x: x != 'NA', values))))
                if self._debug3: print("\nBFC5 values for feature %s are %s" % (feature_name, values))
                entropy = 0
                for value in values:                              
                    feature_value_string = feature_name + "=" + str(convert(value))
                    if self._debug3: print("\nBFC6 feature_value_string: " + feature_value_string)
                    extended_attributes = deep_copy_array(features_and_values_or_thresholds_on_branch)
                    if features_and_values_or_thresholds_on_branch:
                        extended_attributes.append(feature_value_string)
                    else:
                        extended_attributes = [feature_value_string]
                    entropy += \
         self.class_entropy_for_a_given_sequence_of_features_and_values_or_thresholds(extended_attributes) * \
         self.probability_of_a_sequence_of_features_and_values_or_thresholds(extended_attributes)
                    if self._debug3:
                        print("\nBFC7 Entropy calculated for symbolic feature value choice (%s, %s) is %s" % \
                                                                   (feature_name,value,entropy))       
                    entropies_for_different_values_of_symbolic_feature[feature_name].append(entropy)
                if entropy < existing_node_entropy:
                    entropy_values_for_different_features[feature_name] = entropy
        min_entropy_for_best_feature = None
        best_feature_name = None
        for feature_nom in entropy_values_for_different_features:
            if not best_feature_name:
                best_feature_name = feature_nom
                min_entropy_for_best_feature = entropy_values_for_different_features[feature_nom]
            else:
                if entropy_values_for_different_features[feature_nom] < min_entropy_for_best_feature:
                    best_feature_name = feature_nom                    
                    min_entropy_for_best_feature = entropy_values_for_different_features[feature_nom]
        if  best_feature_name in partitioning_point_threshold:
            threshold_for_best_feature = partitioning_point_threshold[best_feature_name]
        else:
            threshold_for_best_feature = None
        best_feature_entropy = min_entropy_for_best_feature
        val_based_entropies_to_be_returned = None
        decision_val_to_be_returned = None
        if best_feature_name in self._numeric_features_valuerange_dict and \
                              self._feature_values_how_many_uniques_dict[best_feature_name] > \
                                                    self._symbolic_to_numeric_cardinality_threshold:
            val_based_entropies_to_be_returned = \
                 partitioning_point_child_entropies_dict[best_feature_name][threshold_for_best_feature]
        else:
            val_based_entropies_to_be_returned = None
        if  best_feature_name in partitioning_point_threshold:
            decision_val_to_be_returned = partitioning_point_threshold[best_feature_name]
        else:
            decision_val_to_be_returned = None
        if self._debug3:
            print("\nBFC8 Val based entropies to be returned for feature %s are %s" % \
                             (best_feature_name, str(val_based_entropies_to_be_returned)))
        return best_feature_name, best_feature_entropy, val_based_entropies_to_be_returned, \
                                                                         decision_val_to_be_returned

#-----------------------------------  Entropy Calculators  ------------------------------------

    def class_entropy_on_priors(self):
        if 'priors' in self._entropy_cache:
            return self._entropy_cache['priors']
        entropy = None
        for class_name in self._class_names:
            prob = self.prior_probability_for_class(class_name)
            if (prob >= 0.0001) and (prob <= 0.999):
                log_prob = math.log(prob,2)
            if prob < 0.0001:
                log_prob = 0 
            if prob > 0.999:
                log_prob = 0 
            if entropy is None:
                entropy = -1.0 * prob * log_prob
                continue
            entropy += -1.0 * prob * log_prob
        if abs(entropy) < 0.0000001: entropy = 0.0
        self._entropy_cache['priors'] = entropy
        return entropy

    def entropy_scanner_for_a_numeric_feature(self, feature):
        all_sampling_points = self._sampling_points_for_numeric_feature_dict[feature]
        entropies_for_less_than_thresholds = []
        entropies_for_greater_than_thresholds = []
        for point in  all_sampling_points:
            entropies_for_less_than_thresholds.append(\
                    self.class_entropy_for_less_than_threshold_for_feature([], feature, point))
            entropies_for_greater_than_thresholds.append(\
                    self.class_entropy_for_greater_than_threshold_for_feature([], feature, point))
        print("\nSCANNER: All entropies less than thresholds for feature %s are: %s" % \
                                (feature, entropies_for_less_than_thresholds))
        print("\nSCANNER: All entropies greater than thresholds for feature %s are: %s" % \
                                (feature, entropies_for_greater_than_thresholds))


    def class_entropy_for_less_than_threshold_for_feature(self, \
                array_of_features_and_values_or_thresholds, feature, threshold):
        threshold = convert(threshold)
        feature_threshold_combo = feature + '<' + str(threshold)
        sequence = ":".join(array_of_features_and_values_or_thresholds) + ":" + feature_threshold_combo
        if sequence in self._entropy_cache:
            return self._entropy_cache[sequence]
        copy_of_array_of_features_and_values_or_thresholds = \
                        deep_copy_array(array_of_features_and_values_or_thresholds)
        copy_of_array_of_features_and_values_or_thresholds.append(feature_threshold_combo)
        entropy = None
        for class_name in self._class_names:
            prob = self.probability_of_a_class_given_sequence_of_features_and_values_or_thresholds(\
                                        class_name, copy_of_array_of_features_and_values_or_thresholds)
            if (prob >= 0.0001) and (prob <= 0.999):
                log_prob = math.log(prob,2)
            if prob < 0.0001:
                log_prob = 0 
            if prob > 0.999:
                log_prob = 0 
            if entropy is None:
                entropy = -1.0 * prob * log_prob
                continue
            entropy += -1.0 * prob * log_prob
        self._entropy_cache[sequence] = entropy
        if abs(entropy) < 0.0000001: entropy = 0.0
        return entropy

    def class_entropy_for_greater_than_threshold_for_feature(self, \
                    array_of_features_and_values_or_thresholds, feature, threshold):
        threshold = convert(threshold)
        feature_threshold_combo = feature + '>' + str(threshold)
        sequence = ":".join(array_of_features_and_values_or_thresholds) + ":" + feature_threshold_combo
        if sequence in self._entropy_cache:
            return self._entropy_cache[sequence]
        copy_of_array_of_features_and_values_or_thresholds = \
                               deep_copy_array(array_of_features_and_values_or_thresholds)
        copy_of_array_of_features_and_values_or_thresholds.append(feature_threshold_combo)
        entropy = None
        for class_name in self._class_names:
            prob = self.probability_of_a_class_given_sequence_of_features_and_values_or_thresholds(\
                                        class_name, copy_of_array_of_features_and_values_or_thresholds)
            if (prob >= 0.0001) and (prob <= 0.999):
                log_prob = math.log(prob,2)
            if prob < 0.0001:
                log_prob = 0 
            if prob > 0.999:
                log_prob = 0 
            if entropy is None:
                entropy = -1.0 * prob * log_prob
                continue
            entropy += -1.0 * prob * log_prob
        if abs(entropy) < 0.0000001: entropy = 0.0
        self._entropy_cache[sequence] = entropy
        return entropy

    def class_entropy_for_a_given_sequence_of_features_and_values_or_thresholds(self, \
                                             array_of_features_and_values_or_thresholds):
        sequence = ":".join(array_of_features_and_values_or_thresholds)
        if sequence in self._entropy_cache:
            return self._entropy_cache[sequence]
        entropy = None    
        for class_name in self._class_names:
            prob = self.probability_of_a_class_given_sequence_of_features_and_values_or_thresholds(\
                                                class_name, array_of_features_and_values_or_thresholds)
            if (prob >= 0.0001) and (prob <= 0.999):
                log_prob = math.log(prob,2)
            if prob < 0.0001:
                log_prob = 0 
            if prob > 0.999:
                log_prob = 0 
            if entropy is None:
                entropy = -1.0 * prob * log_prob
                continue
            entropy += -1.0 * prob * log_prob
        if abs(entropy) < 0.0000001: entropy = 0.0
        self._entropy_cache[sequence] = entropy
        return entropy


#---------------------------------  Probability Calculators  ----------------------------------

    def prior_probability_for_class(self, class_name):
        class_name_in_cache = "".join(["prior::", class_name])
        if class_name_in_cache in self._probability_cache:
            return self._probability_cache[class_name_in_cache]
        total_num_of_samples = len( self._samples_class_label_dict )
        all_values = self._samples_class_label_dict.values()
        for this_class_name in self._class_names:
            trues = list(filter( lambda x: x == this_class_name, all_values ))
            prior_for_this_class = (1.0 * len(trues)) / total_num_of_samples
            self._class_priors_dict[this_class_name] = prior_for_this_class
            this_class_name_in_cache = "".join(["prior::", this_class_name])
            self._probability_cache[this_class_name_in_cache] = prior_for_this_class
        return self._probability_cache[class_name_in_cache]

    def calculate_class_priors(self):
        if len(self._class_priors_dict) > 1: return
        for class_name in self._class_names:
            class_name_in_cache = "".join(["prior::", class_name])
            total_num_of_samples = len( self._samples_class_label_dict )
            all_values = self._samples_class_label_dict.values()
            trues = list(filter( lambda x: x == class_name, all_values ))
            prior_for_this_class = (1.0 * len(trues)) / total_num_of_samples
            self._class_priors_dict[class_name] = prior_for_this_class
            this_class_name_in_cache = "".join(["prior::", class_name])
            self._probability_cache[this_class_name_in_cache] = prior_for_this_class
        if self._debug2:
            print(str(self._class_priors_dict))

    def probability_of_feature_value(self, feature_name, value):
        value = convert(value)
        if value and (feature_name in self._sampling_points_for_numeric_feature_dict):
            value = closest_sampling_point(convert(value), \
                       self._sampling_points_for_numeric_feature_dict[feature_name])
        if value:
            feature_and_value = "".join([feature_name, "=", str(convert(value))])
        if value and (feature_and_value in self._probability_cache):
            return self._probability_cache[feature_and_value]
        histogram_delta = num_of_hist_bins = valuerange = diffrange = None
        if feature_name in self._numeric_features_valuerange_dict:
            if self._feature_values_how_many_uniques_dict[feature_name] > \
                                      self._symbolic_to_numeric_cardinality_threshold:
                if feature_name not in self._sampling_points_for_numeric_feature_dict:
                    valuerange = self._numeric_features_valuerange_dict[feature_name] 
                    diffrange = valuerange[1] - valuerange[0]
                    unique_values_for_feature = \
                      sorted(list(set(filter(lambda x: x != 'NA', \
                                   self._features_and_values_dict[feature_name]))))
                    diffs = sorted([unique_values_for_feature[i] - unique_values_for_feature[i-1] \
                                            for i in range(1,len(unique_values_for_feature))])
                    median_diff = diffs[int(len(diffs)/2) - 1]
                    histogram_delta =  median_diff * 2
                    self._histogram_delta_dict[feature_name] = histogram_delta
                    num_of_histogram_bins = int(diffrange / histogram_delta) + 1
                    self._num_of_histogram_bins_dict[feature_name] = num_of_histogram_bins
                    sampling_points_for_feature = \
                                [valuerange[0] + histogram_delta * j for j in range(num_of_histogram_bins)]
                    self._sampling_points_for_numeric_feature_dict[feature_name] = \
                                                             sampling_points_for_feature
        if feature_name in self._numeric_features_valuerange_dict:
            if self._feature_values_how_many_uniques_dict[feature_name] > \
                                       self._symbolic_to_numeric_cardinality_threshold:
                sampling_points_for_feature = \
                         self._sampling_points_for_numeric_feature_dict[feature_name]
                counts_at_sampling_points = [0] * len(sampling_points_for_feature)
                actual_values_for_feature = self._features_and_values_dict[feature_name]
                actual_values_for_feature = \
                          list(filter(lambda x: x != 'NA', actual_values_for_feature))
                for i in range(len(sampling_points_for_feature)):
                    for j in range(len(actual_values_for_feature)):
                        if abs(sampling_points_for_feature[i] - \
                           convert(actual_values_for_feature[j])) < (histogram_delta):
                            counts_at_sampling_points[i] += 1
                total_counts =  functools.reduce(lambda x,y:x+y, counts_at_sampling_points)
                probs = [x / (1.0 * total_counts) for x in counts_at_sampling_points]
                sum_probs =  functools.reduce(lambda x,y:x+y, probs)
                bin_prob_dict = {sampling_points_for_feature[i] : probs[i] \
                               for i in range(len(sampling_points_for_feature))}
                self._prob_distribution_numeric_features_dict[feature_name] = bin_prob_dict
                values_for_feature = list(map(lambda x: feature_name + "=" + x, \
                                                      map(str, sampling_points_for_feature)))
                for i in range(0, len(values_for_feature)):
                    self._probability_cache[values_for_feature[i]] = probs[i]
                if value and feature_and_value in self._probability_cache:
                    return self._probability_cache[feature_and_value]
                else:
                    return 0
            else:
                # This section is for those numeric features that are treated symbolically
                values_for_feature = list(set(self._features_and_values_dict[feature_name]))
                values_for_feature = list(filter(lambda x: x != 'NA', values_for_feature))
                values_for_feature = \
                       list(map(lambda x: "".join([feature_name,"=",x]), \
                                                     map(str,values_for_feature)))
                value_counts = [0] * len(values_for_feature)
                for sample in sorted(self._training_data_dict.keys(), \
                                           key = lambda x: sample_index(x) ):
                    features_and_values = self._training_data_dict[sample]
                    for i in range(0, len(values_for_feature)):
                        for current_value in (features_and_values):
                            if values_for_feature[i] == current_value:
                                value_counts[i] += 1 
                total_counts = functools.reduce(lambda x,y:x+y, value_counts)
                if total_counts == 0:
                    sys.exit('''PFV Something is wrong with your training file. '''
                             '''It contains no training samples for feature named %s ''' % feature_name)
                probs = [x / (1.0 * total_counts) for x in value_counts]
                for i in range(0, len(values_for_feature)):
                    self._probability_cache[values_for_feature[i]] = probs[i]
                if value and feature_and_value in self._probability_cache:
                    return self._probability_cache[feature_and_value]
                else:
                    return 0
        else:
            # This section is only for purely symbolic features:  
            values_for_feature = self._features_and_values_dict[feature_name]
            values_for_feature = list(map(lambda x: feature_name + "=" + x, values_for_feature))
            value_counts = [0] * len(values_for_feature)
            for sample in sorted(self._training_data_dict.keys(), \
                    key = lambda x: sample_index(x) ):
                features_and_values = self._training_data_dict[sample]
                for i in range(0, len(values_for_feature)):
                    for current_value in features_and_values:
                        if values_for_feature[i] == current_value:
                            value_counts[i] += 1 
            for i in range(0, len(values_for_feature)):
                self._probability_cache[values_for_feature[i]] = \
                          value_counts[i] / (1.0 * len(self._training_data_dict))
            if value and feature_and_value in self._probability_cache:
                return self._probability_cache[feature_and_value]
            else:
                return 0

    def probability_of_feature_value_given_class(self,feature_name,feature_value,class_name):
        feature_value = convert(feature_value)
        histogram_delta = num_of_histogram_bins = valuerange = diffrange = None
        if feature_name in self._sampling_points_for_numeric_feature_dict:
            feature_value = closest_sampling_point(convert(feature_value), \
                       self._sampling_points_for_numeric_feature_dict[feature_name])
        feature_value_class = "".join([feature_name,"=",str(feature_value),"::",class_name])
        if feature_value_class in self._probability_cache:
            return self._probability_cache[feature_value_class]
        if feature_name in self._numeric_features_valuerange_dict:
            if self._feature_values_how_many_uniques_dict[feature_name] > \
                                       self._symbolic_to_numeric_cardinality_threshold:
                histogram_delta = self._histogram_delta_dict[feature_name]
                num_of_histogram_bins = self._num_of_histogram_bins_dict[feature_name]
                valuerange = self._numeric_features_valuerange_dict[feature_name]
                diffrange = valuerange[1] - valuerange[0]
        samples_for_class = []
        # Accumulate all samples names for the given class:
        for sample_name in self._samples_class_label_dict.keys():
            if self._samples_class_label_dict[sample_name] == class_name:
                samples_for_class.append(sample_name) 
        if feature_name in self._numeric_features_valuerange_dict:
            if self._feature_values_how_many_uniques_dict[feature_name] > \
                                        self._symbolic_to_numeric_cardinality_threshold:
                sampling_points_for_feature = self._sampling_points_for_numeric_feature_dict[feature_name]
                counts_at_sampling_points = [0] * len(sampling_points_for_feature)
                actual_feature_values_for_samples_in_class = []
                for sample in samples_for_class:
                    for feature_and_value in self._training_data_dict[sample]:
                        pattern = r'(.+)=(.+)'
                        m = re.search(pattern, feature_and_value)
                        feature,value = m.group(1),m.group(2)
                        if feature == feature_name and value != 'NA':
                            actual_feature_values_for_samples_in_class.append(convert(value))
                for i in range(len(sampling_points_for_feature)):
                    for j in range(len(actual_feature_values_for_samples_in_class)):
                        if abs(sampling_points_for_feature[i] - \
                     actual_feature_values_for_samples_in_class[j]) < histogram_delta:
                            counts_at_sampling_points[i] += 1
                total_counts =  functools.reduce(lambda x,y:x+y, counts_at_sampling_points)
                probs = [x / (1.0 * total_counts) for x in counts_at_sampling_points]
                if total_counts == 0:
                    sys.exit('''PFVC1 Something is wrong with your training file. '''
                             '''It contains no training samples for Class %s and '''
                             '''Feature %s''' % class_name, feature_name)
                values_for_feature_and_class = list(map(lambda x: feature_name + "=" + x + \
                                "::" + class_name, map(str, sampling_points_for_feature)))
                for i in range(0, len(values_for_feature_and_class)):
                    self._probability_cache[values_for_feature_and_class[i]] = probs[i]
                if feature_value_class in self._probability_cache:
                    return self._probability_cache[feature_value_class]
                else:
                    return 0
            else:
                # We now take care of numeric features with a small number of unique values
                values_for_feature = list(set(self._features_and_values_dict[feature_name]))
                values_for_feature = list(filter(lambda x: x != 'NA', values_for_feature))
                values_for_feature = \
                 list(map(lambda x: "".join([feature_name,"=",x]), map(str,map(convert, values_for_feature))))
                value_counts = [0] * len(values_for_feature)
                for sample in samples_for_class:
                    features_and_values = self._training_data_dict[sample]
                    for i in range(0, len(values_for_feature)):
                        for current_value in (features_and_values):
                            if values_for_feature[i] == current_value:
                                value_counts[i] += 1 
                total_count = functools.reduce(lambda x,y:x+y, value_counts)
                if total_count == 0:
                    sys.exit('''PFVC2 Something is wrong with your training file. '''
                             '''It contains no training samples for Class %s and '''
                             '''Feature %s''' % (class_name, feature_name))
                # We normalize by total_count because the probabilities are
                # conditioned on a given class
                for i in range(len(values_for_feature)):
                    feature_and_value_and_class =  values_for_feature[i] + "::" + class_name
                    self._probability_cache[feature_and_value_and_class] = \
                                               value_counts[i] / (1.0 * total_count)
                if feature_value_class in self._probability_cache:
                    return self._probability_cache[feature_value_class]
                else:
                    return 0
        else:
            # This section is for purely symbolic features
            values_for_feature = list(set(self._features_and_values_dict[feature_name]))
            values_for_feature = \
                        list(map(lambda x: "".join([feature_name,"=",x]), map(str,values_for_feature)))
            value_counts = [0] * len(values_for_feature)
            for sample in samples_for_class:
                features_and_values = self._training_data_dict[sample]
                for i in range(len(values_for_feature)):
                    for current_value in (features_and_values):
                        if values_for_feature[i] == current_value:
                            value_counts[i] += 1 
            total_count = functools.reduce(lambda x,y:x+y, value_counts)
            if total_count == 0:
                sys.exit('''PFVC3 Something is wrong with your training file. '''
                         '''It contains no training samples for Class %s and '''
                         '''Feature %s''' % (class_name, feature_name))
            # We normalize by total_count because the probabilities are
            # conditioned on a given class
            for i in range(0, len(values_for_feature)):
                feature_and_value_for_class = "".join([values_for_feature[i],"::",class_name])
                self._probability_cache[feature_and_value_for_class] = \
                                           value_counts[i] / (1.0 * total_count)
            feature_and_value_and_class =\
                          "".join([feature_name,"=", feature_value,"::",class_name])
            if feature_and_value_and_class in self._probability_cache:
                return self._probability_cache[feature_and_value_and_class]
            else:
                return 0

    def probability_of_feature_less_than_threshold(self, feature_name, threshold):
        threshold = convert(threshold)
        feature_threshold_combo = feature_name + '<' + str(threshold)
        if feature_threshold_combo in self._probability_cache:
            return self._probability_cache[feature_threshold_combo]
        all_values = list(filter(lambda x: x != 'NA', self._features_and_values_dict[feature_name]))
        all_values_less_than_threshold = list(filter(lambda x: x <= threshold, all_values))
        probability = 1.0 * len(all_values_less_than_threshold) / len(all_values)
        self._probability_cache[feature_threshold_combo] = probability
        return probability

    def probability_of_feature_less_than_threshold_given_class(self, feature_name, threshold, class_name):
        threshold = convert(threshold)
        feature_threshold_class_combo = "".join([feature_name,'<',str(threshold),"::",class_name])
        if feature_threshold_class_combo in self._probability_cache:
            return self._probability_cache[feature_threshold_class_combo]
        data_samples_for_class = []
        # Accumulate all samples names for the given class:
        for sample_name in self._samples_class_label_dict.keys():
            if self._samples_class_label_dict[sample_name] == class_name:
                data_samples_for_class.append(sample_name) 
        actual_feature_values_for_samples_in_class = []
        for sample in data_samples_for_class:
            for feature_and_value in self._training_data_dict[sample]:
                pattern = r'(.+)=(.+)'
                m = re.search(pattern, feature_and_value)
                feature,value = m.group(1),m.group(2)
                if feature == feature_name and value != 'NA':
                    actual_feature_values_for_samples_in_class.append(convert(value))
        actual_points_for_feature_less_than_threshold = list(filter(lambda x: x <= threshold, \
                                          actual_feature_values_for_samples_in_class))
        probability = 1.0 * len(actual_points_for_feature_less_than_threshold) /    \
                                           len(actual_feature_values_for_samples_in_class)
        self._probability_cache[feature_threshold_class_combo] = probability
        return probability


    def probability_of_a_sequence_of_features_and_values_or_thresholds(self, \
                                        array_of_features_and_values_or_thresholds):
        '''
        This method requires that all truly numeric types only be expressed as '<' or '>'
        constructs in the array of branch features and thresholds
        '''
        if len(array_of_features_and_values_or_thresholds) == 0: return
        sequence = ":".join(array_of_features_and_values_or_thresholds)
        if sequence in self._probability_cache:
            return self._probability_cache[sequence]
        probability = None
        pattern1 = r'(.+)=(.+)'
        pattern2 = r'(.+)<(.+)'
        pattern3 = r'(.+)>(.+)'
        true_numeric_types = []        
        true_numeric_types_feature_names = []
        symbolic_types = []
        symbolic_types_feature_names = []
        for item in array_of_features_and_values_or_thresholds:
            if re.search(pattern2, item):
                true_numeric_types.append(item)
                m = re.search(pattern2, item)
                feature,value = m.group(1),m.group(2)
                true_numeric_types_feature_names.append(feature)
            elif re.search(pattern3, item): 
                true_numeric_types.append(item)
                m = re.search(pattern3, item)
                feature,value = m.group(1),m.group(2)
                true_numeric_types_feature_names.append(feature)
            else:
                symbolic_types.append(item) 
                m = re.search(pattern1, item)
                feature,value = m.group(1),m.group(2)
                symbolic_types_feature_names.append(feature)
        true_numeric_types_feature_names = list(set(true_numeric_types_feature_names))
        symbolic_types_feature_names = list(set(symbolic_types_feature_names))
        bounded_intervals_numeric_types = self.find_bounded_intervals_for_numeric_features(true_numeric_types)
        # Calculate the upper and the lower bounds to be used when searching for the best
        # threshold for each of the numeric features that are in play at the current node:
        upperbound = {feature : None for feature in true_numeric_types_feature_names}
        lowerbound = {feature : None for feature in true_numeric_types_feature_names}
        for item in bounded_intervals_numeric_types:
            if item[1] == '>':
                lowerbound[item[0]] = float(item[2])
            else:
                upperbound[item[0]] = float(item[2])
        for feature_name in true_numeric_types_feature_names:
            if lowerbound[feature_name] and upperbound[feature_name] \
                          and upperbound[feature_name] <= lowerbound[feature_name]:
                return 0
            elif lowerbound[feature_name] and upperbound[feature_name]:
                if not probability:
                    probability = self.probability_of_feature_less_than_threshold(feature_name, \
                                  upperbound[feature_name]) - \
                          self.probability_of_feature_less_than_threshold(feature_name, \
                                                                      lowerbound[feature_name])
                else:
                    probability *= (self.probability_of_feature_less_than_threshold(feature_name, \
                                  upperbound[feature_name]) - \
                          self.probability_of_feature_less_than_threshold(feature_name, \
                                                                      lowerbound[feature_name]))
            elif upperbound[feature_name] and not lowerbound[feature_name]:
                if not probability:
                    probability = self.probability_of_feature_less_than_threshold(feature_name, \
                                                                     upperbound[feature_name])
                else:
                    probability *= self.probability_of_feature_less_than_threshold(feature_name, \
                                                                     upperbound[feature_name])
            elif lowerbound[feature_name] and not upperbound[feature_name]:
                if not probability:
                    probability = 1.0 -self.probability_of_feature_less_than_threshold(feature_name, \
                                                                     lowerbound[feature_name])
                else:
                    probability *= (1.0 - self.probability_of_feature_less_than_threshold(feature_name, \
                                                                     lowerbound[feature_name]))
            else:
                sys.exit("Ill formatted call to 'probability_of_sequence' method")
        for feature_and_value in symbolic_types:
            if re.search(pattern1, feature_and_value):      
                m = re.search(pattern1, feature_and_value)
                feature,value = m.group(1),m.group(2)
                if not probability:
                    probability = self.probability_of_feature_value(feature, value)
                else:
                    probability *= self.probability_of_feature_value(feature, value)
        self._probability_cache[sequence] = probability
        return probability

    def probability_of_a_sequence_of_features_and_values_or_thresholds_given_class(self, 
                                              array_of_features_and_values_or_thresholds, class_name):
        '''
        This method requires that all truly numeric types only be expressed as '<' or '>'
        constructs in the array of branch features and thresholds
        '''
        if len(array_of_features_and_values_or_thresholds) == 0: return
        sequence = ":".join(array_of_features_and_values_or_thresholds)
        sequence_with_class = sequence + "::" + class_name
        if sequence_with_class in self._probability_cache:
            return self._probability_cache[sequence_with_class]
        probability = None
        pattern1 = r'(.+)=(.+)'
        pattern2 = r'(.+)<(.+)'
        pattern3 = r'(.+)>(.+)'
        true_numeric_types = []        
        true_numeric_types_feature_names = []        
        symbolic_types = []
        symbolic_types_feature_names = []
        for item in array_of_features_and_values_or_thresholds:
            if re.search(pattern2, item):
                true_numeric_types.append(item)
                m = re.search(pattern2, item)
                feature,value = m.group(1),m.group(2)
                true_numeric_types_feature_names.append(feature)
            elif re.search(pattern3, item): 
                true_numeric_types.append(item)
                m = re.search(pattern3, item)
                feature,value = m.group(1),m.group(2)
                true_numeric_types_feature_names.append(feature)
            else:
                symbolic_types.append(item) 
                m = re.search(pattern1, item)
                feature,value = m.group(1),m.group(2)
                symbolic_types_feature_names.append(feature)
        true_numeric_types_feature_names = list(set(true_numeric_types_feature_names))
        symbolic_types_feature_names = list(set(symbolic_types_feature_names))
        bounded_intervals_numeric_types = self.find_bounded_intervals_for_numeric_features(true_numeric_types)
        # Calculate the upper and the lower bounds to be used when searching for the best
        # threshold for each of the numeric features that are in play at the current node:
        upperbound = {feature : None for feature in true_numeric_types_feature_names}
        lowerbound = {feature : None for feature in true_numeric_types_feature_names}
        for item in bounded_intervals_numeric_types:
            if item[1] == '>':
                lowerbound[item[0]] = float(item[2])
            else:
                upperbound[item[0]] = float(item[2])
        for feature_name in true_numeric_types_feature_names:
            if lowerbound[feature_name] and upperbound[feature_name] \
                          and upperbound[feature_name] <= lowerbound[feature_name]:
                return 0
            elif lowerbound[feature_name] and upperbound[feature_name]:
                if not probability:
                    probability = self.probability_of_feature_less_than_threshold_given_class(feature_name, \
                                                upperbound[feature_name], class_name) - \
                          self.probability_of_feature_less_than_threshold_given_class(feature_name, \
                                                lowerbound[feature_name], class_name)
                else:
                    probability *= (self.probability_of_feature_less_than_threshold_given_class(feature_name, \
                                  upperbound[feature_name], class_name) - \
                          self.probability_of_feature_less_than_threshold_given_class(feature_name, \
                                                        lowerbound[feature_name], class_name))
            elif upperbound[feature_name] and not lowerbound[feature_name]:
                if not probability:
                    probability = self.probability_of_feature_less_than_threshold_given_class(feature_name, \
                                                         upperbound[feature_name], class_name)
                else:
                    probability *= self.probability_of_feature_less_than_threshold_given_class(feature_name, \
                                                         upperbound[feature_name], class_name)
            elif lowerbound[feature_name] and not upperbound[feature_name]:
                if not probability:
                    probability = 1.0 - \
                          self.probability_of_feature_less_than_threshold_given_class(feature_name, \
                                                         lowerbound[feature_name], class_name)
                else:
                    probability *= (1.0 - \
                          self.probability_of_feature_less_than_threshold_given_class(feature_name, \
                                                         lowerbound[feature_name], class_name))
            else:
                sys.exit("Ill formatted call to 'probability of sequence with class' method")
        for feature_and_value in symbolic_types:
            if re.search(pattern1, feature_and_value):      
                m = re.search(pattern1, feature_and_value)
                feature,value = m.group(1),m.group(2)
                if not probability:
                    probability = self.probability_of_feature_value_given_class(feature, value, class_name)
                else:
                    probability *= self.probability_of_feature_value_given_class(feature, value, class_name)
        self._probability_cache[sequence_with_class] = probability
        return probability


    def probability_of_a_class_given_sequence_of_features_and_values_or_thresholds(self, \
                                  class_name, array_of_features_and_values_or_thresholds):
        sequence = ":".join(array_of_features_and_values_or_thresholds)
        class_and_sequence = "".join([class_name, "::", sequence])
        if class_and_sequence in self._probability_cache:
            return self._probability_cache[class_and_sequence]
        array_of_class_probabilities = [0] * len(self._class_names)
        for i in range(len(self._class_names)):
            class_name = self._class_names[i]
            prob = self.probability_of_a_sequence_of_features_and_values_or_thresholds_given_class(\
                                     array_of_features_and_values_or_thresholds, class_name) 
            if prob < 0.000001:
                array_of_class_probabilities[i] = 0.0
                continue
            prob_of_feature_sequence = self.probability_of_a_sequence_of_features_and_values_or_thresholds(\
                                                      array_of_features_and_values_or_thresholds)
            if not prob_of_feature_sequence: 
                sys.exit('''PCS Something is wrong with your sequence of feature values and thresholds in '''
                         '''probability_of_a_class_given_sequence_of_features_and_values_or_thresholds()''')
            prior = self._class_priors_dict[self._class_names[i]] 
            array_of_class_probabilities[i] = prob * prior / prob_of_feature_sequence
        sum_probability = functools.reduce(lambda x,y:x+y, array_of_class_probabilities)
        if sum_probability == 0:
            array_of_class_probabilities = [1.0/len(self._class_names)] * len(self._class_names)
        else:
            array_of_class_probabilities = \
                             list(map(lambda x: x / sum_probability, array_of_class_probabilities))
        for i in range(len(self._class_names)):
            this_class_and_sequence = "".join([self._class_names[i], "::", sequence])
            self._probability_cache[this_class_and_sequence] = array_of_class_probabilities[i]
        return self._probability_cache[class_and_sequence]

    
#---------------------------------  Class Based Utilities  ------------------------------------

    def determine_data_condition(self):
        '''
        This method estimates the worst-case fan-out of the decision tree taking into
        account the number of values (and therefore the number of branches emanating
        from a node) for the symbolic features.
        '''
        num_of_features = len(self._feature_names)
        values = []
        for feature in self._features_and_unique_values_dict:  
            if feature not in self._numeric_features_valuerange_dict:
                values.append(self._features_and_unique_values_dict[feature])
        print("Number of features: " + str(num_of_features))
        max_num_values = max(list(map(len, values)))
        print("Largest number of values for symbolic features is: " + str(max_num_values))
        estimated_number_of_nodes = max_num_values ** num_of_features
        print('''\nWORST CASE SCENARIO: The decision tree COULD have as many as %s '''
              ''' nodes. The exact number of nodes created depends critically on '''
              '''the entropy_threshold used for node expansion (the default value '''
              '''for this threshold is 0.01) and on the value set for max_depth_desired '''
              '''for the depth of the tree\n''' % estimated_number_of_nodes)
        if estimated_number_of_nodes > 10000:
            print('''THIS IS WAY TOO MANY NODES. Consider using a relatively '''
                  '''large value for entropy_threshold and/or a small value for '''
                  '''for max_depth_desired to reduce the number of nodes created''')
            ans = None
            if sys.version_info[0] == 3:
                ans = input("\nDo you wish to continue? Enter 'y' if yes:  ")
            else:
                ans = raw_input("\nDo you wish to continue? Enter 'y' if yes:  ")
            ans = ans.strip()
            if ans != 'y':
                sys.exit(0)
 
    def _check_names_used(self, features_and_values_test_data):
        '''
        This method is used to verify that you used legal feature names in the test
        sample that you want to classify with the decision tree.
        '''
        for feature_and_value in features_and_values_test_data:
            pattern = r'(\S+)\s*=\s*(\S+)'
            m = re.search(pattern, feature_and_value)
            feature,value = m.group(1),m.group(2)
            if feature is None or value is None:
                raise ValueError("Your test data has formatting error")
            if feature not in self._feature_names:
                return 0
        return 1

    def get_class_names(self):
        return self._class_names

    def find_bounded_intervals_for_numeric_features(self, arr):
        '''
        Given a list of branch attributes for the numeric features of the form, say,
        ['g2<1','g2<2','g2<3','age>34','age>36','age>37'], this method returns the
        smallest list that is relevant for the purpose of calculating the
        probabilities.  To explain, the probability that the feature `g2' is less
        than 1 AND, at the same time, less than 2, AND, at the same time, less than
        3, is the same as the probability that the feature less than 1. Similarly,
        the probability that 'age' is greater than 34 and also greater than 37 is the
        same as `age' being greater than 37.
        '''       
        features = self._feature_names
        arr1 = list(map(lambda x: re.split(r'(>|<)', x, 0), arr))
        # make a separate list for each feature name:                                    
        arr3 = list(filter(lambda x: len(x)>0, [list(filter(lambda x: x[0]==y, arr1)) for y in features]))
        # Sort each list so that '<' entries occur before '>' entries:                   
        arr4 = [sorted(li, key=lambda x: x[1]) for li in arr3]
        arr5 = [[list(filter(lambda x: x[1]==y, alist))] for alist in arr4 for y in ['<','>']]
        arr6 = []
        for i in range(len(arr5)):
            arr6 += [sorted(li, key=lambda x: float(x[2])) for li in arr5[i]]
        arr7 = list(itertools.chain(*arr6))
        arr8 = list(filter(lambda x: len(x)>0, [list(filter(lambda x: x[0]==y, arr7)) for y in features]))
        arr9 = []
        for alist in arr8:
            newalist = []
            if alist[0][1] == '<':
                newalist.append(alist[0])
            else:
                newalist.append(alist[-1])
            if alist[0][1] != alist[-1][1]:
                newalist.append(alist[-1])
            arr9.append(newalist)
        arr10 = list(itertools.chain(*arr9))
        return arr10


#---------------------------  Evaluate Quality of Training Data  ------------------------------------

class EvalTrainingData(DecisionTree):
    def __init__(self, *args, **kwargs ):
        DecisionTree.__init__(self, *args, **kwargs)

    def evaluate_training_data(self):
        evaldebug = 0
        if not self._training_datafile.endswith('.csv'):
            sys.exit('''The data evaluation function in the module can only be used when your '''
                     '''training data is in a CSV file''')
        print('''\nWill run a 10-fold cross-validation test on your training data to test its '''
              '''class-discriminatory power:''')
        all_training_data = self._training_data_dict
        all_sample_names = sorted(all_training_data.keys(), key = lambda x: sample_index(x))
        fold_size = int(0.1 * len(all_training_data))
        confusion_matrix = {class_name : {class_name : 0 for class_name in self._class_names} \
                                                                  for class_name in self._class_names}
        for fold_index in range(10):
            print("\nStarting the iteration indexed %d of the 10-fold cross-validation test" % fold_index)
            testing_samples = all_sample_names[fold_size * fold_index : fold_size * (fold_index+1)]
            training_samples = all_sample_names[0 : fold_size * fold_index] + \
                                                      all_sample_names[fold_size * (fold_index+1):] 
            testing_data = { x : all_training_data[x] for x in testing_samples }
            training_data = { x : all_training_data[x] for x in training_samples }
            trainingDT = DecisionTree('evalmode')
            trainingDT._training_data_dict = training_data
            trainingDT._class_names = self._class_names
            trainingDT._feature_names = self._feature_names
            trainingDT._entropy_threshold = self._entropy_threshold
            trainingDT._max_depth_desired = self._max_depth_desired
            trainingDT._symbolic_to_numeric_cardinality_threshold =   \
                                                     self._symbolic_to_numeric_cardinality_threshold
            trainingDT._samples_class_label_dict = {sample_name : self._samples_class_label_dict[sample_name] \
                                                           for sample_name in training_samples}
            trainingDT._features_and_values_dict = {feature : [] for feature in self._features_and_values_dict}
            pattern = r'(\S+)\s*=\s*(\S+)'        
            for item in sorted(trainingDT._training_data_dict.items(), key = lambda x: sample_index(x[0])):
                for feature_and_value in item[1]:
                    m = re.search(pattern, feature_and_value)
                    feature,value = m.group(1),m.group(2)
                    if value != 'NA':
                        trainingDT._features_and_values_dict[feature].append(convert(value))
            trainingDT._features_and_unique_values_dict = {feature : \
                                      sorted(list(set(trainingDT._features_and_values_dict[feature]))) for \
                                                            feature in trainingDT._features_and_values_dict}
            trainingDT._numeric_features_valuerange_dict = {feature : [] \
                                                         for feature in self._numeric_features_valuerange_dict}
            trainingDT._numeric_features_valuerange_dict = {feature : 
                                      [min(trainingDT._features_and_unique_values_dict[feature]), \
                                       max(trainingDT._features_and_unique_values_dict[feature])] \
                                                         for feature in self._numeric_features_valuerange_dict}
            if evaldebug:
                print("\n\nprinting samples in the testing set: " + str(testing_samples))            
                print("\n\nPrining features and their values in the training set:\n")
                for item in sorted(trainingDT._features_and_values_dict.items()):
                    print(item[0]  + "  =>  "  + str(item[1]))
                print("\n\nPrinting unique values for features:\n")
                for item in sorted(trainingDT._features_and_unique_values_dict.items()):
                    print(item[0]  + "  =>  "  + str(item[1]))
                print("\n\nPrinting unique value ranges for features:\n")
                for item in sorted(trainingDT._numeric_features_valuerange_dict.items()):
                    print(item[0]  + "  =>  "  + str(item[1]))
            trainingDT._feature_values_how_many_uniques_dict = {feature : [] 
                                                    for  feature in self._features_and_unique_values_dict}
            trainingDT._feature_values_how_many_uniques_dict = {feature : \
                                     len(trainingDT._features_and_unique_values_dict[feature]) \
                                                    for  feature in self._features_and_unique_values_dict}
            if evaldebug: trainingDT._debug2 = 1
            trainingDT.calculate_first_order_probabilities()
            trainingDT.calculate_class_priors()
            root_node = trainingDT.construct_decision_tree_classifier()
            if evaldebug:
                root_node.display_decision_tree("     ")
            for test_sample_name in testing_samples:
                test_sample_data = all_training_data[test_sample_name]
                if evaldebug: 
                    print("original data in test sample:", str(test_sample_data))  
                test_sample_data = [x for x in test_sample_data if not x.endswith('=NA')]
                if evaldebug: 
                    print("data in test sample:", str(test_sample_data))  
                classification = trainingDT.classify(root_node, test_sample_data)
                solution_path = classification['solution_path']                                  
                del classification['solution_path']                                              
                which_classes = list( classification.keys() )                                    
                which_classes = sorted(which_classes, key=lambda x: classification[x], reverse=True)
                most_likely_class_label = which_classes[0]
                if evaldebug:
                    print("\nClassification:\n")                                                     
                    print("     "  + str.ljust("class name", 30) + "probability")                    
                    print("     ----------                    -----------")                          
                    for which_class in which_classes:                                                
                        if which_class is not 'solution_path':                                       
                            print("     "  + str.ljust(which_class, 30) +  str(classification[which_class])) 
                    print("\nSolution path in the decision tree: " + str(solution_path))             
                    print("\nNumber of nodes created: " + str(root_node.how_many_nodes()))
                true_class_label_for_test_sample = self._samples_class_label_dict[test_sample_name]
                if evaldebug: 
                    print("%s:   true_class: %s    estimated_class: %s\n" % \
                             (test_sample_name, true_class_label_for_test_sample, most_likely_class_label))
                confusion_matrix[true_class_label_for_test_sample][most_likely_class_label] += 1    
        print("\n\n       DISPLAYING THE CONFUSION MATRIX FOR THE 10-FOLD CROSS-VALIDATION TEST:\n")
        matrix_header = " " * 30
        for class_name in self._class_names:  
            matrix_header += '{:^30}'.format(class_name)
        print("\n" + matrix_header + "\n")
        for row_class_name in sorted(confusion_matrix.keys()):
            row_display = str.rjust(row_class_name, 30)
            for col_class_name in sorted(confusion_matrix[row_class_name].keys()):
                row_display += '{:^30}'.format(str(confusion_matrix[row_class_name][col_class_name]) )
            print(row_display + "\n")
        diagonal_sum, off_diagonal_sum = 0,0
        for row_class_name in sorted(confusion_matrix.keys()):
            for col_class_name in sorted(confusion_matrix[row_class_name].keys()):
                if row_class_name == col_class_name:
                    diagonal_sum += confusion_matrix[row_class_name][col_class_name]
                else:
                    off_diagonal_sum += confusion_matrix[row_class_name][col_class_name]
        data_quality_index = 100.0 * diagonal_sum / (diagonal_sum + off_diagonal_sum)
        print("\nTraining Data Quality Index: %s   (out of a possible maximum of 100)" % data_quality_index)
        if data_quality_index <= 80:
            print( '''\nYour training data does not possess much class discriminatory '''
                   '''information.  It could be that the classes are inherently not well '''
                   '''separable or that your constructor parameter choices are not appropriate.''')
        elif 80 < data_quality_index <= 90:
            print( '''\nYour training data possesses some class discriminatory information '''
                   '''but it may not be sufficient for real-world applications.  You might '''
                   '''try tweaking the constructor parameters to see if that improves the '''
                   '''class discriminations.''')
        elif 90 < data_quality_index <= 95:
            print( '''\nYour training data appears to possess good class discriminatory '''
                   '''information.  Whether or not it is acceptable would depend on your '''
                   '''application.''')
        elif 95 < data_quality_index < 98:        
            print( '''\nYour training data is of very high quality.''')
        else:
            print('''\nYour training data is excellent.''')


#------------------------------  Generate Your Own Numeric Training Data  -----------------------------

class TrainingDataGeneratorNumeric(object):
    '''
    See the example script generate_training_data_numeric.py on how to use this class
    for generating your numeric training data.  The training data is generator in
    accordance with the specifications you place in a parameter file.
    '''
    def __init__(self, *args, **kwargs ):
        if args:
            raise ValueError(  
                   '''TrainingDataGeneratorNumeric can only be called
                      with keyword arguments for the following
                      keywords: output_csv_file, parameter_file,
                      number_of_samples_per_class, and debug1''') 
        allowed_keys = 'output_csv_file','parameter_file','number_of_samples_per_class','debug1'
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise ValueError("Wrong keyword used --- check spelling") 
        output_csv_file = parameter_file = number_of_samples_per_class = debug = None

        if 'output_csv_file' in kwargs : output_csv_file = kwargs.pop('output_csv_file')
        if 'parameter_file' in kwargs : parameter_file = kwargs.pop('parameter_file')
        if 'number_of_samples_per_class' in kwargs : \
                              number_of_samples_per_class = kwargs.pop('number_of_samples_per_class')
        if 'debug1' in kwargs  :  debug1 = kwargs.pop('debug1')
        if output_csv_file:
            self._output_csv_file = output_csv_file
        else:
            raise ValueError('''You must specify the name for a csv file for the training data''')
        if parameter_file: 
            self._parameter_file =  parameter_file
        else:
            raise ValueError('''You must specify a parameter file''')
        if number_of_samples_per_class:
            self._number_of_samples_per_class = number_of_samples_per_class
        else:
            raise ValueError('''You forgot to specify the number of training samples needed per class''')
        if debug1:
            self._debug1 = debug1
        else:
            self._debug1 = 0
        self._class_names                 = []
        self._class_names_and_priors      = {}
        self._features_with_value_range   = {}
        self._classes_and_their_param_values = {}

    def read_parameter_file_numeric( self ):
        '''
        The training data generated by an instance of the class
        TrainingDataGeneratorNumeric is based on the specs you place in a parameter
        that you supply to the class constructor through a constructor variable
        called `parameter_file.  This method is for parsing the parameter file in
        order to order to determine the names to be used for the different data
        classes, their means, and their variances.
        '''
        class_names = []
        class_names_and_priors = {}
        features_with_value_range = {}
        classes_and_their_param_values = {}
#        regex8 =  '[+-]?\ *(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?';
        FILE = open(self._parameter_file)
        params = FILE.read()
        regex = r'class names: ([\w ]+)\W*class priors: ([\d. ]+)'
        classes = re.search(regex, params, re.DOTALL | re.IGNORECASE)
        if (classes != None):
            class_names = classes.group(1).strip().split(' ')        
            class_priors = classes.group(2).strip().split(' ')
        class_names_and_priors = {class_names[i] : float(class_priors[i]) for i in range(len(class_names))}
        if self._debug1: print("\nClass names and priors: " + class_names_and_priors)
        regex = r'feature name: \w*.*?value range: [\d\. -]+'
        features = re.findall(regex, params, re.DOTALL | re.IGNORECASE)
        regex = r'feature name: (\w+)\W*?value range:\s*([\d. -]+)'
        for feature in features:
            feature_groups = re.match(regex, feature, re.IGNORECASE)
            feature_name = feature_groups.group(1)
            value_range = feature_groups.group(2).split()
            value_range = [float(value_range[0]), float(value_range[2])]
            features_with_value_range[feature_name] = value_range
        if self._debug1: print("\nFeature and their value ranges: "+ features_with_value_range)
        classes_and_their_param_values = {class_names[i] : {} for i in range(len(class_names))}
        regex = r'params for class: \w*?\W+?mean:[\d\. ]+\W*?covariance:\W+?(?:[ \d.]+\W+?)+'
        class_params = re.findall(regex, params, re.DOTALL | re.IGNORECASE)
        regex = r'params for class: (\w+)\W*?mean:\s*([\d. -]+)\W*covariance:\s*([\s\d.]+)'
        for class_param in class_params:
            class_params_groups = re.match(regex, class_param, re.IGNORECASE)
            class_name = class_params_groups.group(1)
            class_mean = class_params_groups.group(2).split()
            class_mean = list(map(float, class_mean))
            classes_and_their_param_values[class_name]['mean'] =  class_mean
            vector_size = len(class_mean)
            class_param_string = class_params_groups.group(3)
            covar_rows = list(map(lambda x: x.strip().split(), class_param_string.splitlines()))
            covar_matrix = []
            for row in covar_rows:
                row = list(map(float, row))
                covar_matrix.append(row)
            classes_and_their_param_values[class_name]['covariance'] =  covar_matrix
        if self._debug1: print("\nThe class parameters are: "+ classes_and_their_param_values)
        self._class_names                 = class_names
        self._class_names_and_priors      = class_names_and_priors
        self._features_with_value_range   = features_with_value_range
        self._classes_and_their_param_values = classes_and_their_param_values

    def gen_numeric_training_data_and_write_to_csv(self):
        '''
        After the parameter file is parsed by the previous method, this method calls
        on `numpy.random.multivariate_normal()' to generate the training data
        samples. Your training data can be of any number of of dimensions, can have
        any mean, and any covariance.
        '''
        import numpy
        import random
        samples_for_class = {class_name : [] for class_name in self._class_names}
        for class_name in self._classes_and_their_param_values:
            mean = self._classes_and_their_param_values[class_name]['mean']
            covariance = self._classes_and_their_param_values[class_name]['covariance']
            samples = numpy.random.multivariate_normal(mean, covariance, self._number_of_samples_per_class)
            samples = [map(float, map(lambda x: "%.3f" % x, list_of_samples)) for list_of_samples in samples]
            samples_for_class[class_name] = samples
        data_records = []     
        for class_name in samples_for_class:
            for sample_index in range(self._number_of_samples_per_class):
                data_record = class_name + "," + \
                             ','.join(map(lambda x: str(x), samples_for_class[class_name][sample_index]))
                if self._debug1: print("data record: " + data_record)
                data_records.append(data_record)
        random.shuffle(data_records)
        sample_records = []
        sample_records.append('""' + ',' + '"' + 'class_name' + '"' + ',' + \
                   ','.join(map(lambda x: '"'+x+'"', self._features_with_value_range.keys())) + "\n")
        for i in range(len(data_records)):
            i1 = i+1
            sample_records.append('"' + str(i1) + '"' + ',' + data_records[i] +"\n")
        try:
            FILE = open(self._output_csv_file, 'w') 
        except IOError:
            print("Unable to open file: " + self._output_csv_file)
            sys.exit(1)
        map(FILE.write, sample_records)
        FILE.close()    


#------------------------  Generate Your Own Symbolic Training Data  --------------------------------

class TrainingDataGeneratorSymbolic(object):
    '''
    See the sample script generate_training_data_symbolic.py for how to use this
    class for generating symbolic training data.  The data is generated according to
    the specifications you place in a parameter file.
    '''
    def __init__(self, *args, **kwargs ):
        if args:
            raise ValueError(  
                   '''TrainingDataGeneratorSymbolic can only be called
                      with keyword arguments for the following
                      keywords: output_datafile, parameter_file,
                      number_of_training_samples, write_to_file,
                      debug1, and debug2''') 
        allowed_keys = 'output_datafile','parameter_file', \
                       'number_of_training_samples', 'write_to_file', \
                       'debug1','debug2'
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise ValueError("Wrong keyword used --- check spelling") 
        output_datafile = parameter_file = number_of_training_samples = None
        write_to_file = debug1 = debug2 = None

        if 'output_datafile' in kwargs : \
                           output_datafile = kwargs.pop('output_datafile')
        if 'parameter_file' in kwargs : \
                           parameter_file = kwargs.pop('parameter_file')
        if 'number_of_training_samples' in kwargs : \
          number_of_training_samples = kwargs.pop('number_of_training_samples')
        if 'write_to_file' in kwargs : \
                                   write_to_file = kwargs.pop('write_to_file')
        if 'debug1' in kwargs  :  debug1 = kwargs.pop('debug1')
        if 'debug2' in kwargs  :  debug2 = kwargs.pop('debug2')

        if output_datafile:
            self._output_datafile = output_datafile
        else:
            raise ValueError('''You must specify an output datafile''')
        if parameter_file: 
            self._parameter_file =  parameter_file
        else:
            raise ValueError('''You must specify a parameter file''')
        if number_of_training_samples:
            self._number_of_training_samples = number_of_training_samples
        else:
            raise ValueError('''You forgot to specify the number of training samples needed''')
        if write_to_file:
            self._write_to_file = write_to_file
        else:
            self._write_to_file = 0          
        if debug1:
            self._debug1 = debug1
        else:
            self._debug1 = 0
        if debug2:
            self._debug2 = debug2
        else:
            self._debug2 = 0
        self._training_sample_records     = {}
        self._features_and_values_dict    = {}
        self._bias_dict                   = {}
        self._class_names                 = []
        self._class_priors                = []


    def read_parameter_file_symbolic( self ):
        '''
        Read the parameter file for generating symbolic training data. See the script
        generate_training_data_symbolic.py in the Examples directory for how to pass
        the name of the parameter file to the constructor of the
        TrainingDataGeneratorSymbolic class.
        '''
        debug1 = self._debug1
        debug2 = self._debug2
        write_to_file = self._write_to_file
        number_of_training_samples = self._number_of_training_samples
        input_parameter_file = self._parameter_file
        all_params = []
        param_string = ''
        try:
            FILE = open(input_parameter_file, 'r')
        except IOError:
            print("unable to open %s" % input_parameter_file)
            sys.exit(1)
        all_params = FILE.read()
        all_params = re.split(r'\n', all_params)
        FILE.close()
        pattern = r'^(?![ ]*#)'
        try:
            regex = re.compile( pattern )
        except:
            print("error in your pattern")
            sys.exit(1)
        all_params = list( filter( regex.search, all_params ) )
        all_params = list( filter( None, all_params ) )
        all_params = [x.rstrip('\n') for x in all_params]
        param_string = ' '.join( all_params )

        pattern = '^\s*class names:(.*?)\s*class priors:(.*?)(feature: .*)'
        m = re.search( pattern, param_string )
        rest_params = m.group(3)
        self._class_names = list( filter(None, re.split(r'\s+', m.group(1))) )
        self._class_priors = list( filter(None, re.split(r'\s+', m.group(2))) )
        pattern = r'(feature:.*?) (bias:.*)'
        m = re.search( pattern, rest_params  )
        feature_string = m.group(1)
        bias_string = m.group(2)
        features_and_values_dict = {}
        features = list( filter( None, re.split( r'(feature[:])', feature_string ) ) )
        for item in features:
            if re.match(r'feature', item): continue
            splits = list( filter(None, re.split(r' ', item)) )
            for i in range(0, len(splits)):
                if i == 0: features_and_values_dict[splits[0]] = []
                else:
                    if re.match( r'values', splits[i] ): continue
                    features_and_values_dict[splits[0]].append( splits[i] )
        self._features_and_values_dict = features_and_values_dict
        bias_dict = {}
        biases = list( filter(None, re.split(r'(bias[:]\s*class[:])', bias_string )) )
        for item in biases:
            if re.match(r'bias', item): continue
            splits = list( filter(None, re.split(r' ', item)) )
            feature_name = ''
            for i in range(0, len(splits)):
                if i == 0:
                    bias_dict[splits[0]] = {}
                elif ( re.search( r'(^.+)[:]$', splits[i] ) ):
                    m = re.search(  r'(^.+)[:]$', splits[i] )
                    feature_name = m.group(1)
                    bias_dict[splits[0]][feature_name] = []
                else:
                    if not feature_name: continue
                    bias_dict[splits[0]][feature_name].append( splits[i] )
        self._bias_dict = bias_dict
        if self._debug1:
            print("\n\n") 
            print("Class names: " + str(self._class_names))
            print("\n") 
            num_of_classes = len(self._class_names)
            print("Number of classes: " + str(num_of_classes))
            print("\n")
            print("Class priors: " + str(self._class_priors))
            print("\n\n")
            print("Here are the features and their possible values")
            print("\n")
            items = self._features_and_values_dict.items()
            for item in items:
                print(item[0] + " ===> " + str(item[1]))
            print("\n")
            print("Here is the biasing for each class:")
            print("\n")          
            items = self._bias_dict.items()
            for item in items:
                print("\n")
                print(item[0])
                items2 = list( item[1].items() )
                for i in range(0, len(items2)):
                    print( items2[i])

    def gen_symbolic_training_data( self ):
        '''
        This method generates the training data according to the specifications
        placed in the parameter file that is read by the previous method.
        '''
        class_names = self._class_names
        class_priors = self._class_priors
        training_sample_records = {}
        features_and_values_dict = self._features_and_values_dict
        bias_dict  = self._bias_dict
        how_many_training_samples = self._number_of_training_samples
        class_priors_to_unit_interval_map = {}
        accumulated_interval = 0
        for i in range(0, len(class_names)):
            class_priors_to_unit_interval_map[class_names[i]] = \
            (accumulated_interval, accumulated_interval+float(class_priors[i]))
            accumulated_interval += float(class_priors[i])
        if self._debug1:
            print("Mapping of class priors to unit interval:")
            print("\n")
            items = class_priors_to_unit_interval_map.items()
            for item in items:
                print(item[0] + " ===> " + str(item[1]))
        class_and_feature_based_value_priors_to_unit_interval_map = {}
        for class_name  in class_names:
            class_and_feature_based_value_priors_to_unit_interval_map[class_name] = {}
            for feature in features_and_values_dict.keys():
                class_and_feature_based_value_priors_to_unit_interval_map[class_name][feature] = {}
        for class_name  in class_names:
            for feature in features_and_values_dict.keys():
                values = features_and_values_dict[feature]
                if len(bias_dict[class_name][feature]) > 0:
                    bias_string = bias_dict[class_name][feature][0]
                else:
                    no_bias = 1.0 / len(values)
                    bias_string = values[0] +  "=" + str(no_bias)
                value_priors_to_unit_interval_map = {}
                splits = list( filter( None, re.split(r'\s*=\s*', bias_string) ) )
                chosen_for_bias_value = splits[0]
                chosen_bias = splits[1]
                remaining_bias = 1 - float(chosen_bias)
                remaining_portion_bias = remaining_bias / (len(values) -1)
                accumulated = 0;
                for i in range(0, len(values)):
                    if (values[i] == chosen_for_bias_value):
                        value_priors_to_unit_interval_map[values[i]] = \
                          [accumulated, accumulated + float(chosen_bias)]
                        accumulated += float(chosen_bias)
                    else:
                        value_priors_to_unit_interval_map[values[i]] = \
                          [accumulated, accumulated + remaining_portion_bias]
                        accumulated += remaining_portion_bias
                class_and_feature_based_value_priors_to_unit_interval_map[class_name][feature] = \
                                                                   value_priors_to_unit_interval_map
                if self._debug2:
                    print("\n")
                    print( "For class " + class_name + \
                       ": Mapping feature value priors for feature '" + \
                       feature + "' to unit interval: ")
                    print("\n")
                    items = value_priors_to_unit_interval_map.items()
                    for item in items:
                        print("    " + item[0] + " ===> " + str(item[1]))
        ele_index = 0
        while (ele_index < how_many_training_samples):
            sample_name = "sample" + "_" + str(ele_index)
            training_sample_records[sample_name] = []
            # Generate class label for this training sample:                
            import random
            ran = random.Random()
            roll_the_dice  = ran.randint(0,1000) / 1000.0
            class_label = ''
            for class_name  in class_priors_to_unit_interval_map.keys():
                v = class_priors_to_unit_interval_map[class_name]
                if ( (roll_the_dice >= v[0]) and (roll_the_dice <= v[1]) ):
                    training_sample_records[sample_name].append( 
                                             "class=" + class_name )
                    class_label = class_name
                    break
            for feature in sorted(list(features_and_values_dict.keys())):
                roll_the_dice  = ran.randint(0,1000) / 1000.0
                value_label = ''
                value_priors_to_unit_interval_map = \
                  class_and_feature_based_value_priors_to_unit_interval_map[class_label][feature]
                for value_name in value_priors_to_unit_interval_map.keys():
                    v = value_priors_to_unit_interval_map[value_name]
                    if ( (roll_the_dice >= v[0]) and (roll_the_dice <= v[1]) ):
                        training_sample_records[sample_name].append( \
                                            feature + "=" + value_name )
                        value_label = value_name;
                        break
            ele_index += 1
        self._training_sample_records = training_sample_records
        if self._debug2:
            print("\n\n")
            print("TERMINAL DISPLAY OF TRAINING RECORDS:")
            print("\n\n")
            sample_names = training_sample_records.keys()
            sample_names = sorted( sample_names, key=lambda x: int(x.lstrip('sample_')) )
            for sample_name in sample_names:
                print(sample_name + " = " + str(training_sample_records[sample_name]))

    def find_longest_feature_or_value(self):
        features_and_values_dict = self._features_and_values_dict
        max_length = 0
        for feature in features_and_values_dict.keys():
            if not max_length:
                max_length = len(str(feature))
            if len(str(feature)) > max_length:
                max_length = len(str(feature)) 
            values = features_and_values_dict[feature]
            for value in values:
                if len(str(value)) > max_length:
                    max_length = len(str(value)) 
        return max_length

    def write_training_data_to_file( self ):
        features_and_values_dict = self._features_and_values_dict
        class_names = self._class_names
        output_file = self._output_datafile
        training_sample_records = self._training_sample_records
        try:
            FILE = open(self._output_datafile, 'w') 
        except IOError:
            print("Unable to open file: " + self._output_datafile)
            sys.exit(1)
        class_names_string = ''
        for aname in class_names:
            class_names_string += aname + " "
        class_names_string.rstrip()
        FILE.write("Class names: %s\n\n" % class_names_string ) 
        FILE.write("Feature names and their values:\n")
        features = list( features_and_values_dict.keys() )
        if len(features) == 0:
            print("You probably forgot to call gen_training_data() before " + \
                          "calling write_training_data_to_file()") 
            sys.exit(1)
        for i in range(0, len(features)):
            values = features_and_values_dict[features[i]]
            values_string = ''
            for aname in values:
                values_string += aname + " "
            values_string.rstrip()
            FILE.write("     %(s1)s = %(s2)s\n" % {'s1':features[i], 's2':values_string} )
        FILE.write("\n\nTraining Data:\n\n")
        num_of_columns = len(features) + 2
        field_width = self.find_longest_feature_or_value() + 2
        if field_width < 12: field_width = 12
        title_string = str.ljust( "sample", field_width ) + \
                       str.ljust( "class", field_width )
        features.sort()
        for feature_name in features:
            title_string += str.ljust( str(feature_name), field_width )
        FILE.write(title_string + "\n")
        separator = '=' * len(title_string)
        FILE.write(separator + "\n")
        sample_names = list( training_sample_records.keys() )
        sample_names = sorted( sample_names, key=lambda x: int(x.lstrip('sample_')) )
        record_string = ''
        for sample_name in sample_names:
            sample_name_string = str.ljust(sample_name, field_width)
            record_string += sample_name_string
            record = training_sample_records[sample_name]
            item_parts_dict = {}
            for item in record:
                splits = list( filter(None, re.split(r'=', item)) )
                item_parts_dict[splits[0]] = splits[1]
            record_string += str.ljust(item_parts_dict["class"], field_width)
            del item_parts_dict["class"]
            kees = list(item_parts_dict.keys())
            kees.sort()
            for kee in kees:
                record_string += str.ljust(item_parts_dict[kee], field_width)
            FILE.write(record_string + "\n")
            record_string = ''
        FILE.close()


#-------------------  Generate Your Own Test Data For Purely Symbolic Case  -----------------------

class TestDataGeneratorSymbolic(object):
    '''
    This convenience class does basically the same thing as the
    TrainingDataGeneratorSymbolic except that it places the class labels for the
    sample records in a separate file.  Let's say you have already created a DT
    classifier and you would like to test its class discriminatory power.  You can
    use the classifier to calculate the class labels for the data records by the
    class shown here.  And then you can you can compare those class labels with those
    placed originally by this class in a separate file.  See the script
    generate_test_data_symbolic.py for how to use this class.
    '''
    def __init__(self, *args, **kwargs ):
        if args:
            raise ValueError(  
                   '''TestDataGenerator can only be called
                      with keyword arguments for the following
                      keywords: parameter_file, output_test_datafile,
                      output_class_labels_file, number_of_test_samples, 
                      write_to_file, debug1, and debug2''') 
        allowed_keys = 'output_test_datafile','parameter_file', \
                       'number_of_test_samples', 'write_to_file', \
                       'output_class_labels_file', 'debug1', 'debug2'
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise ValueError("Wrong keyword used --- check spelling") 
        output_test_datafile = parameter_file = number_of_test_samples = None
        write_to_file = debug1 = debug2 = None
        if 'output_test_datafile' in kwargs : \
                    output_test_datafile = kwargs.pop('output_test_datafile')
        if 'output_class_labels_file' in kwargs : \
             output_class_labels_file =  kwargs.pop('output_class_labels_file')
        if 'parameter_file' in kwargs : \
                           parameter_file = kwargs.pop('parameter_file')
        if 'number_of_test_samples' in kwargs : \
          number_of_test_samples = kwargs.pop('number_of_test_samples')
        if 'write_to_file' in kwargs : \
                                   write_to_file = kwargs.pop('write_to_file')
        if 'debug1' in kwargs  :  debug1 = kwargs.pop('debug1')
        if 'debug2' in kwargs  :  debug2 = kwargs.pop('debug2')
        if output_test_datafile:
            self._output_test_datafile = output_test_datafile
        else:
            raise ValueError('''You must specify an output test datafile''')
        if output_class_labels_file:
            self._output_class_labels_file = output_class_labels_file
        else:
            raise ValueError('''You must specify an output file for class labels''')
        if parameter_file: 
            self._parameter_file =  parameter_file
        else:
            raise ValueError('''You must specify a parameter file''')
        if number_of_test_samples:
            self._number_of_test_samples = number_of_test_samples
        else:
            raise ValueError('''You forgot to specify the number of test samples needed''')
        if write_to_file:
            self._write_to_file = write_to_file
        else:
            self._write_to_file = 0          
        if debug1: self._debug1 = debug1
        else: self._debug1 = 0
        if debug2: self._debug2 = debug2
        else: self._debug2 = 0
        self._test_sample_records         = {}
        self._features_and_values_dict    = {}
        self._bias_dict                   = {}
        self._class_names                 = []
        self._class_priors                = []

    def read_parameter_file( self ):
        '''
        This methods reads the parameter file for generating the test data.
        ''' 
        debug1 = self._debug1
        debug2 = self._debug2
        write_to_file = self._write_to_file
        number_of_test_samples = self._number_of_test_samples
        input_parameter_file = self._parameter_file
        all_params = []
        param_string = ''
        try:
            FILE = open(input_parameter_file, 'r') 
        except IOError:
            print("Unable to open file: " + input_parameter_file)
            sys.exit(1)
        all_params = FILE.read()

        all_params = re.split(r'\n', all_params)
        FILE.close()
        pattern = r'^(?![ ]*#)'
        try:
            regex = re.compile( pattern )
        except:
            print("error in your pattern")
            sys.exit(1)
        all_params = list( filter( regex.search, all_params ) )
        all_params = list( filter( None, all_params ) )
        all_params = [x.rstrip('\n') for x in all_params]
        param_string = ' '.join( all_params )
        pattern = '^\s*class names:(.*?)\s*class priors:(.*?)(feature: .*)'
        m = re.search( pattern, param_string )
        rest_params = m.group(3)
        self._class_names = list( filter(None, re.split(r'\s+', m.group(1))) )
        self._class_priors = list( filter(None, re.split(r'\s+', m.group(2))) )
        pattern = r'(feature:.*?) (bias:.*)'
        m = re.search( pattern, rest_params  )
        feature_string = m.group(1)
        bias_string = m.group(2)
        features_and_values_dict = {}
        features = list( filter( None, re.split( r'(feature[:])', feature_string ) ) )
        for item in features:
            if re.match(r'feature', item): continue
            splits = list( filter(None, re.split(r' ', item)) )
            for i in range(0, len(splits)):
                if i == 0: features_and_values_dict[splits[0]] = []
                else:
                    if re.match( r'values', splits[i] ): continue
                    features_and_values_dict[splits[0]].append( splits[i] )
        self._features_and_values_dict = features_and_values_dict
        bias_dict = {}
        biases = list( filter(None, re.split(r'(bias[:]\s*class[:])', bias_string )) )
        for item in biases:
            if re.match(r'bias', item): continue
            splits = list( filter(None, re.split(r' ', item)) )
            feature_name = ''
            for i in range(0, len(splits)):
                if i == 0:
                    bias_dict[splits[0]] = {}
                elif ( re.search( r'(^.+)[:]$', splits[i] ) ):
                    m = re.search(  r'(^.+)[:]$', splits[i] )
                    feature_name = m.group(1)
                    bias_dict[splits[0]][feature_name] = []
                else:
                    if not feature_name: continue
                    bias_dict[splits[0]][feature_name].append( splits[i] )
        self._bias_dict = bias_dict
        if self._debug1:
            print("\n\n")
            print("Class names: " + str(self._class_names))
            print("\n")
            num_of_classes = len(self._class_names)
            print("Number of classes: " + str(num_of_classes))
            print("\n")
            print("Class priors: " + str(self._class_priors))
            print("\n\n")
            print("Here are the features and their possible values:")
            print("\n")
            items = self._features_and_values_dict.items()
            for item in items:
                print(item[0] + " ===> " + str(item[1]))
            print("\n")
            print("Here is the biasing for each class:")
            print("\n")            
            items = self._bias_dict.items()
            for item in items:
                print("\n")
                print(item[0])
                items2 = list( item[1].items() )
                for i in range(0, len(items2)):
                    print( items2[i])

    def gen_test_data( self ):
        '''
        This method generates the test data according to the specifications
        laid out in the parameter file read by the previous method.
        '''
        class_names = self._class_names
        class_priors = self._class_priors
        test_sample_records = {}
        features_and_values_dict = self._features_and_values_dict
        bias_dict  = self._bias_dict
        how_many_test_samples = self._number_of_test_samples
        file_for_class_labels = self._output_class_labels_file
        class_priors_to_unit_interval_map = {}
        accumulated_interval = 0
        for i in range(0, len(class_names)):
            class_priors_to_unit_interval_map[class_names[i]] = \
            (accumulated_interval, accumulated_interval+float(class_priors[i]))
            accumulated_interval += float(class_priors[i])
        if self._debug1:
            print("Mapping of class priors to unit interval:")
            print("\n")
            items = class_priors_to_unit_interval_map.items()
            for item in items:
                print(item[0] + " ===> " + str(item[1]))
        class_and_feature_based_value_priors_to_unit_interval_map = {}
        for class_name  in class_names:
            class_and_feature_based_value_priors_to_unit_interval_map[class_name] = {}
            for feature in features_and_values_dict.keys():
                class_and_feature_based_value_priors_to_unit_interval_map[class_name][feature] = {}
        for class_name  in class_names:
            for feature in features_and_values_dict.keys():
                values = features_and_values_dict[feature]
                if len(bias_dict[class_name][feature]) > 0:
                    bias_string = bias_dict[class_name][feature][0]
                else:
                    no_bias = 1.0 / len(values)
                    bias_string = values[0] +  "=" + str(no_bias)
                value_priors_to_unit_interval_map = {}
                splits = list( filter( None, re.split(r'\s*=\s*', bias_string) ) )
                chosen_for_bias_value = splits[0]
                chosen_bias = splits[1]
                remaining_bias = 1 - float(chosen_bias)
                remaining_portion_bias = remaining_bias / (len(values) -1)
                accumulated = 0;
                for i in range(0, len(values)):
                    if (values[i] == chosen_for_bias_value):
                        value_priors_to_unit_interval_map[values[i]] = \
                          [accumulated, accumulated + float(chosen_bias)]
                        accumulated += float(chosen_bias)
                    else:
                        value_priors_to_unit_interval_map[values[i]] = \
                          [accumulated, accumulated + remaining_portion_bias]
                        accumulated += remaining_portion_bias
                class_and_feature_based_value_priors_to_unit_interval_map[class_name][feature] = \
                                                                     value_priors_to_unit_interval_map
                if self._debug1:
                    print("\n")
                    print("For class " + class_name + \
                       ": Mapping feature value priors for feature '" + \
                       feature + "' to unit interval:")
                    print("\n")
                    items = value_priors_to_unit_interval_map.items()
                    for item in items:
                        print("    " + item[0] + " ===> " + str(item[1]))
        ele_index = 0
        while (ele_index < how_many_test_samples):
            sample_name = "sample" + "_" + str(ele_index)
            test_sample_records[sample_name] = []
            # Generate class label for this test sample:                
            import random
            ran = random.Random()
            roll_the_dice  = ran.randint(0,1000) / 1000.0
            class_label = ''
            for class_name  in class_priors_to_unit_interval_map.keys():
                v = class_priors_to_unit_interval_map[class_name]
                if ( (roll_the_dice >= v[0]) and (roll_the_dice <= v[1]) ):
                    test_sample_records[sample_name].append( 
                                             "class=" + class_name )
                    class_label = class_name
                    break
            for feature in sorted(list(features_and_values_dict.keys())):
                roll_the_dice  = ran.randint(0,1000) / 1000.0
                value_label = ''
                value_priors_to_unit_interval_map = \
                  class_and_feature_based_value_priors_to_unit_interval_map[class_label][feature]
                for value_name in value_priors_to_unit_interval_map.keys():
                    v = value_priors_to_unit_interval_map[value_name]
                    if ( (roll_the_dice >= v[0]) and (roll_the_dice <= v[1]) ):
                        test_sample_records[sample_name].append( \
                                            feature + "=" + value_name )
                        value_label = value_name;
                        break
            ele_index += 1
        self._test_sample_records = test_sample_records
        if self._debug1:
            print("\n\n")
            print("TERMINAL DISPLAY OF TEST RECORDS:")
            print("\n\n")
            sample_names = test_sample_records.keys()
            sample_names = sorted(sample_names, key=lambda x: int(x.lstrip('sample_')))
            for sample_name in sample_names:
                print(sample_name + " => " + \
                                 str(test_sample_records[sample_name]))

    def find_longest_value(self):
        features_and_values_dict = self._features_and_values_dict
        max_length = 0
        for feature in features_and_values_dict.keys():
            values = features_and_values_dict[feature]
            for value in values:
                if not max_length:
                    max_length = len(str(value))
                if len(str(value)) > max_length:
                    max_length = len(str(value)) 
        return max_length

    def write_test_data_to_file(self):
        features_and_values_dict = self._features_and_values_dict
        class_names = self._class_names
        output_file = self._output_test_datafile
        test_sample_records = self._test_sample_records
        try:
            FILE = open(self._output_test_datafile, 'w') 
        except IOError:
            print("Unable to open file: " + self._output_test_datafile)
            sys.exit(1)
        try:
            FILE2 = open(self._output_class_labels_file, 'w') 
        except IOError:
            print("Unable to open file: " + self._output_class_labels_file)
            sys.exit(1)
        header = '''
# REQUIRED LINE FOLLOWS (the first uncommented line below):
# This line shown below must begin with the string 
#
#             "Feature Order For Data:"  
#
# What comes after this string can be any number of feature labels.  
# The feature values shown in the table in the rest of the file will 
# be considered to be in same order as shown in the next line.
                '''
        FILE.write(header + "\n\n\n")       
        title_string = "Feature Order For Data: "
        features = list(features_and_values_dict.keys())
        features.sort()
        for feature_name in features:
            title_string += str(feature_name) + " "
        title_string.rstrip()
        FILE.write(title_string + "\n\n")
        num_of_columns = len(features) + 1
        field_width = self.find_longest_value() + 2
        sample_names = test_sample_records.keys()
        sample_names = sorted(sample_names, key=lambda x: int(x.lstrip('sample_')))
        record_string = ''
        for sample_name in sample_names:
            sample_name_string = str.ljust(sample_name, 13 )
            record_string += sample_name_string
            record = test_sample_records[sample_name]
            item_parts_dict = {}
            for item in record:
                splits = list( filter(None, re.split(r'=', item)) )
                item_parts_dict[splits[0]] = splits[1]
            label_string = sample_name + " " + item_parts_dict["class"]
            FILE2.write(label_string + "\n")
            del item_parts_dict["class"]
            kees = list(item_parts_dict.keys())
            kees.sort()
            for kee in kees:
                record_string += str.ljust(item_parts_dict[kee], field_width)
            FILE.write(record_string + "\n")
            record_string = ''
        FILE.close()
        FILE2.close

#---------------------------------------  Class DTNode   --------------------------------------

class DTNode(object):
    '''
    The nodes of the decision tree are instances of this class:
    '''
    nodes_created = -1
    class_names = None

    def __init__(self, feature, entropy, class_probabilities, branch_features_and_values_or_thresholds, \
                                                                              root_or_not=None):
        if root_or_not == 'root':
            DTNode.initialize_DTNode_class()
        self._serial_number               = self.get_next_serial_num()
        self._feature                     = feature
        self._node_creation_entropy       = entropy
        self._class_probabilities = class_probabilities
        self._branch_features_and_values_or_thresholds =branch_features_and_values_or_thresholds
        self._linked_to = []

    @staticmethod
    def initialize_DTNode_class():
        DTNode.nodes_created = -1
        DTNode.class_names = None

    @staticmethod
    def how_many_nodes():
        return DTNode.nodes_created + 1

    @staticmethod
    def set_class_names(class_names_list):
        DTNode.class_names = class_names_list
    
    @staticmethod
    def get_class_names():
        return DTNode.class_names

    def get_next_serial_num(self):
        DTNode.nodes_created += 1
        return DTNode.nodes_created

    def get_serial_num(self):
        return self._serial_number

    def get_feature(self):
        '''
        Returns the feature test at the current node
        '''
        return self._feature

    def set_feature(self, feature):
        self._feature = feature

    def get_node_entropy(self):
        return self._node_creation_entropy

    def get_class_probabilities(self):
        return self._class_probabilities

    def get_branch_features_and_values_or_thresholds(self):
        return self._branch_features_and_values_or_thresholds

    def get_children(self):
        return self._linked_to

    def add_child_link(self, new_node):
        self._linked_to.append(new_node)                  

    def delete_all_links(self):
        self._linked_to = None

    def display_node(self):
        feature_at_node = self.get_feature() or " "
        node_creation_entropy_at_node = self.get_node_entropy()
        print_node_creation_entropy_at_node = "%.3f" % node_creation_entropy_at_node
        class_probabilities = self.get_class_probabilities()
        class_probabilities_for_display = ["%0.3f" % x for x in class_probabilities]
        serial_num = self.get_serial_num()
        branch_features_and_values_or_thresholds = self.get_branch_features_and_values_or_thresholds()
        print("\n\nNODE " + str(serial_num) + \
              ":\n   Branch features and values to this node: " + \
              str(branch_features_and_values_or_thresholds) + 
              "\n   Class probabilities at current node: " + \
              str(class_probabilities_for_display) + \
              "\n   Entropy at current node: " + \
              print_node_creation_entropy_at_node + \
              "\n   Best feature test at current node: " + feature_at_node + "\n\n")

    def display_decision_tree(self, offset):
        serial_num = self.get_serial_num()
        if len(self.get_children()) > 0:
            feature_at_node = self.get_feature() or " "
            node_creation_entropy_at_node = self.get_node_entropy()
            print_node_creation_entropy_at_node = "%.3f" % node_creation_entropy_at_node
            branch_features_and_values_or_thresholds = self.get_branch_features_and_values_or_thresholds()
            class_probabilities = self.get_class_probabilities()
            print_class_probabilities = ["%.3f" % x for x in class_probabilities]
            print_class_probabilities_with_class = [DTNode.get_class_names()[i] + " => " + \
                         print_class_probabilities[i] for i in range(len(DTNode.get_class_names()))]
            print("NODE " + str(serial_num) + ":  " + offset +  "BRANCH TESTS TO NODE: " +\
                                           str(branch_features_and_values_or_thresholds))
            second_line_offset = offset + " " * (8 + len(str(serial_num)))
            print(second_line_offset +  "Decision Feature: " + \
                  feature_at_node + "   Node Creation Entropy: " + print_node_creation_entropy_at_node +  \
                  "   Class Probs: " + str(print_class_probabilities_with_class) + "\n")
            offset += "   "
            for child in self.get_children():
                child.display_decision_tree(offset)
        else:
            node_creation_entropy_at_node = self.get_node_entropy()
            print_node_creation_entropy_at_node = "%.3f" % node_creation_entropy_at_node
            branch_features_and_values_or_thresholds = self.get_branch_features_and_values_or_thresholds()
            class_probabilities = self.get_class_probabilities()
            print_class_probabilities = ["%.3f" % x for x in class_probabilities]
            print_class_probabilities_with_class = [DTNode.get_class_names()[i] + " => " + \
                   print_class_probabilities[i] for i in range(len(DTNode.get_class_names()))]
            print("NODE " + str(serial_num) + ":  " + offset +  "BRANCH TESTS TO LEAF NODE: " +\
                                           str(branch_features_and_values_or_thresholds))
            second_line_offset = offset + " " * (8 + len(str(serial_num)))
            print(second_line_offset + "Node Creation Entropy: " + print_node_creation_entropy_at_node +  \
                  "   Class Probs: " + str(print_class_probabilities_with_class) + "\n")


#------------------------------------  Test Code Follows  -------------------------------------

if __name__ == '__main__':

    dt = DecisionTree( training_datafile = "Examples/training.dat",  
                        max_depth_desired = 5,
                        entropy_threshold = 0.1,
                     )
    dt.get_training_data()

    dt.show_training_data()

    prob = dt.prior_probability_for_class( 'benign' )
    print("prior for benign: ", prob)
    prob = dt.prior_probability_for_class( 'malignant' )
    print("prior for malignant: ", prob)

    prob = dt.probability_of_feature_value( 'smoking', 'heavy')
    print(prob)

    dt.determine_data_condition()

    root_node = dt.construct_decision_tree_classifier()
    root_node.display_decision_tree("   ")

    test_sample = ['exercising=never', 'smoking=heavy', 'fatIntake=heavy', 'videoAddiction=heavy']
    classification = dt.classify(root_node, test_sample)
    print("Classification: " + str(classification))

    test_sample = ['videoAddiction=none', 'exercising=occasionally', 'smoking=never', 'fatIntake=medium']
    classification = dt.classify(root_node, test_sample)
    print("Classification: " + str(classification))

    print("Number of nodes created: " + str(root_node.how_many_nodes()))
