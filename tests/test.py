from tensorflow.contrib.learn.python.learn.estimators import dnn_test

import automated_tests
import basic_tests

dnn_test.FeatureColumnTest().testTrain()
basic_tests.test_all_algos_regression()

