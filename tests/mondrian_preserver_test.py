import unittest
import numpy as np
import pandas as pd
import sys
import os
from color import bcolors as C 
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from preserver.mondrian_preserver import Preserver

from  functools import wraps 

def test_case(func):
    @wraps(func)
    def test_case(*args, **kwargs):
        print (C.HEADER, func.__name__, C.NC)
        return func(*args, **kwargs)
    return test_case


def failed(msg):
    print(C.FAIL, msg , C.NC)

def passed(msg):
    print(C.OKGREEN, msg , C.NC)
   

def is_equal(df1, df2):
    result = pd.concat([df1,df2]).drop_duplicates(keep=False)
    if len(result.index) == 0:
        return True
    else:
        print('diff:\n',result)
        return False


def init():
    data = np.array([[6, '1', 'test1', 'x', 20],
            [6, '1', 'test1', 'y', 30],
            [8, '2', 'test2', 'x', 50],
            [8, '2', 'test2', 'x', 45],
            [4, '1', 'test2', 'y', 35],
            [4, '2', 'test3', 'y', 20]])
    schema = [
        "column1",
        "column2",
        "column3",
        "column4",
        "column5",
    ]
    df = pd.DataFrame(data, 
        columns=schema)
    categorical = set((
        'column2',
        'column3',
        'column4'
    ))
    feature_columns = ['column1', 'column2', 'column3']
    return df, feature_columns, categorical


class functionTest(unittest.TestCase):
    @test_case
    def test1_k_anonymize(self):
        df, feature_columns, categorical = init()
        sensitive_column = 'column4'
        schema = [
            "column1",
            "column2",
            "column3",
            "column4",
            "count",
        ]

        resultdf = Preserver.k_anonymize(df, 3, feature_columns,
                                         sensitive_column, categorical, schema)
        
        testdata = [["0-10", '1', 'test1,test2', 'x', 1],
                    ["0-10", '1', 'test1,test2', 'y', 2],
                    ["0-10", '2', 'test2,test3', 'x', 2],
                    ["0-10", '2', 'test2,test3', 'y', 1]]
        testdf = pd.DataFrame(testdata, columns=schema)
        try:
            self.assertTrue(is_equal(testdf, resultdf))
            passed("K-Anonymity function 1 - Passed")
        except AssertionError:
            failed("K-Anonymity function 1 - Failed")

    @test_case
    def test2_k_anonymize(self):
        df, feature_columns, categorical = init()
        sensitive_column = 'column5'
        schema = [
            "column1",
            "column2",
            "column3",
            "column5",
            "count",
        ]
        resultdf = Preserver.k_anonymize(df, 3, feature_columns,
                                         sensitive_column, categorical, schema)

        testdata = [["0-10", '1', 'test1,test2', 20.0, 1],
                    ["0-10", '1', 'test1,test2', 30.0, 1],
                    ["0-10", '1', 'test1,test2', 35.0, 1],
                    ["0-10", '2', 'test2,test3', 20.0, 1],
                    ["0-10", '2', 'test2,test3', 45.0, 1],
                    ["0-10", '2', 'test2,test3', 50.0, 1]]
        testdf = pd.DataFrame(testdata, columns=schema)

        try:
            self.assertTrue(is_equal(testdf,resultdf))
            passed("K-Anonymity function 2 - Passed")
        except AssertionError:
            failed("K-Anonymity function 2 - Failed")

    @test_case
    def test_k_anonymize_w_user(self):
        df, feature_columns, categorical = init()
        feature_columns = ['column2', 'column3']
        sensitive_column = 'column4'
        schema = [
            "column1",
            "column2",
            "column3",
            "column4",
            "column5"
        ]
        resultdf = Preserver.k_anonymize_w_user(df, 3, feature_columns,
                                         sensitive_column, categorical, schema)

        testdata = [[6, '1', 'test1,test2', 'x', 20],
                    [6, '1', 'test1,test2', 'y', 30],
                    [4, '1', 'test1,test2', 'y', 35],
                    [8, '2', 'test2,test3', 'x', 50],
                    [8, '2', 'test2,test3', 'x', 45],
                    [4, '2', 'test2,test3', 'y', 20]]

        testdf = pd.DataFrame(testdata, columns=schema)

        try:
            self.assertTrue(is_equal(testdf, resultdf))
            passed("K-Anonymity function with user - Passed")
        except AssertionError:
            failed("K-Anonymity function with user - Failed")

    @test_case
    def test1_l_diversity(self):
        df, feature_columns, categorical = init()
        sensitive_column = 'column4'
        schema = [
            "column1",
            "column2",
            "column3",
            "column4",
            "count"
        ]
        resultdf = Preserver.l_diversity(df, 3, 2, feature_columns,
                                         sensitive_column, categorical, schema)

        testdata = [["0-10", '1', 'test1,test2', 'x', 1],
                    ["0-10", '1', 'test1,test2', 'y', 2],
                    ["0-10", '2', 'test2,test3', 'x', 2],
                    ["0-10", '2', 'test2,test3', 'y', 1]]
        testdf = pd.DataFrame(testdata, columns=schema)

        try:
            self.assertTrue(is_equal(testdf,resultdf))
            passed("L-Diversity function 1 - Passed")
        except AssertionError:
            failed("L-Diversity function 1 - Failed")

    @test_case
    def test2_l_diversity(self):
        df, feature_columns, categorical = init()
        sensitive_column = 'column5'
        schema = [
            "column1",
            "column2",
            "column3",
            "column5",
            "count",
        ]
        resultdf = Preserver.l_diversity(df, 3, 2, feature_columns,
                                         sensitive_column, categorical, schema)

        testdata = [["0-10", '1', 'test1,test2', 20.0, 1],
                    ["0-10", '1', 'test1,test2', 30.0, 1],
                    ["0-10", '1', 'test1,test2', 35.0, 1],
                    ["0-10", '2', 'test2,test3', 20.0, 1],
                    ["0-10", '2', 'test2,test3', 45.0, 1],
                    ["0-10", '2', 'test2,test3', 50.0, 1]]
        testdf = pd.DataFrame(testdata, columns=schema)

        try:
            self.assertTrue(is_equal(testdf,resultdf))
            passed(f"L-Diversity function 2 - Passed")
        except AssertionError:
            failed("L-Diversity function 2 - Failed")

    @test_case
    def test_l_diversity_w_user(self):
        df, feature_columns, categorical = init()
        feature_columns = ['column2', 'column3']
        sensitive_column = 'column4'
        schema = [
            "column1",
            "column2",
            "column3",
            "column4",
            "column5",
        ]
        resultdf = Preserver.l_diversity_w_user(df, 3,2, feature_columns,
                                         sensitive_column, categorical, schema)

        testdata = [[6, '1', 'test1,test2', 'x', 20],
                    [6, '1', 'test1,test2', 'y', 30],
                    [4, '1', 'test1,test2', 'y', 35],
                    [8, '2', 'test2,test3', 'x', 50],
                    [8, '2', 'test2,test3', 'x', 45],
                    [4, '2', 'test2,test3', 'y', 20]]
        testdf = pd.DataFrame(testdata, columns=schema)

        try:
            self.assertTrue(is_equal(testdf,resultdf))
            passed("L-Diversity function with user - Passed")
        except AssertionError:
            failed("L-Diversity function with user - Failed")

    @test_case
    def test_t_closeness(self):
        df, feature_columns, categorical = init()
        sensitive_column = 'column4'
        schema = [
            "column1",
            "column2",
            "column3",
            "column4",
            "count",
        ]
        resultdf = Preserver.t_closeness(df, 3, 0.2, feature_columns,
                                         sensitive_column, categorical, schema)

        testdata = [["0-10", '1', 'test1,test2', 'x', 1],
                    ["0-10", '1', 'test1,test2', 'y', 2],
                    ["0-10", '2', 'test2,test3', 'x', 2],
                    ["0-10", '2', 'test2,test3', 'y', 1]]
        testdf = pd.DataFrame(testdata, columns=schema)

        try:
            self.assertTrue(is_equal(testdf,resultdf))
            passed("T-closeness function - Passed")
        except AssertionError:
            failed("T-closeness function - Failed")

    @test_case
    def test_t_closeness_w_user(self):
        df, feature_columns, categorical = init()
        feature_columns = ['column2', 'column3']
        sensitive_column = 'column4'
        schema = [
            "column1",
            "column2",
            "column3",
            "column4",
            "column5",
        ]
        resultdf = Preserver.t_closeness_w_user(df, 3, 0.2, feature_columns,
                                                sensitive_column, categorical, schema)

        testdata = [[6, '1', 'test1,test2', 'x', 20],
                    [6, '1', 'test1,test2', 'y', 30],
                    [4, '1', 'test1,test2', 'y', 35],
                    [8, '2', 'test2,test3', 'x', 50],
                    [8, '2', 'test2,test3', 'x', 45],
                    [4, '2', 'test2,test3', 'y', 20]]
        testdf = pd.DataFrame(testdata, columns=schema)

        try:
            self.assertTrue(is_equal(testdf,resultdf))
            passed("T-closeness function wiht user - Passed")
        except AssertionError:
            failed("T-closeness function with user - Failed")

    @test_case
    def test_user_anonymize(self):
        df, feature_columns, categorical = init()

        sensitive_column = 'column4'
        schema = [
            "column1",
            "column2",
            "column3",
            "column4",
            "column5",
        ]
        user = 4
        usercolumn_name = "column1"
        k = 2

        resultdf = Preserver.anonymize_user(
            df, k, user, usercolumn_name, sensitive_column, categorical, schema)

        testdata = [['6', '1', 'test1', 'x', '20'],
                    ['6', '1', 'test1', 'y', '30'],
                    ['8', '1,2', 'test2,test3', 'x', '20-55'],
                    ['8', '1,2', 'test2,test3', 'x', '20-55'],
                    ['4', '1,2', 'test2,test3', 'y', '20-55'],
                    ['4', '1,2', 'test2,test3', 'y', '20-55']]
        testdf = pd.DataFrame(testdata, columns=schema)

        try:
            self.assertTrue(is_equal(testdf,resultdf))
            passed("User anonymize function - Passed")
        except AssertionError:
            failed("User anonymize function - Failed")


if __name__ == '__main__':
    unittest.main()
