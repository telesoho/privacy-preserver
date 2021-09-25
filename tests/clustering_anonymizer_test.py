import pandas as pd
import unittest

import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 


from preserver import gv

from preserver.clustering_anonymizer import *
from preserver.clustering_utils.cluster_init import *
from preserver.clustering_utils.clustering import *
from preserver.clustering_utils.data_loss import *
from preserver.clustering_utils.distance_calculation import *
from preserver.clustering_utils.input_validate import *
from preserver.clustering_utils.kmodes import *

import logging


logger = logging.getLogger(__name__)
logger.level = logging.DEBUG
stream_handler = logging.StreamHandler(sys.stderr)
logger.addHandler(stream_handler)


class KmodeTest(unittest.TestCase):

    def setUp(self):
        self._df = pd.read_csv('data/reduced_adult.csv', index_col=False).dropna().reset_index(drop=True)
        self.QI = ['age', 'workclass', 'education_num', 'matrital_status',
                   'occupation', 'race', 'sex', 'native_country']
        self.SA = ['class']
        self.CAT_INDEXES = [1, 3, 4, 5, 6, 7]
        self.anonymizer1 = Kanonymizer(
            self._df.copy(), self.QI, self.SA, self.CAT_INDEXES)
        self.anonymizer1.n_clusters = 10

    def tearDown(self):
        del self.anonymizer1

    def test_komode_clustering(self):
        self.anonymizer1._komode_clustering(
            catergorical_indexes=self.CAT_INDEXES)
        self.assertTrue(self.anonymizer1.df.groupby('cluster_number').ngroups <= 10)


class KanonymizerTest(unittest.TestCase):

    def setUp(self):
        self._df = pd.read_csv('data/reduced_adult.csv', index_col=False).dropna().reset_index(drop=True)
        self.QI = ['age', 'workclass', 'education_num', 'matrital_status',
                   'occupation', 'race', 'sex', 'native_country']
        self.SA = ['class']
        self.CAT_INDEXES = [1, 3, 4, 5, 6, 7]
        self.anonymizer1 = Kanonymizer(
            self._df.copy(), self.QI, self.SA, self.CAT_INDEXES)
        self.anonymizer1.n_clusters = 10
        self.anonymizer1.k = 10
        self.df1 = self._df.copy().loc[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]
        self.df1['cluster_number'] = [1, 0, 1, 2, 1, 3, 1, 4, 1, 0, 1, 1, 1, 0, 1, 0, 1]
        self.anonymizer2 = Kanonymizer(self.df1.copy(), self.QI, self.SA, self.CAT_INDEXES)
        self.anonymizer2.n_clusters = 10
        self.anonymizer2.k = 10
        self.anonymizer2.centroids = self._df.copy().loc[[0, 1, 2, 3, 4]]

        self.df2 = self._df.copy().loc[[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]]
        self.df2['cluster_number'] = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 1, 2, 3]
        self.anonymizer3 = Kanonymizer(
            self.df2.copy(), self.QI, self.SA, self.CAT_INDEXES)
        self.anonymizer3.n_clusters = 10
        self.anonymizer3.k = 5
        self.anonymizer3.centroids = self._df.copy().loc[[0, 1, 2, 3, 4]]

        self.df3 = self._df.copy().loc[[
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]]
        self.df3['cluster_number'] = [1, 2, 3, 4, 1, 2,
                                      3, 2, 1, 2, 3, 4, 1, 1, 2, 3, 3, 3, 4, 4, 3]
        self.anonymizer4 = Kanonymizer(
            self.df3.copy(), self.QI, self.SA, self.CAT_INDEXES)
        self.anonymizer4.n_clusters = 10
        self.anonymizer4.k = 5
        self.anonymizer4.centroids = self._df.copy().loc[[0, 1, 2, 3, 4]]

    def tearDown(self):
        del self.anonymizer1
        del self.anonymizer2
        del self.anonymizer3
        del self.anonymizer4

    
    def test_find_best_cluster_gens(self):
        result = self.anonymizer1._find_best_cluster_gens(self._df)
        self.assertIsInstance(result, pd.core.frame.DataFrame)
        self.assertEqual(len(result), 10)

    
    def test_select_centroids_using_weighted_column(self):
        result = self.anonymizer1._select_centroids_using_weighted_column(
            self._df)
        self.assertIsInstance(result, pd.core.frame.DataFrame)
        self.assertEqual(len(result), 10)

    
    def test_random_sample_centroids(self):
        result = self.anonymizer1._random_sample_centroids(self._df)
        self.assertIsInstance(result, pd.core.frame.DataFrame)
        self.assertEqual(len(result), 10)

    
    def test_mark_less_n_kcentroids(self):
        self.anonymizer2.mark_less_n_kcentroids(dataframe='df')
        self.assertEqual(len(self.anonymizer2.less_centroids), 4)
        self.assertEqual(len(self.anonymizer2.k_centroids), 1)

    
    def test_cluster_data_loss(self):
        dataloss = self.anonymizer2._cluster_data_loss()
        self.assertIsInstance(dataloss, pd.core.frame.DataFrame)
        self.assertEqual(len(dataloss), 4)
        self.assertEqual(len(dataloss.columns), 5)

    
    def test_anon_k_clusters(self):
        self.anonymizer2.anon_k_clusters()
        dataframe_QI = self.anonymizer2.df_second_copy[self.QI]
        self.assertEqual(len(dataframe_QI.drop_duplicates()), 5)

    
    def test_mark_less_clusters_to_kclusters(self):
        self.anonymizer2.cluster_distances = self.anonymizer2._cluster_data_loss()
        self.anonymizer3.cluster_distances = self.anonymizer3._cluster_data_loss()
        self.anonymizer4.cluster_distances = self.anonymizer4._cluster_data_loss()
        self.anonymizer2.mark_less_clusters_to_kclusters()
        self.anonymizer3.mark_less_clusters_to_kclusters()
        self.anonymizer4.mark_less_clusters_to_kclusters()
        self.assertEqual(
            len(self.anonymizer2.df_second_copy['cluster_number'].unique()), 1)
        self.assertEqual(
            len(self.anonymizer3.df_second_copy['cluster_number'].unique()), 2)
        self.assertEqual(
            len(self.anonymizer4.df_second_copy['cluster_number'].unique()), 3)

    
    def test_mark_less_clusters_to_close_clusters(self):
        self.anonymizer2.cluster_distances = self.anonymizer2._cluster_data_loss()
        self.anonymizer3.cluster_distances = self.anonymizer3._cluster_data_loss()
        self.anonymizer4.cluster_distances = self.anonymizer4._cluster_data_loss()
        self.anonymizer2.mark_less_clusters_to_close_clusters()
        self.anonymizer3.mark_less_clusters_to_close_clusters()
        self.anonymizer4.mark_less_clusters_to_close_clusters()
        self.assertTrue(
            len(self.anonymizer2.df_second_copy['cluster_number'].unique()) == 1)
        self.assertTrue(
            len(self.anonymizer3.df_second_copy['cluster_number'].unique()) == 2)
        self.assertTrue(
            len(self.anonymizer4.df_second_copy['cluster_number'].unique()) == 3)

    
    def test_mark_clusters(self):
        self.anonymizer2.cluster_distances = self.anonymizer2._cluster_data_loss()
        self.anonymizer3.cluster_distances = self.anonymizer3._cluster_data_loss()
        self.anonymizer4.cluster_distances = self.anonymizer4._cluster_data_loss()
        self.anonymizer2._mark_clusters()
        self.anonymizer3._mark_clusters()
        self.anonymizer4._mark_clusters()
        self.assertTrue(
            len(self.anonymizer2.df_second_copy['cluster_number'].unique()) == 1)
        self.assertTrue(
            len(self.anonymizer3.df_second_copy['cluster_number'].unique()) == 2)
        self.assertTrue(
            len(self.anonymizer4.df_second_copy['cluster_number'].unique()) == 3)

    
    def test_make_anonymize(self):
        self.anonymizer2.cluster_distances = self.anonymizer2._cluster_data_loss()
        self.anonymizer3.cluster_distances = self.anonymizer3._cluster_data_loss()
        self.anonymizer4.cluster_distances = self.anonymizer4._cluster_data_loss()
        self.anonymizer2._make_anonymize()
        self.anonymizer3._make_anonymize()
        self.anonymizer4._make_anonymize()
        self.assertTrue(
            len(self.anonymizer2.df_second_copy['cluster_number'].unique()) == 1)
        self.assertTrue(
            len(self.anonymizer3.df_second_copy['cluster_number'].unique()) == 2)
        self.assertTrue(
            len(self.anonymizer4.df_second_copy['cluster_number'].unique()) == 3)


class TestAnonymizationClass(unittest.TestCase):
    def setUp(self):
        self._df = pd.read_csv('data/reduced_adult.csv', index_col=False).dropna().reset_index(drop=True)
        self.QI = ['age', 'workclass', 'education_num', 'matrital_status',
                   'occupation', 'race', 'sex', 'native_country']
        self.SA = ['class']
        self.CAT_INDEXES = [1, 3, 4, 5, 6, 7]

    def tearDown(self):
        pass

    def test_anonymize(self):
        _df = self._df
        QI = self.QI
        cat_index = self.CAT_INDEXES
        SA = self.SA
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='kmode', k=10, center_type='hung', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= 10).all(axis=None))
        
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='kmode', k=5, center_type='cao', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= 5).all(axis=None))

        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='kmode', k=10, center_type='random', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= 10).all(axis=None))

        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='kmode', k=5, center_type='hung', return_mode='equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= 5).all(axis=None))

        k = 10
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='kmode', k=k, center_type='cao', return_mode='equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k = 5
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='kmode', k=k, center_type='random', return_mode='equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k = 10
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=10, center_type='fbcg', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= 10).all(axis=None))

        k = 5
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=5, center_type='rsc', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k = 10
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=10, center_type='random', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k = 5
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=5, center_type='fbcg', return_mode='equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))


        k = 10
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=10, center_type='rsc', return_mode='equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k = 5
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=5, center_type='random', return_mode='equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))



class SecondTestAnonymizationClass(unittest.TestCase):
    def setUp(self):
        self._df = pd.read_csv('data/reduced_adult.csv', index_col=False).dropna().reset_index(drop=True)
        self.QI = ['age', 'education_num']
        self.SA = ['class', 'workclass']
        self.CAT_INDEXES = []

    def tearDown(self):
        pass


    def test_anonymize(self):
        _df = self._df
        QI = self.QI
        cat_index = self.CAT_INDEXES
        SA = self.SA

        k = 10
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='kmode', k=10, center_type='hung', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))


        k = 5
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='kmode', k=5, center_type='cao', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k = 10
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='kmode', k=10, center_type='random', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k = 5
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='kmode', k=5, center_type='hung', return_mode='equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k = 10
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='kmode', k=10, center_type='cao', return_mode='equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k = 5
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='kmode', k=5, center_type='random', return_mode='equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k = 10
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=10, center_type='fbcg', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k = 5
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=5, center_type='rsc', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k = 10
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=10, center_type='random', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k = 5
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=5, center_type='fbcg', return_mode='equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k = 10
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=10, center_type='rsc', return_mode='equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k = 5
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=5, center_type='random', return_mode='equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))



class ThirdTestAnonymizationClass(unittest.TestCase):
    def setUp(self):
        self._df = pd.read_csv('data/reduced_adult.csv', index_col=False).dropna().reset_index(drop=True)
        self.QI = ['matrital_status', 'occupation',
                   'race', 'sex', 'native_country']
        self.SA = ['age', 'education_num']
        self.CAT_INDEXES = [0, 1, 2, 3, 4]


    def test_anonymize(self):

        _df = self._df
        QI = self.QI
        cat_index = self.CAT_INDEXES
        SA = self.SA

        k=10
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='kmode', k=10, center_type='hung', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k=5
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='kmode', k=5, center_type='cao', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k=10
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='kmode', k=10, center_type='random', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k=5
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='kmode', k=5, center_type='hung', return_mode='equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k=10
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='kmode', k=10, center_type='cao', return_mode='equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k=5
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='kmode', k=5, center_type='random', return_mode='equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k=10
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=10, center_type='fbcg', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k=5
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=5, center_type='rsc', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k=10
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=10, center_type='random', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k=5
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=5, center_type='fbcg', return_mode='equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k=10
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=10, center_type='rsc', return_mode='equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= k).all(axis=None))

        k=5
        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=5, center_type='random', return_mode='equal')




class FourthTestAnonymizationClass(unittest.TestCase):
    def setUp(self):
        self._df = pd.read_csv('data/reduced_adult.csv', index_col=False).dropna().reset_index(drop=True)
        self.QI = ['education_num', 'matrital_status',
                   'occupation', 'race', 'sex', 'native_country']
        self.SA = ['age']
        self.CAT_INDEXES = [0, 1, 2, 3, 4, 5]


    def tearDown(self):
        pass


    def test_anonymize(self):
        _df = self._df
        QI = self.QI
        cat_index = self.CAT_INDEXES
        SA = self.SA

        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='kmode', k=10, center_type='hung', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= 10).all(axis=None))

        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        self.dataframe = anonymizer.anonymize(mode='kmode', k=5, center_type='cao', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= 5).all(axis=None))

        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='kmode', k=10, center_type='random', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= 10).all(axis=None))


        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='kmode', k=5, center_type='hung', return_mode='equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= 5).all(axis=None))


        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='kmode', k=10, center_type='cao', return_mode='equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= 10).all(axis=None))

        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='kmode', k=5, center_type='random', return_mode='equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= 5).all(axis=None))

        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=10, center_type='fbcg', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= 10).all(axis=None))

        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=5, center_type='rsc', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= 5).all(axis=None))

        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=10, center_type='random', return_mode='Not_equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= 10).all(axis=None))

        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=5, center_type='fbcg', return_mode='equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= 5).all(axis=None))

        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=10, center_type='rsc', return_mode='equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= 10).all(axis=None))

        anonymizer = Kanonymizer(_df.copy(), QI, SA, cat_index)
        dataframe = anonymizer.anonymize(mode='', k=5, center_type='random', return_mode='equal')
        self.assertTrue((dataframe[self.QI].groupby(self.QI, as_index=False).size()['size'] >= 5).all(axis=None))


class LDiverseTestClass(unittest.TestCase):

    def setUp(self):
        self._df = pd.read_csv('data/reduced_adult.csv', index_col=False).dropna().reset_index(drop=True)
        self.QI = ['age', 'workclass', 'education_num',
                   'matrital_status', 'occupation', 'sex', 'native_country']
        self.SA = ['class', 'race']
        self.CAT_INDEXES = [1, 3, 4, 5, 6]
        _df = self._df
        QI = self.QI
        cat_index = self.CAT_INDEXES
        SA = self.SA
        self.anonymizer1 = Kanonymizer(_df.copy(), QI, SA, cat_index)
        self.dataframe1 = self.anonymizer1.anonymize(mode='', k=10, center_type='fbcg', return_mode='Not_equal')
        self.ldiverser1 = LDiversityAnonymizer(self.dataframe1, QI, SA)
        self.dataframe2 = self.ldiverser1.anonymize(l=4)
        self.dataframe3 = self.ldiverser1.anonymize(l=3)
        self.dataframe4 = self.ldiverser1.anonymize(l=5)

    def tearDown(self):
        del self._df
        del self.QI
        del self.SA
        del self.CAT_INDEXES
        del self.anonymizer1
        del self.dataframe1

    def test_l_anonymizer(self):
        cluster_groups = self.dataframe2.groupby(self.QI)
        filter_frame = cluster_groups.filter(
            lambda x: len(x['class'].unique()) < 4)
        self.assertTrue(filter_frame.empty)
        cluster_groups = self.dataframe2.groupby(self.QI)
        filter_frame = cluster_groups.filter(
            lambda x: len(x['race'].unique()) < 4)
        self.assertTrue(filter_frame.empty)
        cluster_groups = self.dataframe3.groupby(self.QI)
        filter_frame = cluster_groups.filter(
            lambda x: len(x['class'].unique()) < 3)
        self.assertTrue(filter_frame.empty)
        cluster_groups = self.dataframe3.groupby(self.QI)
        filter_frame = cluster_groups.filter(
            lambda x: len(x['race'].unique()) < 3)
        self.assertTrue(filter_frame.empty)
        cluster_groups = self.dataframe4.groupby(self.QI)
        filter_frame = cluster_groups.filter(
            lambda x: len(x['class'].unique()) < 5)
        self.assertTrue(filter_frame.empty)
        cluster_groups = self.dataframe4.groupby(self.QI)
        filter_frame = cluster_groups.filter(
            lambda x: len(x['race'].unique()) < 5)
        self.assertTrue(filter_frame.empty)


class TClosenessTestClass(unittest.TestCase):

    def setUp(self):
        self._df = pd.read_csv('data/reduced_adult.csv', index_col=False).dropna().reset_index(drop=True)
        self.QI = ['age', 'workclass', 'education_num',
                   'matrital_status', 'occupation', 'sex', 'native_country']
        self.SA = ['class', 'race']
        self.CAT_INDEXES = [1, 3, 4, 5, 6]
        _df = self._df
        QI = self.QI
        cat_index = self.CAT_INDEXES
        SA = self.SA
        self.anonymizer1 = Kanonymizer(_df.copy(), QI, SA, cat_index)
        self.dataframe1 = self.anonymizer1.anonymize(mode='', k=10, center_type='fbcg', return_mode='Not_equal')
        self.ldiverser1 = LDiversityAnonymizer(self.dataframe1, QI, SA)
        self.dataframe2 = self.ldiverser1.anonymize(l=4)
        self.tcloseness1 = TClosenessAnonymizer(self.dataframe2, QI, SA)
        self.dataframe2 = self.tcloseness1.anonymize(t=0.2)

    def tearDown(self):
        del self._df
        del self.QI
        del self.SA
        del self.CAT_INDEXES
        del self.anonymizer1
        del self.dataframe1
        del self.dataframe2

    def test_t_anonymizer(self):
        class_distribution = self.dataframe2['class'].value_counts()
        cluster_groups = self.dataframe2.groupby(self.QI)
        filter_frame = cluster_groups.filter(lambda x: x['class'].value_counts()[
                                             "<=50K"]/len(x) + 0.2 < class_distribution["<=50K"])
        self.assertTrue(filter_frame.empty)
        cluster_groups = self.dataframe2.groupby(self.QI)
        filter_frame = cluster_groups.filter(lambda x: x['class'].value_counts()[
                                             ">50K"]/len(x) + 0.2 < class_distribution["<=50K"])
        self.assertTrue(filter_frame.empty)


if __name__ == '__main__':
    unittest.main()

