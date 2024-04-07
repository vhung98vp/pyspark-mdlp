import time
from pyspark.sql import SparkSession
from pyspark.ml.linalg import DenseVector, SparseVector
from pyspark import StorageLevel
import logging
from .InitialThresholdsFinder import InitialThresholdsFinder
from .ManyValuesThresholdFinder import ManyValuesThresholdFinder
from .FewValuesThresholdFinder import FewValuesThresholdFinder
from .DiscretizerModel import DiscretizerModel


class MDLPDiscretizer:
    """
    * Entropy minimization discretizer based on Minimum Description Length Principle (MDLP) proposed by Fayyad and Irani in 1993 [1].
    * [1] Fayyad, U., & Irani, K. (1993). "Multi-interval discretization of continuous-valued attributes for classification learning."
    * Note: Approximate version may generate some non-boundary points when processing limits in partitions.
    """

    DEFAULT_STOPPING_CRITERION = 0
    """
    The original paper suggested 0 for the stopping criterion, but smaller values like -1e-3 yield more splits
    """

    DEFAULT_MIN_BIN_PERCENTAGE = 0
    """
    * Don't allow less that this percent of the total number of records in a single bin.
    * The default is 0, meaning that its OK to have as few as a single record in a bin.
    * A value of 0.1 means that no fewer than 0.1% of records will be in a single bin.
    """

    DEFAULT_MAX_BY_PART = 100000
    """
    * Maximum number of elements in a partition.
    * If this number gets too big, you could run out of memory depending on resources.
    """

    def __init__(self, data, stopping_criterion=DEFAULT_STOPPING_CRITERION,
                 max_by_part=DEFAULT_MAX_BY_PART, min_bin_percentage=DEFAULT_MIN_BIN_PERCENTAGE,
                 approximate=True):
        """      
        * @param data RDD of LabeledPoint
        * @param stopping_criterion (optional) used to determine when to stop recursive splitting
        * @param max_by_part (optional) used to determine maximum number of elements in a partition
        * @param min_bin_percentage (optional) minimum percent of total dataset allowed in a single bin.
        * @param approximate If true, boundary points are computed faster but in an approximate manner.
        """

        self.data = data
        self.stopping_criterion = stopping_criterion
        self.max_by_part = max_by_part
        self.min_bin_percentage = min_bin_percentage
        self.approximate = approximate
        # Dictionary maps labels (classes) to integer indices
        self.labels2int = self.data.select('label').distinct().rdd.map(lambda x: x[0]).zipWithIndex().collectAsMap()
        self.n_labels = len(self.labels2int)

    def process_continuous_attributes(self, cont_indices, n_features):
        """
        * Get information about the attributes before performing discretization.
        * @param cont_indices Indexes to discretize (if not specified, they are calculated).
        * @param n_features Total number of input features.
        * @return Indexes of continuous features.
        """
        if cont_indices is not None:
            intersect = set(range(n_features)).intersection(cont_indices)
            if len(intersect) != len(cont_indices):
                raise ValueError("Invalid continuous feature indices provided")
            return cont_indices
        else:
            return list(range(n_features))
    
    def get_sorted_distinct_values(self, b_class_distrib, feature_values):
        """
        * Group elements by feature and point (get distinct points).
        * Since values like (0, Float.NaN) are not considered unique when calling reduceByKey, use the serialized version of the tuple.
        * @return sorted list of unique feature values
        """

        non_zeros = (feature_values
            .map(lambda y: ((y[0][0], y[0][1]), y[1]))
            .reduceByKey(lambda x, y: [a + b for a, b in zip(x, y)]))
        
        zeros = (non_zeros
            .map(lambda x: (x[0][0], x[1])) 
            .reduceByKey(lambda x, y: [a + b for a, b in zip(x, y)]) 
            .map(lambda x: ((x[0], 0.0), [b_class_distrib.value.get(i, 0) - v for i, v in enumerate(x[1])])) 
            #.map(lambda x: ((x[0], 0.0), x[1])) instead
            .filter(lambda x: sum(x[1]) > 0) )
        distinct_values = non_zeros.union(zeros)
        return distinct_values.sortByKey()
    
    def initial_thresholds(self, points, n_features):
        """
        * Computes the initial candidate points by feature.
        * @param points RDD with distinct points by feature ((feature, point), class values).
        * @param n_features the number of continuous features to bin
        * @return RDD of candidate points.
        """
        finder = InitialThresholdsFinder()
        if self.approximate:
            return finder.find_fast_initial_thresholds(points, self.n_labels, self.max_by_part)
        else:
            return finder.find_initial_thresholds(points, n_features, self.n_labels, self.max_by_part)   

    def find_all_thresholds(self, initial_candidates, sc, max_bins):
        """
        * Divide RDD into two categories according to the number of points by feature.
        * @return find threshold for both sorts of attributes - those with many values, and those with few.
        """
        big_indexes = initial_candidates.countByKey()
        big_thresholds = self.find_big_thresholds(initial_candidates, big_indexes)

        small_thresholds = initial_candidates \
            .filter(lambda x: x[0] not in big_indexes) \
            .groupByKey() \
            .mapValues(list) \
            .mapValues(lambda x: self.find_small_thresholds(x, max_bins))

        all_thresholds = small_thresholds.union(sc.parallelize(big_thresholds.items())).collect()

        return all_thresholds

    def find_big_thresholds(self, initial_candidates, big_indexes):
        """
        * Features with too many unique points must be processed iteratively (rare condition)
        * @return the splits for features with more values than will fit in a partition.
        """
        big_thresholds = {}
        big_thresholds_finder = ManyValuesThresholdFinder(self.n_labels, self.stopping_criterion,
                                                          self.max_by_part, self.min_bin_percentage, self.max_by_part)
        for k in big_indexes:
            cands = initial_candidates.filter(lambda x: x[0] == k).values().sortByKey()
            big_thresholds[k] = big_thresholds_finder.find_thresholds(cands)
        return big_thresholds

    def find_small_thresholds(self, points, max_bins):
        """
        * The features with a small number of points can be processed in a parallel way
        * @return the splits for features with few values
        """
        small_thresholds_finder = FewValuesThresholdFinder(self.n_labels, self.stopping_criterion, max_bins,
                                                            self.min_bin_percentage)
        return small_thresholds_finder.find_thresholds(sorted(points, key=lambda x: x[0]))

    def build_model_from_thresholds(self, n_features, continuous_vars, all_thresholds):
        """
        * @return the discretizer model that can be used to bin data
        """
        thresholds = [None] * n_features
        for idx in continuous_vars:
            thresholds[idx] = [float('-inf'), float('inf')]
        for k, vth in all_thresholds:
            thresholds[k] = vth
        logging.info("Number of features with thresholds computed: {}".format(len(all_thresholds)))
        logging.debug("thresholds = {}".format(";\n".join([", ".join(map(str, t)) for t in thresholds])))
        return DiscretizerModel(thresholds)
    
    def train(self, cont_feat=None, max_bins=15):
        """
        * Run the entropy minimization discretizer on input data. (train model)
        * @param cont_feat Indices to discretize (if not specified, the algorithm tries to figure it out).
        * @param max_bins Maximum number of thresholds per feature.
        * @return A discretization model with the thresholds by feature.
        """

        if self.data.storageLevel == StorageLevel.NONE:
            print("The input data is not directly cached, which may hurt performance if its parent RDDs are also uncached.")

        if self.data.filter(self.data["label"].isNull()).count() > 0:
            raise ValueError("Some NaN values have been found in the label Column. This problem must be fixed before continuing with discretization.")

        sc = self.data.rdd.context

        b_labels2int = sc.broadcast(self.labels2int)
        class_distrib = self.data.rdd.map(lambda x: b_labels2int.value[x.label]).countByValue()
        b_class_distrib = sc.broadcast(class_distrib)
        print(type(b_labels2int))

        n_features = len(self.data.first().features)

        continuous_vars = self.process_continuous_attributes(cont_feat, n_features)
        print("Number of continuous attributes:", len(continuous_vars))
        print("Total number of attributes:", n_features)

        if not continuous_vars:
            print("Discretization aborted. No continuous attributes in the dataset!")

        sc.broadcast(continuous_vars)

        def map_to_feature_values(lp): 
            label = lp.label    # Labeled Point
            c = [0] * self.n_labels
            c[b_labels2int.value(label)] = 1
            if isinstance(lp.features, DenseVector):
                return [((i, v), c) for i, v in enumerate(lp.features.toArray())]
            elif isinstance(lp.features, SparseVector):
                return [((i, v), c) for i, v in zip(lp.features.indices, lp.features.toArray())]

        feature_values = self.data.rdd.flatMap(map_to_feature_values)

        print(type(feature_values))        
        feature_values.foreach(lambda x: print(x))
        
        sorted_values = self.get_sorted_distinct_values(b_class_distrib, feature_values)

        arr = [False] * n_features
        for idx in continuous_vars:
            arr[idx] = True
        b_arr = sc.broadcast(arr)

        # Get only boundary points from the whole set of distinct values
        start1 = time.time()
        initial_candidates = (sorted_values 
            .map(lambda x: (x[0][0], (x[0][1], x[1]))) 
            .filter(lambda x: b_arr.value[x[0]]) )
            #.cache())
        print("Done finding initial thresholds in", time.time() - start1)

        start2 = time.time()
        all_thresholds = self.find_all_thresholds(initial_candidates, sc, max_bins)
        print("Done finding MDLP thresholds in", time.time() - start2, "Now returning model")

        return self.build_model_from_thresholds(n_features, continuous_vars, all_thresholds)