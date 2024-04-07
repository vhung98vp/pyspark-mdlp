from typing import List, Tuple
from pyspark import RDD
from math import ceil

class InitialThresholdsFinder:
    def __init__(self):
        pass

    @staticmethod
    def is_boundary(f1: List[int], f2: List[int]) -> bool:
        """
        * @return true if f1 and f2 define a boundary.
        * It is a boundary if there is more than one class label present when the two are combined.
        """
        return sum(1 for a, b in zip(f1, f2) if a + b != 0) > 1

    @staticmethod
    def midpoint(x1: float, x2: float) -> float:
        """
        If one of the unique values is NaN, use the other one, otherwise take the midpoint.
        """

        if x1 != x1:
            return x2
        elif x2 != x2:
            return x1
        else:
            return (x1 + x2) / 2.0

    def find_initial_thresholds(self, points: RDD[Tuple[Tuple[int, float], List[int]]],
                                n_features: int, n_labels: int, max_by_part: int) -> RDD[Tuple[Tuple[int, float], List[int]]]:
        """
        * Computes the initial candidate cut points by feature.
        * @param points RDD with distinct points by feature ((feature, point), class values)
        * @param n_features expected number of features
        * @param n_labels number of class labels
        * @param max_by_part maximum number of values allowed in a partition
        * @return RDD of candidate points.
        """
        feature_info = self.create_feature_info_list(points, max_by_part, n_features)
        total_partitions = feature_info[-1][-1] + feature_info[-1][-2]

        # This custom partitioner will partition by feature and subdivide features into smaller partitions if large
        class FeaturePartitioner():
            def getPartition(self, key):
                feature_idx, cut, sort_idx = key
                _, _, sum_values_before, partition_size, _, sum_previous_num_parts = feature_info[feature_idx]
                part_key = sum_previous_num_parts + max(0, sort_idx - sum_values_before - 1) // partition_size
                return part_key

            def numPartitions(self):
                return total_partitions

        points_with_index = points.zipWithIndex().map(lambda v: ((v[0][0][0], v[0][0][1], v[1]), v[0][1]))
        partitioned_points = points_with_index.partitionBy(FeaturePartitioner())

        def map_partitions_with_index(index, it):
            if it:
                last_feature_idx, last_x, _ = next(it)[0]
                result = []
                accum_freqs = [0] * n_labels
                
                for (f_idx, x, _), freqs in it:
                    if self.is_boundary(freqs, last_freqs):
                        result.append(((last_feature_idx, self.midpoint(x, last_x)), accum_freqs.copy()))
                        accum_freqs = [0] * n_labels

                    last_x = x
                    last_feature_idx = f_idx
                    last_freqs = freqs
                    accum_freqs = [accum_freq + freq for accum_freq, freq in zip(accum_freqs, freqs)]

                result.append(((last_feature_idx, last_x), accum_freqs.copy()))
                return result[::-1]  # Reverse the list and return as an iterator
            else:
                return iter([])  # Empty iterator if the partition is empty

        return partitioned_points.mapPartitionsWithIndex(map_partitions_with_index)
    
    def find_fast_initial_thresholds(self, sorted_values: RDD, n_labels: int, max_by_part: int):
        """
        * Computes the initial candidate cut points by feature. This is a non-deterministic, but and faster version.
        * This version may generate some non-boundary points when processing limits in partitions.
        * This approximate solution may slightly affect the final set of cut_points, which will provoke some unit tests to fail. 
          It should not be relevant in large scenarios, where performance is more valuable.
        * @param sortedValues RDD with distinct points by feature ((feature, point), class values).
        * @param n_labels number of class labels
        * @param max_by_part maximum number of values allowed in a partition
        * @return RDD of candidate points.
        """
        num_partitions = sorted_values.getNumPartitions()
        sc = sorted_values.context

        # Get the first elements by partition for the boundary points evaluation
        first_elements = sc.runJob(sorted_values, lambda it: next(it, None))

        bc_firsts = sc.broadcast(first_elements)

        def map_partitions_with_index(index, it):
            if it:
                last_feature_idx, last_x = next(it)[0]
                result = []
                accum_freqs = [0] * n_labels

                for ((feature_idx, x), freqs) in it:
                    if feature_idx != last_feature_idx:
                        result.append(((last_feature_idx, last_x), accum_freqs.copy()))
                        accum_freqs = [0] * n_labels
                    elif self.is_boundary(freqs, last_freqs):
                        result.append(((last_feature_idx, (x + last_x) / 2), accum_freqs.copy()))
                        accum_freqs = [0] * n_labels

                    last_feature_idx = feature_idx
                    last_x = x
                    last_freqs = freqs
                    accum_freqs = [accum_freq + freq for accum_freq, freq in zip(accum_freqs, freqs)]

                last_point = last_x if index == (num_partitions - 1) else bc_firsts.value[index + 1][0][1]
                result.append(((last_feature_idx, last_point), accum_freqs.copy()))
                return result[::-1]  # Reverse the list and return as an iterator
            else:
                return iter([])  # Empty iterator if the partition is empty

        return sorted_values.mapPartitionsWithIndex(map_partitions_with_index)

    def create_feature_info_list(self, points: RDD[Tuple[Tuple[int, float], List[int]]],
                                  max_by_part: int, n_features: int) -> List[Tuple[int, int, int, int, int, int]]:
        """
        * @param points all unique points
        * @param max_by_part maximum number of values in a partition
        * @param n_features expected number of features.
        * @return a list of info for each partition. The values in the info tuple are:
        *  (feature_idx, num_unique_vals, sum_vals_before_first, part_size, num_partitions_for_feature, sum_previous_num_parts)
        """

        # Find the number of points in each partition, ordered by feature_idx
        counts_by_feature_idx = points.map(lambda x: x[0][0]).countByValue().items()

        # If there are features not represented, add them manually.
        # This can happen if there are some features with all 0 values (rare, but we need to handle it).
        represented_features = set(map(lambda x: x[0], counts_by_feature_idx))
        if len(counts_by_feature_idx) < n_features:
            for i in range(n_features):
                if i not in represented_features:
                    counts_by_feature_idx.append((i, 1))

        counts_by_feature_idx.sort(key=lambda x: x[0])

        last_count = 0
        total_sum = 0
        sum_previous_num_parts = 0
        feature_info = []

        for feature_idx, count in counts_by_feature_idx:
            part_size = ceil(count / ceil(count / max_by_part))
            num_parts = ceil(count / part_size)
            info = (feature_idx, count, total_sum + last_count, part_size, num_parts, sum_previous_num_parts)
            total_sum += last_count
            sum_previous_num_parts += num_parts
            last_count = count
            feature_info.append(info)

        return feature_info