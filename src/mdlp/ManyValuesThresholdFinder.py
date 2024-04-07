from pyspark import RDD
from typing import List, Tuple
from collections import deque
from .ThresholdFinder import ThresholdFinder
from .BucketInfo import BucketInfo

class ManyValuesThresholdFinder(ThresholdFinder):
    """
    Use this version when the feature to discretize has more values that will fit in a partition (see max_by_part param).
    """

    def __init__(self, n_labels: int, stopping_criterion: float, max_bins: int, min_bin_weight: int, elements_by_part: int):
        """  
        * @param n_labels the number of class labels
        * @param stopping_criterion influences when to stop recursive splits
        * @param max_bins Maximum number of cut points to select
        * @param min_bin_weight don't generate bins with fewer than this many records
        * @param elements_by_part Maximum number of elements to evaluate in each partition
        """
        self.n_labels = n_labels
        self.stopping_criterion = stopping_criterion
        self.max_bins = max_bins
        self.min_bin_weight = min_bin_weight
        self.elements_by_part = elements_by_part

    def find_thresholds(self, candidates: RDD[Tuple[float, List[int]]]) -> List[float]:
        """
        * Evaluate boundary points and select the most relevant. This version is used when the number of candidates exceeds 
        the maximum size per partition (distributed version).
        * @param candidates RDD of candidates points (point, class histogram).
        * @return Sequence of threshold values.
        """

        stack = deque([((float("-inf"), float("inf")), None)])
        result = [float("-inf")]

        while stack and len(result) < self.max_bins:
            bounds, last_threshold = stack.popleft()
            new_candidates = candidates.filter(lambda x: bounds[0] < x[0] < bounds[1])
            if new_candidates:
                thresholds = self.eval_thresholds(new_candidates, last_threshold)
                if thresholds:
                    result.append(thresholds[0])
                    stack.append(((bounds[0], thresholds[0]), thresholds[0]))
                    stack.append(((thresholds[0], bounds[1]), thresholds[0]))

        return sorted(result) + [float("inf")]

    def eval_thresholds(self, candidates: RDD[Tuple[float, List[int]]], last_selected: float) -> List[float]:
        """
        * Compute entropy minimization for candidate points in a range, and select the best one according to the MDLP criterion (RDD version).
        * @param candidates RDD of candidate points (point, class histogram).
        * @param last_selected Last selected threshold.
        * @return The minimum-entropy candidate.
        """
        sc = candidates.context

        # Compute the accumulated frequencies by partition
        totals_by_part = candidates.mapPartitions(lambda it: [sum(freqs) for freqs in zip(*[x[1] for x in it])]).collect()
        totals = [sum(x) for x in zip(*totals_by_part)]

        # Compute the total frequency for all partitions
        bc_totals_by_part = sc.broadcast(totals_by_part)
        bc_totals = sc.broadcast(totals)

        def map_partitions_with_index(slice_idx, it):
            left_total = [sum(x) for x in zip(*bc_totals_by_part.value[:slice_idx])]
            entropy_freqs = []
            for cand, freqs in it:
                left_total = [x + y for x, y in zip(left_total, freqs)]
                right_total = [x - y for x, y in zip(bc_totals.value, left_total)]
                entropy_freqs.append((cand, freqs, left_total, right_total))
            return entropy_freqs

        result = candidates.mapPartitionsWithIndex(map_partitions_with_index)

        bucket_info = BucketInfo(totals)

        def flat_map(cand_freqs):
            cand, freqs, left_freqs, right_freqs = cand_freqs
            duplicate = cand == last_selected if last_selected is not None else False
            if duplicate:
                return []
            else:
                criterion_value, weighted_hs, left_sum, right_sum = self.calc_criterion_value(bucket_info, left_freqs, right_freqs)
                criterion = criterion_value > self.stopping_criterion and left_sum > self.min_bin_weight and right_sum > self.min_bin_weight
                if criterion:
                    return [(weighted_hs, cand)]
                else:
                    return []

        final_candidates = result.flatMap(flat_map)
        # Select the candidate with the minimum weighted_hs
        return min(final_candidates, key=lambda x: x[0])[1] if final_candidates else []
