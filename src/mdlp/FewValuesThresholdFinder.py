from typing import List, Tuple, Optional
from collections import deque
from .ThresholdFinder import ThresholdFinder
from .BucketInfo import BucketInfo

class FewValuesThresholdFinder(ThresholdFinder):
    """
    Use when the feature to discretize has relatively few unique values.
    """

    def __init__(self, n_labels: int, stopping_criterion: float, max_bins: int, min_bin_weight: int):
        """
        * @param n_labels the number of class labels
        * @param max_bins Maximum number of cut points to select.
        * @param min_bin_weight don't generate bins with fewer than this many records.
        * @param stopping_criterion influences when to stop recursive splits
        """
        self.n_labels = n_labels
        self.stopping_criterion = stopping_criterion
        self.max_bins = max_bins
        self.min_bin_weight = min_bin_weight

    def find_thresholds(self, candidates: List[Tuple[float, List[int]]]) -> List[float]:
        """
        * Evaluates boundary points and selects the most relevant candidates (sequential version).
        * Here, the evaluation is bounded by partition as the number of points is small enough.
        * @param candidates RDD of candidates points (point, class histogram).
        * @return Sequence of threshold values.
        """
        stack = deque([((float("-inf"), float("inf")), None)])
        result = [float("-inf")]

        while stack and len(result) < self.max_bins:
            bounds, last_threshold = stack.popleft()
            new_candidates = list(filter(lambda x: bounds[0] < x[0] < bounds[1], candidates))
            if new_candidates:
                thresholds = self.eval_thresholds(new_candidates, last_threshold, self.n_labels)
                if thresholds:
                    result.append(thresholds)
                    stack.append(((bounds[0], thresholds), thresholds))
                    stack.append(((thresholds, bounds[1]), thresholds))

        return sorted(result) + [float("inf")]

    def eval_thresholds(self, candidates: List[Tuple[float, List[int]]], last_selected: Optional[float], n_labels: int) -> List[float]:
        """
        * Compute entropy minimization for candidate points in a range, and select the best one according to MDLP criterion (sequential version).
        * @param candidates Array of candidate points (point, class histogram).
        * @param last_selected last selected threshold.
        * @param n_labels Number of classes.
        * @return The minimum-entropy cut point.
        """

        # Compute the accumulated frequencies (both left and right) by label
        totals = [sum(freqs) for freqs in zip(*[freq for _, freq in candidates])]

        left_accum = [0] * n_labels
        entropy_freqs = []

        for cand, freq in candidates:
            left_accum = [x + y for x, y in zip(left_accum, freq)]
            right_total = [x - y for x, y in zip(totals, left_accum)]
            entropy_freqs.append((cand, freq, left_accum, right_total))

        bucket_info = BucketInfo(totals)

        # Select the best threshold according to MDLP
        def filter_candidates(cand_freqs):
            cand, _, left_freqs, right_freqs = cand_freqs
            duplicate = cand == last_selected if last_selected is not None else False
            if not duplicate:
                criterion_value, weighted_hs, left_sum, right_sum = self.calc_criterion_value(bucket_info, left_freqs, right_freqs)
                criterion = criterion_value > self.stopping_criterion and left_sum > self.min_bin_weight and right_sum > self.min_bin_weight
                if criterion:
                    return (weighted_hs, cand)
            return []
        
        final_candidates = list(filter(None, map(filter_candidates, entropy_freqs)))       
        print(f"Final cands with few vals: {final_candidates}")
        
        return min(final_candidates, key=lambda x: x[0])[1] if final_candidates else []