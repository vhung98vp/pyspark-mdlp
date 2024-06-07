import math
from typing import List, Tuple
from pyspark.broadcast import Broadcast
from .DiscretizationUtils import DiscretizationUtils
from .BucketInfo import BucketInfo

class ThresholdFinder:
    @staticmethod
    def calc_criterion_value(bucket_info: BucketInfo, left_freqs: List[int], right_freqs: List[int]) -> Tuple[float, float, int, int]:
        """
        * @param bucket_info info about the parent bucket
        * @param left_freqs frequencies to the left
        * @param right_freqs frequencies to the right
        * @return the MDLP criterion value, the weighted entropy value, sum of leftFreqs, and sum of rightFreqs
        """

        print(f'Bucket: {bucket_info}')
        print(f'freqs: {left_freqs} <<>> {right_freqs}')
        k1 = len([count for count in left_freqs if count != 0])     # number of non-zero frequencies in left_freqs
        s1 = sum(left_freqs) if k1 > 0 else 0                       # sum of frequencies in left_freqs
        hs1 = DiscretizationUtils.entropy(left_freqs, s1)           # entropy of left_freqs
        k2 = len([count for count in right_freqs if count != 0])    # number of non-zero frequencies in right_freqs
        s2 = sum(right_freqs) if k2 > 0 else 0                      # sum of frequencies in right_freqs
        hs2 = DiscretizationUtils.entropy(right_freqs, s2)          # entropy of right_freqs
        weighted_hs = (s1 * hs1 + s2 * hs2) / bucket_info.s         # weighted entropy value
        gain = bucket_info.hs - weighted_hs                         # information gain
        delta = DiscretizationUtils.log2(math.pow(3, bucket_info.k) - 2) - (bucket_info.k * bucket_info.hs - k1 * hs1 - k2 * hs2)
        criterion_value = gain - (DiscretizationUtils.log2(bucket_info.s - 1) + delta) / bucket_info.s
        return criterion_value, weighted_hs, s1, s2

    @staticmethod
    def sum_by_column(a: List[List[int]], num_cols: int, initial_totals: List[int] = None) -> List[int]:
        """
        * @param a array of arrays to sum by column
        * @param num_cols number of columns to sum
        * @param initial_totals total to add by columns
        * @return 1D element-wise sum of the arrays passed in
        """
        
        total = initial_totals if initial_totals is not None else [0] * num_cols
        for row in a:
            for i in range(num_cols):
                total[i] += row[i]
        return total

    @staticmethod
    def sum_by_column_broadcast(a: Broadcast[List[List[int]]], num_rows: int, num_cols: int) -> List[int]:
        """
        * @param a array of arrays to sum by column
        * @param num_rows the number of rows to sum (from 0)
        * @param num_cols number of columns to sum
        * @return 1D element-wise sum of the arrays passed in
        """
        
        total = [0] * num_cols
        for row_idx in range(num_rows):
            row = a.value[row_idx]
            for i in range(num_cols):
                total[i] += row[i]
        return total