from typing import List
from .DiscretizationUtils import DiscretizationUtils

class BucketInfo:
    def __init__(self, totals: List[int]):
        self.totals = totals

    @property
    def s(self) -> int:
        """
        Number of elements in bucket
        """
        return sum(self.totals)

    @property
    def hs(self) -> float:
        """
        Calculated entropy for the bucket
        """
        return DiscretizationUtils.entropy(self.totals, self.s)

    @property
    def k(self) -> int:
        """
        Number of distinct classes in bucket
        """
        return len([count for count in self.totals if count != 0])