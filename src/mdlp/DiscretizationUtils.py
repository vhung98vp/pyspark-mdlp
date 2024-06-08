import math
from typing import List, Union

class DiscretizationUtils:

    @staticmethod
    def log2(x: Union[int, float]) -> float:
        """
        Log base 2 of x
        """
        return math.log(x) / math.log(2) if x > 0 else 0

    @staticmethod
    def entropy(frequencies: List[int], n: int) -> float:
        """
        * Entropy is a measure of disorder. The higher the value, the closer to a purely random distribution.
        * The MDLP algorithm tries to find splits that will minimize entropy.
        * @param frequencies sequence of integer frequencies.
        * @param n the sum of all the frequencies in the list.
        * @return the total entropy
        """
        return -sum([0 if q == 0 or n == 0 else q/n * DiscretizationUtils.log2(q/n) for q in frequencies])
        # return -sum(h + (0 if q == 0 else q/n * DiscretizationUtils.log2(q/n))
        #        for h, q in zip(frequencies, frequencies) if h > 0)