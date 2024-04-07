from pyspark.ml.linalg import SparseVector, DenseVector, Vector, Vectors
from pyspark import RDD
from typing import List
import bisect

class DiscretizerModel:
    """
    Generic discretizer model that transforms data given a list of thresholds by feature.
    """
    
    def __init__(self, thresholds: List[List[float]]):
        """
        @param thresholds Thresholds defined for each feature (must be sorted).
        """
        self.thresholds = thresholds

    def transform(self, data: Vector) -> Vector:
        """
        * Discretizes values in a given dataset using thresholds.
        * @param data A single continuous-valued vector.
        * @return A resulting vector with its values discretized (from 1 to n).
        """

        if isinstance(data, SparseVector):
            new_values = [self.assign_discrete_value(float(data.values[i]), self.thresholds[data.indices[i]]) for i in range(len(data.indices))]
            return Vectors.sparse(data.size, data.indices, new_values)
        elif isinstance(data, DenseVector):
            new_values = [self.assign_discrete_value(float(data[i]), self.thresholds[i]) for i in range(len(data))]
            return Vectors.dense(new_values)
        else:
            raise TypeError("Data type not supported for transformation.")

    def transform_rdd(self, data_rdd: RDD):
        """
        * Discretizes values in a given dataset using thresholds.
        * @param data_RDD RDD with continuous-valued vectors.
        * @return RDD with discretized data (from 1 to n).
        """

        bc_thresholds = data_rdd.context.broadcast(self.thresholds)

        def map_vector(v):
            if isinstance(v, SparseVector):
                new_values = [self.assign_discrete_value(v.values[i], bc_thresholds.value[v.indices[i]]) for i in range(len(v.indices))]
                return Vectors.sparse(v.size, v.indices, new_values)
            elif isinstance(v, DenseVector):
                new_values = [self.assign_discrete_value(v[i], bc_thresholds.value[i]) for i in range(len(v))]
                return Vectors.dense(new_values)
            else:
                raise TypeError("Data type not supported for transformation.")

        return data_rdd.map(map_vector)

    @staticmethod
    def assign_discrete_value(value: float, thresholds: List[float]) -> float:
        """
        * Discretizes a value with a set of intervals.
        * @param value Value to be discretized.
        * @param thresholds Thresholds used to assign a discrete value
        """
        
        if thresholds:
            return float(bisect.bisect_left(thresholds, value))
        else:
            return value