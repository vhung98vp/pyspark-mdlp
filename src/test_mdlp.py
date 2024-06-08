import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.feature import VectorAssembler
from mdlp.MDLPDiscretizer import MDLPDiscretizer

spark = SparkSession.builder.appName("TestMdlp").getOrCreate()

data = np.random.rand(100, 4) * 1000

schema = ["Class", "V1", "V2", "V3"]

# Create DataFrame
df = spark.createDataFrame(data, schema)
assembler = VectorAssembler(inputCols=["V1", "V2", "V3"], outputCol="features")
df = assembler.transform(df)
df = df.select(df.Class.alias("label"), "features")
df.show()

discretizer = MDLPDiscretizer(df)
discretizer_model = discretizer.train()

# Define a UDF to apply the transformation
@udf(returnType=VectorUDT())
def discretize_udf(vector):
    return discretizer_model.transform(vector)

# Apply the UDF to the DataFrame
df = df.withColumn("discretized_features", discretize_udf("features"))

# Show the transformed DataFrame
df.show(truncate=False)
