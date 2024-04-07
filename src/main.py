from pyspark.ml.linalg import DenseVector
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from mdlp import MDLPDiscretizer
import pandas as pd

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

path = "/opt/bitnami/spark/src/creditcard_sample.csv"
csv = spark.read.option("header", True).csv(path)
#csv = pd.read_csv(path, header=0)
# label_column = csv.select("Class")
# csv = csv.drop("Class")
label_column = csv['Class']
csv.drop(columns=['Class'], inplace=True)

print(csv.columns)
print(label_column)
print(csv)

# data = [
#     (0.1, DenseVector([1.0, 2.0, 3.0])),
#     (0.0, DenseVector([4.0, 5.0, 6.0])),
#     (1.0, DenseVector([-1.0, -2.0, -3.0])),
# ]
#df = spark.createDataFrame(data, ["label", "features"])

df = label_column.rdd.zip(csv.rdd).map(lambda x: (x[0][0], DenseVector(x[1]))).toDF(["label", "features"])
df.show()

discretizer = MDLPDiscretizer.MDLPDiscretizer(df)
model = discretizer.train()

result = model.transform(df)
result.show()