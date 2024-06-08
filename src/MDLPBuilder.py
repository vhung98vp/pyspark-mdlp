from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from mdlp.MDLPDiscretizer import MDLPDiscretizer


def MDLPBuilder(app_name, csv_path, label_col):
    """
    Build MDLP Discretizer from CSV file
    * @param app_name Spark session
    * @param csv_path Path to csv file
    * @param label_col label (target) column
    * @param max_by_part max number of elements in a partition
    * @param min_bin_percentage minimum percent of total dataset allowed in a single bin.
    """

    # Initialize Spark session
    spark = SparkSession.builder.appName(app_name).getOrCreate()

    # Read CSV file
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    df.printSchema()

    # Identify columns with numeric and boolean types
    numeric_cols = [col_name for col_name, dtype in df.dtypes if dtype in ["int", "bigint", "float", "double"]]
    bool_cols = [col_name for col_name, dtype in df.dtypes if dtype in ["boolean"]]
    selected_cols = numeric_cols + bool_cols

    # Fill label column first
    if (label_col not in selected_cols):
        selected_cols.append(label_col)
        df.na.fill("", subset=[label_col])
    
    # Calculate mean values
    # numeric_values = {col_name: df.select(mean(col(col_name))).first()[0] for col_name in numeric_cols}
    numeric_values = {col_name: 0 for col_name in numeric_cols}
    bool_values = {col_name: True for col_name in bool_cols}
    mean_values = {**numeric_values, **bool_values}

    # Select only these columns and fill
    df = df.select(selected_cols)
    df = df.na.fill(mean_values)

    # Create DataFrame
    cols = [col_name for col_name, dtype in df.dtypes]
    assembler = VectorAssembler(inputCols=[col_name for col_name in cols if col_name != label_col], outputCol="features", handleInvalid="skip")
    df = assembler.transform(df)
    df = df.withColumnRenamed(label_col, "label").select("label", "features")
    df.show(20)

    # MDLP Discretizer
    discretizer = MDLPDiscretizer(df)
    discretizer_model = discretizer.train()

    # Define a UDF to apply the transformation
    @udf(returnType=VectorUDT())
    def discretize_udf(vector):
        return discretizer_model.transform(vector)

    # Apply the UDF to the DataFrame
    df = df.withColumn("discretized_features", discretize_udf("features"))

    # Show the transformed DataFrame
    df.show(20) 

    # Split df to cols to test
    df_array = df.withColumn("features_array", vector_to_array(col("discretized_features")))
    df_final = df_array.drop("features", "discretized_features")
    num_features = len(df_array.select("features_array").first()[0])

    for i in range(num_features):
        df_array = df_array.withColumn(f"feature_{i}", col("features_array")[i])

    df_final = df_array.drop("features_array")

    return df_final
