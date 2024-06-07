from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("SumColumn").getOrCreate()

# Path to the CSV file in GCS
csv_path = "gs://hungp-spark-bucket-1/users_neighborhood_anon.csv"

# Read CSV file
df = spark.read.csv(csv_path, header=True, inferSchema=True)

# Calculate the sum of the specific column (replace 'your_column' with the column name)
column_sum = df.agg({"followers_count": "sum"}).collect()[0][0]

# Print the sum
print(f"The sum of the column is: {column_sum}")


# Stop the Spark session
spark.stop()