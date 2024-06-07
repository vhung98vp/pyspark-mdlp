from MDLPBuilder import MDLPBuilder

# Initialize params
app_name = "CreditCard Discretizer"
csv_path = "gs://hungp-spark-bucket-1/creditcard.csv"
label_col = "Time"
feature_cols = ["Time", "V1", "V4"]

model = MDLPBuilder(app_name, csv_path, label_col, feature_cols)

print(f"Data transformed!")