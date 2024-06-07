from MDLPBuilder import MDLPBuilder

# Initialize params
app_name = "CreditCard Discretizer"
csv_path = "gs://hungp-spark-bucket-1/creditcard.csv"
label_col = "Class"

model = MDLPBuilder(app_name, csv_path, label_col, 28000)

print(f"Data transformed!")