from MDLPBuilder import MDLPBuilder

# Initialize params
app_name = "InsuranceData Discretizer"
csv_path = "gs://hungp-spark-bucket-1/insurance_data.csv"
label_col = "PROD_LINE"

model = MDLPBuilder(app_name, csv_path, label_col, 1000)

print(f"Data transformed!")