from MDLPBuilder import MDLPBuilder

# Initialize params
app_name = "Hateful User Discretizer"
csv_path = "gs://hungp-spark-bucket-1/users_neighborhood_anon.csv"
label_col = "hate_neigh"

model = MDLPBuilder(app_name, csv_path, label_col, 10000)

print(f"Data transformed!")