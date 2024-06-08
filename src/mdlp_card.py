from MDLPBuilder import MDLPBuilder
from test_model import evaluate_models

# Initialize params
app_name = "InsuranceData Discretizer"
csv_path = "gs://hungp-spark-bucket-1/creditcard.csv"
label_col = "Class"

df = MDLPBuilder(app_name, csv_path, label_col).toPandas()

y = df['label']
X = df.drop(columns=['label'])
results = evaluate_models(X, y)        
print(f"Results for {csv_path}:")
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.4f}")
print()

print(f"Data transformed!")