from MDLPBuilder import MDLPBuilder
from test_model import evaluate_models

# Initialize params
app_name = "InsuranceData Discretizer"
csv_path = "gs://hungp-spark-bucket-1/insurance_data.csv"
label_col = "PROD_LINE"

df = MDLPBuilder(app_name, csv_path, label_col).toPandas()

df['label'] = df['label'].map({'CL': 0, 'PL': 1})
y = df['label']
X = df.drop(columns=['label'])
results = evaluate_models(X, y)        
print(f"Results for {csv_path}:")
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.4f}")
print()

print(f"Data transformed!")