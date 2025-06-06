import pickle


file_path = "_Plot/fitted_distributions.pkl"  


with open(file_path, "rb") as f:
    data = pickle.load(f)


print("Contenuto del file Pickle:")
for key, value in data.items():
    print(f"\nVariabile: {key}")
    for sub_key, sub_value in value.items():
        print(f"  {sub_key}: {sub_value}")

