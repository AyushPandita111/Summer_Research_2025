import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv("gpu_power_time_data.csv")


print("Columns in CSV:", df.columns.tolist())


df.rename(columns=lambda x: x.strip().replace(" ", ""), inplace=True)

le_model = LabelEncoder()
le_dataset = LabelEncoder()
le_gpu = LabelEncoder()


df['ModelType_enc'] = le_model.fit_transform(df['ModelType'])
df['Dataset_enc'] = le_dataset.fit_transform(df['Dataset'])
df['GPU_enc'] = le_gpu.fit_transform(df['GPU'])


df['Efficiency'] = 0.5 * df['Power'] + 0.5 * df['InferenceTime']


best_gpus = df.loc[df.groupby(['ModelType_enc', 'Dataset_enc'])['Efficiency'].idxmin()]


df = df.merge(best_gpus[['ModelType_enc', 'Dataset_enc', 'GPU_enc']],
              on=['ModelType_enc', 'Dataset_enc'],
              suffixes=('', '_best'))


X = df[['ModelType_enc', 'Dataset_enc']]
y = df['GPU_enc_best']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


def recommend_gpu(model_name, dataset_name):
    try:
        model_enc = le_model.transform([model_name])[0]
        dataset_enc = le_dataset.transform([dataset_name])[0]
        
        prediction = clf.predict([[model_enc, dataset_enc]])[0]
        best_gpu = le_gpu.inverse_transform([prediction])[0]
        
        print(f"\n For Model '{model_name}' on Dataset '{dataset_name}', "
              f"the recommended GPU is: **{best_gpu}**\n")
    except ValueError:
        print("\nError: Model or Dataset not found in training data.\n")


model_input = input("Enter your Model: ").strip()
dataset_input = input("Enter your Dataset: ").strip()

recommend_gpu(model_input, dataset_input)
