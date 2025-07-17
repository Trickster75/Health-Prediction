import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("D:/python/project/health_dataset.csv")
data.dropna(inplace=True)

data["Label"] = data["Label"].map({"Good": 0, "Warning": 1, "Critical": 2})

X = data.drop("Label", axis=1)
y = data["Label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(6,3), activation='relu', max_iter=300, random_state=42)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

print("\nEnter your health parameters for prediction:")

heart_rate = float(input("Heart Rate: "))
sleep = float(input("Sleep Hours: "))
steps = float(input("Steps: "))
calories = float(input("Calories: "))
spo2 = float(input("SpO2 (%): "))
stress = float(input("Stress Level: "))

user_features = [[heart_rate, sleep, steps, calories, spo2, stress]]
user_scaled = scaler.transform(user_features)
user_pred = mlp.predict(user_scaled)[0]

label_map = {0: "Good", 1: "Warning", 2: "Critical"}
print(f"\nPredicted Health Status: {label_map[user_pred]}")