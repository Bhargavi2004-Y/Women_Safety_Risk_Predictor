import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

data = pd.read_csv("dataset.csv")

le_location = LabelEncoder()
le_time = LabelEncoder()
le_risk = LabelEncoder()

data['location'] = le_location.fit_transform(data['location'])
data['time'] = le_time.fit_transform(data['time'])
data['risk'] = le_risk.fit_transform(data['risk'])

X = data[['location','time','crime_rate']]
y = data['risk']

model = DecisionTreeClassifier()
model.fit(X,y)

joblib.dump(model,"model.pkl")
joblib.dump(le_location,"le_location.pkl")
joblib.dump(le_time,"le_time.pkl")
joblib.dump(le_risk,"le_risk.pkl")

print("Model trained")
