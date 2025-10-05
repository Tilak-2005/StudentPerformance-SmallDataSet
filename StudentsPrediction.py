import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns

data = pd.read_csv("Students.csv")
print(data.head())
print(f"(rows,columns) : {data.shape}")
print(f" Missing Values : \n {data.isnull().sum()}")
le = LabelEncoder()
data["Internet"] = le.fit_transform(data["Internet"]) #Encoding Internet Column
data["Passed"] = le.fit_transform(data["Passed"])   #Encoding Passed Column
print(data["Internet"])
print(data["Passed"])
print(data.dtypes) #check the datatypes

features = ["StudyHours","Attendance","PastScore","SleepHours"]
scalar = StandardScaler()
df = data.copy()
df[features] = scalar.fit_transform(df[features])

X = df[features]
y = df["Passed"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LogisticRegression()
model.fit(X_train,y_train)

y_predicted = model.predict(X_test)

print("Classification Report : ")
print(classification_report(y_test,y_predicted))

conf_mat = confusion_matrix(y_test,y_predicted)

plt.figure(figsize=(6,4))

sns.heatmap(conf_mat,annot=True,fmt="d",cmap="Blues",xticklabels=["Fail","Pass"],yticklabels=["Fail","Pass"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


print("----------------------PREDICTED-RESULTS----------------------")
try:
    study_hours = float(input("Enter the study hours: "))
    attendance = float(input("Enter the attendance: "))
    past_scores = float(input("Enter the past scores: "))
    sleep_hours = float(input("Enter the sleep hours: "))
    
    user_input = pd.DataFrame([{
        "StudyHours" : study_hours,
        "Attendance" : attendance,
        "PastScore" : past_scores,
        "SleepHours" : sleep_hours
    }])
    
    user_input_scaled = scalar.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]
    
    result = "Pass" if prediction == 1 else "Fail"
    print(f"Prediction Based On I/P : {result}")
    
finally:
    print("Program Ended")
    
    
