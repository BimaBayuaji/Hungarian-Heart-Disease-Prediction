import streamlit as st
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Column names
columns = [
  'age', 'sex', 'cp', 'trestbps', 'chol',
  'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
  'slope', 'ca', 'thal', 'num'
]

# Read dataset
df = pd.read_csv("processed.hungarian.data", names=columns, header=None)

# Replacing '?' with 'NaN'
df.replace('?', np.nan, inplace=True)

# Drop column that have >50 row null value
for column in df.columns:
  if df[column].isnull().sum() > 50:
    df.drop(column, axis=1, inplace=True)

# Drop insignificant column
df.drop('chol', axis=1, inplace=True)

updated_df = df.dropna()

# Change the data type for each column
for column in updated_df.columns:
  if updated_df[column].dtypes == 'O':
    if column == "oldpeak":
      updated_df[column] = updated_df[column].astype(float)
    else:
      updated_df[column] = updated_df[column].astype(int)
  elif updated_df[column].dtypes == 'int64':
    updated_df[column] = updated_df[column].astype(int)

features = updated_df.drop('num', axis=1)
target = updated_df['num']

# Oversampling target 0, 2, 3, 4 based on target 1
oversample = SMOTE(k_neighbors=4)
X, y = oversample.fit_resample(features, target)

filled_df = X
filled_df['num'] = y

features = filled_df.drop('num', axis=1)
target = filled_df['num']

# Train - Test split
X_train ,X_test, y_train ,y_test = train_test_split(features, target, test_size = 0.2)

model = DecisionTreeClassifier()

accuracy_list = np.array([])

for i in range(0, 10):
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  accuracy = round((accuracy * 100), 2)

  accuracy_list = np.append(accuracy_list, accuracy)

min_accuracy = np.min(accuracy_list)
max_accuracy = np.max(accuracy_list)

# STREAMLIT
st.set_page_config(
  page_title = "Hungarian Heart Disease",
  page_icon = ":purple_heart:"
)

st.title("Hungarian Heart Disease Prediction")
st.write("_Using Decision Tree Classifier_, _Random Forest_, _Logistic Regression_, _Support Vector Machine_")
st.write("\n")

age = st.number_input(label=":violet[**Age**]", min_value=filled_df['age'].min(), max_value=filled_df['age'].max())
st.write(f":orange[Min] value: :orange[**{filled_df['age'].min()}**], :red[Max] value: :red[**{filled_df['age'].max()}**]")
st.write("")

sex_sb = st.selectbox(label=":violet[**Sex**]", options=["Male", "Female"])
st.write("")
st.write("")
if sex_sb == "Male":
  sex = 1
elif sex_sb == "Female":
  sex = 0
# -- Value 0: Female
# -- Value 1: Male

cp_sb = st.selectbox(label=":violet[**Chest pain type**]", options=["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
st.write("")
st.write("")
if cp_sb == "Typical angina":
  cp = 1
elif cp_sb == "Atypical angina":
  cp = 2
elif cp_sb == "Non-anginal pain":
  cp = 3
elif cp_sb == "Asymptomatic":
  cp = 4
# -- Value 1: typical angina
# -- Value 2: atypical angina
# -- Value 3: non-anginal pain
# -- Value 4: asymptomatic

trestbps = st.number_input(label=":violet[**Resting blood pressure** (in mm Hg on admission to the hospital)]", min_value=filled_df['trestbps'].min(), max_value=filled_df['trestbps'].max())
st.write(f":orange[Min] value: :orange[**{filled_df['trestbps'].min()}**], :red[Max] value: :red[**{filled_df['trestbps'].max()}**]")
st.write("")

restecg_sb = st.selectbox(label=":violet[**Resting electrocardiographic results**]", options=["Normal", "Having ST-T wave abnormality", "Showing left ventricular hypertrophy"])
st.write("")
st.write("")
if restecg_sb == "Normal":
  restecg = 0
elif restecg_sb == "Having ST-T wave abnormality":
  restecg = 1
elif restecg_sb == "Showing left ventricular hypertrophy":
  restecg = 2
# -- Value 0: normal
# -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST  elevation or depression of > 0.05 mV)
# -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

thalach = st.number_input(label=":violet[**Maximum heart rate achieved**]", min_value=filled_df['thalach'].min(), max_value=filled_df['thalach'].max())
st.write(f":orange[Min] value: :orange[**{filled_df['thalach'].min()}**], :red[Max] value: :red[**{filled_df['thalach'].max()}**]")
st.write("")

exang_sb = st.selectbox(label=":violet[**Exercise induced angina**]", options=["No", "Yes"])
st.write("")
st.write("")
if exang_sb == "No":
  exang = 0
elif exang_sb == "Yes":
  exang = 1
# -- Value 0: No
# -- Value 1: Yes

oldpeak = st.number_input(label=":violet[**ST depression induced by exercise relative to rest**]", min_value=filled_df['oldpeak'].min(), max_value=filled_df['oldpeak'].max())
st.write(f":orange[Min] value: :orange[**{filled_df['oldpeak'].min()}**], :red[Max] value: :red[**{filled_df['oldpeak'].max()}**]")
st.write("")

slope_sb = st.selectbox(label=":violet[**Exercise induced angina**]", options=["Upsloping", "Flat", "Downsloping"])
st.write("")
st.write("")
if slope_sb == "Upsloping":
  slope = 1
elif slope_sb == "Flat":
  slope = 2
elif slope_sb == "Downsloping":
  slope = 3
# -- Value 1: upsloping
# -- Value 2: flat
# -- Value 3: downsloping

result = ":violet[-]"

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Use the scaled data to train the SVM model
svm_model_scaled = SVC(C=1.0, kernel='rbf', gamma='scale')  # Adjust hyperparameters
svm_model_scaled.fit(X_train_scaled, y_train)
svm_preds_scaled = svm_model_scaled.predict(X_test_scaled)
svm_accuracy_scaled = accuracy_score(y_test, svm_preds_scaled)

# Train the Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)

# Train the Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_preds)

# Display results for all models
st.write('### Model Comparison')

def get_result(prediction):
    if prediction == 0:
        return ":green[**Healthy**]"
    elif prediction == 1:
        return ":orange[**Heart disease level 1**]"
    elif prediction == 2:
        return ":orange[**Heart disease level 2**]"
    elif prediction == 3:
        return ":red[**Heart disease level 3**]"
    elif prediction == 4:
        return ":red[**Heart disease level 4**]"
    else:
        return ":violet[**Unknown**]"


st.write("")
st.write("")


# Predictions for all models
if st.button("Predict Decision Tree"):
    inputs = [[age, sex, cp, trestbps, restecg, thalach, exang, oldpeak, slope]]
    dt_prediction = model.predict(inputs)[0]
    dt_result = get_result(dt_prediction)
    st.subheader("Decision Tree Prediction:")
    st.write(dt_result)
    st.write(f"Decision Tree Accuracy: {max_accuracy}%")

# Prediksi untuk model Random Forest
if st.button("Predict Random Forest"):
    inputs = [[age, sex, cp, trestbps, restecg, thalach, exang, oldpeak, slope]]
    rf_prediction = rf_model.predict(inputs)[0]
    rf_result = get_result(rf_prediction)
    st.subheader("Random Forest Prediction:")
    st.write(rf_result)
    st.write(f"SVM Accuracy: {round((svm_accuracy_scaled * 100), 2)}%")

# Prediksi untuk model Logistic Regression
if st.button("Predict Logistic Regression"):
    inputs = [[age, sex, cp, trestbps, restecg, thalach, exang, oldpeak, slope]]
    lr_prediction = lr_model.predict(inputs)[0]
    lr_result = get_result(lr_prediction)
    st.subheader("Logistic Regression Prediction:")
    st.write(lr_result)
    st.write(f"Random Forest Accuracy: {round((rf_accuracy * 100), 2)}%")

# Prediksi untuk model Support Vector Machine (SVM)
if st.button("Predict Support Vector Machine (SVM)"):
    inputs = [[age, sex, cp, trestbps, restecg, thalach, exang, oldpeak, slope]]
    svm_prediction = svm_model_scaled.predict(inputs)[0]
    svm_result = get_result(svm_prediction)
    st.subheader("SVM Prediction:")
    st.write(svm_result)
    st.write(f"Logistic Regression Accuracy: {round((lr_accuracy * 100), 2)}%")