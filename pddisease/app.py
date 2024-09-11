from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# Load the dataset
df = pd.read_csv("parkinsons.data")

# Prepare the data
X = df.drop(columns=['name', 'status'], axis=1)
Y = df['status']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Standardize the data
SS = StandardScaler()
SS.fit(X_train)
X_train = SS.transform(X_train)
X_test = SS.transform(X_test)

# Train the model
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/result.html')
def result():
    return send_from_directory(app.static_folder, 'result.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json['input_data']
    input_data_np = np.asarray(input_data)
    input_data_re = input_data_np.reshape(1, -1)
    input_df = pd.DataFrame(input_data_re, columns=X.columns)
    s_data = SS.transform(input_df)
    pred = model.predict(s_data)
    result = "Positive, Parkinson's Disease found" if pred[0] == 1 else "Negative, no Parkinson's disease found"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
