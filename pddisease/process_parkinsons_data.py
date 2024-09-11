import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("parkinsons.data")

# Display the dataset info
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df.head())
print(df.info())
print(df.describe())
print("Shape of the dataset:", df.shape)
print("Missing values in each column:\n", df.isnull().sum())
print("Value counts of 'status':\n", df['status'].value_counts())

# Prepare the data
X = df.drop(columns=['name', 'status'], axis=1)
Y = df['status']
print("Features (X):\n", X)
print("Labels (Y):\n", Y)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print("Shapes - X: {}, X_train: {}, X_test: {}".format(X.shape, X_train.shape, X_test.shape))

# Standardize the data
SS = StandardScaler()
SS.fit(X_train)
X_train = SS.transform(X_train)
X_test = SS.transform(X_test)
print("Standardized X_train:\n", X_train)
print("Standardized X_test:\n", X_test)

# Train the model
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

# Evaluate the model
X_train_pred = model.predict(X_train)
train_data_acc = accuracy_score(Y_train, X_train_pred)
print("Accuracy of training data: ", train_data_acc*100,"%")

X_test_pred = model.predict(X_test)
test_data_acc = accuracy_score(Y_test, X_test_pred)
print("Accuracy of testing data: ", test_data_acc*100,"%")

# Test with a new input data
input_data = (260.10500, 264.91900, 237.30300, 0.00339, 0.00001, 0.00205, 0.00186, 0.00616, 0.19700, 0.01186, 0.01230, 0.01367, 0.03557, 0.00910, 21.08300, 0, 0.440988, 0.628058, -7.517934, 0.160414, 1.881767, 0.075587)
input_data_np = np.asarray(input_data)
input_data_re = input_data_np.reshape(1, -1)

# Ensure the input data has the correct structure with feature names
input_df = pd.DataFrame(input_data_re, columns=X.columns)

# Transform the input data
s_data = SS.transform(input_df)

# Predict the result
pred = model.predict(s_data)
print("Prediction result:", pred)

if pred[0] == 0:
    print("Negative, no Parkinson's disease found")
else:
    print("Positive, Parkinson's Disease found")