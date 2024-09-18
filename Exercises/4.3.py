import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

# Load the dataset from a CSV file
data = pd.read_csv('./thyroid+disease/sick-euthyroid.data', header=None)

# Rename columns (as per dataset understanding)
columns = ['class', 'age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
           'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
           'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary',
           'psych', 'TSH', 'TSH_measured', 'T3', 'T3_measured', 'TT4', 'TT4_measured',
           'T4U', 'T4U_measured', 'FTI', 'FTI_measured']

data.columns = columns

# Handle missing values (represented as '?')
data.replace('?', np.nan, inplace=True)

# Separate the columns with excessive missing or unsuitable data
dropped_columns = data.columns[data.isnull().mean() > 0.2]  # Drop columns with more than 20% missing data
processed_data = data.drop(columns=dropped_columns)

# Fill missing numeric values in the remaining dataset
processed_data.fillna(processed_data.median(numeric_only=True), inplace=True)

# Encode categorical variables in the remaining dataset
le = LabelEncoder()
for col in processed_data.columns:
    if processed_data[col].dtype == 'object':
        processed_data[col] = le.fit_transform(processed_data[col].astype(str))

# Drop any rows with remaining missing data
processed_data.dropna(inplace=True)

# Split the dataset into features (X) and target (y)
X = processed_data.drop(columns=['class'])
y = processed_data['class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine X_train and y_train for Bayesian Network
train_data = pd.concat([X_train, y_train], axis=1)

# Define the Bayesian Network structure
model = BayesianNetwork([('age', 'class'), ('sex', 'class'), ('TSH', 'class')])

# Train the model using Maximum Likelihood Estimation
model.fit(train_data, estimator=MaximumLikelihoodEstimator)

# Make predictions on the test set
y_pred = []
for _, row in X_test.iterrows():
    pred = model.predict([row])[0]
    y_pred.append(pred['class'])

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Plot the analysis of dropped columns (unprocessed columns)
plt.figure(figsize=(10, 6))
missing_values = data[dropped_columns].isnull().mean() * 100
sns.barplot(x=missing_values.index, y=missing_values.values)
plt.xticks(rotation=90)
plt.title('Percentage of Missing Values in Dropped Columns')
plt.ylabel('Percentage Missing')
plt.show()
