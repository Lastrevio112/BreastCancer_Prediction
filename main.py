import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

import functions as func

#Loading input data
df = pd.read_csv('dataIN/data.csv')
X = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
y = df['diagnosis']

#Exploratory data analysis
corr_matrix = X.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
#plt.show()

# remove highly correlated features in X
# Get the upper triangle of the correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column].abs() > 0.95)]
X = X.drop(to_drop, axis=1)

#Splitting data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# # Random Forest
# rf = RandomForestClassifier()
# rf_params = {
#     'n_estimators': [10, 50, 100, 200],
#     'max_depth': [None, 5, 10],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 5, 10]
# }
# rf_grid = GridSearchCV(rf, rf_params, cv=5)
# rf_grid.fit(X_train, y_train)
#
# # Decision Tree
# dt = DecisionTreeClassifier()
# dt_params = {
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [None, 5, 10],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 5, 10]
# }
# dt_grid = GridSearchCV(dt, dt_params, cv=5)
# dt_grid.fit(X_train, y_train)
#
# # K-Nearest Neighbors
# knn = KNeighborsClassifier()
# knn_params = {
#     'n_neighbors': [3, 5, 7, 10],
#     'weights': ['uniform', 'distance'],
#     'p': [1, 2]
# }
# knn_grid = GridSearchCV(knn, knn_params, cv=5)
# knn_grid.fit(X_train, y_train)
#
# # Logistic Regression
# lr = LogisticRegression()
# lr_params = {
#     'penalty': ['none', 'l2'],
#     'C': [0.1, 0.3, 1, 10]
# }
# lr_grid = GridSearchCV(lr, lr_params, cv=5)
# lr_grid.fit(X_train, y_train)
#
# # Print the best parameters and scores for each model
# print("Best Parameters for Random Forest:", rf_grid.best_params_)
# print("Best Score for Random Forest:", rf_grid.best_score_)
# print("Best Parameters for Decision Tree:", dt_grid.best_params_)
# print("Best Score for Decision Tree:", dt_grid.best_score_)
# print("Best Parameters for K-Nearest Neighbors:", knn_grid.best_params_)
# print("Best Score for K-Nearest Neighbors:", knn_grid.best_score_)
# print("Best Parameters for Logistic Regression:", lr_grid.best_params_)
# print("Best Score for Logistic Regression:", lr_grid.best_score_)

# #CONCLUSION: LOGISTIC REGRESSION HAS BEST ACCURACY (97.36%) WITH PARAMETERS {'C': 0.3, 'penalty': 'l2'}
#
# # save best model to disk
# model_filename = "logistic_regression_model.pkl"
# with open(model_filename, 'wb') as f:
#     pickle.dump(LogisticRegression(**lr_grid.best_params_), f)
# print(f"Model saved to {model_filename}")

lr = LogisticRegression(penalty='l2', C=0.3)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='M')
recall = recall_score(y_test, y_pred, pos_label='M')
f1 = f1_score(y_test, y_pred, pos_label='M')

print("Accuracy:", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)

feature_list = list(X.columns)





