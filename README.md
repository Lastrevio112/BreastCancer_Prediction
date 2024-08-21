# BreastCancer_Prediction
A classification machine learning model that predicts whether a breast tumor is malignant or benign depending on its radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension.

To use the application, access this website: https://breastcancer-prediction-vzex.onrender.com/

The application uses a Logistic Regression model to predict whether a breast tumor is malignant or benign. 

Initially, feature engineering was done by removing unnecessary columns as well as the features which are highly correlated with each other (R > 0.95).

After that, the data was split into a training set and a test set and GridSearchCV was used for hyperparameter tuning. The following models were tested with various parameters: Random Forest, Logistic Regression, Decision Tree, KNearestNeighbors.

GridSearchCV found out that LogisticRegression is the best model for this problem. Furthermore, the model.py file computes the accuracy, precision, recall and f1_score of this model.

The model was dumped using pickle in the LR_model_pkl file. After that, I built the back-end of a web app using Flask in the backEndApp.py file. The front-end HTML can be found in the templates/index.html file and the CSS and Javascript can be found in the static directory.