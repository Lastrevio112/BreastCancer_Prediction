from sklearn.preprocessing import StandardScaler
import pandas as pd

#Function that predicts whether a tumor is malign or benign (returns M or B)
#based on a list of 23 values
def predict_tumor(model, values):
    #scaler = StandardScaler()
    #input = scaler.fit_transform(pd.DataFrame(values).transpose())
    input = pd.DataFrame(values).transpose()
    prediction = model.predict(input)
    return prediction