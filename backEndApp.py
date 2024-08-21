import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained logistic regression model
with open('LR_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve the features from the POST request
        features = [float(x) for x in request.form.values()]

        # Ensure the correct number of features
        if len(features) != 23:
            return jsonify({'error': 'Incorrect number of features provided'})

        # Make the prediction using the model
        prediction = model.predict([features])[0]
        result = 'Malignant' if prediction == 'M' else 'Benign'

        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)