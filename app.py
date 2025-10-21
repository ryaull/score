from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('wife_score_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get values from form
        height = float(request.form['height'])
        humor = float(request.form['humor'])
        dating_exp = float(request.form['dating_exp'])
        fitness = float(request.form['fitness'])
        cleanliness = float(request.form['cleanliness'])
        extroversion = float(request.form['extroversion'])
        finance = float(request.form['finance'])

        # Prepare input and predict
        input_features = np.array([[height, humor, dating_exp, fitness, cleanliness, extroversion, finance]])
        prediction = model.predict(input_features)[0]

        return render_template('index.html', prediction=f"{prediction:.2f}")

    return render_template('index.html', prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
