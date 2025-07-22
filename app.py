from flask import Flask, render_template, request
import joblib
import pandas as pd



app = Flask(__name__)

model = joblib.load('TitanicModel.pkl')

@app.route('/')
def form():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    sex = 1 if request.form["Sex"] == "Female" else 0
    fare = float(request.form["Fare"])

    # Pass a DataFrame with correct column names
    input_df = pd.DataFrame([[sex, fare]], columns=["Sex", "Fare"])
    prediction = model.predict(input_df)[0]

    result = "Survived" if prediction >= 0.5 else "Did Not Survive"
    return render_template('index.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
