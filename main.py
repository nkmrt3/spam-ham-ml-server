import pickle
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model
model_filename = 'model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load the TfidfVectorizer
vectorizer_filename = 'tfd.pkl'
with open(vectorizer_filename, 'rb') as file:
    feature_extraction = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form.get("message")  # Get the message from the form
    input_data_features = feature_extraction.transform([message])  # Transform the message to features
    prediction = model.predict(input_data_features)
    
    if prediction[0] == 1:
        result = 'Ham mail'
    else:
        result = 'Spam mail'
  
    return render_template("index.html", prediction_text=f"The SMS is {result}")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
