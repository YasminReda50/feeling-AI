from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


app = Flask(__name__)
CORS(app) 


data = pd.read_excel("C:\\Users\\H.DATA\\Desktop\\Book1.xlsx") 
X = data['Review']  
y = data['Liked']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = LogisticRegression()
model.fit(X_train_vec, y_train)


@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json.get('text', '')  
    if not input_data:
        return jsonify({"error": "No text provided"}), 400  
    input_vec = vectorizer.transform([input_data])  
    prediction = model.predict(input_vec)[0]  
    if prediction == 1: 
        sentiment = "Positive"
    else:
         sentiment = "Negative" 

    return jsonify({"sentiment": sentiment})

if __name__ == '__main__':
    app.run(debug=True)