# Simple Flask app that returns a JSON response with POST
# request data.

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)

# enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the Random Forest model
with open("ada_boost_model-v2.pkl", "rb") as model_file:
    model = pickle.load(model_file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyse", methods=["POST"])
def index():
    data = request.get_json()

    # Make predictions using the Ada Boost model
    prediction = model.predict(
        [
            [
                data["seed"],
                data["venture"],
                data["age_first_funding"],
                data["round_D"],
                data["funding_rounds"],
                data["round_A"],
                data["round_B"],
                data["founded_year"],
                data["grant"],
                data["angel"],
                data["debt_financing"],
                data["market_Clean Technology"],
                data["private_equity"],
                data["convertible_note"],
                data["market_Advertising"],
                data["market_Analytics"],
                data["market_Biotechnology"],
                data["round_E"],
                data["market_Other"],
                data["market_Enterprise Software"],
                data["market_Finance"],
                data["market_Health Care"],
                data["market_Consulting"],
                data["secondary market"],
                data["undisclosed"],
                data["equity_crowdfunding"],
                data["market_Security"],
                data["founded_month"],
                data["market_Manufacturing"],
                data["product_crowdfunding"],
                data["round_C"],
                data["market_Mobile"],
                data["market_Hospitality"],
                data["round_F"],
                data["market_Social Media"],
                data["market_Curated Web"],
                data["market_E-Commerce"],
                data["market_Education"],
                data["market_Games"],
                data["market_Hardware + Software"],
                data["market_Health and Wellness"],
                data["market_Software"],
            ],
        ]
    )

    outcome = prediction[0]

    # Return JSON response to client 
    response = {"outcome": int(np.int64(outcome))}
        
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
