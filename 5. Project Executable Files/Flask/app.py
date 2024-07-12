import os
import pickle
from flask import Flask, request, render_template, redirect, url_for
import logging

app = Flask(__name__)

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# Get the absolute path of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model and normalizer using the absolute path
model_path = os.path.join(base_dir, "rf_acc_68.pkl")
norms_path = os.path.join(base_dir, "normalizer.pkl")
model = pickle.load(open(model_path, "rb"))
norms = pickle.load(open(norms_path, "rb"))

# Define mapping dictionaries for categorical variables
warehouse_block_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4}
mode_of_shipment_mapping = {'Flight': 0, 'Ship': 1, 'Road': 2}
product_importance_mapping = {'low': 0, 'medium': 1, 'high': 2}
gender_mapping = {'F': 0, 'M': 1}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form and apply mappings
        Warehouse_block = warehouse_block_mapping[request.form["Warehouse_block"]]
        Mode_of_Shipment = mode_of_shipment_mapping[request.form["Mode_of_Shipment"]]
        Customer_care_calls = int(request.form["Customer_care_calls"])
        Customer_rating = int(request.form["Customer_rating"])
        Cost_of_the_Product = int(request.form["Cost_of_the_Product"])
        Prior_purchases = int(request.form["Prior_purchases"])
        Product_importance = product_importance_mapping[request.form["Product_importance"]]
        Gender = gender_mapping[request.form["Gender"]]
        Discount_offered = int(request.form["Discount_offered"])
        Weight_in_gms = int(request.form["Weight_in_gms"])

        # Log the form data
        app.logger.debug(f"Form Data: {request.form}")

        # Prepare data for prediction
        data = [[Warehouse_block, Mode_of_Shipment, Customer_care_calls, Customer_rating, Cost_of_the_Product,
                 Prior_purchases, Product_importance, Gender, Discount_offered, Weight_in_gms]]
        
        # Print the raw data
        app.logger.debug(f"Raw Data: {data}")
        
        data = norms.transform(data)
        
        # Print the transformed data
        app.logger.debug(f"Transformed Data: {data}")

        # Predict and get probabilities
        prediction = model.predict(data)
        probabilities = model.predict_proba(data)[0]

        result = 'Order will reach on time' if prediction[0] == 1 else 'Order will not reach on time'
        reach_prob = probabilities[1] * 100
        no_reach_prob = probabilities[0] * 100

        # Redirect to result page with prediction details
        return redirect(url_for('show_result', result=result, reach_prob=reach_prob, no_reach_prob=no_reach_prob))
    
    except KeyError as ke:
        app.logger.error(f"KeyError: {ke}")
        return "Invalid form data received. Please check your inputs and try again.", 400
    
    except ValueError as ve:
        app.logger.error(f"ValueError: {ve}")
        return "Invalid data type received. Please enter valid numeric values.", 400
    
    except Exception as e:
        # Log any other exceptions
        app.logger.error(f"Error: {e}")
        return "Internal Server Error", 500

@app.route('/result')
def show_result():
    result = request.args.get('result')
    reach_prob = float(request.args.get('reach_prob'))
    no_reach_prob = float(request.args.get('no_reach_prob'))

    return render_template('result.html', result=result, reach_prob="{:.2f}".format(reach_prob), no_reach_prob="{:.2f}".format(no_reach_prob))

if __name__ == '__main__':
    app.run(debug=False)
