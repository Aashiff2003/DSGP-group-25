from flask import Flask, jsonify, Response, request
from config import create_app, db
from database import add_user, password_check, add_bird_strike, fetch_records
from visualization import plot_alert_level_distribution, plot_bird_quantity_vs_time
import joblib
import pandas as pd
# load Alert model
alert_model = joblib.load('random_forest_model.joblib')

# Define the feature order (as used during training)
alert_feature_order = [
    'NumberStruckActual', 'WildlifeSize', 'ConditionsSky_No Cloud',
    'ConditionsSky_Overcast', 'ConditionsSky_Some Cloud'
]
# Create Flask app using config
app = create_app()

# Flask Routes
@app.route('/')
def home():
    return "Bird Strike Detection System API is Running!"

@app.route('/add_user/<string:username>/<string:password>')
def add_user_route(username, password):
    return add_user(username, password)

@app.route('/login/<string:username>/<string:password>')
def login(username, password):
    return password_check(username, password)

@app.route('/insert/<string:weather>/<string:bird_size>/<string:bird_species>/<int:bird_quantity>/<string:alert_level>')
def insert(weather, bird_size, bird_species, bird_quantity, alert_level):
    return add_bird_strike(weather, bird_size, bird_species, bird_quantity, alert_level)

@app.route('/records')
def records():
    return jsonify(fetch_records())

@app.route('/visualize')
def visualize():
    # Get the image responses as base64-encoded strings
    alert_level_img = plot_alert_level_distribution()
    bird_quantity_img = plot_bird_quantity_vs_time()

    # Return images separately in HTML format with base64-encoded image data
    html_content = f'''
    <html>
        <body>
            <h1>Visualization</h1>
            <h2>Alert Level Distribution</h2>
            <img src="data:image/png;base64,{alert_level_img}" alt="Alert Level Distribution">
            <h2>Bird Quantity Over Time</h2>
            <img src="data:image/png;base64,{bird_quantity_img}" alt="Bird Quantity Over Time">
        </body>
    </html>
    '''
    return Response(html_content, mimetype='text/html')
    return render_template('index.html')


@app.route('/predict_alert', methods=['POST'])
def predict_alert():
    try:
        # Get input data from the request
        input_data = request.json

        # Convert input data into a DataFrame with the correct feature order
        input_df = pd.DataFrame([input_data], columns=alert_feature_order)

        # Make prediction
        prediction = alert_model.predict(input_df)

        # Return the predicted alert level as a JSON response
        return jsonify({'predicted_alert_level': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run Flask Apppyhon
if __name__ == "__main__":
    app.run(debug=True)
