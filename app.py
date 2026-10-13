import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the exported model pipeline
model = joblib.load('netflix_movie_duration_regressor.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Expecting JSON input: {"release_year": 2021, "rating": "PG-13"}
        data = request.get_json(force=True)
        
        # Convert request data into a pandas DataFrame
        input_df = pd.DataFrame([data])
        
        # Make a prediction using our pipeline
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            'status': 'success',
            'predicted_duration_minutes': round(float(prediction), 2)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
