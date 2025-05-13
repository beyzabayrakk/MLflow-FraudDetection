from flask import Flask, request, jsonify, render_template
import mlflow.pyfunc
import mlflow
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("http://localhost:5000")

app = Flask(__name__)

model_name = "FraudDetectionModel"
try:
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Production")
    logger.info("Model successfully loaded from Production stage.")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            inputs = []
            for i in range(29):
                if f'feature_{i}' not in request.form:
                    raise ValueError(f"Feature {i} is missing in the form submission.")
                value = request.form[f'feature_{i}']
                if value == "":
                    raise ValueError(f"Feature {i} cannot be empty.")
                inputs.append(float(value))
            inputs = np.array([inputs])
            logger.info(f"Input shape: {inputs.shape}")
            predictions = model.predict(inputs)
            prediction = predictions.tolist()[0]
            logger.info(f"Prediction successful: {prediction}, type: {type(prediction)}")
        except Exception as e:
            prediction = f"Error: {str(e)}"
            logger.error(f"Prediction error: {str(e)}")
    return render_template('index.html', prediction=prediction)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            data = request.get_json(force=True)
        else:
            data = {"inputs": [[-1.35980713, -0.07278117, 2.53634674, 1.37815522, -0.33832077, 0.46238778, 0.23959855, 0.0986979, 0.36378697, 0.09079417, -0.55159953, -0.61780086, -0.99138985, -0.31116935, 1.46817697, -0.47040053, 0.20797124, 0.02579058, 0.40399296, 0.2514121, -0.01830678, 0.27783758, -0.11047391, 0.06692807, 0.12853936, -0.18911484, 0.13355838, -0.02105305, 0.0]]}
        inputs = np.array(data["inputs"])
        logger.info(f"Input shape for API: {inputs.shape}")
        predictions = model.predict(inputs)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)