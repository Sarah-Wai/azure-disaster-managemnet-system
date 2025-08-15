import os
import joblib
import json
import traceback

model = None

def init():
    global model
    try:
        model_path = os.getenv("AZUREML_MODEL_DIR")
        print(f"Model directory: {model_path}")
        print(f"Files in model directory: {os.listdir(model_path)}")

        # Navigate into the subfolder
        subfolder = "INPUT_model_folder"
        model_file = os.path.join(model_path, subfolder, "model.pkl")

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Expected model file not found: {model_file}")

        model = joblib.load(model_file)
        print("Model loaded successfully.")

    except Exception as e:
        print("Exception during model loading:")
        traceback.print_exc()
        raise


def run(raw_data):
    try:
        # raw_data is a JSON string, parse it
        data = json.loads(raw_data)
        # Assuming input data in {"data": [[feature1, feature2, ...]]} format
        features = data.get("data")
        if features is None:
            raise ValueError("Input JSON must have 'data' key with feature array")

        preds = model.predict(features)
        preds_list = preds.tolist()

        return json.dumps({"predictions": preds_list})
    except Exception as e:
        print("Exception during scoring:")
        traceback.print_exc()
        return json.dumps({"error": str(e)})
