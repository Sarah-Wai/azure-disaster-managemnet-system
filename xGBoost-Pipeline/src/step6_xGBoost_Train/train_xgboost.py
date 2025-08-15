import argparse
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# If using Azure ML run context, import and get run:
try:
    from azureml.core import Run
    run = Run.get_context()
except ImportError:
    run = None

# -------------------- Data Preparation --------------------
def encode_categorical_columns(df, columns):
    for col in columns:
        df[f"{col}_encoded"] = LabelEncoder().fit_transform(df[col])
    return df

def normalize_features(df, features):
    scaler = MinMaxScaler()

    for feature in features:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    invalid_mask = df[features].isnull().any(axis=1)
    invalid_count = invalid_mask.sum()
    logging.info(f"Number of rows with NaNs/Infs in features: {invalid_count}")

    if invalid_count > 0:
        logging.warning("Sample rows with invalid values:\n{}".format(df[invalid_mask].head()))
        df[invalid_mask].to_csv("debug_invalid_rows.csv", index=False)

    df.dropna(subset=features, inplace=True)

    if df.empty:
        raise ValueError(f"All rows were dropped after removing invalid values in features: {features}. "
                         "Check debug_invalid_rows.csv for examples.")

    try:
        df[features] = scaler.fit_transform(df[features])
    except Exception as e:
        logging.error(f"Normalization failed for features {features}: {e}", exc_info=True)
        raise

    return df

def compute_risk_score(df):
    weights = {
        'damage_level': 0.4,
        'population_density': 0.3,
        'infrastructure_value': 0.2,
        'rainfall': 0.05,
        'wind_speed': 0.05
    }

    df['risk_score'] = sum(weights[col] * df[col] for col in weights)

    bins = [0, 0.3, 0.5, 0.7, 1.0]
    labels = ['Low', 'Medium', 'High', 'Critical']
    df['risk_level'] = pd.cut(df['risk_score'], bins=bins, labels=labels)

    return df

def load_and_prepare_data(csv_path):
    logging.info(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    raw_df = df.copy()

    logging.info(f"Loaded CSV with shape: {df.shape}")
    logging.info(f"Columns: {df.columns.tolist()}")
    logging.info(f"Missing values:\n{df.isnull().sum()}")

    df['location_id'] = df['lat'].round(3).astype(str) + '_' + df['lon'].round(3).astype(str)
    df['infrastructure_value'] = df.get('infrastructure_value', 100000)
    df['economic_activity'] = df.get('economic_activity', 1.0)

    required_columns = [
        "damage_level", "population_density", "rainfall", 
        "wind_speed", "lat", "lon", "country", "year", 
        "disaster_type", "temperature", "region"
    ]
    df = df.dropna(subset=required_columns).fillna(0)

    if df['damage_level'].dtype == object:
        df['damage_level'] = df['damage_level'].str.extract(r'(\d+)').astype(float)
        logging.info("Converted 'damage_level' from class labels to numeric.")

    features_to_normalize = [
        'damage_level', 'population_density', 
        'rainfall', 'wind_speed', 'infrastructure_value'
    ]
    logging.info(f"Features to normalize: {features_to_normalize}")
    logging.info(f"Sample before normalization:\n{df[features_to_normalize].head()}")

    df = normalize_features(df, features_to_normalize)
    df = compute_risk_score(df)

    le_country = LabelEncoder()
    le_disaster = LabelEncoder()
    le_region = LabelEncoder()
    le_target = LabelEncoder()

    df['country_encoded'] = le_country.fit_transform(df['country'])
    df['disaster_type_encoded'] = le_disaster.fit_transform(df['disaster_type'])
    df['region_encoded'] = le_region.fit_transform(df['region'])
    df['risk_level_enc'] = le_target.fit_transform(df['risk_level'])

    return df, le_target, le_country, le_disaster, le_region, raw_df

# -------------------- Plotting Function --------------------
def plot_training_curves(eval_results):
    train_loss = eval_results['validation_0']['mlogloss']
    val_loss = eval_results['validation_1']['mlogloss']
    iterations = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8,6))
    plt.plot(iterations, train_loss, label='Train Loss')
    plt.plot(iterations, val_loss, label='Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Log Loss')
    plt.title('XGBoost Training Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.close()

    if run:
        run.upload_file(name='training_curves.png', path_or_stream='training_curves.png')
    logging.info("Training curves plot saved and uploaded as training_curves.png")

# -------------------- Model Training --------------------
def train_model(X_train, y_train, X_test, y_test, num_classes):
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes,
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        random_state=42,
        verbosity=1
    )

    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    results = model.evals_result()
    for i, loss in enumerate(results['validation_0']['mlogloss']):
        if run:
            run.log(f"Train Loss Iteration {i+1}", loss)
    for i, loss in enumerate(results['validation_1']['mlogloss']):
        if run:
            run.log(f"Validation Loss Iteration {i+1}", loss)

    if run:
        run.log("Final Train Loss", results['validation_0']['mlogloss'][-1])
        run.log("Final Validation Loss", results['validation_1']['mlogloss'][-1])

    plot_training_curves(results)

    return model

# --- Added evaluation functions ---
def evaluate_model_performance(y_true, y_pred, le_target, run=None, dataset_name=""):
    # Decode numeric labels to original risk level names
    y_true_labels = le_target.inverse_transform(y_true)
    y_pred_labels = le_target.inverse_transform(y_pred)

    # Classification report
    report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
    print(f"\nClassification Report ({dataset_name}):\n", classification_report(y_true_labels, y_pred_labels))
    if run:
        # Log per-class metrics to Azure ML run
        for class_label, metrics in report.items():
            if isinstance(metrics, dict):
                run.log(f"{dataset_name}_Precision_{class_label}", metrics.get('precision', 0))
                run.log(f"{dataset_name}_Recall_{class_label}", metrics.get('recall', 0))
                run.log(f"{dataset_name}_F1-score_{class_label}", metrics.get('f1-score', 0))

    # Confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=le_target.classes_)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_target.classes_)

    plt.figure(figsize=(8,6))
    cm_display.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title(f'Confusion Matrix - {dataset_name} Risk Prediction Model')
    plt.tight_layout()

    cm_filename = f"confusion_matrix_{dataset_name.lower()}.png"
    plt.savefig(cm_filename)
    plt.close()

    if run:
        run.upload_file(name=cm_filename, path_or_stream=cm_filename)
        print(f"Confusion matrix ({dataset_name}) saved and uploaded.")

# -------------------- Power BI Output --------------------
def prepare_powerbi_outputs(df, predictions, probabilities, le_target, le_country, le_disaster, le_region):
    df['predicted_risk_level'] = le_target.inverse_transform(predictions)
    df['country'] = le_country.inverse_transform(df['country_encoded'])
    df['disaster_type'] = le_disaster.inverse_transform(df['disaster_type_encoded'])
    df['region'] = le_region.inverse_transform(df['region_encoded'])

    for i, risk_class in enumerate(le_target.classes_):
        df[f'prob_{risk_class}'] = probabilities[:, i]

    df['priority_score'] = df['risk_score'] * 100
    df['resource_priority'] = np.where(
        df['predicted_risk_level'] == 'Critical', 'Immediate',
        np.where(df['predicted_risk_level'] == 'High', 'High', 'Medium')
    )

    selected_columns = [
        'disaster', 'year', 'country', 'region', 'disaster_type',
        'location_id', 'lat', 'lon',
        'risk_score', 'priority_score',
        'predicted_risk_level', 'resource_priority',
        'prob_Critical', 'prob_High', 'prob_Medium', 'prob_Low',
        'damage_level', 'population_density',
        'infrastructure_value', 'economic_activity'
    ]
    powerbi_df = df[selected_columns].copy()

    def get_likelihood_label(row):
        probs = {
            "1\nRare": row.get("prob_Low", 0),
            "3\nPossible": row.get("prob_Medium", 0),
            "4\nLikely": row.get("prob_High", 0),
            "5\nAlmost Certain": row.get("prob_Critical", 0),
        }
        label = max(probs, key=probs.get)
        score = int(label.split("\n")[0])
        return label, score

    def get_impact_label(damage_level):
        if damage_level == 0:
            return "1\nNo Damage", 1
        elif damage_level <= 0.33:
            return "2\nMinor Damage", 2
        elif damage_level <= 0.66:
            return "3\nMajor Damage", 3
        else:
            return "4\nCritical", 4

    def get_color(score):
        if score <= 3:
            return "#93D150"
        elif score <= 6:
            return "#FFFF00"
        elif score <= 10:
            return "#FFC100"
        else:
            return "#FF0000"

    matrix_rows = []
    for i, row in powerbi_df.iterrows():
        likelihood_label, likelihood_score = get_likelihood_label(row)
        impact_label, impact_score = get_impact_label(row["damage_level"])
        score = likelihood_score * impact_score
        risk_status = f"{row['disaster_type'].capitalize()} - {score}"
        color = get_color(score)

        matrix_rows.append({
            "Risk Temp": i + 1,
            "Category": "Disaster Risk",
            "Rows Headers": likelihood_label,
            "Columns Headers": impact_label,
            "Score": score,
            "Risk Status": risk_status,
            "Total risks": 1,
            "Color": color
        })

    risk_matrix_df = pd.DataFrame(matrix_rows)

    return powerbi_df, risk_matrix_df

def prepare_response_priority_matrix_extended(df):
    df = df.copy()

    def categorize_exposure_fixed(val):
        if val < 5:
            return 'Low'
        elif val <= 15:
            return 'Medium'
        else:
            return 'High'

    df['exposure_level'] = df['population_density'].apply(categorize_exposure_fixed)

    damage_mapping = {0: 'No Damage', 1: 'Minor Damage', 2: 'Major Damage', 3: 'Destroyed'}
    df['damage_category'] = df['damage_class'].map(damage_mapping)

    damage_weight = {'No Damage': 0, 'Minor Damage': 1, 'Major Damage': 2, 'Destroyed': 3}
    exposure_weight = {'Low': 1, 'Medium': 2, 'High': 3}

    df['damage_weight'] = df['damage_category'].map(damage_weight)
    df['exposure_weight'] = df['exposure_level'].map(exposure_weight)

    df['combined_score'] = df['damage_weight'] * df['exposure_weight']

    grouped = df.groupby(
        ['disaster', 'year', 'country', 'damage_category', 'exposure_level']
    ).agg(
        count=('damage_class', 'size'),
        avg_combined_score=('combined_score', 'mean')
    ).reset_index()

    return grouped

# -------------------- Main Pipeline --------------------
def main(input_csv, output_model, output_le, powerbi_output, risk_matrix_output, priority_matrix_output):
    logging.info("Loading and preparing data")
    df, le_target, le_country, le_disaster, le_region, raw_df = load_and_prepare_data(input_csv)

    feature_columns = [
        # Removed 'damage_level' to prevent leakage
        "population_density", "rainfall", "wind_speed",
        "lat", "lon", "country_encoded", "year", 
        "disaster_type_encoded", "temperature"
    ]
    X = df[feature_columns]
    y = df['risk_level_enc']

    logging.info("Splitting data into training and test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logging.info("Training XGBoost model")
    model = train_model(X_train, y_train, X_test, y_test, num_classes=len(le_target.classes_))

    # Accuracy logging
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    logging.info(f"Train Accuracy: {train_acc:.4f}")
    logging.info(f"Test Accuracy: {test_acc:.4f}")
    if run:
        run.log("Train Accuracy", train_acc)
        run.log("Test Accuracy", test_acc)

    logging.info("Generating predictions for evaluation")
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # --- Added evaluation ---
    # Train set evaluation
    evaluate_model_performance(y_train, y_pred_train, le_target, run, dataset_name="Train")

    # Test set evaluation
    evaluate_model_performance(y_test, y_pred, le_target, run, dataset_name="Test")

    logging.info("Generating Power BI outputs")
    powerbi_df, risk_matrix_df = prepare_powerbi_outputs(df.copy(), model.predict(X), model.predict_proba(X), le_target, le_country, le_disaster, le_region)
    powerbi_df.to_csv(powerbi_output, index=False)
    logging.info(f"Power BI data saved to {powerbi_output}")

    risk_matrix_df.to_csv(risk_matrix_output, index=False)
    logging.info(f"Risk Matrix data saved to {risk_matrix_output}")

    priority_matrix_df = prepare_response_priority_matrix_extended(raw_df)
    priority_matrix_df.to_csv(priority_matrix_output, index=False)
    logging.info(f"Disaster Response Priority Matrix saved to {priority_matrix_output}")

    logging.info("Saving model and label encoder")
    joblib.dump(model, output_model)
    joblib.dump(le_target, output_le)

    logging.info("Model training and export complete")

# -------------------- Entry Point --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost Risk Model with Power BI Outputs")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_model", type=str, required=True)
    parser.add_argument("--output_le", type=str, required=True)
    parser.add_argument("--powerbi_output", type=str, required=True)
    parser.add_argument("--risk_matrix_output", type=str, required=True)
    parser.add_argument("--priority_matrix_output", type=str, required=True)

    args = parser.parse_args()
    print("Main....")

    try:
        main(args.input_csv, args.output_model, args.output_le, args.powerbi_output, args.risk_matrix_output, args.priority_matrix_output)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        if run:
            run.log("Error", str(e))
        exit(1)
