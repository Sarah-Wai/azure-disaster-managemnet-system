#  Automated Disaster Management System (Azure ML + End-to-End Pipelines)

## Overview

This project presents a **production-style, end-to-end disaster management system** that integrates deep learning, big data enrichment, and cloud deployment using Microsoft Azure.

The system transforms **raw satellite imagery into real-world risk intelligence**, enabling faster and data-driven disaster response.

---

## System Highlights

* 🛰 Image-based damage classification (ResNet50 ensemble)
* 🌦 Multi-source data enrichment (Weather, Population, Geo)
* 📊 Risk prediction using XGBoost
* ☁️ Fully automated Azure ML pipelines
* 🔁 MLOps workflow (training → deployment → API → dashboard)
* 📡 Real-time inference via AKS endpoint
* 📈 Power BI + Streamlit visualization

---

##  End-to-End Pipeline Architecture

```
Raw xBD Dataset
   ↓
Preprocessing Pipeline
   ↓
ResNet50 Classification Pipeline
   ↓
Prediction CSV (registered as Azure ML Data Asset)
   ↓
Merge Metadata + Predictions
   ↓
Geo Enrichment (OpenCage API)
   ↓
Weather Enrichment (NASA POWER API)
   ↓
Population Enrichment (WorldPop GeoTIFF)
   ↓
XGBoost Risk Model
   ↓
Deployment (AKS API + Power BI + GitHub SAS)
```

---

##  Project Structure

```
azure-disaster-managemnet-system/
│
├── preprocessing-Pipeline/
│   ├── data_preprocessing.py
│   ├── pipeline_preprocess.py
│
├── resnet50-Pipeline/
│   ├── pipeline_resnet50.py
│   ├── register_asset.py
│
├── xGBoost-Pipeline/
│   ├── pipeline.py
│   ├── submit_pipeline.py
│   ├── train_xgboost.py
│   ├── enrich_data.py
│   ├── enrich_weather.py
│   ├── enrich_population.py
│   ├── merge_metadata_prediction.py
│   ├── deploy_to_aks.py
│   ├── score.py
│
├── environments/
│   ├── custom_env.yaml
│   ├── xgboost-conda.yaml
│   ├── xgboost-pipeline-env.yaml
│
└── README.md
```

---

##  Key Pipelines

### 1️⃣ Preprocessing Pipeline

* Resize images to 224×224
* Extract damage class from JSON labels
* Compute geospatial features (lat/lon)
* Generate metadata CSV for downstream tasks

---

### 2️⃣ Classification Pipeline (ResNet50 Ensemble)

* Models:

  * ResNet50
  * DenseNet121
  * EfficientNet-B0
* Ensemble prediction via logit averaging
* Handles class imbalance with augmentation

Result:

* Train Accuracy: **93.6%**
* Test Accuracy: **67.5%**

---

### 3️⃣ Data Enrichment Pipeline

####  Geo Enrichment

* OpenCage API
* Adds country, region, ISO code

####  Weather Enrichment

* NASA POWER API
* Adds:

  * Temperature
  * Rainfall
  * Wind speed
* Includes caching + retry logic

####  Population Enrichment

* WorldPop GeoTIFF data
* Extracts population density using raster processing

---

### 4️⃣ XGBoost Risk Prediction Pipeline

#### Risk Formula:

```
R = 0.4D + 0.3P + 0.2I + 0.05Rf + 0.05W
```

#### Features:

* Damage level
* Population density
* Weather variables
* Location data

Result:

* Train Accuracy: **80.4%**
* Test Accuracy: **70.1%**

---

## ☁️ Azure Architecture

###  Azure Services Used

* Azure ML → pipeline orchestration
* Azure Blob Storage → data storage
* Azure Functions → automation trigger
* Azure Container Registry → environment images
* Azure Kubernetes Service → model deployment

---

## 🚀 Deployment

###  AKS Endpoint

```
POST /xgboost-risk-endpoint/score
```

###  Model Serving

* Dockerized environment
* Real-time inference API
* Scalable via Kubernetes

---

##  Outputs

### Power BI Dashboard

* Global disaster risk visualization
* Resource allocation insights
* Trend analysis

### Streamlit App

* Interactive map
* Risk matrix
* Weather correlation
* Probability explorer

---

##  MLOps Features

* Modular pipeline design
* YAML-based reusable components
* Environment version control
* Data asset registration
* Automated pipeline execution
* GitHub + SAS integration for data sharing

---

##  Limitations

* Limited dataset (~5,500 images)
* Minor damage classification difficulty
* External API dependency (weather, geo)
* Moderate overfitting in models

---

##  Future Improvements

* Real-time image upload UI
* Pre + post disaster comparison
* Advanced augmentation (GANs)
* Active learning / human-in-the-loop
* API fallback mechanisms

---

##  Author

**Sarah Wai (Wai Phu Paing)**
MSc Data Science

GitHub: https://github.com/Sarah-Wai

---

##  Why This Project Matters

This project goes beyond model building by delivering a **complete AI system** that connects:

* Computer Vision
* Tabular ML
* Big Data
* Cloud Deployment

 Transforming raw disaster data into **actionable intelligence in hours instead of days**
