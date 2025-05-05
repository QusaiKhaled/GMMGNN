# Semi-Supervised Graph-Based Anomaly Detection in River Monitoring Stations

This repository contains the code and documentation for a semi-supervised anomaly detection framework using Graph Neural Networks (GNNs) and Gaussian Mixture Models (GMMs) to monitor water quality in river systems. The approach models the spatial relationships between monitoring stations and detects anomalies in real-time with minimal labeled data.

The repo was submitted based on the abstract submitted to the Netherlands Center for River Studies titled A semi-supervised graph-based approach for anomaly
detection in river monitoring stations. Citation details will be added once abstract is published.


## 🧠 Overview

Traditional water quality monitoring methods often miss sudden pollution events. This project proposes a graph-based machine learning framework that:

- Represents river monitoring stations as nodes in a graph
- Uses GMMs to identify initial anomalies in an unsupervised way
- Trains a GNN encoder-decoder on normal data to reconstruct expected water quality patterns
- Flags deviations as anomalies based on reconstruction error

## 📊 Dataset

The model uses publicly available data from:

- **Southern Bug River (Ukraine)** – 22 monitoring stations
- **Time period** – 2000 to 2021
- **Water quality indicators**:
  - NH₄, NO₃, NO₂, SO₄, PO₄, Cl, O₂, suspended solids, BSK5

Dataset Source: [Kaggle - River Water Quality EDA and Forecasting (Mokin, 2020)](https://www.kaggle.com/datasets)

## 🔧 Methodology

1. **Anomaly Identification**:
   - Fit GMM to water quality features
   - Flag data in the lowest 5% of log-likelihood scores as anomalies

2. **Graph Construction**:
   - Create a graph from upstream to downstream connections using station metadata

3. **GNN Architecture**:
   - Encoder: Graph Attention Convolution layers
   - Decoder: Fully connected layers
   - Loss: Mean Absolute Error (MAE)

4. **Training**:
   - Train only on non-anomalous data (90%) to learn normal behavior
   - Use the remaining data for validation/testing

5. **Inference**:
   - Compare reconstruction error to a threshold
   - Flag values exceeding the threshold as anomalies

## 📈 Results

- **Accuracy**: 97.95%
- **Precision**: 79.71%
- **Recall**: 79.14%
- **F1-Score**: 79.42%
- Performance evaluated on spatially varying anomalies across stations

## 📂 Repository Structure
