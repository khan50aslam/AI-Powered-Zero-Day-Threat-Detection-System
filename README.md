# AI Powered Zero-Day Threat Detection System

This project detects unknown ("zero-day") network threats by using an AI algorithm — Isolation Forest — to identify anomalies in network flow data.

## 🚀 Features:
- Detects unseen, unknown threats in real-time.
- Uses Isolation Forest, an unsupervised machine learning algorithm.
- No need for labeled data (it learns normal behavior automatically).
- Provides Anomaly Scores and Threat Classifications.
- Lightweight, fast, and scalable.

## 🔥 How it Works:
1. Collects network flow data (Source IP, Destination Port, Flow Duration, etc.).
2. Trains an Isolation Forest model on normal and mixed traffic.
3. New incoming network flows are evaluated:
   - If Anomaly Score > threshold → Marked as **Threat**.
   - Else → Marked as **Normal**.
4. Helps in early detection of zero-day attacks without needing manual signatures.

## 📈 Data Used:
- **Source IP**
- **Destination Port**
- **Flow Duration**
- **Total Forward Packets**
- **Total Length of Forward Packets**
- **Flow Bytes per Second**

These features are enough to detect suspicious activities in network traffic.

## 🛠 Tech Stack:
- Python
- pandas
- scikit-learn
- Isolation Forest (ML algorithm)

## 📦 Requirements:
```bash
pip install pandas scikit-learn

