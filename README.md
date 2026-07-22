# AML-GNN (aml-gnn-upi)

**Graph Neural Network-based Anti-Money Laundering (AML) Fraud Detection for UPI Transactions**

AML-GNN is an end-to-end machine learning system designed to detect fraudulent patterns (Smurfing, Layering, Round-tripping) in Unified Payments Interface (UPI) transaction graphs using PyTorch Geometric (PyG), FastAPI, Streamlit, and NetworkX.

---

## 🏛️ System Architecture

```
[ Synthetic UPI Generator ] ──> [ Transaction Graph Builder ] (9 Node / 5 Edge Features)
                                           │
       ┌───────────────────────────────────┴───────────────────────────────────┐
       ▼                                                                       ▼
[ GNN PyG Models ]                                                  [ Pattern Detection Engine ]
 (GraphSAGE, GAT, Hetero-GNN)                                        (Smurfing, Layering, Cycles)
       │                                                                       │
       ├───────────────────────────────────┬───────────────────────────────────┤
       ▼                                   ▼                                   ▼
[ FastAPI Inference Engine ]       [ GNN Explainer / SHAP ]             [ Streamlit Dashboard ]
 (/predict, /explain, /alerts)      (Feature Attribution)                (5 Interactive Pages)
```

---

## 🚀 Key Features

- **PyTorch Geometric Architectures**: Production-ready GraphSAGE, GAT (Graph Attention Networks), and Hetero-GNN implementations with batch normalization, dropout, and residual connections.
- **Stratified Split Evaluation**: Node classification trained on 70% train split and evaluated on stratified val/test splits (15%/15%) to report honest out-of-sample accuracy, precision, recall, F1-score, and AUC-ROC.
- **Graph Topology Mining**: High-performance detection of:
  - **Smurfing**: High out-degree fan-out split transactions.
  - **Layering**: Multi-hop transaction chains.
  - **Round-Tripping**: Cycle detection with amount ratio checking.
- **FastAPI Endpoint Service**:
  - `/api/v1/predict`: Batch & single-account risk scoring with persistent graph context and automated `AlertGenerator` triggers.
  - `/api/v1/explain`: Feature attributions via `GNNExplainer` and `FeatureAttribution`.
  - `/api/v1/alerts`: Active alert history and summary reports.
- **Streamlit Interactive Dashboard**:
  - Reusable components (`GraphVisualizationComponent`, `risk_meter`).
  - `@st.cache_resource` / `@st.cache_data` performance caching.
  - 5 complete pages: Dashboard, Live Alerts, Graph Explorer, Pattern Catalog, Multi-Model Comparison.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+

### Setup Steps
```bash
# 1. Clone the repository and navigate into it
cd aml-gnn-upi

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install in editable mode with dependencies
pip install -e .
```

---

## 🚦 Usage & Running Instructions

### 1. Run Multi-Model Training & Benchmark Comparison
Train GraphSAGE, GAT, and Hetero-GNN on stratified node splits and save evaluation metrics & checkpoints:
```bash
python scripts/train_and_compare.py
```

### 2. Run Full Pipeline Demonstration
```bash
# Standard pipeline
python scripts/run_pipeline.py

# Fast pipeline demo
python scripts/run_pipeline.py --fast
```

### 3. Launch FastAPI REST Service
```bash
uvicorn src.api.main:app --reload --port 8000
```
- Interactive API Docs: `http://localhost:8000/docs`

### 4. Launch Streamlit Web Dashboard
```bash
streamlit run src/dashboard/app.py
```

### 5. Run Scalability Benchmarks & Export Model
```bash
# Run cycle detection scalability benchmark
python scripts/benchmark_scalability.py

# Export model checkpoint & metadata
python scripts/export_model.py
```

### 6. Run Test Suite
```bash
pytest -v
```

---

## 📊 Directory Layout

```
aml-gnn-upi/
├── config/                  # Configuration & settings
├── data/                    # Raw & processed synthetic graph datasets
├── models_saved/            # Saved model checkpoints & comparison CSVs
├── notebooks/               # EDA, Feature Engineering & Pattern Analysis notebooks
├── scripts/                 # Execution scripts (train, benchmark, export)
├── src/
│   ├── api/                 # FastAPI routes (predict, explain, health)
│   ├── dashboard/           # Streamlit app & reusable UI components
│   ├── data/                # Data generator & PyG graph builder
│   ├── detection/           # Pattern detector & Alert Generator
│   ├── explainability/      # GNNExplainer & FeatureAttribution
│   ├── models/              # PyG GraphSAGE, GAT, Hetero-GNN & Trainer
│   ├── streaming/           # Optional Redis transaction streaming
│   └── utils/               # PDF/CSV/JSON report generators
├── tests/                   # Pytest automated test suite
├── pyproject.toml           # Standard package definition
└── requirements.txt         # Dependency manifest
```
