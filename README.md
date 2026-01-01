<div align="center">

# Spark ETL Pipeline

🚀 **Production-grade data engineering for financial transaction processing and fraud detection.**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PySpark](https://img.shields.io/badge/PySpark-3.5-orange?logo=apache-spark)
![Delta Lake](https://img.shields.io/badge/Delta_Lake-3.3-00ADD8)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52-FF4B4B?logo=streamlit)
![Tests](https://img.shields.io/badge/Tests-32_Passing-green)

</div>

## ✨ What It Does

- 🔄 Processes **500K+ financial transactions** using Apache Spark
- 🔍 Detects fraud patterns through **analytics and machine learning**
- 📊 Generates **professional reports** and **interactive dashboards**

## 📋 Prerequisites

- **Python 3.11+**
- **Java 17+** (required by PySpark)
  - Windows: Download from [Eclipse Adoptium](https://adoptium.net/temurin/releases/?version=17)
  - Or via winget: `winget install EclipseAdoptium.Temurin.17.JDK`
  - After install, set `JAVA_HOME` (see Troubleshooting)
- **[uv](https://github.com/astral-sh/uv)** package manager: `pip install uv`

## 🚀 Quick Start

```bash
git clone https://github.com/VijayAdithyaBK/spark-etl-pipeline.git
cd spark-etl-pipeline
pip install uv && uv sync

# Run the pipeline
uv run python main.py

# Launch the dashboard
uv run streamlit run dashboard/app.py
```

## 🛠️ Features

| Module                    | Description                                                 |
| ------------------------- | ----------------------------------------------------------- |
| **ETL Pipeline**          | CSV extraction, transformation chains, Delta Lake storage   |
| **Feature Engineering**   | 26 derived features (time, amount, velocity, merchant risk) |
| **Analytics**             | 6 fraud analysis modules with HTML/JSON reports             |
| **ML Model**              | Random Forest classifier (0.94 AUC-ROC)                     |
| **Customer Segmentation** | K-Means clustering for risk profiling                       |
| **Geographic Analysis**   | Haversine distance, fraud hotspots                          |
| **Dashboard**             | Streamlit + Plotly interactive visualizations               |

## 🏗️ Architecture

```
spark-etl-pipeline/
├── src/
│   ├── config/       # Pydantic settings
│   ├── core/         # Decorators, generators, context managers
│   ├── spark/        # Session, transformations, UDFs
│   ├── etl/          # Extractors, transformers, Delta Lake
│   ├── features/     # 26 engineered features
│   ├── analytics/    # Fraud analytics, segmentation, geographic
│   ├── ml/           # Random Forest model
│   └── quality/      # Validators, reports
├── dashboard/        # Streamlit app
├── tests/            # 32 tests
└── main.py
```

## 📈 Results

| Metric                 | Value                   |
| ---------------------- | ----------------------- |
| Transactions Processed | **555,719**             |
| Fraud Rate             | **0.39%** (2,145 cases) |
| Amount at Risk         | **$1,133,324**          |
| Model AUC-ROC          | **0.94**                |

## 🧪 Testing

```bash
uv run pytest tests/ -v
```

## 🔧 Troubleshooting

**Java not found error:**
```powershell
# Windows: Set JAVA_HOME before running
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.17.10-hotspot"
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
uv run python main.py
```

**winutils.exe warning (Windows only):**
This is a Hadoop dependency warning. The pipeline handles it gracefully by skipping Parquet output and using CSV/JSON instead. No action needed.

## 📚 Documentation

- [Development Journey](DEVELOPMENT.md) - Thought process and design decisions

## 📄 License

MIT

---

<div align="center">

**⭐ If you find this project interesting, please consider giving it a star! ⭐**

---

### 💼 Open to Opportunities

**Data Engineer** | 4+ years in production data platforms

📧 [vijayadithyabk@gmail.com](mailto:vijayadithyabk@gmail.com) | 🔗 [LinkedIn](https://www.linkedin.com/in/vijayadithyabk/) | 💻 [GitHub](https://github.com/VijayAdithyaBK)

---

*⚡ Crafted by Vijay Adithya B K*

</div>
