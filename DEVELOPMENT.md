# Development Journey

This document explains my thought process, research, and design decisions while building this project.

---

## Why This Project?

I wanted to work with Apache Spark, but I needed a real problem to solve. So I asked: *"Where is Spark actually used in production?"*

**Research findings:**
- Financial services process billions of transactions daily. Banks, payment processors, and fintech companies run Spark clusters to handle this volume.
- Fraud detection is a critical use case. Every major card network (Visa, Mastercard) uses real-time and batch analytics to flag suspicious transactions.
- The challenge isn't just processing data; it's extracting patterns from massive datasets where fraud is less than 1% of transactions.

**Why finance and fraud detection specifically?**
- It's a real business problem with measurable impact (money saved from prevented fraud)
- The data has clear structure: transactions, merchants, customers, timestamps
- It requires both analytics (understanding patterns) and ML (predicting fraud)
- Data quality matters hugely; false positives block legitimate customers, false negatives cost money

This gave me my core challenges:
1. **Scale**: Process 500K+ records without performance degradation
2. **Pattern Detection**: Find the 0.4% of fraudulent transactions hidden in legitimate ones
3. **Actionable Output**: Produce insights that fraud analysts can actually use

---

## Research Phase: What Tools Should I Use?

I evaluated several approaches:

| Requirement            | Options Considered           | Final Choice      | Why                                                  |
| ---------------------- | ---------------------------- | ----------------- | ---------------------------------------------------- |
| Distributed Processing | Pandas, Dask, PySpark        | **PySpark**       | Industry standard, scales to petabytes, SQL-like API |
| Data Storage           | CSV, Parquet, Delta Lake     | **Delta Lake**    | ACID transactions, time travel, schema evolution     |
| Configuration          | ENV vars, YAML, Pydantic     | **Pydantic**      | Type safety, validation, IDE autocomplete            |
| ML Framework           | scikit-learn, MLlib, XGBoost | **PySpark MLlib** | Native Spark integration, distributed training       |
| Dashboard              | Flask, Dash, Streamlit       | **Streamlit**     | Rapid prototyping, Python-native, built-in charts    |

**Key Learning:** The best tool isn't always the most powerful. It's the one that fits the ecosystem and scales with the problem.

---

## Architecture Decision: How Should the Code Be Organized?

I started by sketching the data flow:

```
Raw Data -> Extract -> Transform -> Validate -> Analyze -> Report
```

But this linear thinking was too simplistic. I realized:
- Transformations need to be **composable** (chain multiple operations)
- Validation should be **pluggable** (add new rules without changing core code)
- Analytics should be **modular** (run independently or together)

This led to my design principles:

1. **Single Responsibility**: Each class does one thing well
2. **Open/Closed Principle**: Extend through composition, not modification
3. **Dependency Injection**: Pass configuration, don't hardcode

**Example:** Instead of one monolithic `process_data()` function, I created:
- `TransformerChain`: Compose transformers like LEGO blocks
- `CompositeValidator`: Stack validation rules flexibly
- `FraudAnalytics`: Run any subset of 6 analytics modules

---

## Learning: Advanced Python Patterns

I identified patterns that senior engineers use and implemented each:

| Pattern              | What I Learned                                                    | Where I Applied It                                                |
| -------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Decorators**       | Functions that wrap functions, perfect for cross-cutting concerns | `@timer`, `@retry`, `@validate_input` in `src/core/decorators.py` |
| **Generators**       | Memory-efficient iteration, critical for large datasets           | Batch processing in `src/core/generators.py`                      |
| **Context Managers** | Resource cleanup with `with` statements                           | Spark session lifecycle in `src/core/context_managers.py`         |
| **Metaclasses**      | Control class creation                                            | Singleton pattern for Spark session, auto-registration            |
| **Type Hints**       | Self-documenting code, IDE support                                | Every function signature in the project                           |

**Key Insight:** These patterns aren't academic. They solve real problems. Decorators eliminated duplicate timing code. Context managers prevented Spark session leaks.

---

## Feature Engineering: What Makes a Good Fraud Detector?

I researched fraud detection literature and identified that fraud often correlates with:
- **Time anomalies**: Transactions at unusual hours
- **Amount anomalies**: Deviation from customer's normal spending
- **Velocity anomalies**: Rapid successive transactions
- **Location anomalies**: Transactions far from usual location

This drove my feature engineering design:

```
26 Features Across 4 Categories:

Time (7 features)
├── hour, day_of_week, day_of_month, month
├── is_weekend, is_night, is_business_hours

Amount (8 features)
├── amount_zscore, log_amount, amount_bucket
├── is_round_amount
├── customer_avg_amount, customer_std_amount
└── amount_vs_customer_avg

Velocity (6 features)
├── customer_txn_count, customer_total_amount
├── txn_number, seconds_since_last_txn
└── is_rapid_txn

Merchant Risk (5 features)
├── merchant_txn_count, merchant_fraud_rate
├── merchant_avg_amount, merchant_risk_tier
└── amount_vs_merchant_avg
```

**Technical Challenge:** Computing customer-level statistics efficiently.

**Solution:** Spark Window Functions. They allow aggregations over partitions without shuffling all data to one node.

```python
customer_window = Window.partitionBy("cc_num")
df = df.withColumn("customer_avg_amount", F.avg("amt").over(customer_window))
```

---

## ML Model: Why Random Forest?

I considered several algorithms:

| Algorithm           | Pros                                      | Cons                         | Verdict           |
| ------------------- | ----------------------------------------- | ---------------------------- | ----------------- |
| Logistic Regression | Fast, interpretable                       | Assumes linear relationships | Too simple        |
| Random Forest       | Handles non-linearity, feature importance | Slower training              | **Chosen**        |
| Gradient Boosting   | Best accuracy                             | Black box, overfits easily   | Overkill for demo |
| Neural Network      | Powerful                                  | Needs more data, hardware    | Wrong tool        |

**Why Random Forest?**
1. **Interpretability**: I can show which features matter most
2. **Robustness**: Handles outliers and missing data well
3. **No feature scaling required**: Simplifies preprocessing
4. **Native PySpark support**: `RandomForestClassifier` in MLlib

---

## Dashboard Design: Who Is the User?

I asked: *"Who will actually use this dashboard?"*

Answer: Business analysts and fraud investigators, not data scientists.

This changed my approach:

| Developer Mindset | Business User Needs                                 |
| ----------------- | --------------------------------------------------- |
| Show all metrics  | Show only **actionable** metrics                    |
| Raw numbers       | **Relative comparisons** (7.8x higher than average) |
| Static tables     | **Interactive filters** (drill down by category)    |
| Technical jargon  | **Plain language** ("High Risk" not "p < 0.05")     |

---

## The Build Journey: Phase by Phase

### Phase 1: Foundation
**Goal:** Get data flowing through a basic pipeline

**Implemented:**
- Spark session with singleton pattern
- CSV extractor with schema inference
- Basic transformer chain

**Challenge:** Spark session kept creating duplicates in tests.
**Solution:** Implemented singleton pattern via metaclass.

---

### Phase 2: Data Quality
**Goal:** Validate data before expensive processing

**Implemented:**
- Null ratio validator
- Duplicate detector
- Composite validator pattern

**Learning:** Validation should fail fast. Running analytics on bad data wastes hours.

---

### Phase 3: Analytics Engine
**Goal:** Extract fraud insights from raw data

**Implemented 6 modules:**
1. Executive Summary: Overall fraud metrics
2. Category Analysis: Which product categories have highest fraud?
3. Merchant Analysis: Which merchants are risky?
4. Pattern Detection: Round amounts, high values, etc.
5. Anomaly Detection: Z-score based outlier identification
6. Time Trends: When does fraud peak?

**Key Decision:** Each module returns a standardized `AnalyticsResult` dataclass. This made report generation trivial, just iterate over results.

---

### Phase 4: Report Design
**Goal:** Present insights professionally

**First attempt:** Colorful dashboard with gradients and emojis.
**Problem:** Looked unprofessional, distracted from data.

**Redesign principles:**
- Clean white background
- Numbered academic sections
- Inline bar charts in tables
- Statistical language

**Outcome:** Report you'd present to a CFO.

---

### Phase 5: Feature Engineering
**Goal:** Create ML-ready features

**Research:** Read papers on fraud detection indicators.
**Implementation:** 26 features using Spark window functions.

**Technical insight:** Window functions are the secret weapon for feature engineering at scale. They compute aggregates without expensive joins.

---

### Phase 6: Customer Segmentation
**Goal:** Group customers by risk profile

**Approach:** K-Means clustering on behavioral metrics.
**Output:** 4 segments (Premium, Frequent, Budget, High-Risk).

**Why clustering?** Even without labeled data, you can find patterns. High-Risk segment had 3x fraud rate.

---

### Phase 7: Geographic Analysis
**Goal:** Add location-based fraud signals

**Implemented:**
- Haversine distance (customer location to merchant)
- Home location detection (median lat/long)
- Unusual transaction flagging

**Discovery:** Transactions >200km from customer's home have 5x fraud rate.

---

### Phase 8: ML Model
**Goal:** Predict fraud probability

**Pipeline:**
1. Feature assembly (VectorAssembler)
2. Scaling (StandardScaler)
3. Training (RandomForest, 100 trees)
4. Evaluation (AUC-ROC, confusion matrix)

**Result:** AUC-ROC of 0.94, strong discrimination.

---

### Phase 9: Dashboard
**Goal:** Self-service exploration for business users

**Tech:** Streamlit + Plotly

**Features:**
- Interactive filters
- Drill-down tables
- Temporal visualizations

---

### Phase 10: Delta Lake
**Goal:** Production-ready data storage

**Why Delta over Parquet?**
- ACID transactions (no corrupted writes)
- Time travel (query historical versions)
- Upsert support (update existing records)

---

## Key Learnings

1. **Validation saves debugging time.** Catching nulls at ingestion beats tracing them through 10 tables.
2. **Window functions are powerful.** They compute aggregates without expensive joins.
3. **Design for the user.** Business analysts need "High Risk" labels, not p-values.
4. **Modular code is testable code.** If I can't unit test it, I can't trust it.
5. **Iterate on design.** The first report version looked unprofessional. Redesigning it made the output usable.

---

## Challenges Faced

### Java Configuration on Windows

**Problem:** PySpark requires Java, but running the pipeline threw errors because Java wasn't found in the PATH.

**Investigation:** Spark looks for `JAVA_HOME` environment variable. On Windows, even with Java installed, this isn't always set correctly.

**Solution:** Explicitly set environment variables before running:
```powershell
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.17.10-hotspot"
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
uv run python main.py
```

**Learning:** Always document environment requirements. What works on your machine might break on someone else's.

---

### winutils.exe Warning on Windows

**Problem:** Spark threw warnings about missing `winutils.exe`, a Hadoop utility for Windows file system operations.

**Investigation:** This is a known issue when running Spark on Windows without a full Hadoop installation. It affects operations that write to the local filesystem using Hadoop's file system abstraction.

**Workaround:** Modified the pipeline to:
1. Skip Parquet output (which triggers the Hadoop file system)
2. Use CSV/JSON output instead for reports
3. Handle the warning gracefully without crashing

```python
# In main.py - graceful handling
logger.info("NOTE: Parquet output skipped (requires winutils.exe on Windows)")
logger.info("      The data was successfully processed and validated!")
```

**Learning:** Not every warning needs to be fixed. Sometimes the right approach is to work around it and document the limitation.
