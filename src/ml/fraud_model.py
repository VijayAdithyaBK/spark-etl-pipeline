"""
Fraud Detection Model Module.
Implements ML-based fraud classification using PySpark MLlib.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


@dataclass
class ModelMetrics:
    """Model evaluation metrics."""
    auc_roc: float
    auc_pr: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    feature_importance: list[dict] = field(default_factory=list)


class FraudDetectionModel:
    """
    Fraud detection using Random Forest classifier.
    
    Features:
    - Automatic feature selection and preprocessing
    - Train/test split with stratification
    - Model evaluation with multiple metrics
    - Feature importance analysis
    """
    
    # Default numeric features to use
    DEFAULT_NUMERIC_FEATURES = [
        "amt", "log_amount", "amount_zscore", "hour", "day_of_week",
        "is_weekend", "is_night", "customer_avg_amount", "customer_txn_count",
        "merchant_fraud_rate", "transaction_distance_km"
    ]
    
    # Default categorical features
    DEFAULT_CATEGORICAL_FEATURES = ["category", "amount_bucket"]
    
    def __init__(
        self,
        df: DataFrame,
        label_column: str = "is_fraud",
        numeric_features: Optional[list[str]] = None,
        categorical_features: Optional[list[str]] = None
    ):
        self.df = df
        self.label_col = label_column
        self.numeric_features = numeric_features or self.DEFAULT_NUMERIC_FEATURES
        self.categorical_features = categorical_features or self.DEFAULT_CATEGORICAL_FEATURES
        self._model = None
        self._train_df = None
        self._test_df = None
        self._metrics: Optional[ModelMetrics] = None
        self._predictions = None
    
    def prepare_features(self) -> DataFrame:
        """Prepare features for modeling."""
        logger.info("Preparing features for modeling...")
        
        df = self.df
        
        # Filter to only available features
        available_numeric = [f for f in self.numeric_features if f in df.columns]
        available_categorical = [f for f in self.categorical_features if f in df.columns]
        
        logger.info(f"Using {len(available_numeric)} numeric features: {available_numeric}")
        logger.info(f"Using {len(available_categorical)} categorical features: {available_categorical}")
        
        # Store for later use
        self._available_numeric = available_numeric
        self._available_categorical = available_categorical
        
        # Handle missing values in numeric features
        for col in available_numeric:
            df = df.fillna({col: 0})
        
        # Ensure label is integer
        df = df.withColumn(self.label_col, F.col(self.label_col).cast("integer"))
        
        return df
    
    def split_data(self, test_ratio: float = 0.2, seed: int = 42) -> tuple[DataFrame, DataFrame]:
        """Split data into train and test sets."""
        logger.info(f"Splitting data with {test_ratio:.0%} test ratio...")
        
        df = self.prepare_features()
        
        # Stratified split based on fraud label
        fraud_df = df.filter(F.col(self.label_col) == 1)
        non_fraud_df = df.filter(F.col(self.label_col) == 0)
        
        fraud_train, fraud_test = fraud_df.randomSplit([1-test_ratio, test_ratio], seed=seed)
        non_fraud_train, non_fraud_test = non_fraud_df.randomSplit([1-test_ratio, test_ratio], seed=seed)
        
        self._train_df = fraud_train.union(non_fraud_train)
        self._test_df = fraud_test.union(non_fraud_test)
        
        train_count = self._train_df.count()
        test_count = self._test_df.count()
        
        logger.info(f"Train set: {train_count:,} samples")
        logger.info(f"Test set: {test_count:,} samples")
        
        return self._train_df, self._test_df
    
    def train(self, n_trees: int = 100, max_depth: int = 10) -> "FraudDetectionModel":
        """Train the fraud detection model."""
        logger.info("Training Random Forest classifier...")
        
        if self._train_df is None:
            self.split_data()
        
        # Build preprocessing stages
        stages = []
        feature_cols = []
        
        # Index categorical features
        for cat_col in self._available_categorical:
            indexer = StringIndexer(
                inputCol=cat_col,
                outputCol=f"{cat_col}_indexed",
                handleInvalid="keep"
            )
            stages.append(indexer)
            feature_cols.append(f"{cat_col}_indexed")
        
        # Add numeric features
        feature_cols.extend(self._available_numeric)
        
        # Assemble features
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features_raw",
            handleInvalid="keep"
        )
        stages.append(assembler)
        
        # Scale features
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=True
        )
        stages.append(scaler)
        
        # Random Forest classifier
        rf = RandomForestClassifier(
            labelCol=self.label_col,
            featuresCol="features",
            numTrees=n_trees,
            maxDepth=max_depth,
            seed=42
        )
        stages.append(rf)
        
        # Build and fit pipeline
        pipeline = Pipeline(stages=stages)
        self._model = pipeline.fit(self._train_df)
        
        # Store feature columns for importance analysis
        self._feature_cols = feature_cols
        
        logger.info("Model training complete.")
        return self
    
    def evaluate(self) -> ModelMetrics:
        """Evaluate model performance."""
        logger.info("Evaluating model performance...")
        
        if self._model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make predictions
        self._predictions = self._model.transform(self._test_df)
        
        # AUC-ROC
        roc_evaluator = BinaryClassificationEvaluator(
            labelCol=self.label_col,
            metricName="areaUnderROC"
        )
        auc_roc = roc_evaluator.evaluate(self._predictions)
        
        # AUC-PR
        pr_evaluator = BinaryClassificationEvaluator(
            labelCol=self.label_col,
            metricName="areaUnderPR"
        )
        auc_pr = pr_evaluator.evaluate(self._predictions)
        
        # Accuracy
        acc_evaluator = MulticlassClassificationEvaluator(
            labelCol=self.label_col,
            metricName="accuracy"
        )
        accuracy = acc_evaluator.evaluate(self._predictions)
        
        # Precision, Recall, F1
        precision_evaluator = MulticlassClassificationEvaluator(
            labelCol=self.label_col,
            metricName="weightedPrecision"
        )
        precision = precision_evaluator.evaluate(self._predictions)
        
        recall_evaluator = MulticlassClassificationEvaluator(
            labelCol=self.label_col,
            metricName="weightedRecall"
        )
        recall = recall_evaluator.evaluate(self._predictions)
        
        f1_evaluator = MulticlassClassificationEvaluator(
            labelCol=self.label_col,
            metricName="f1"
        )
        f1 = f1_evaluator.evaluate(self._predictions)
        
        # Feature importance
        rf_model = self._model.stages[-1]
        importances = rf_model.featureImportances.toArray()
        
        feature_importance = [
            {"feature": feat, "importance": round(float(imp), 4)}
            for feat, imp in sorted(
                zip(self._feature_cols, importances),
                key=lambda x: x[1],
                reverse=True
            )
        ][:15]  # Top 15 features
        
        self._metrics = ModelMetrics(
            auc_roc=round(auc_roc, 4),
            auc_pr=round(auc_pr, 4),
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            feature_importance=feature_importance
        )
        
        logger.info(f"Model AUC-ROC: {auc_roc:.4f}")
        logger.info(f"Model Accuracy: {accuracy:.4f}")
        logger.info(f"Model F1 Score: {f1:.4f}")
        
        return self._metrics
    
    def predict(self, df: DataFrame) -> DataFrame:
        """Score new transactions."""
        if self._model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self._model.transform(df)
    
    def get_confusion_matrix(self) -> dict:
        """Get confusion matrix from predictions."""
        if self._predictions is None:
            raise ValueError("No predictions. Call evaluate() first.")
        
        # Calculate confusion matrix values
        tp = self._predictions.filter(
            (F.col(self.label_col) == 1) & (F.col("prediction") == 1)
        ).count()
        
        tn = self._predictions.filter(
            (F.col(self.label_col) == 0) & (F.col("prediction") == 0)
        ).count()
        
        fp = self._predictions.filter(
            (F.col(self.label_col) == 0) & (F.col("prediction") == 1)
        ).count()
        
        fn = self._predictions.filter(
            (F.col(self.label_col) == 1) & (F.col("prediction") == 0)
        ).count()
        
        return {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "total": tp + tn + fp + fn
        }
    
    def get_model_summary(self) -> dict:
        """Get complete model summary for reporting."""
        if self._metrics is None:
            self.evaluate()
        
        confusion = self.get_confusion_matrix()
        
        return {
            "model_type": "Random Forest Classifier",
            "metrics": {
                "auc_roc": self._metrics.auc_roc,
                "auc_pr": self._metrics.auc_pr,
                "accuracy": self._metrics.accuracy,
                "precision": self._metrics.precision,
                "recall": self._metrics.recall,
                "f1_score": self._metrics.f1_score
            },
            "confusion_matrix": confusion,
            "feature_importance": self._metrics.feature_importance,
            "train_samples": self._train_df.count() if self._train_df else 0,
            "test_samples": self._test_df.count() if self._test_df else 0
        }
