
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import (VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler)
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import subprocess, shlex

# -----------------------
# 1. Spark Session
# -----------------------
spark = (SparkSession.builder
         .appName("AccidentSeverityRF")
         .config("spark.executor.memory", "4g")
         .config("spark.driver.memory", "4g")
         .getOrCreate())

# -----------------------
# 2. Load cleaned dataset
# -----------------------
DATA_PATH = "gs://us_accidents_tasks/cleaned_accidents/parquet/"
df = spark.read.parquet(DATA_PATH)

# Cast boolean columns to string so StringIndexer can handle them
bool_cols = ["Amenity", "Crossing", "Junction", "Traffic_Signal"]
for b in bool_cols:
    if b in df.columns:
        df = df.withColumn(b, col(b).cast("string"))

# -----------------------
# 3. Feature Engineering
# -----------------------
NUMERIC_COLS = [
    "Distance(mi)", "Temperature(F)", "Humidity(%)", "Visibility(mi)",
    "Wind_Speed(mph)", "Precipitation(in)"
]
CATEGORICAL_COLS = [
    "Weather_Condition", "Amenity", "Crossing", "Junction",
    "Traffic_Signal", "Sunrise_Sunset"
]

# Fill remaining nulls in numeric columns to avoid VectorAssembler failure
fill_dict = {col_name: -1 for col_name in NUMERIC_COLS}
df = df.fillna(fill_dict)

indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
            for c in CATEGORICAL_COLS]
encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe", handleInvalid="keep")
            for c in CATEGORICAL_COLS]

assembler_inputs = NUMERIC_COLS + [f"{c}_ohe" for c in CATEGORICAL_COLS]
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="assembled_features", handleInvalid="keep")
scaler = StandardScaler(inputCol="assembled_features", outputCol="features")
label_indexer = StringIndexer(inputCol="Severity", outputCol="label", handleInvalid="keep")

# -----------------------
# 4. Model Definition
# -----------------------
rf = RandomForestClassifier(labelCol="label", featuresCol="features",
                           numTrees=100, maxDepth=10, seed=42)

pipeline = Pipeline(stages=indexers + encoders +
                    [assembler, scaler, label_indexer, rf])

# -----------------------
# 5. Train‑Test Split
# -----------------------
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# -----------------------
# 6. Fit Model
# -----------------------
model = pipeline.fit(train_df)

# -----------------------
# 7. Predictions & Metrics
# -----------------------
predictions = model.transform(test_df)

accuracy = MulticlassClassificationEvaluator(labelCol="label",
                                            predictionCol="prediction",
                                            metricName="accuracy").evaluate(predictions)
f1 = MulticlassClassificationEvaluator(labelCol="label",
                                       predictionCol="prediction",
                                       metricName="f1").evaluate(predictions)

print("=== Model Evaluation ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1‑Score: {f1:.4f}")

# -----------------------
# 8A. Confusion Matrix Heatmap
# -----------------------
num_classes = predictions.select("label").distinct().count()
pl_rdd = predictions.select("prediction", "label").rdd.map(lambda r: (r[0], r[1]))
conf_mtx = MulticlassMetrics(pl_rdd).confusionMatrix().toArray().astype(int)

plt.figure(figsize=(8, 6))
plt.imshow(conf_mtx, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix – Severity (Random Forest)")
plt.colorbar()
plt.xticks(np.arange(num_classes), labels=[str(i) for i in range(num_classes)])
plt.yticks(np.arange(num_classes), labels=[str(i) for i in range(num_classes)])
plt.xlabel("Predicted")
plt.ylabel("Actual")

thresh = conf_mtx.max() / 2
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, format(conf_mtx[i, j], "d"), ha="center", va="center",
                 color="white" if conf_mtx[i, j] > thresh else "black")
plt.tight_layout()
local_conf_png = "/tmp/confusion_matrix_rf.png"
plt.savefig(local_conf_png, dpi=150)
plt.close()


# -----------------------
# 8. Persist Artefacts to GCS
# -----------------------
MODEL_PATH = "gs://us_accidents_tasks/ML/models/random_forest_severity"
PREDS_PATH = "gs://us_accidents_tasks/ML/predictions_random_forest"
CONF_PNG_GCS = "gs://us_accidents_tasks/ML/confusion_matrix_rf.png"

model.write().overwrite().save(MODEL_PATH)
(predictions.select("ID", "Severity", "prediction")
            .coalesce(1)
            .write.mode("overwrite").option("header", True).csv(PREDS_PATH))

for local_file, gcs_target in [(local_conf_png, CONF_PNG_GCS)]:
    try:
        subprocess.check_call(shlex.split(f"gsutil cp {local_file} {gcs_target}"))
        print(f"Uploaded {gcs_target}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to upload {local_file} to GCS", e)

# -----------------------
# 10. Clean‑up
# -----------------------
spark.stop()
