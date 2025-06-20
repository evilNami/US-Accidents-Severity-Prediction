from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("EDA_spark") \
    .getOrCreate()

# Path to the cleaned dataset in GCS
file_path = "gs://us_accidents_tasks/cleaned_accidents/csv/"

# Load the cleaned dataset into a DataFrame
df = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv(file_path)

# 1. Accident Count by Severity
eda_q1 = df.groupBy("Severity").count() \
    .withColumnRenamed("count", "Accident_Count")
eda_q1.coalesce(1) \
    .write \
    .option("header", True) \
    .mode("overwrite") \
    .csv("gs://us_accidents_tasks/EDA/SPARK/accident_count_by_severity")

# 2. Top 10 States by Number of Accidents
eda_q2 = df.groupBy("State").count() \
    .withColumnRenamed("count", "State_Accident_Count") \
    .orderBy("State_Accident_Count", ascending=False) \
    .limit(10)
eda_q2.coalesce(1) \
    .write \
    .option("header", True) \
    .mode("overwrite") \
    .csv("gs://us_accidents_tasks/EDA/SPARK/top_10_states")

# 3. Average Distance by Severity
eda_q3 = df.groupBy("Severity").avg("Distance(mi)") \
    .withColumnRenamed("avg(Distance(mi))", "Average_Distance")
eda_q3.coalesce(1) \
    .write \
    .option("header", True) \
    .mode("overwrite") \
    .csv("gs://us_accidents_tasks/EDA/SPARK/avg_distance_by_severity")

# 4. Accidents by Hour of Day
from pyspark.sql.functions import hour
eda_q4 = df.withColumn("Hour_of_Day", hour("Start_Time")) \
    .groupBy("Hour_of_Day").count() \
    .withColumnRenamed("count", "Accident_Count") \
    .orderBy("Hour_of_Day")
eda_q4.coalesce(1) \
    .write \
    .option("header", True) \
    .mode("overwrite") \
    .csv("gs://us_accidents_tasks/EDA/SPARK/accidents_by_hour")

# 5. Top 5 Cities with Highest Severity-4+ Accidents
eda_q5 = df.filter(df.Severity >= 4) \
    .groupBy("City").count() \
    .withColumnRenamed("count", "Severe_Accident_Count") \
    .orderBy("Severe_Accident_Count", ascending=False) \
    .limit(5)
eda_q5.coalesce(1) \
    .write \
    .option("header", True) \
    .mode("overwrite") \
    .csv("gs://us_accidents_tasks/EDA/SPARK/top_5_cities_severity4_plus")

# Stop the Spark session
spark.stop()
