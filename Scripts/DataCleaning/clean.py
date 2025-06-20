
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, trim, lower


def main():
    # Initialize Spark session
    spark = (SparkSession.builder
        .appName("CleanedAccidentsJob") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate())

    # Input and output paths 
    input_path = "gs://us_accidents_tasks/US_Accidents_March23.csv"
    output_csv = "gs://us_accidents_tasks/cleaned_accidents/csv/"

    # Read raw CSV with header and infer schema
    df = (spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(input_path))

    # Display the inferred schema and a few sample rows for inspection
    print("=== Initial Schema ===")
    df.printSchema()
    print("=== Sample Rows ===")
    df.show(5, truncate=False)

    # 1. Drop fully duplicate rows based on unique ID
    df = df.dropDuplicates(["ID"] )

    # 2. Parse timestamp fields to proper TimestampType
    df = (df
        .withColumn("Start_Time", to_timestamp(col("Start_Time"), "yyyy-MM-dd HH:mm:ss"))
        .withColumn("End_Time",   to_timestamp(col("End_Time"),   "yyyy-MM-dd HH:mm:ss")))

    # 3. Filter out rows with null critical fields
    df = df.filter(
        col("ID").isNotNull() &
        col("Start_Time").isNotNull() &
        col("Severity").isNotNull()
    )

    # 4. Fill nulls in numeric columns where a default makes sense
    numeric_defaults = {
        "Distance(mi)": 0.0,
        "Temperature(F)": 32.0,
        "Humidity(%)": 1.0
    }
    df = df.fillna(numeric_defaults)

    # 5. Standardize text columns: trim whitespace and lowercase
    text_cols = ["State", "City", "Description"]
    for col_name in text_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, trim(lower(col(col_name))))

    # 6. remove outliers, e.g., zero-distance severe accidents
    df = df.filter(~((col("Severity") >= 4) & (col("Distance(mi)") == 0)))

    # Write cleaned dataset as Parquet for efficient analytics
    df.write \
        .mode("overwrite") \
        .parquet(output_parquet)
    
     # Coalesce into a single partition and write as CSV with header
    df.coalesce(1) \
      .write \
      .mode("overwrite") \
      .option("header", True) \
      .csv(output_csv)
    

    # Stop Spark session
    spark.stop()


if __name__ == "__main__":
    main()
