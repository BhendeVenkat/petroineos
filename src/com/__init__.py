from powerservice import trading
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import pyspark
from pyspark.sql import SparkSession, DataFrame
from datetime import datetime
import pytz
import numpy as np

schema = ["check_type", "sub type", "input count", "valid record count", "check fail record count"]


def get_spark_session():
    spark = SparkSession.builder.master("local[1]").appName("Petroineos_trading_reporting").getOrCreate()
    return spark

# driver method to call others
def calculate_average():
    spark = get_spark_session()
    extract_date = ""
    trades = trading.get_trades(extract_date)
    print(trades)
    df = spark.sparkContext.parallelize(trades).toDF()
    flatten_records = generate_flatten(df)
    rejected_records = generate_audit(flatten_records)
    valid_records = generate_valid_record(flatten_records, rejected_records)
    valid_records = valid_records.withColumn("hr", split(col("time"), ":").getItem(0).cast("int"))
    valid_records = valid_records.filter(col("hr") < lit(23))

    dbutils = get_db_utils(spark)
    root_target_dir = "dbfs:/FileStore/python_powerservice_report/"

    agg_sum_per_hour = get_agg_sum_per_hour(valid_records)
    save_output(
        dbutils,
        agg_sum_per_hour,
        "csv",
        f"{root_target_dir}tmp/agg_sum_per_hour",
        f"{root_target_dir}agg_sum_per_hour",
        True)
    save_output(
        dbutils,
        agg_sum_per_hour,
        "delta",
        f"{root_target_dir}tmp/agg_sum_per_hour",
        f"{root_target_dir}agg_sum_per_hour",
        True)

    agg_count_per_hour = get_agg_count_per_hour(valid_records)
    save_output(
        dbutils,
        agg_count_per_hour,
        "csv",
        f"{root_target_dir}tmp/agg_count_per_hour",
        f"{root_target_dir}agg_count_per_hour",
        True)
    save_output(
        dbutils,
        agg_count_per_hour,
        "delta",
        f"{root_target_dir}tmp/agg_count_per_hour",
        f"{root_target_dir}agg_count_per_hour",
        True)

    agg_avg_per_hour = get_agg_avg_per_hour(valid_records)
    save_output(
        dbutils,
        agg_avg_per_hour,
        "csv",
        f"{root_target_dir}tmp/agg_avg_per_hour",
        f"{root_target_dir}agg_avg_per_hour",
        True)
    save_output(
        dbutils,
        agg_avg_per_hour,
        "delta",
        f"{root_target_dir}tmp/agg_avg_per_hour",
        f"{root_target_dir}agg_avg_per_hour",
        True)

    check_fail_data = perform_audit_check(spark, flatten_records)
    save_output(
        dbutils,
        check_fail_data,
        "csv",
        f"{root_target_dir}tmp/record_check",
        f"{root_target_dir}agg_sum_per_hour",
        False)
    save_output(
        dbutils,
        check_fail_data,
        "csv",
        f"{root_target_dir}tmp/record_check",
        f"{root_target_dir}agg_count_per_hour",
        False)
    save_output(
        dbutils,
        check_fail_data,
        "csv",
        f"{root_target_dir}tmp/record_check",
        f"{root_target_dir}agg_avg_per_hour",
        False)


# Audit generation
def generate_audit(df: DataFrame):
    return df.filter(((col("time") == lit(np.nan)) | (col("volume").isNull())))


def generate_valid_record(df: DataFrame, rejected_record_df: DataFrame):
    return df.join(rejected_record_df, ["pk"], "anti")


# used to flat the records from list to rows
def generate_flatten(df: DataFrame):
    df = df\
        .select(col("date"), col("id"), arrays_zip(col("time"), col("volume")).alias("tmp"))\
        .withColumn("tmp", explode(col("tmp")))
    exploded_df = df\
        .withColumn("time", col("tmp.time"))\
        .withColumn("volume", col("tmp.volume"))\
        .drop("tmp")\
        .withColumn("pk", monotonically_increasing_id())
    return exploded_df


# Method used to calculate hourly sum
def get_agg_sum_per_hour(df: DataFrame):
    df = df\
        .groupBy("hr")\
        .agg(sum("volume").alias("Volume"))\
        .orderBy("hr")\
        .withColumn("hr", when(col("hr") < 10, concat(lit("0"), str(col("hr")))).otherwise(col("hr")))\
        .select(concat_ws(":", col("hr"), lit("00")).alias("Local Time"),"Volume")\
        .drop("hr")
    return df


# Method used to calculate hourly count
def get_agg_count_per_hour(df: DataFrame):
    df = df \
        .groupBy("hr") \
        .agg(count("volume").alias("Volume")) \
        .orderBy("hr") \
        .withColumn("hr", when(col("hr") < 10, concat(lit("0"), str(col("hr")))).otherwise(col("hr"))) \
        .select(concat_ws(":", col("hr"), lit("00")).alias("Local Time"),"Volume")\
        .drop("hr")
    return df


# Method used to calculate hourly average
def get_agg_avg_per_hour(df: DataFrame):
    df = df \
        .groupBy("hr") \
        .agg(avg("volume").cast("decimal(24,2)").alias("Volume")) \
        .orderBy("hr") \
        .withColumn("hr", when(col("hr") < 10, concat(lit("0"), str(col("hr")))).otherwise(col("hr"))) \
        .select(concat_ws(":", col("hr"), lit("00")).alias("Local Time"),"Volume")\
        .drop("hr")
    return df


# write dataframe to temp first and then move to actual directory
def save_output(dbutils, df: DataFrame, output_format: str, tmp_target_dir: str, target_dir: str, data: bool):
    df.coalesce(1).write.format(output_format).option("header", True).mode("overwrite").save(tmp_target_dir)
    file_name = get_file_name()
    if not data:
        file_name = file_name+"_data_quality"
    dbutils.fs.mv(f"{tmp_target_dir}*.{output_format}", f"{target_dir}{file_name}.{output_format}")


def get_db_utils(spark):
    dbutils = None
    if spark.conf.get("spark.databricks.service.client.enabled") == "true":
        from pyspark.dbutils import DBUtils
        dbutils = DBUtils(spark)
    else:
        import IPython
        dbutils = IPython.get_ipython().user_ns["dbutils"]

    return dbutils


# Generate file name from current timestamp in UK time zone
def get_file_name():
    uk_tz = pytz.timezone("Europe/London")
    local_time = datetime.now(uk_tz).strftime("%Y%m%d_%H%M")
    file_name = "PowerPosition_"+local_time

    return file_name


# performing different data quality check
def perform_audit_check(spark: SparkSession, df: DataFrame):
    audit_df = start_end_time_check(spark, df).\
        unionAll(time_format_check).\
        unionAll(missing_values_check).\
        unionAll(time_interval_check)
    return audit_df


def start_end_time_check(spark: SparkSession, df: DataFrame):
    total_count = df.count()
    df = df\
        .withColumn("date_time", concat_ws(" ", col("date"), col("time")))\
        .withColumn("date_time", to_timestamp(col("date_time"), "MM/dd/yyyy HH:mm"))
    df = df.groupBy("id").agg(min("date_time").alias("start_time"), max("date_time").alias("end_time"))
    df = df.filter(col("start_time").isNull() | col("start_time").isNull() | (hour(col("start_time")) != 0) | (min(col("start_time")) != 0) | (hour(col("end_time")) != 23) | (min(col("end_time")) != 55))
    record_count = df.count()
    return_data = [("start_end_time_check", "Either start time is not 00:00 or end time is not 23:55", total_count, (total_count-record_count), record_count)]

    return spark.sparkContext.parallelize(return_data).toDF(schema)


def time_format_check(spark: SparkSession, df: DataFrame):
    total_count = df.count()
    df = df\
        .withColumn("hr", (split(col("time"), lit(":"))).getItem(0).cast("int"))\
        .withColumn("mn", (split(col("time"), lit(":"))).getItem(1).cast("int"))\
        .filter(col("hr").isNull() | col("mn").isNull() | ~col("hr").between(0, 23) | ~col("mn").between(0, 55))
    hour_exceed_23_count = df.filter(~col("hr").between(0, 23)).count()
    min_exceed_55_count = df.filter(~col("mn").between(0, 55)).count()
    invalid_hour_min = df.filter(col("hr").isNull() | col("mn").isNull()).count()
    return_data = [("time_format_check", "hour exceeded 23", total_count,(total_count - hour_exceed_23_count), hour_exceed_23_count),
                   ("time_format_check", "minute exceeded 55", total_count,(total_count - min_exceed_55_count), min_exceed_55_count),
                   ("time_format_check", "either hour or minute is invalid", total_count,(total_count - invalid_hour_min), invalid_hour_min)]
    return spark.sparkContext.parallelize(return_data).toDF(schema)


def missing_values_check(spark: SparkSession, df: DataFrame):
    total_count = df.count()
    df = df.filter(
        col("id").isNull() |
        col("date").isNull() |
        col("time").isNull() |
        col("volume").isNull() |
        col("time") == lit(np.nan) |
        col("volume") == lit(np.nan))
    missing_value_count = df.count()
    return_data = [("missing_values_check", "ALl column should have values and not NaN or None or empty", total_count,
                    (total_count - missing_value_count), missing_value_count)]

    return spark.sparkContext.parallelize(return_data).toDF(schema)


def time_interval_check(spark: SparkSession, df: DataFrame):
    total_count = df.count()
    window_Spec = Window.partitionBy(col("id")).orderBy(col("date_time").asc())
    df = df \
        .withColumn("date_time", concat_ws(" ", col("date"), col("time"))) \
        .withColumn("date_time", to_timestamp(col("date_time"), "MM/dd/yyyy HH:mm"))\
        .withColumn("date_time_lag", lag(col("date_time").over(window_Spec))) \
        .withColumn("rank", rank().over(window_Spec)) \
        .withColumn("interval_in_min", (unix_timestamp(col("date_time")) - unix_timestamp(col("date_time_lag")))/60)\
        .filter((col("rank") != lit(1)) & (col("interval_in_min").isNull() | col("interval_in_min") != lit(5)))
    time_interval_check_fail_count = df.count()
    return_data = [("time_interval_check", "time interval is not 5 minutes", total_count, (total_count - time_interval_check_fail_count), time_interval_check_fail_count)]
    return spark.sparkContext.parallelize(return_data).toDF(schema)


if __name__ == '__main__':
    calculate_average()
