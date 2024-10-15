import pyarrow.fs as fs 
from pyspark.sql import SparkSession


spark = SparkSession.builder\
        .master("yarn")\
        .appName("pyarrow_to_df")\
        .getOrCreate()
s3_bucket = "ubs-datasets"
s3_path = "bigearthnet/" 

s3 = fs.S3FileSystem()

file_paths = s3.get_file_info(fs.FileSelector(f"{s3_bucket}/{s3_path}"),
                recursive=True)

file_paths = [info.path for info in file_paths]

