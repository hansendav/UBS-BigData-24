import pyarrow.fs as fs
import pyarrow.parquet as pq
import pandas as pd


parquet_path = "s3://ubs-datasets/bigearthnet/metadata.parquet"
meta = pq.read_table(parquet_path).to_pandas()

s3 = fs.S3FileSystem()

s1_file_info = s3.get_file_info(fs.FileSelector(
   'ubs-datasets/bigearthnetBigEarthNet-S1/', recursive=True
))

print(s1_file_info)