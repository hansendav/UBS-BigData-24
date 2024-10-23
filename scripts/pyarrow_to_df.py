import pyarrow.fs as fs
from pyarrow.fs import FileSelector

import pyarrow.parquet as pq
import pandas as pd

s3 = fs.S3FileSystem()

# paths to parquet and directories
parquet_path = "s3://ubs-datasets/bigearthnet/metadata.parquet"

s1_path = "BigEarthNet-S1"
s2_path = "BigEarthNet-S2"
ref_maps_paths = "Reference_Maps"

def extract_filenames(fs, s3_bucket, s3_path):
        selector = fs.FileSelector(f"{s3_bucket}/{s3_path}/", recursive=True)
        file_info = fs.get_file_info(selector)
        return [info.path.split('/')[-1].split('.')[0] for info in file_info if info.is_file]

s1_paths = extract_filenames(s3, "ubs-datasets/bigearthnet", s1_path)
s2_paths = extract_filenames(s3, "ubs-datasets/bigearthnet", s2_path)
ref_maps_paths = extract_filenames(s3, "ubs-datasets/bigearthnet", ref_maps_paths)

# read parquet file
meta = pq.read_table(parquet_path).to_pandas()

print(s1_paths[:5])
print(s2_paths[:5])
print(ref_maps_paths[:5])     

print(meta)