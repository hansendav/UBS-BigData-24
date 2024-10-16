import pyarrow.fs as fs
import pyarrow.parquet as pq
import pandas as pd

# Set up S3 
s3 = fs.S3FileSystem()

"""
# Read parquet file to pandas df
parquet_path = "s3://ubs-datasets/bigearthnet/metadata.parquet"
meta = pq.read_table(parquet_path).to_pandas()
"""

selector = fs.FileSelector("ubs-datasets/bigearthnet/BigEarthNet-S1/", recursive=True)
file_info = s3.get_file_info(selector)
file_names = [info.path.split('/')[-1].split('.')[0] for info in file_info]
print(file_names[:5]) 

"""
s1_path = "BigEarthNet-S1/"
s2_path = "BigEarthNet-S2/"
ref_maps_paths = "Reference_Maps/"

def extract_filenames(fs, s3_bucket, s3_path):
        selector = fs.FileSelector(f"{s3_bucket}/{s3_path}", recursive=True)
        file_info = fs.get_file_info(selector)
        file_names = [info.path.split('/')[-1].split('.')[0] for info in file_info if info.is_file]
        return file_names

s1_files = extract_filenames(s3, "ubs-datasets/bigearthnet", s1_path)
s2_files = extract_filenames(s3, "ubs-datasets/bigearthnet", s2_path)
ref_files = extract_filenames(s3, "ubs-datasets/bigearthnet", ref_maps_paths)

print(s1_files[:5])
"""