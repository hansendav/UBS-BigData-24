import pyarrow.fs as fs
import pyarrow.parquet as pq
import pandas as pd

# Set up S3 
s3 = fs.S3FileSystem()

# Read parquet file to pandas df
parquet_path = "s3://ubs-datasets/bigearthnet/metadata.parquet"
meta = pq.read_table(parquet_path).to_pandas()

s1_path = "BigEarthNet-S1/"
s2_path = "BigEarthNet-S2/"
ref_maps_paths = "Reference_Maps/"

def extract_filenames_to_df(fs, s3_bucket, s3_path):
        selector = fs.FileSelector(f"{s3_bucket}/{s3_path}", recursive=True)
        file_info = fs.get_file_info(selector)
        file_names = [info.path.split('/')[-1].split('.')[0] for info in file_info if info.is_file]
        return pd.DataFrame(file_names, columns="filename")

s1_files = extract_filenames(s3, "ubs-datasets/bigearthnet", s1_path)
s2_files = extract_filenames(s3, "ubs-datasets/bigearthnet", s2_path)
ref_files = extract_filenames(s3, "ubs-datasets/bigearthnet", ref_maps_paths)
merged_df = pd.concat([s1_files, s2_files, ref_files], axis=1)

print(merged_df.info())