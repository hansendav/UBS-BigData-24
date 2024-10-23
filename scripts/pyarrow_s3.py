import pyarrow as pa
import pyarrow.fs as fs
import pyarrow.parquet as pq
import pandas as pd
import re 

# Set up S3 
s3 = fs.S3FileSystem()

bigearthnet_bucket = "s3://ubs-datasets/bigearthnet/"


# Read parquet file to pandas df
parquet_path = "s3://ubs-datasets/bigearthnet/metadata.parquet"
meta = pq.read_table(parquet_path).to_pandas()

meta['patch_id_path'] = meta['patch_id'].apply(lambda x: re.match(r'(.*)_[0-9]+_[0-9]+$', x).group(1))
meta['patch_id_path_s1'] = meta['s1_name'].apply(lambda x: re.match(r'(.*)_[A-Z0-9]+_[0-9]+_[0-9]+$', x).group(1))

meta['s1_path'] = bigearthnet_bucket + 'BigEarthNet-S1/'  + meta['patch_id_path_s1'] + '/' + meta['s1_name'] + '/'
meta['s2_path'] = bigearthnet_bucket + 'BigEarthNet-S2/'  + meta['patch_id_path'] + '/' + meta['patch_id'] +  '/' 
meta['label_path'] = bigearthnet_bucket + 'Reference_Maps/' + meta['patch_id_path'] + '/' + meta['patch_id'] + '/'

# Write to S3 
table = pa.Table.from_pandas(meta)
pq.write_table(table, 'ubs-cde/home/e2405193/bigdata/meta_with_image_paths.parquet', filesystem=s3)

print(
        f"Saved updated meta.parquet to:"
        f"\n s3://ubs-cde/home/e2405193/bigdata/meta_with_image_paths.parquet"
)