import pyarrow.fs as fs

import pyarrow.parquet as pq
import pandas as pd


parquet_path = "s3://ubs-datasets/bigearthnet/metadata.parquet"
meta = pq.read_table(parquet_path).to_pandas()
print(meta)