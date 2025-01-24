import pybaseballstats as pyb
import polars as pl
data = pyb.statcast_date_range('2015-01-01', '2024-12-31', extra_stats=True)

data.collect().write_parquet('data/raw/2015_2024_statcast_train.parquet')