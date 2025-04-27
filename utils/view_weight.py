import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.normpath(os.path.join(script_dir, '..', 'data'))

weights_file = os.path.join(data_dir, "raw_csi300_constituent_weights.parquet")

weights = pd.read_parquet(weights_file)

# 按TRADE_DT分组，并按I_WEIGHT加权，计算每日的权重
daily_sumed_weights = weights.groupby('TRADE_DT').apply(lambda x: x.set_index('S_INFO_WINDCODE')['I_WEIGHT'].sum()).reset_index()