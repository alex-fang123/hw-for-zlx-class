"""
    把行情数据和财务数据合并
"""

import pandas as pd

financial_data = pd.read_parquet("./data/raw_csi300_constituent_financials.parquet")
trading_data = pd.read_parquet("./data/raw_csi300_constituent_prices.parquet")