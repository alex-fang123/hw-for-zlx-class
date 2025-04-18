"""
    测试对单个股票进行行情数据和财务数据的合并
"""

import pandas as pd

financial_data = pd.read_parquet("./data/raw_csi300_constituent_financials.parquet")
trading_data = pd.read_parquet("./data/raw_csi300_constituent_prices.parquet")

financial_data['REPORT_PERIOD'] = pd.to_datetime(financial_data['REPORT_PERIOD'])
trading_data['TRADE_DT'] = pd.to_datetime(trading_data['TRADE_DT'])

financial_data = financial_data.rename(columns={'REPORT_PERIOD': 'DATE'})
trading_data = trading_data.rename(columns={'TRADE_DT': 'DATE'})

stock_code = '000001.SZ'

sample_financial = financial_data[financial_data['S_INFO_WINDCODE'] == stock_code]
sample_trading = trading_data[trading_data['S_INFO_WINDCODE'] == stock_code]

merged_group = pd.merge_asof(
    sample_trading,
    sample_financial,
    on='DATE',
)