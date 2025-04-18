"""
    把行情数据和财务数据合并
"""

import pandas as pd

financial_data = pd.read_parquet("./data/raw_csi300_constituent_financials.parquet")
trading_data = pd.read_parquet("./data/raw_csi300_constituent_prices.parquet")

# --- 数据预处理 ---

# 1. 转换日期列为 datetime 对象
financial_data['REPORT_PERIOD'] = pd.to_datetime(financial_data['REPORT_PERIOD'])
trading_data['TRADE_DT'] = pd.to_datetime(trading_data['TRADE_DT'])

# 1.5 (新增) 将财务数据日期推迟一天，以避免前瞻性偏差
# 假设报告期 T 的数据在 T+1 才可用
financial_data['REPORT_PERIOD'] = financial_data['REPORT_PERIOD'] + pd.Timedelta(days=1)

# 2. 重命名日期列以便合并
financial_data = financial_data.rename(columns={'REPORT_PERIOD': 'DATE'})
trading_data = trading_data.rename(columns={'TRADE_DT': 'DATE'})

# --- 合并数据 (Groupby 方法) ---

# 先对 financial_data 按股票和日期排序一次，这有助于后续 merge_asof 的效率
financial_data = financial_data.sort_values(by=['S_INFO_WINDCODE', 'DATE'])

# 定义合并操作函数
def merge_single_stock(group_trading, financial_all):
    """
    对单个股票的交易数据(group_trading)和对应的财务数据进行合并。
    """
    stock_code = group_trading['S_INFO_WINDCODE'].iloc[0]
    financial_subset = financial_all[financial_all['S_INFO_WINDCODE'] == stock_code].copy()

    # 确保子集内按日期排序 (关键步骤)
    group_trading = group_trading.sort_values(by='DATE')
    financial_subset = financial_subset.sort_values(by='DATE') # 再次确认排序

    # 如果该股票没有财务数据，则返回原始交易数据，并填充 NaN
    if financial_subset.empty:
        # 获取 financial_data 中独有的列名 (除了 'S_INFO_WINDCODE' 和 'DATE')
        fin_cols_to_add = financial_all.columns.difference(group_trading.columns).tolist()
        if 'S_INFO_WINDCODE' in fin_cols_to_add: fin_cols_to_add.remove('S_INFO_WINDCODE')
        if 'DATE' in fin_cols_to_add: fin_cols_to_add.remove('DATE')

        # --- 优化：一次性添加所有 NA 列以避免碎片化 ---
        if fin_cols_to_add: # 只有当确实有列需要添加时才执行
            # 创建一个包含所有新列的 DataFrame，填充 NA，并使用 group_trading 的索引
            na_df = pd.DataFrame(pd.NA, index=group_trading.index, columns=fin_cols_to_add)
            # 使用 concat 横向合并，效率更高
            group_trading = pd.concat([group_trading, na_df], axis=1)
        # --- 优化结束 ---

        print(f"警告: 股票 {stock_code} 没有找到对应的财务数据，已填充 NA。")
        return group_trading

    # 执行 merge_asof
    merged_group = pd.merge_asof(
        group_trading,
        financial_subset,
        on='DATE',
        by='S_INFO_WINDCODE', # 确保按股票代码匹配
        direction='backward'  # 确保是前向填充
    )
    return merged_group

# 按股票代码分组，并应用合并函数
# group_keys=False 避免将分组键添加到索引中
print("开始按股票分组进行合并...")
all_merged_data = trading_data.groupby('S_INFO_WINDCODE', group_keys=False).apply(
    lambda x: merge_single_stock(x, financial_data)
)
print("合并完成。")

# 重置索引，使 DataFrame 结构更规整
merged_data = all_merged_data.reset_index(drop=True)

# --- 查看结果 (可选) ---
print("合并后数据的形状:", merged_data.shape)
print("合并后数据的前几行:")
print(merged_data.head())
print("\n合并后数据的列名:")
print(merged_data.columns)
print("\n合并后数据的缺失值统计 (部分列):")
# 选择几个可能来自 financial_data 的列查看缺失情况
financial_cols_sample = financial_data.columns.difference(trading_data.columns).tolist()[:5]
if financial_cols_sample:
    print(merged_data[financial_cols_sample].isnull().sum())
else:
    # 如果没有独有列（例如 financial_data 列完全被 trading_data 包含）
    # 随机选几列看看
    cols_to_check = merged_data.columns[2:7] # 示例：检查第3到第7列
    print(merged_data[cols_to_check].isnull().sum())


# 可以选择保存合并后的数据
print("正在保存合并后的数据...")
merged_data.to_parquet("./data/merged_daily_data_grouped.parquet", index=False)
print("数据已保存到 ./data/merged_daily_data_grouped.parquet")
