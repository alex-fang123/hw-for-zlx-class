# 数据文件说明

本目录包含由 `oracle_download.py` 脚本从 Oracle 数据库直接下载的原始数据文件，格式为 Parquet。脚本未进行合并或日度化处理。

如果你在我的虚拟局域网内，可以访问http://10.181.253.68:5244/ 下载

## 1. `raw_csi300_index_prices.parquet`

*   **内容**: 沪深300指数 (`000300.SH`) 的原始日度行情数据。
*   **来源表**: `FILESYNC.AINDEXEODPRICES`
*   **时间范围**: 从 `oracle_download.py` 中 `START_YEAR_FILTER` (当前为 2000) 开始的所有数据。
*   **数据结构**: 每行代表指数在一个交易日的行情信息。
*   **列说明**:
    *   `OBJECT_ID`: 对象ID
    *   `S_INFO_WINDCODE`: Wind代码 (指数代码)
    *   `TRADE_DT`: 交易日期
    *   `CRNCY_CODE`: 货币代码
    *   `S_DQ_PRECLOSE`: 昨收盘价(点)
    *   `S_DQ_OPEN`: 开盘价(点)
    *   `S_DQ_HIGH`: 最高价(点)
    *   `S_DQ_LOW`: 最低价(点)
    *   `S_DQ_CLOSE`: 收盘价(点)
    *   `S_DQ_CHANGE`: 涨跌(点)
    *   `S_DQ_PCTCHANGE`: 涨跌幅(%)
    *   `S_DQ_VOLUME`: 成交量(手)
    *   `S_DQ_AMOUNT`: 成交金额(千元)
    *   `SEC_ID`: 证券ID
    *   `OPDATE`: 操作日期 (数据库内部使用)
    *   `OPMODE`: 操作模式 (数据库内部使用)

## 2. `raw_csi300_constituent_membership.parquet`

*   **内容**: 沪深300指数 (`000300.SH`) 的历史成分股纳入和剔除信息。
*   **来源表**: `FILESYNC.AINDEXMEMBERS`
*   **时间范围**: 所有历史记录。
*   **数据结构**: 每行代表一只股票的一次纳入或剔除事件。
*   **列说明**:
    *   `OBJECT_ID`: 对象ID
    *   `S_INFO_WINDCODE`: 指数Wind代码
    *   `S_CON_WINDCODE`: 成份股Wind代码
    *   `S_CON_INDATE`: 纳入日期
    *   `S_CON_OUTDATE`: 剔除日期 (如果当前仍在指数内，则为空)
    *   `CUR_SIGN`: 最新标志 (1表示最新记录)
    *   `OPDATE`: 操作日期 (数据库内部使用)
    *   `OPMODE`: 操作模式 (数据库内部使用)

## 3. `raw_csi300_constituent_prices.parquet`

*   **内容**: 沪深300指数历史成分股的原始日度行情数据。
*   **来源表**: `FILESYNC.ASHAREEODPRICES`
*   **时间范围**: 从 `oracle_download.py` 中 `START_YEAR_FILTER` (当前为 2000) 开始的所有数据。
*   **数据结构**: 每行代表一只成分股在一个交易日的行情信息。
*   **列说明**: (列出部分关键列)
    *   `OBJECT_ID`: 对象ID
    *   `S_INFO_WINDCODE`: Wind代码 (股票代码)
    *   `TRADE_DT`: 交易日期
    *   `CRNCY_CODE`: 货币代码
    *   `S_DQ_PRECLOSE`: 昨收盘价(元)
    *   `S_DQ_OPEN`: 开盘价(元)
    *   `S_DQ_HIGH`: 最高价(元)
    *   `S_DQ_LOW`: 最低价(元)
    *   `S_DQ_CLOSE`: 收盘价(元)
    *   `S_DQ_CHANGE`: 涨跌(元)
    *   `S_DQ_PCTCHANGE`: 涨跌幅(%)
    *   `S_DQ_VOLUME`: 成交量(手)
    *   `S_DQ_AMOUNT`: 成交金额(千元)
    *   `S_DQ_ADJPRECLOSE`: 复权昨收盘价(元)
    *   `S_DQ_ADJOPEN`: 复权开盘价(元)
    *   `S_DQ_ADJHIGH`: 复权最高价(元)
    *   `S_DQ_ADJLOW`: 复权最低价(元)
    *   `S_DQ_ADJCLOSE`: 复权收盘价(元)
    *   `S_DQ_ADJFACTOR`: 复权因子
    *   `S_DQ_AVGPRICE`: 均价(VWAP)
    *   `S_DQ_TRADESTATUS`: 交易状态
    *   `OPDATE`: 操作日期 (数据库内部使用)
    *   `OPMODE`: 操作模式 (数据库内部使用)
    *   *(包含其他行情相关列)*

## 4. `raw_csi300_constituent_weights.parquet`

*   **内容**: 沪深300指数成分股的原始日度权重数据。
*   **来源表**: `FILESYNC.AINDEXHS300CLOSEWEIGHT`
*   **时间范围**: 从 `oracle_download.py` 中 `START_YEAR_FILTER` (当前为 2000) 开始的所有数据。
*   **数据结构**: 每行代表一只成分股在一个交易日的权重信息。
*   **列说明**:
    *   `OBJECT_ID`: 对象ID
    *   `S_INFO_WINDCODE`: 指数Wind代码
    *   `S_CON_WINDCODE`: 成份股Wind代码
    *   `TRADE_DT`: 交易日期
    *   `I_WEIGHT`: 权重 (%)
    *   `S_IN_INDEX`: 计算用股本(股)
    *   `I_WEIGHT_11`: 总股本(股)
    *   `I_WEIGHT_14`: 权重因子
    *   `I_WEIGHT_15`: 收盘价
    *   `I_WEIGHT_17`: 总市值
    *   `I_WEIGHT_18`: 计算用市值
    *   `OPDATE`: 操作日期 (数据库内部使用)
    *   `OPMODE`: 操作模式 (数据库内部使用)
    *   *(包含其他权重相关列，部分可能为空)*

## 5. `raw_csi300_constituent_financials.parquet`

*   **内容**: 沪深300指数历史成分股的原始财务指标数据。
*   **来源表**: `FILESYNC.ASHAREFINANCIALINDICATOR`
*   **时间范围**: 所有历史记录。
*   **数据结构**: 每行代表一只股票在一个报告期的一次财务指标公告。
*   **列说明**: (包含大量列，此处仅列出核心部分)
    *   `OBJECT_ID`: 对象ID
    *   `S_INFO_WINDCODE`: Wind代码 (股票代码)
    *   `ANN_DT`: 公告日期
    *   `REPORT_PERIOD`: 报告期 (例如 '20230331', '20230630')
    *   `CRNCY_CODE`: 货币代码
    *   `S_FA_EPS_BASIC`: 基本每股收益
    *   `S_FA_BPS`: 每股净资产
    *   `S_FA_ROE`: 净资产收益率
    *   `S_FA_ROA`: 总资产净利率
    *   `S_FA_DEBTTOASSETS`: 资产负债率
    *   `S_FA_GROSSPROFITMARGIN`: 销售毛利率
    *   `S_FA_NETPROFITMARGIN`: 销售净利率
    *   `S_FA_YOYNETPROFIT`: 同比增长率-归属母公司股东的净利润(%)
    *   `S_FA_YOY_TR`: 营业总收入同比增长率(%)
    *   `S_FA_YOYOP`: 同比增长率-营业利润(%)
    *   `S_FA_YOYOCF`: 同比增长率-经营活动产生的现金流量净额(%)
    *   `OPDATE`: 操作日期 (数据库内部使用)
    *   `OPMODE`: 操作模式 (数据库内部使用)
    *   *(包含非常多的其他财务指标列)*

**注意**: 财务指标 (`raw_csi300_constituent_financials.parquet`) 文件包含了大量列。具体每个财务指标的精确含义建议参考数据源 (Wind数据库) 的字段说明文档或使用 `get_column_comments.py` 脚本查询。
