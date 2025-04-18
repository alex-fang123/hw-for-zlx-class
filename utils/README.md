# 工具脚本说明 (./utils/)

本目录包含一系列用于辅助数据处理、查询和检查的 Python 脚本。

## 脚本列表

*   **`check_merged_data.py`**:
    *   **功能**: 用于检查 `../data/merged_daily_data_grouped.parquet` 文件。可能包含数据完整性、格式或特定数值的验证逻辑。
    *   **依赖**: `../data/merged_daily_data_grouped.parquet`
    *   **运行**: `python utils/check_merged_data.py` (在项目根目录运行)

*   **`get_column_comments.py`**:
    *   **功能**: 连接到 `../db_config.ini` 中配置的 Oracle 数据库，查询指定表的列名及其注释。
    *   **依赖**: `../db_config.ini`, Oracle Instant Client, `oracledb` Python 包。
    *   **运行**: `python utils/get_column_comments.py <OWNER> <TABLE_NAME>` (在项目根目录运行)
    *   **示例**: `python utils/get_column_comments.py FILESYNC ASHAREFINANCIALINDICATOR`

*   **`get_columns.py`**:
    *   **功能**: 连接到 `../db_config.ini` 中配置的 Oracle 数据库，查询指定表的列名。
    *   **依赖**: `../db_config.ini`, Oracle Instant Client, `oracledb` Python 包。
    *   **运行**: `python utils/get_columns.py <OWNER> <TABLE_NAME>` (在项目根目录运行)
    *   **示例**: `python utils/get_columns.py FILESYNC AINDEXEODPRICES`

*   **`list_tables.py`**:
    *   **功能**: 连接到 `../db_config.ini` 中配置的 Oracle 数据库，列出数据库中可能与本项目相关的表名（基于一些预定义的模式或名称）。
    *   **依赖**: `../db_config.ini`, Oracle Instant Client, `oracledb` Python 包。
    *   **运行**: `python utils/list_tables.py` (在项目根目录运行)

*   **`merge_single_stock.py`**:
    *   **功能**: 包含 `merge_single_stock` 函数，该函数被 `../merge_data.py` 用于对单个股票的日度行情数据和季度财务数据进行 `merge_asof` 合并。此脚本本身可能不直接运行，而是作为模块被导入。
    *   **依赖**: `pandas` Python 包。
    *   **注意**: 此脚本通常不单独执行，其功能由 `../merge_data.py` 调用。

*   **`view_data.py`**:
    *   **功能**: 读取并显示 `../data/` 目录下生成的 Parquet 数据文件的内容概览（例如，文件形状、前几行、列名）。
    *   **依赖**: `pandas`, `pyarrow` 或 `fastparquet` Python 包。
    *   **运行**: `python utils/view_data.py` (在项目根目录运行)
