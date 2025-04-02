# 献给玛丽卡.zlx的代码

本项目包含一系列用于从 Oracle 数据库下载、处理和查看金融数据的 Python 脚本。

## 根目录脚本说明

*   **`db_config.ini`**: (配置文件) 存储数据库连接所需的凭证和配置信息（主机、端口、服务名、用户名、密码、Instant Client 路径）。脚本会从此文件读取配置，避免在代码中硬编码敏感信息。**请务必在使用前填入正确的用户名和密码。**
*   **`list_tables.py`**: 连接到 `db_config.ini` 中配置的 Oracle 数据库，并列出用户可访问的、名称可能与金融数据相关的表（基于预定义的关键词过滤）。用于初步探索数据库结构。
*   **`get_columns.py`**: 连接到 `db_config.ini` 中配置的 Oracle 数据库，并查询指定表（默认为 `FILESYNC.ASHAREFINANCIALINDICATOR`，也可通过命令行参数 `python get_columns.py <OWNER> <TABLE_NAME>` 指定）的列名和数据类型。用于了解特定表的详细结构。
*   **`get_column_comments.py`**: 连接到 `db_config.ini` 中配置的 Oracle 数据库，并查询指定表（通过命令行参数 `python get_column_comments.py <OWNER> <TABLE_NAME>` 指定）的列名及其注释（来自 `ALL_COL_COMMENTS`）。用于获取列的详细含义。
*   **`oracle_download.py`**: 核心数据下载脚本。**注意：当前版本仅下载原始数据，不进行合并或日度化处理。**
    *   连接到 `db_config.ini` 中配置的 Oracle 数据库。
    *   下载沪深300指数 (`000300.SH`) 的行情数据 (`AINDEXEODPRICES`)。
    *   下载沪深300指数 (`000300.SH`) 的历史成分股列表 (`AINDEXMEMBERS`)。
    *   基于成分股列表，下载所有涉及成分股的：
        *   日度行情数据 (`ASHAREEODPRICES`)。
        *   日度权重数据 (`AINDEXHS300CLOSEWEIGHT`)。
        *   财务指标数据 (`ASHAREFINANCIALINDICATOR`)。
    *   将这五类原始数据分别保存到 `./data/` 目录下的 Parquet 文件中（文件名以 `raw_` 开头）。
*   **`view_data.py`**: 用于快速查看 `oracle_download.py` 生成的五个原始 Parquet 数据文件的内容。它会读取文件并打印每个文件的前5行、数据形状（行数、列数）以及列名和数据类型信息。
*   **`hello.py`**: 一个简单的 "Hello, World!" 脚本，可能用于测试 Python 环境。

## 数据目录

*   **`./data/`**: 存放由 `oracle_download.py` 生成的原始数据文件。
    *   `readme.md`: 描述该目录下数据文件的详细信息（包括列注释）。
