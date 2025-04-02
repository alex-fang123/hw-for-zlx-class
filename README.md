# 献给玛丽卡.zlx的代码

本项目包含一系列用于从 Oracle 数据库下载、处理和查看金融数据的 Python 脚本。

## 根目录文件/目录说明

*   **`db_config.ini`**: (配置文件) 存储数据库连接所需的凭证和配置信息（主机、端口、服务名、用户名、密码、Instant Client 路径）。脚本会从此文件读取配置，避免在代码中硬编码敏感信息。**请务必在使用前填入正确的用户名和密码。**
*   **`oracle_download.py`**: 核心数据下载脚本。**注意：当前版本仅下载原始数据，不进行合并或日度化处理。**
    *   连接到 `db_config.ini` 中配置的 Oracle 数据库。
    *   下载沪深300指数 (`000300.SH`) 的行情数据 (`AINDEXEODPRICES`)。
    *   下载沪深300指数 (`000300.SH`) 的历史成分股列表 (`AINDEXMEMBERS`)。
    *   基于成分股列表，下载所有涉及成分股的：
        *   日度行情数据 (`ASHAREEODPRICES`)。
        *   日度权重数据 (`AINDEXHS300CLOSEWEIGHT`)。
        *   财务指标数据 (`ASHAREFINANCIALINDICATOR`)。
    *   将这五类原始数据分别保存到 `./data/` 目录下的 Parquet 文件中（文件名以 `raw_` 开头）。
*   **`hello.py`**: 一个简单的 "Hello, World!" 脚本，可能用于测试 Python 环境。
*   **`./data/`**: 存放由 `oracle_download.py` 生成的原始数据文件。
    *   `readme.md`: 描述该目录下数据文件的详细信息（包括列注释）。
*   **`./utils/`**: 存放辅助性的 Python 脚本。
    *   `list_tables.py`: 连接数据库并列出可能相关的表。
    *   `get_columns.py`: 连接数据库并查询指定表的列名。
    *   `get_column_comments.py`: 连接数据库并查询指定表的列名和注释。
    *   `view_data.py`: 读取并显示 `./data/` 目录下生成的 Parquet 文件内容概览。

## 如何运行

1.  **配置数据库连接**: 编辑根目录下的 `db_config.ini` 文件，填入正确的 Oracle 用户名和密码。
2.  **下载数据**: 在根目录下运行 `python oracle_download.py`。这将连接数据库并下载五类原始数据到 `./data/` 目录。
3.  **查看数据 (可选)**: 在根目录下运行 `python utils/view_data.py` 来查看已下载数据文件的概览。
4.  **探索数据库 (可选)**:
    *   运行 `python utils/list_tables.py` 查看可能相关的表。
    *   运行 `python utils/get_columns.py <OWNER> <TABLE_NAME>` 查看指定表的列名。
    *   运行 `python utils/get_column_comments.py <OWNER> <TABLE_NAME>` 查看指定表的列名和注释。
