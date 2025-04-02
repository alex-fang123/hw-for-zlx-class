import oracledb
import pandas as pd
import getpass
import datetime
import sys
import warnings
import os
import configparser

# Suppress specific UserWarning from pandas about DBAPI2 connection
warnings.filterwarnings('ignore', message='pandas only supports SQLAlchemy connectable.*')

# --- Configuration ---
CONFIG_FILE = 'db_config.ini'

# Output filenames for RAW data
OUTPUT_INDEX_PRICES_FILENAME = "./data/raw_csi300_index_prices.parquet"
OUTPUT_CONSTITUENT_MEMBERSHIP_FILENAME = "./data/raw_csi300_constituent_membership.parquet"
OUTPUT_CONSTITUENT_PRICES_FILENAME = "./data/raw_csi300_constituent_prices.parquet" 
OUTPUT_CONSTITUENT_WEIGHTS_FILENAME = "./data/raw_csi300_constituent_weights.parquet" 
OUTPUT_CONSTITUENT_FINANCIALS_FILENAME = "./data/raw_csi300_constituent_financials.parquet"

# --- Table and Index Information ---
SCHEMA_PREFIX = "FILESYNC."
INDEX_CODE = "000300.SH"
CONSTITUENTS_TABLE = f"{SCHEMA_PREFIX}AINDEXMEMBERS"
STOCK_PRICE_TABLE = f"{SCHEMA_PREFIX}ASHAREEODPRICES"
FINANCIAL_INDICATOR_TABLE = f"{SCHEMA_PREFIX}ASHAREFINANCIALINDICATOR"
INDEX_PRICE_TABLE = f"{SCHEMA_PREFIX}AINDEXEODPRICES"
WEIGHT_TABLE = f"{SCHEMA_PREFIX}AINDEXHS300CLOSEWEIGHT" 
WEIGHT_COLUMN = "I_WEIGHT" 

# Define overall date range (optional, can be adjusted if needed)
# For raw download, we might not need START_YEAR/END_YEAR unless filtering is desired
# Let's keep START_YEAR for filtering weights and prices if needed, but download all financials/membership
START_YEAR_FILTER = 2000 # Or adjust as needed, e.g., 2024 for recent data only
START_DATE_FILTER_STR = f"{START_YEAR_FILTER}0101"

connection = None

# --- Helper Function for IN Clause ---
def build_in_clause(items):
    if not items:
        return "IN ('')" 
    if len(items) == 1:
        return f"= '{items[0]}'"
    return f"IN {tuple(items)}"

try:
    # Read configuration from file
    if not os.path.exists(CONFIG_FILE):
        print(f"错误: 配置文件 '{CONFIG_FILE}' 不存在。请先创建并配置该文件。")
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(CONFIG_FILE, encoding='utf-8') 

    try:
        db_user = config.get('database', 'user')
        db_password = config.get('database', 'password')
        db_host = config.get('database', 'host')
        db_port = config.get('database', 'port')
        db_service_name = config.get('database', 'service_name')
        instant_client_dir = config.get('database', 'instant_client_dir')
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        print(f"错误: 配置文件 '{CONFIG_FILE}' 格式不正确或缺少必要的键: {e}")
        sys.exit(1)

    if not db_user or db_user == 'YOUR_USERNAME_HERE' or not db_password or db_password == 'YOUR_PASSWORD_HERE':
        print(f"错误: 请在 '{CONFIG_FILE}' 中配置有效的数据库用户名和密码。")
        sys.exit(1)

    dsn = f"{db_host}:{db_port}/{db_service_name}"
    print(f"正在从 '{CONFIG_FILE}' 读取配置并连接到数据库: {dsn} 用户: {db_user}...")

    try:
        print(f"尝试使用指定的 Instant Client 目录初始化 Thick Mode: {instant_client_dir}")
        oracledb.init_oracle_client(lib_dir=instant_client_dir)
        connection = oracledb.connect(user=db_user, password=db_password, dsn=dsn)
        print("数据库连接成功 (Thick Mode)！")
    except Exception as thick_error:
        print(f"Thick mode connection failed: {thick_error}")
        sys.exit(1)

    # --- Check for pyarrow library ---
    try:
        import pyarrow
        print("检测到 'pyarrow' 库，将使用 Parquet 格式输出。")
    except ImportError:
        print("\n错误: 未找到 'pyarrow' 库，无法输出 Parquet 格式。")
        print("请先安装: pip install pyarrow")
        sys.exit(1)
        
    # --- Remove existing output files ---
    for filename in [OUTPUT_INDEX_PRICES_FILENAME, OUTPUT_CONSTITUENT_MEMBERSHIP_FILENAME, 
                     OUTPUT_CONSTITUENT_PRICES_FILENAME, OUTPUT_CONSTITUENT_WEIGHTS_FILENAME, 
                     OUTPUT_CONSTITUENT_FINANCIALS_FILENAME]:
        if os.path.exists(filename):
            print(f"删除已存在的数据文件: {filename}")
            os.remove(filename)

    # --- 1. Download Index Prices ---
    print(f"\n--- 正在下载指数 {INDEX_CODE} 的行情数据 ---")
    index_price_query = f"""
    SELECT * 
    FROM {INDEX_PRICE_TABLE}
    WHERE s_info_windcode = '{INDEX_CODE}'
      AND trade_dt >= '{START_DATE_FILTER_STR}' 
    ORDER BY trade_dt
    """
    try:
        df_index_prices = pd.read_sql(index_price_query, connection)
        print(f"获取到 {len(df_index_prices)} 条指数行情记录。")
        if not df_index_prices.empty:
            df_index_prices.to_parquet(OUTPUT_INDEX_PRICES_FILENAME, index=False, engine='pyarrow')
            print(f"指数行情数据已保存到: {OUTPUT_INDEX_PRICES_FILENAME}")
        else:
            print("未找到指数行情数据。")
    except Exception as e:
        print(f"下载指数行情数据时出错: {e}")

    # --- 2. Download Constituent Membership ---
    print(f"\n--- 正在下载指数 {INDEX_CODE} 的所有历史成分股信息 ---")
    constituent_query = f"""
    SELECT * 
    FROM {CONSTITUENTS_TABLE}
    WHERE S_INFO_WINDCODE = '{INDEX_CODE}'
    ORDER BY S_CON_INDATE, S_CON_WINDCODE
    """
    try:
        df_membership = pd.read_sql(constituent_query, connection)
        print(f"获取到 {len(df_membership)} 条成分股成员记录。")
        if not df_membership.empty:
            all_unique_constituent_codes = tuple(df_membership['S_CON_WINDCODE'].unique())
            print(f"涉及 {len(all_unique_constituent_codes)} 个不同的股票代码。")
            df_membership.to_parquet(OUTPUT_CONSTITUENT_MEMBERSHIP_FILENAME, index=False, engine='pyarrow')
            print(f"成分股成员数据已保存到: {OUTPUT_CONSTITUENT_MEMBERSHIP_FILENAME}")
        else:
            print("错误：未找到成分股成员数据。后续依赖此数据的下载将跳过。")
            all_unique_constituent_codes = tuple() # Empty tuple
    except Exception as e:
        print(f"下载成分股成员数据时出错: {e}")
        all_unique_constituent_codes = tuple() # Ensure it's empty on error

    # --- Proceed only if constituents were found ---
    if all_unique_constituent_codes:
        in_clause_all_stocks = build_in_clause(all_unique_constituent_codes)

        # --- 3. Download Constituent Prices ---
        print(f"\n--- 正在下载 {len(all_unique_constituent_codes)} 个成分股的行情数据 ---")
        stock_price_query = f"""
        SELECT * 
        FROM {STOCK_PRICE_TABLE}
        WHERE s_info_windcode {in_clause_all_stocks}
          AND trade_dt >= '{START_DATE_FILTER_STR}' 
        ORDER BY s_info_windcode, trade_dt
        """
        try:
            df_constituent_prices = pd.read_sql(stock_price_query, connection)
            print(f"获取到 {len(df_constituent_prices)} 条成分股行情记录。")
            if not df_constituent_prices.empty:
                df_constituent_prices.to_parquet(OUTPUT_CONSTITUENT_PRICES_FILENAME, index=False, engine='pyarrow')
                print(f"成分股行情数据已保存到: {OUTPUT_CONSTITUENT_PRICES_FILENAME}")
            else:
                print("未找到成分股行情数据。")
        except Exception as e:
            print(f"下载成分股行情数据时出错: {e}")

        # --- 4. Download Constituent Weights ---
        print(f"\n--- 正在下载 {len(all_unique_constituent_codes)} 个成分股的权重数据 (表: {WEIGHT_TABLE}) ---")
        weight_query = f"""
        SELECT * 
        FROM {WEIGHT_TABLE}
        WHERE S_INFO_WINDCODE = '{INDEX_CODE}' 
          AND S_CON_WINDCODE {in_clause_all_stocks}
          AND TRADE_DT >= '{START_DATE_FILTER_STR}' 
        ORDER BY S_CON_WINDCODE, TRADE_DT
        """
        try:
            df_weights = pd.read_sql(weight_query, connection)
            print(f"获取到 {len(df_weights)} 条成分股权重记录。")
            if not df_weights.empty:
                df_weights.to_parquet(OUTPUT_CONSTITUENT_WEIGHTS_FILENAME, index=False, engine='pyarrow')
                print(f"成分股权重数据已保存到: {OUTPUT_CONSTITUENT_WEIGHTS_FILENAME}")
            else:
                print(f"未从 {WEIGHT_TABLE} 获取到权重数据。")
        except Exception as e:
            print(f"下载成分股权重数据时出错: {e}")

        # --- 5. Download Constituent Financials ---
        print(f"\n--- 正在下载 {len(all_unique_constituent_codes)} 个成分股的财务指标数据 ---")
        financial_query = f"""
        SELECT * 
        FROM {FINANCIAL_INDICATOR_TABLE}
        WHERE s_info_windcode {in_clause_all_stocks}
        ORDER BY s_info_windcode, REPORT_PERIOD, ANN_DT /* Order by report period and announcement date */
        """
        try:
            df_financials = pd.read_sql(financial_query, connection)
            print(f"获取到 {len(df_financials)} 条成分股财务指标记录。")
            if not df_financials.empty:
                df_financials.to_parquet(OUTPUT_CONSTITUENT_FINANCIALS_FILENAME, index=False, engine='pyarrow')
                print(f"成分股财务指标数据已保存到: {OUTPUT_CONSTITUENT_FINANCIALS_FILENAME}")
            else:
                print("未找到成分股财务指标数据。")
        except Exception as e:
            print(f"下载成分股财务指标数据时出错: {e}")

    else:
        print("\n由于未找到成分股成员信息，跳过成分股相关数据的下载。")


except oracledb.DatabaseError as e:
    error, = e.args
    print(f"\nOracle 数据库错误: {error.code} - {error.message}")
except ImportError as e:
    if 'pyarrow' in str(e):
         print("\n错误: 未找到 'pyarrow' 库，无法输出 Parquet 格式。")
         print("请先安装: pip install pyarrow")
    else:
        print("\n错误: 缺少必要的库。请先安装 'oracledb' 和 'pandas'。")
        print("运行: pip install oracledb pandas")
except Exception as e:
    print(f"\n发生未知错误: {e}")
    import traceback
    traceback.print_exc()

finally:
    if connection:
        connection.close()
        print("\n数据库连接已关闭。")

print("\n脚本执行完毕。")
