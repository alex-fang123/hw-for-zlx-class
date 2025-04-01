import oracledb
import pandas as pd
import getpass
import datetime
import sys
import warnings
import os

# Suppress specific UserWarning from pandas about DBAPI2 connection
warnings.filterwarnings('ignore', message='pandas only supports SQLAlchemy connectable.*')

# --- Configuration ---
DB_HOST = "219.223.208.52"
DB_PORT = 1521
DB_SERVICE_NAME = "ORCL"
INSTANT_CLIENT_DIR = r"D:\instantclient-basic-windows.x64-23.7.0.25.01\instantclient_23_7"

OUTPUT_FULL_DATA_FILENAME = "./data/csi300_historical_data_full.csv"
OUTPUT_INDEX_FILENAME = "./data/csi300_index_data.csv"

# --- Table and Index Information ---
SCHEMA_PREFIX = "FILESYNC."
INDEX_CODE = "000300.SH"
CONSTITUENTS_TABLE = f"{SCHEMA_PREFIX}AINDEXMEMBERS"
STOCK_PRICE_TABLE = f"{SCHEMA_PREFIX}ASHAREEODPRICES"
FINANCIAL_INDICATOR_TABLE = f"{SCHEMA_PREFIX}ASHAREFINANCIALINDICATOR"
INDEX_PRICE_TABLE = f"{SCHEMA_PREFIX}AINDEXEODPRICES"
START_YEAR = 2000
END_YEAR = datetime.datetime.now().year

connection = None

# --- Helper Function for IN Clause ---
def build_in_clause(items):
    if not items:
        return "IN ('')" # Avoid empty IN clause error
    if len(items) == 1:
        return f"= '{items[0]}'"
    # Oracle limit is 1000, chunking might be needed for extreme cases, but usually fine for stocks per year
    return f"IN {tuple(items)}"

try:
    db_user = input("请输入 Oracle 数据库用户名: ")
    print("请输入 Oracle 数据库密码:")
    db_password = getpass.getpass()

    dsn = f"{DB_HOST}:{DB_PORT}/{DB_SERVICE_NAME}"
    print(f"正在连接到数据库: {dsn} 用户: {db_user}...")

    try:
        print(f"尝试使用指定的 Instant Client 目录初始化 Thick Mode: {INSTANT_CLIENT_DIR}")
        oracledb.init_oracle_client(lib_dir=INSTANT_CLIENT_DIR)
        connection = oracledb.connect(user=db_user, password=db_password, dsn=dsn)
        print("数据库连接成功 (Thick Mode)！")
    except Exception as thick_error:
        print(f"Thick mode connection failed: {thick_error}")
        sys.exit(1)

    # --- 1. Get ALL Historical Constituents (Once) ---
    print(f"正在获取指数 {INDEX_CODE} 的所有历史成分股信息...")
    constituent_query = f"""
    SELECT
        S_CON_WINDCODE,
        S_CON_INDATE,
        S_CON_OUTDATE
    FROM {CONSTITUENTS_TABLE}
    WHERE S_INFO_WINDCODE = '{INDEX_CODE}'
    """
    all_members_df = pd.read_sql(constituent_query, connection)
    print(f"找到 {len(all_members_df)} 条总历史成分股记录。")

    if all_members_df.empty:
        print(f"错误：未找到指数 {INDEX_CODE} 的历史成分股。")
        sys.exit(1)

    all_members_df['S_CON_INDATE'] = pd.to_datetime(all_members_df['S_CON_INDATE'], format='%Y%m%d')
    all_members_df['S_CON_OUTDATE'] = pd.to_datetime(all_members_df['S_CON_OUTDATE'], format='%Y%m%d', errors='coerce')
    future_date_marker = pd.Timestamp.max # Use a very future date for active members
    all_members_df['S_CON_OUTDATE'].fillna(future_date_marker, inplace=True)

    all_unique_constituent_codes = tuple(all_members_df['S_CON_WINDCODE'].unique())
    print(f"涉及 {len(all_unique_constituent_codes)} 个不同的股票代码。")

    # --- 2. Get ALL Financial Indicator Data (Once) ---
    print(f"正在获取 {len(all_unique_constituent_codes)} 个股票的所有财务指标数据...")
    in_clause_all_stocks = build_in_clause(all_unique_constituent_codes)
    financial_query = f"""
    SELECT *
    FROM {FINANCIAL_INDICATOR_TABLE}
    WHERE s_info_windcode {in_clause_all_stocks}
    """
    all_financials_df = pd.read_sql(financial_query, connection)
    print(f"获取到 {len(all_financials_df)} 条总财务指标记录。")

    # Prepare financial data for merge_asof
    all_financials_df['ANN_DT'] = pd.to_datetime(all_financials_df['ANN_DT'], format='%Y%m%d', errors='coerce')
    all_financials_df.dropna(subset=['ANN_DT'], inplace=True)
    all_financials_df.sort_values(by=['S_INFO_WINDCODE', 'ANN_DT'], inplace=True)
    all_financials_df.drop_duplicates(subset=['S_INFO_WINDCODE', 'ANN_DT'], keep='last', inplace=True)
    all_financials_df_renamed = all_financials_df.rename(columns={'ANN_DT': 'fin_date'}) # Rename for merge_asof

    # --- Remove existing output files before starting loop ---
    if os.path.exists(OUTPUT_FULL_DATA_FILENAME):
        print(f"删除已存在的合并数据文件: {OUTPUT_FULL_DATA_FILENAME}")
        os.remove(OUTPUT_FULL_DATA_FILENAME)
    if os.path.exists(OUTPUT_INDEX_FILENAME):
        print(f"删除已存在的指数数据文件: {OUTPUT_INDEX_FILENAME}")
        os.remove(OUTPUT_INDEX_FILENAME)

    # --- 3. Loop Through Years ---
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n--- 开始处理年份: {year} ---")
        year_start_str = f"{year}0101"
        year_end_str = f"{year}1231"
        year_start_dt = pd.to_datetime(year_start_str, format='%Y%m%d')
        year_end_dt = pd.to_datetime(year_end_str, format='%Y%m%d')

        # --- 3a. Filter Constituents for the Current Year ---
        # Stock is a constituent in this year if its membership period overlaps with the year
        members_this_year_df = all_members_df[
            (all_members_df['S_CON_INDATE'] <= year_end_dt) &
            (all_members_df['S_CON_OUTDATE'] > year_start_dt) # Use > for OUTDATE
        ].copy()
        # Adjust IN/OUT dates to be within the year for daily generation
        members_this_year_df['year_start_bound'] = members_this_year_df['S_CON_INDATE'].clip(lower=year_start_dt)
        members_this_year_df['year_end_bound'] = members_this_year_df['S_CON_OUTDATE'].clip(upper=year_end_dt + pd.Timedelta(days=1)) # Clip OUTDATE to end of year + 1 day

        stocks_this_year = tuple(members_this_year_df['S_CON_WINDCODE'].unique())
        if not stocks_this_year:
            print(f"年份 {year} 没有找到成分股，跳过。")
            continue
        print(f"年份 {year} 涉及 {len(stocks_this_year)} 个股票代码。")

        # --- 3b. Get Stock Price Data for the Year ---
        print(f"正在获取年份 {year} 的日行情数据...")
        in_clause_year_stocks = build_in_clause(stocks_this_year)
        stock_price_query = f"""
        SELECT
            TRADE_DT, S_INFO_WINDCODE, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE, S_DQ_VOLUME, S_DQ_AMOUNT
        FROM {STOCK_PRICE_TABLE}
        WHERE trade_dt >= '{year_start_str}' AND trade_dt <= '{year_end_str}'
        AND s_info_windcode {in_clause_year_stocks}
        """
        prices_df_year = pd.read_sql(stock_price_query, connection)
        print(f"获取到 {len(prices_df_year)} 条年份 {year} 的股票日行情数据。")
        if prices_df_year.empty:
             print(f"警告：年份 {year} 未获取到股票行情数据。")
             # continue # Decide if you want to skip year if no price data

        prices_df_year['TRADE_DT'] = pd.to_datetime(prices_df_year['TRADE_DT'], format='%Y%m%d')
        prices_df_year.rename(columns={'TRADE_DT': 'date'}, inplace=True)

        # --- 3c. Create Base Daily Constituent DataFrame for the Year ---
        print(f"正在生成年份 {year} 的每日成分股基础表...")
        date_range_year = pd.date_range(start=year_start_dt, end=year_end_dt, freq='D')
        date_df_year = pd.DataFrame({'date': date_range_year})
        all_combinations_year = date_df_year.merge(pd.DataFrame({'S_INFO_WINDCODE': list(stocks_this_year)}), how='cross')

        constituents_daily_list_year = []
        for _, member_row in members_this_year_df.iterrows():
            mask = (
                (all_combinations_year['S_INFO_WINDCODE'] == member_row['S_CON_WINDCODE']) &
                (all_combinations_year['date'] >= member_row['year_start_bound']) &
                (all_combinations_year['date'] < member_row['year_end_bound']) # Use < adjusted end bound
            )
            constituents_daily_list_year.append(all_combinations_year[mask])

        if not constituents_daily_list_year:
            print(f"错误：年份 {year} 未能生成每日成分股列表。")
            continue

        constituents_daily_df_year = pd.concat(constituents_daily_list_year).drop_duplicates().sort_values(by=['S_INFO_WINDCODE', 'date'])
        print(f"年份 {year} 生成了 {len(constituents_daily_df_year)} 条有效的 '日期-成分股' 记录。")

        # --- 3d. Merge Price Data for the Year ---
        print(f"正在合并年份 {year} 的日行情数据...")
        merged_df_year = pd.merge(
            constituents_daily_df_year,
            prices_df_year,
            on=['date', 'S_INFO_WINDCODE'],
            how='left'
        )

        # --- 3e. Merge Financial Data using Forward Fill for the Year ---
        print(f"正在合并年份 {year} 的财务指标数据 (使用向前填充)...")
        # --- 3e. Merge Financial Data using Forward Fill (Alternative Method) ---
        print(f"正在合并年份 {year} 的财务指标数据 (使用向前填充 - 替代方法)...")

        # Prepare DataFrames for concat and ffill
        left_df = merged_df_year.set_index('date')
        # Select only necessary columns from right_df to avoid huge concat, add 'fin_date' for sorting
        # Ensure no duplicate columns except S_INFO_WINDCODE before concat, and explicitly exclude 'date' if it exists
        financial_cols_to_keep = [
            col for col in all_financials_df_renamed.columns
            if col not in left_df.columns and col not in ['S_INFO_WINDCODE', 'fin_date', 'date'] # Exclude potential original 'date'
        ]
        cols_to_keep = ['S_INFO_WINDCODE', 'fin_date'] + financial_cols_to_keep
        right_df = all_financials_df_renamed[cols_to_keep].rename(columns={'fin_date': 'date'}).set_index('date')


        # Combine, sort, ffill
        # Keep 'date' as a column for now
        left_df_reset = merged_df_year # Already has date as column
        right_df_reset = all_financials_df_renamed[cols_to_keep].rename(columns={'fin_date': 'date'})

        combined = pd.concat([left_df_reset, right_df_reset])
        combined.sort_values(by=['S_INFO_WINDCODE', 'date'], inplace=True) # Sort by stock and date

        # Identify financial columns to fill (all columns from right_df except S_INFO_WINDCODE and date)
        financial_cols = [col for col in right_df_reset.columns if col not in ['S_INFO_WINDCODE', 'date']]
        # Perform forward fill within each stock group
        combined[financial_cols] = combined.groupby('S_INFO_WINDCODE')[financial_cols].ffill()

        # Filter back to the original daily structure using an inner merge
        # Select only the columns from the original merged_df_year to avoid duplicate price columns etc.
        original_cols = merged_df_year.columns.tolist()
        final_df_year = pd.merge(
            merged_df_year[original_cols], # Use original structure/columns as the left frame
            combined,                      # Use the forward-filled combined data as the right frame
            on=['date', 'S_INFO_WINDCODE'],# Merge keys
            how='inner'                    # Keep only rows matching the original daily structure
        )
        # Select unique columns, preferring those from combined (which have ffill applied)
        # This handles potential duplicate columns from the merge if names weren't perfectly excluded
        final_df_year = final_df_year.loc[:, ~final_df_year.columns.duplicated(keep='last')]


        print(f"年份 {year} 财务指标数据合并完成 (替代方法 v2)。")


        # --- 3f. Save Combined Data for the Year (Append) ---
        if not final_df_year.empty:
            print(f"正在将年份 {year} 的合并数据追加到文件: {OUTPUT_FULL_DATA_FILENAME}...")
            is_first_chunk = (year == START_YEAR)
            final_df_year.to_csv(
                OUTPUT_FULL_DATA_FILENAME,
                index=False,
                encoding='utf-8-sig',
                mode='a', # Append mode
                header=is_first_chunk # Write header only for the first year
            )
            print(f"年份 {year} 合并数据追加成功！")
        else:
            print(f"年份 {year} 未生成合并数据。")

        # --- 3g. Get and Save Index Data for the Year (Append) ---
        print(f"正在获取指数 {INDEX_CODE} 年份 {year} 的日行情数据...")
        index_data_query_year = f"""
        SELECT *
        FROM {INDEX_PRICE_TABLE}
        WHERE trade_dt >= '{year_start_str}' AND trade_dt <= '{year_end_str}'
        AND s_info_windcode = '{INDEX_CODE}'
        ORDER BY trade_dt
        """
        index_df_year = pd.read_sql(index_data_query_year, connection)
        print(f"获取到 {len(index_df_year)} 条年份 {year} 的指数数据。")

        if not index_df_year.empty:
            print(f"正在将年份 {year} 的指数数据追加到文件: {OUTPUT_INDEX_FILENAME}...")
            is_first_index_chunk = (year == START_YEAR)
            index_df_year.to_csv(
                OUTPUT_INDEX_FILENAME,
                index=False,
                encoding='utf-8-sig',
                mode='a', # Append mode
                header=is_first_index_chunk # Write header only for the first year
            )
            print(f"年份 {year} 指数数据追加成功！")
        else:
            print(f"年份 {year} 未查询到指数数据。")

        print(f"--- 完成处理年份: {year} ---")


except oracledb.DatabaseError as e:
    error, = e.args
    print(f"\nOracle 数据库错误: {error.code} - {error.message}")
except ImportError:
    print("\n错误: 缺少必要的库。请先安装 'oracledb' 和 'pandas'。")
    print("运行: pip install oracledb pandas")
except Exception as e:
    print(f"\n发生未知错误: {e}")
    import traceback
    traceback.print_exc()

finally:
    if connection:
        connection.close()
        print("数据库连接已关闭。")

print("\n脚本执行完毕。")
