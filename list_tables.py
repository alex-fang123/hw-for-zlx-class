import oracledb
import getpass
import sys
import pandas as pd

# --- Configuration ---
# DB_USER = "" # Use input prompt for username
DB_HOST = "219.223.208.52"
DB_PORT = 1521
DB_SERVICE_NAME = "ORCL"
# Use the Instant Client path confirmed earlier
INSTANT_CLIENT_DIR = r"D:\instantclient-basic-windows.x64-23.7.0.25.01\instantclient_23_7"

connection = None
cursor = None

try:
    db_user = input("请输入 Oracle 数据库用户名: ") # Prompt for username
    print("请输入 Oracle 数据库密码:")
    db_password = getpass.getpass() # Securely get password

    dsn = f"{DB_HOST}:{DB_PORT}/{DB_SERVICE_NAME}"
    print(f"正在连接到数据库: {dsn} 用户: {db_user}...")

    try:
        print(f"尝试使用指定的 Instant Client 目录初始化 Thick Mode: {INSTANT_CLIENT_DIR}")
        oracledb.init_oracle_client(lib_dir=INSTANT_CLIENT_DIR)
        connection = oracledb.connect(user=db_user, password=db_password, dsn=dsn) # Use db_user here
        print("数据库连接成功 (Thick Mode)！")
    except Exception as thick_error:
        print(f"Thick mode connection failed: {thick_error}")
        # Add checks for common connection issues like wrong path, architecture mismatch etc.
        print("\n请确保:")
        print(f"1. Instant Client 路径 '{INSTANT_CLIENT_DIR}' 正确且包含所需的库文件。")
        print("2. Instant Client 版本与您的 Python/操作系统架构 (64位) 匹配。")
        print("3. 您已重启 VS Code 或终端以使环境变量生效（如果通过 PATH 设置）。")
        sys.exit(1)

    print("正在查询用户可访问的表 (ALL_TABLES)，这可能需要一些时间...")
    # Query ALL_TABLES to find tables accessible by the user
    # Filtering for names that might be relevant (case-insensitive)
    # Excluding common system schemas
    query = """
    SELECT OWNER, TABLE_NAME
    FROM ALL_TABLES
    WHERE OWNER NOT IN (
        'SYS', 'SYSTEM', 'OUTLN', 'DBSNMP', 'WMSYS', 'ORDSYS', 'ORDDATA',
        'CTXSYS', 'XDB', 'MDSYS', 'OLAPSYS', 'EXFSYS', 'APPQOSSYS', 'DVSYS',
        'OJVMSYS', 'GSMADMIN_INTERNAL', 'DBSFWUSER', 'REMOTE_SCHEDULER_AGENT',
        'AUDSYS', 'SYSBACKUP', 'SYSDG', 'SYSKM', 'SYSRAC', 'PUBLIC', 'SQLTXPLAIN',
        'SPATIAL_CSW_ADMIN_USR', 'SPATIAL_WFS_ADMIN_USR', 'FLOWS_FILES', 'SI_INFORMTN_SCHEMA',
        'ORACLE_OCM', 'XS$NULL', 'SYS$UMF', 'GSMCATUSER', 'GGSYS', 'DIP', 'ANONYMOUS'
        )
    AND (
        UPPER(TABLE_NAME) LIKE '%INDEX%' OR
        UPPER(TABLE_NAME) LIKE '%MEMBER%' OR
        UPPER(TABLE_NAME) LIKE '%EOD%' OR
        UPPER(TABLE_NAME) LIKE '%PRICE%' OR
        UPPER(TABLE_NAME) LIKE '%STOCK%' OR
        UPPER(TABLE_NAME) LIKE '%SHARE%' OR
        UPPER(TABLE_NAME) LIKE '%QUOTE%' OR
        UPPER(TABLE_NAME) LIKE '%DAILY%' OR
        UPPER(TABLE_NAME) LIKE '%WIND%' OR
        UPPER(TABLE_NAME) LIKE '%ASHARE%' OR
        UPPER(TABLE_NAME) LIKE '%AINDEX%' OR
        UPPER(TABLE_NAME) LIKE '%CSI%' OR
        UPPER(TABLE_NAME) LIKE '%HS300%' OR
        UPPER(TABLE_NAME) LIKE '%CONSTITUENT%'
    )
    ORDER BY OWNER, TABLE_NAME
    """
    
    # Using pandas for potentially large results
    try:
        df_tables = pd.read_sql(query, connection)
        print(f"\n查询完成，找到 {len(df_tables)} 个可能相关的表。")
    except Exception as query_error:
         print(f"查询 ALL_TABLES 时出错: {query_error}")
         print(f"可能是用户 {db_user} 权限不足或查询超时。") # Use db_user here
         df_tables = pd.DataFrame(columns=['OWNER', 'TABLE_NAME']) # Empty dataframe


    if not df_tables.empty:
        print("\n找到可能相关的表 (Owner.TableName):")
        for index, row in df_tables.iterrows():
            print(f"- {row['OWNER']}.{row['TABLE_NAME']}")
    else:
        print("在 ALL_TABLES 中未找到符合搜索条件的表。")
        print("尝试查询 USER_TABLES (用户拥有的表)...")
        query_user = "SELECT TABLE_NAME FROM USER_TABLES ORDER BY TABLE_NAME"
        try:
            df_user_tables = pd.read_sql(query_user, connection)
            print(f"\n查询完成，用户 {db_user} 拥有 {len(df_user_tables)} 个表。") # Use db_user here
            if not df_user_tables.empty:
                 print("\n用户拥有的表 (TableName):")
                 for index, row in df_user_tables.iterrows():
                      # Filter user tables based on keywords
                      table_name_user = row['TABLE_NAME']
                      if any(keyword in table_name_user.upper() for keyword in [
                          'INDEX', 'MEMBER', 'EOD', 'PRICE', 'STOCK', 'SHARE', 'QUOTE',
                          'DAILY', 'WIND', 'ASHARE', 'AINDEX', 'CSI', 'HS300', 'CONSTITUENT']):
                          print(f"- {table_name_user}")
            else:
                 print(f"用户 {db_user} 没有拥有的表。") # Use db_user here
        except Exception as user_query_error:
            print(f"查询 USER_TABLES 时出错: {user_query_error}")


except oracledb.DatabaseError as e:
    error, = e.args
    print(f"\nOracle 数据库错误: {error.code} - {error.message}")
    if error.code == 1017:
        print("提示: 无效的用户名或密码。")
except ImportError:
    print("\n错误: 缺少 'pandas' 库。请运行: pip install pandas")
except Exception as e:
    print(f"\n发生未知错误: {e}")
finally:
    if connection:
        connection.close()
        print("数据库连接已关闭。")

print("\n脚本执行完毕。")
