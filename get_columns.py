import oracledb
import getpass
import sys
import pandas as pd

# --- Configuration ---
DB_HOST = "219.223.208.52"
DB_PORT = 1521
DB_SERVICE_NAME = "ORCL"
# Use the Instant Client path confirmed earlier
INSTANT_CLIENT_DIR = r"D:\instantclient-basic-windows.x64-23.7.0.25.01\instantclient_23_7"
TARGET_OWNER = 'FILESYNC'
TARGET_TABLE = 'ASHAREFINANCIALINDICATOR'

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
        connection = oracledb.connect(user=db_user, password=db_password, dsn=dsn)
        print("数据库连接成功 (Thick Mode)！")
    except Exception as thick_error:
        print(f"Thick mode connection failed: {thick_error}")
        print("\n请确保:")
        print(f"1. Instant Client 路径 '{INSTANT_CLIENT_DIR}' 正确且包含所需的库文件。")
        print("2. Instant Client 版本与您的 Python/操作系统架构 (64位) 匹配。")
        print("3. 您已重启 VS Code 或终端以使环境变量生效（如果通过 PATH 设置）。")
        sys.exit(1)

    print(f"正在查询表 {TARGET_OWNER}.{TARGET_TABLE} 的列名...")

    query = f"""
    SELECT column_name
    FROM all_tab_columns
    WHERE owner = '{TARGET_OWNER.upper()}'
      AND table_name = '{TARGET_TABLE.upper()}'
    ORDER BY column_id
    """

    try:
        cursor = connection.cursor()
        cursor.execute(query)
        columns = [row[0] for row in cursor.fetchall()]

        if columns:
            print(f"\n表 {TARGET_OWNER}.{TARGET_TABLE} 的列名:")
            for col in columns:
                print(f"- {col}")
        else:
            print(f"错误：未找到表 {TARGET_OWNER}.{TARGET_TABLE} 的列或无权访问。")

    except Exception as query_error:
         print(f"查询列名时出错: {query_error}")
         print(f"可能是用户 {db_user} 权限不足或表名错误。")

except oracledb.DatabaseError as e:
    error, = e.args
    print(f"\nOracle 数据库错误: {error.code} - {error.message}")
    if error.code == 1017:
        print("提示: 无效的用户名或密码。")
except ImportError:
    print("\n错误: 缺少 'pandas' 库。请运行: pip install pandas") # Although pandas isn't strictly needed here, keep the check consistent
except Exception as e:
    print(f"\n发生未知错误: {e}")
finally:
    if cursor:
        cursor.close()
    if connection:
        connection.close()
        print("数据库连接已关闭。")

print("\n脚本执行完毕。")
