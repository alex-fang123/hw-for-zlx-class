import oracledb
import getpass
import sys
import pandas as pd
import configparser
import os

# --- Configuration ---
CONFIG_FILE = 'db_config.ini'
# Keep target table/owner for now, can be parameterized later if needed
TARGET_OWNER = 'FILESYNC'
TARGET_TABLE = 'ASHAREFINANCIALINDICATOR'

connection = None
cursor = None

try:
    # Read configuration from file
    if not os.path.exists(CONFIG_FILE):
        print(f"错误: 配置文件 '{CONFIG_FILE}' 不存在。请先创建并配置该文件。")
        sys.exit(1)

    config = configparser.ConfigParser()
    # Specify encoding as utf-8 to handle potential BOM or special characters
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
        print("\n请确保:")
        print(f"1. Instant Client 路径 '{instant_client_dir}' 正确且包含所需的库文件。")
        print("2. Instant Client 版本与您的 Python/操作系统架构 (64位) 匹配。")
        print("3. 您已重启 VS Code 或终端以使环境变量生效（如果通过 PATH 设置）。")
        sys.exit(1)

    # Allow specifying table via command line arguments (optional enhancement)
    if len(sys.argv) == 3:
        target_owner_arg = sys.argv[1].upper()
        target_table_arg = sys.argv[2].upper()
        print(f"使用命令行参数指定查询表: {target_owner_arg}.{target_table_arg}")
        TARGET_OWNER = target_owner_arg
        TARGET_TABLE = target_table_arg
    elif len(sys.argv) > 1 and len(sys.argv) != 3:
        print("用法: python get_columns.py [OWNER] [TABLE_NAME]")
        print(f"如果未提供参数，将使用默认表: {TARGET_OWNER}.{TARGET_TABLE}")

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
