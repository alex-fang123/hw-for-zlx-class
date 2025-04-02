import oracledb
import sys
import pandas as pd
import configparser
import os
import configparser # Ensure configparser is imported if not already

# --- Configuration ---
# Construct path relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the root directory and join with the config file name
config_path = os.path.normpath(os.path.join(script_dir, '..', 'db_config.ini'))
CONFIG_FILE = config_path

connection = None
cursor = None

def get_column_comments(owner, table_name):
    """Queries and returns column names and comments for a given table."""
    global connection # Allow modification of the global connection variable
    
    comments_df = pd.DataFrame(columns=['COLUMN_NAME', 'COMMENTS']) # Default empty
    
    if not connection:
        print("错误：数据库连接未建立。")
        return comments_df

    print(f"\n正在查询表 {owner}.{table_name} 的列注释...")
    query = f"""
    SELECT COLUMN_NAME, COMMENTS
    FROM ALL_COL_COMMENTS
    WHERE OWNER = :owner 
      AND TABLE_NAME = :table_name
    ORDER BY COLUMN_ID 
    """ 
    # Using COLUMN_ID from ALL_TAB_COLUMNS might be better for exact order, 
    # but requires a join. Ordering by COLUMN_NAME is usually sufficient.
    # Let's refine the query to join for correct order:
    query_ordered = f"""
    SELECT c.COLUMN_NAME, cc.COMMENTS
    FROM ALL_TAB_COLUMNS c
    LEFT JOIN ALL_COL_COMMENTS cc 
      ON c.OWNER = cc.OWNER 
      AND c.TABLE_NAME = cc.TABLE_NAME 
      AND c.COLUMN_NAME = cc.COLUMN_NAME
    WHERE c.OWNER = :owner 
      AND c.TABLE_NAME = :table_name
    ORDER BY c.COLUMN_ID
    """

    try:
        # Use pandas for easier handling, though direct cursor fetch is also fine
        comments_df = pd.read_sql(query_ordered, connection, params={'owner': owner.upper(), 'table_name': table_name.upper()})
        print(f"查询完成，找到 {len(comments_df)} 列。")
        # Fill missing comments with a placeholder
        comments_df['COMMENTS'].fillna('N/A', inplace=True)
    except Exception as query_error:
         print(f"查询列注释时出错: {query_error}")
         print(f"可能是用户权限不足、表名错误或系统视图不可访问。")
         # Return empty dataframe on error
         comments_df = pd.DataFrame(columns=['COLUMN_NAME', 'COMMENTS']) 

    return comments_df

try:
    # Read configuration from file
    if not os.path.exists(CONFIG_FILE):
        print(f"错误: 配置文件 '{CONFIG_FILE}' 不存在。")
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
        print(f"错误: 配置文件 '{CONFIG_FILE}' 格式不正确: {e}")
        sys.exit(1)

    if not db_user or db_user == 'YOUR_USERNAME_HERE' or not db_password or db_password == 'YOUR_PASSWORD_HERE':
        print(f"错误: 请在 '{CONFIG_FILE}' 中配置有效的用户名和密码。")
        sys.exit(1)

    dsn = f"{db_host}:{db_port}/{db_service_name}"
    print(f"正在从 '{CONFIG_FILE}' 读取配置并连接到数据库: {dsn} 用户: {db_user}...")

    try:
        oracledb.init_oracle_client(lib_dir=instant_client_dir)
        connection = oracledb.connect(user=db_user, password=db_password, dsn=dsn)
        print("数据库连接成功！")
    except Exception as conn_error:
        print(f"数据库连接失败: {conn_error}")
        sys.exit(1)

    # Get owner and table name from command line arguments
    if len(sys.argv) != 3:
        print("\n用法: python get_column_comments.py <OWNER> <TABLE_NAME>")
        sys.exit(1)
        
    target_owner = sys.argv[1]
    target_table = sys.argv[2]

    # Get and print comments
    comments_data = get_column_comments(target_owner, target_table)

    if not comments_data.empty:
        print(f"\n表 {target_owner}.{target_table} 的列注释:")
        # Print in a formatted way
        for index, row in comments_data.iterrows():
            print(f"- {row['COLUMN_NAME']}: {row['COMMENTS']}")
    else:
        print(f"未能获取表 {target_owner}.{target_table} 的列注释。")


except Exception as e:
    print(f"\n发生未知错误: {e}")
    import traceback
    traceback.print_exc()

finally:
    if connection:
        connection.close()
        print("\n数据库连接已关闭。")

print("\n脚本执行完毕。")
