import pandas as pd
import os

# --- Configuration ---
# Construct path relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the root directory and then into the 'data' directory
data_dir = os.path.normpath(os.path.join(script_dir, '..', 'data'))

# Define file paths for RAW data using the calculated data_dir
index_file = os.path.join(data_dir, "raw_csi300_index_prices.parquet")
membership_file = os.path.join(data_dir, "raw_csi300_constituent_membership.parquet")
constituent_prices_file = os.path.join(data_dir, "raw_csi300_constituent_prices.parquet") 
constituent_weights_file = os.path.join(data_dir, "raw_csi300_constituent_weights.parquet") 
constituent_financials_file = os.path.join(data_dir, "raw_csi300_constituent_financials.parquet")

def view_parquet(file_path, file_description):
    """Helper function to read and display info for a Parquet file."""
    print(f"--- 查看 {file_description} 文件: {file_path} ---")
    if os.path.exists(file_path):
        try:
            df = pd.read_parquet(file_path)
            print("文件读取成功！")
            print("\n前 5 行数据:")
            print(df.head())
            print(f"\n数据形状 (行, 列): {df.shape}")
            print("\n列名和数据类型:")
            df.info() 
        except Exception as e:
            print(f"读取或处理文件时出错: {e}")
    else:
        print(f"错误：文件 {file_path} 不存在。")
    print("-" * (len(file_description) + 15)) # Separator

# View each file
view_parquet(index_file, "指数行情数据")
print("\n")
view_parquet(membership_file, "成分股成员信息")
print("\n")
view_parquet(constituent_prices_file, "成分股行情数据")
print("\n")
view_parquet(constituent_weights_file, "成分股权重数据")
print("\n")
view_parquet(constituent_financials_file, "成分股财务指标数据")

print("\n\n查看完毕。")
