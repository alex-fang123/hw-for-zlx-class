import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools
import psutil
import gc
import traceback # 确保导入
import time
from numpy.lib.stride_tricks import as_strided # 确保导入
import argparse # 添加argparse导入

from model_DNN import IndexEnhancementModel # <-- 添加导入语句
from transformer_model import TransformerForTimeSeries # <-- Import Transformer

def get_memory_usage():
    """获取当前进程的内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # 转换为MB

def clear_memory():
    """清理内存"""
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# --- REMOVED OLD process_single_stock and process_date_window ---
# They are replaced by the generator logic

# --- Helper function for parallel indicator calculation ---
def calculate_indicators_for_stock(stock_data):
    """计算单只股票的技术指标 (用于并行处理)"""
    # stock_data 是包含单只股票所有数据的 DataFrame
    stock_code = stock_data['S_INFO_WINDCODE'].iloc[0] # 获取股票代码
    
    # 确保数据按日期排序
    stock_data = stock_data.sort_values('DATE')
    
    # 初始化指标列 (避免 SettingWithCopyWarning)
    stock_data = stock_data.assign(
        RETURN=np.nan,
        MA5=np.nan,
        MA10=np.nan,
        MA20=np.nan,
        VOLUME_MA5=np.nan,
        VOLUME_MA10=np.nan,
        VOLATILITY=np.nan
    )
    
    # 计算收益率
    stock_data['RETURN'] = stock_data['S_DQ_CLOSE'].pct_change()
    
    # 计算移动平均 (使用 min_periods=1 避免开头过多 NaN)
    stock_data['MA5'] = stock_data['S_DQ_CLOSE'].rolling(window=5, min_periods=1).mean()
    stock_data['MA10'] = stock_data['S_DQ_CLOSE'].rolling(window=10, min_periods=1).mean()
    stock_data['MA20'] = stock_data['S_DQ_CLOSE'].rolling(window=20, min_periods=1).mean()
    
    # 计算成交量移动平均
    stock_data['VOLUME_MA5'] = stock_data['S_DQ_VOLUME'].rolling(window=5, min_periods=1).mean()
    stock_data['VOLUME_MA10'] = stock_data['S_DQ_VOLUME'].rolling(window=10, min_periods=1).mean()
    
    # 计算波动率
    stock_data['VOLATILITY'] = stock_data['RETURN'].rolling(window=20, min_periods=1).std()
    
    return stock_data
# -------------------------------------------------------

# --- Helper function for creating windows using stride_tricks ---
def create_windows_for_stock(features_np, targets_np, index, lookback_days):
    """
    为单只股票使用 stride_tricks 高效创建滑动窗口特征和对齐目标。

    Args:
        features_np (np.ndarray): 该股票的日特征数组 (num_days, num_features)。
        targets_np (np.ndarray): 该股票的日目标数组 (num_days,)。
        index (pd.Index): 该股票数据的索引。
        lookback_days (int): 回看窗口大小。

    Returns:
        tuple: (np.ndarray, np.ndarray, list) 包含展平的窗口特征、对齐的目标和对齐的索引元组。
               如果无法创建窗口，则返回 (None, None, None)。
    """
    num_days, num_features = features_np.shape
    if num_days < lookback_days + 1:
        return None, None, None

    # 创建窗口
    feature_windows_shape = (num_days - lookback_days, lookback_days, num_features)
    feature_windows = as_strided(features_np, 
                               shape=feature_windows_shape, 
                               strides=(features_np.strides[0], features_np.strides[0], features_np.strides[1]),
                               writeable=False)

    # 展平窗口
    features_flattened = feature_windows.reshape(feature_windows_shape[0], -1).copy()

    # 对齐目标和索引
    aligned_targets = targets_np[lookback_days:]
    aligned_indices = [(date, code) for date, code in index[lookback_days:]]

    return features_flattened, aligned_targets, aligned_indices
# ---------------------------------------------------------------

class StockDataset(Dataset):
    # Note: This dataset now expects NumPy arrays directly
    def __init__(self, features_np, targets_np):
        """初始化数据集 (使用 NumPy 数组)"""
        self.features = torch.FloatTensor(features_np)
        # Ensure targets are FloatTensor and have shape [N, 1]
        self.targets = torch.FloatTensor(targets_np)
        if self.targets.ndim == 1:
            self.targets = self.targets.unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class IndexEnhancementStrategy:
    def __init__(self, data_dir='./data', transaction_cost_rate=0.001):
        self.data_dir = data_dir
        self.start_date = pd.Timestamp('2015-01-01') # 训练开始日期改为2015年1月1日
        self.end_date = pd.Timestamp('2024-12-31')   # 结束日期改为 2024 年底
        self.validation_start_date = pd.Timestamp('2024-01-01') # 验证集改为 2024 年
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用的设备: {self.device}")
        self.model_dnn = None # DNN Model
        self.model_transformer = None # Transformer Model
        self.scaler = StandardScaler() # Use one scaler for both for simplicity
        self.lookback_days = 60
        self.data = None
        self.index_prices = None
        self.transaction_cost_rate = transaction_cost_rate
        print(f"设置单边交易成本率: {self.transaction_cost_rate:.4f}")
        
    def load_data(self):
        """加载股票和指数数据"""
        print("开始加载数据...")
        stock_file = os.path.join(self.data_dir, 'merged_daily_data_grouped.parquet')
        index_file = os.path.join(self.data_dir, 'raw_csi300_index_prices.parquet')
        
        try:
            # 加载股票数据
            print(f"加载股票数据: {stock_file}")
            self.data = pd.read_parquet(stock_file)
            self.data['DATE'] = pd.to_datetime(self.data['DATE'])
            print(f"股票数据加载完成，总行数: {len(self.data)}")
            print(f"原始股票数据时间范围: {self.data['DATE'].min()} 到 {self.data['DATE'].max()}")

            # 加载指数数据
            print(f"加载指数数据: {index_file}")
            self.index_prices = pd.read_parquet(index_file)
            self.index_prices['TRADE_DT'] = pd.to_datetime(self.index_prices['TRADE_DT'])
            # 重命名日期列以保持一致
            self.index_prices.rename(columns={'TRADE_DT': 'DATE'}, inplace=True)
            # 计算指数日收益率
            self.index_prices = self.index_prices.sort_values('DATE')
            self.index_prices['RETURN'] = self.index_prices['S_DQ_CLOSE'].pct_change()
            # 仅保留需要的列
            self.index_prices = self.index_prices[['DATE', 'RETURN']].dropna()
            print(f"指数数据加载完成，总行数: {len(self.index_prices)}")
            print(f"原始指数数据时间范围: {self.index_prices['DATE'].min()} 到 {self.index_prices['DATE'].max()}")
            
            # 基本数据质量检查 (可选，之前已做过)
            # ... 
            
            print("\n数据文件加载完成")
            return True
            
        except FileNotFoundError as e:
            print(f"数据文件未找到: {e}")
            return False
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            return False
        
    def preprocess_data(self):
        """预处理数据"""
        print("数据预处理...")
        if self.data is None or self.index_prices is None:
            print("错误: 数据未加载。请先调用 load_data()。")
            return False
        
        try:
            # 1. 过滤股票数据时间范围
            stock_date_mask = (self.data['DATE'] >= self.start_date) & (self.data['DATE'] <= self.end_date)
            self.data = self.data[stock_date_mask].copy()
            print(f"\n筛选后的股票数据范围: {self.data['DATE'].min()} 到 {self.data['DATE'].max()}")
            print(f"筛选后的股票数据量: {len(self.data)}")
            if self.data.empty:
                print("错误: 在指定日期范围内没有股票数据。")
                return False

            # 2. 过滤指数数据时间范围
            index_date_mask = (self.index_prices['DATE'] >= self.start_date) & (self.index_prices['DATE'] <= self.end_date)
            self.index_prices = self.index_prices[index_date_mask].copy()
            print(f"筛选后的指数数据范围: {self.index_prices['DATE'].min()} 到 {self.index_prices['DATE'].max()}")
            print(f"筛选后的指数数据量: {len(self.index_prices)}")
            if self.index_prices.empty:
                print("错误: 在指定日期范围内没有指数数据。")
                return False
            
            # 3. 确保股票数据排序 (在分组前做一次，虽然 calculate_indicators_for_stock 内部也会做)
            self.data = self.data.sort_values(['S_INFO_WINDCODE', 'DATE'])
            
            # 4. 计算股票交易天数并过滤 (在计算指标前完成)
            stock_trading_days = self.data.groupby('S_INFO_WINDCODE').size()
            valid_stocks_mask = stock_trading_days >= self.lookback_days
            valid_stocks = stock_trading_days[valid_stocks_mask].index
            print(f"\n交易天数 >= {self.lookback_days} 天的股票数量：{len(valid_stocks)} (共 {self.data['S_INFO_WINDCODE'].nunique()} 只)")
            self.data = self.data[self.data['S_INFO_WINDCODE'].isin(valid_stocks)]
            print(f"过滤低交易天数股票后的数据量: {len(self.data)}")
            if self.data.empty:
                print("错误: 过滤低交易天数股票后没有数据。")
                return False

            # --- 新增：对 S_FA_* 数据进行滞后处理以防前视偏差 ---
            print("\n开始滞后处理财务数据 (S_FA_*) 以防止前视偏差...")
            financial_lag_days = 65 # 假设约 3 个月的交易日滞后期
            s_fa_columns = [col for col in self.data.columns if col.startswith('S_FA_')]
            print(f"找到 {len(s_fa_columns)} 个 S_FA_ 列进行滞后 {financial_lag_days} 天处理。")

            if s_fa_columns:
                # 定义一个函数来处理单个分组
                def lag_financial_data(group):
                    lagged_data = {}
                    # 对每个 S_FA 列进行 shift 操作，存入字典
                    for col in s_fa_columns:
                        lagged_data[f'{col}_lagged'] = group[col].shift(financial_lag_days)
                    # 将所有滞后列一次性合并回group
                    return pd.concat([group, pd.DataFrame(lagged_data, index=group.index)], axis=1)

                # 使用 groupby().apply() 进行分组滞后操作
                # 注意：apply 可能较慢，对于非常大的数据集，可能有更优化的方法
                # 但对于这里的规模应该是可行的，且逻辑清晰
                print(f"按股票分组应用滞后 {financial_lag_days} 天...")
                # 使用 copy() 避免 SettingWithCopyWarning in apply
                self.data = self.data.groupby('S_INFO_WINDCODE', group_keys=False).apply(lambda x: lag_financial_data(x.copy()))
                print("财务数据滞后处理完成。")
            else:
                print("未找到 S_FA_ 列，跳过滞后处理。")
            # -----------------------------------------------------

            # 5. 计算技术指标 (使用多进程)
            print("\n开始并行计算技术指标...")
            self.calculate_technical_indicators_parallel()
            
            # 6. 处理缺失值 (在计算完指标后)
            self.handle_missing_values()
            
            print("数据预处理完成")
            return True
            
        except Exception as e:
            print(f"数据预处理失败: {str(e)}")
            traceback.print_exc()
            return False

    def handle_missing_values(self):
        """处理缺失值"""
        print("\n处理缺失值...")
        
        # 列出需要特别处理的关键价格/成交量列
        # 如果这些列缺失，我们通常认为当天的数据不可靠，应该删除整行
        key_price_vol_columns = ['S_DQ_CLOSE', 'S_DQ_VOLUME', 'S_DQ_AMOUNT']
        
        # 检查关键列的缺失值
        missing_key = self.data[key_price_vol_columns].isnull().sum()
        if missing_key.sum() > 0:
            print(f"关键列 {key_price_vol_columns} 存在缺失值，将删除对应行：")
            print(missing_key[missing_key > 0])
            original_len = len(self.data)
            self.data = self.data.dropna(subset=key_price_vol_columns)
            print(f"删除 {original_len - len(self.data)} 行后，数据量: {len(self.data)}")
    
        # 对所有数值类型的列（包括计算出的指标和原始的 S_DQ/S_FA 列）进行填充
        # 注意：此时应该在计算完指标之后，但在 prepare_features 提取最终特征之前
        numeric_cols = self.data.select_dtypes(include=np.number).columns
        print(f"对以下 {len(numeric_cols)} 列数值列进行 ffill 和 fillna(0) 处理...")
        
        # 按股票分组进行前向填充，然后用 0 填充剩余 NaN
        # GroupBy + transform 通常比循环更高效
        self.data[numeric_cols] = self.data.groupby('S_INFO_WINDCODE')[numeric_cols].ffill()
        self.data[numeric_cols] = self.data[numeric_cols].fillna(0)
        
        # 验证是否还有 NaN (理论上不应该有)
        remaining_nan = self.data[numeric_cols].isnull().sum().sum()
        if remaining_nan > 0:
            print(f"警告：在缺失值处理后，数值列中仍有 {remaining_nan} 个 NaN 值！")
        else:
            print("数值列缺失值处理完成，已使用 ffill + fillna(0)。")

    def calculate_technical_indicators_parallel(self, num_processes=None):
        """使用多进程并行计算技术指标"""
        if num_processes is None:
            num_processes = cpu_count() - 1 if cpu_count() > 1 else 1 # 留一个核心给主进程
        print(f"使用 {num_processes} 个进程进行计算...")

        # 按股票代码分组，创建数据块列表
        grouped = self.data.groupby('S_INFO_WINDCODE')
        stock_data_list = [group for name, group in grouped]
        num_stocks = len(stock_data_list)

        results = []
        try:
            with Pool(processes=num_processes) as pool:
                # 使用 imap_unordered 以便实时更新进度条
                # tqdm 显示处理的股票数量
                results = list(tqdm(pool.imap_unordered(calculate_indicators_for_stock, stock_data_list), 
                                     total=num_stocks, desc="计算技术指标"))
            print("\n技术指标并行计算完成。")
        except Exception as e:
             print(f"\n技术指标并行计算出错: {e}")
             traceback.print_exc()
             # 可以在这里决定是否抛出异常或尝试恢复
             return # 提前退出

        if not results:
             print("错误: 并行计算未能产生任何结果。")
             return # 提前退出
             
        # 合并结果
        print("合并计算结果...")
        self.data = pd.concat(results, ignore_index=True)
        # 并行处理可能打乱顺序，重新排序
        self.data = self.data.sort_values(['S_INFO_WINDCODE', 'DATE'])
        print(f"结果合并完成，数据形状: {self.data.shape}")

    # --- Renamed helper function for sequential window creation ---
    def _create_windows_for_single_stock(self, task_data):
        """
        为单只股票创建窗口化特征和目标 (顺序处理辅助函数)。

        Args:
            task_data (tuple): 包含 (stock_code, features_group, targets_group, lookback_days) 的元组。

        Returns:
            tuple: (str, np.ndarray | None, np.ndarray | None, list | None)
                   包含股票代码、展平的窗口特征、对齐的目标和对齐的索引元组。
                   如果无法创建窗口，则返回 (stock_code, None, None, None)。
        """
        stock_code, features_group, targets_group, lookback_days = task_data

        if features_group.empty or targets_group.empty:
            # print(f"股票 {stock_code}: 特征或目标组为空，跳过窗口化。") # Too verbose
            return stock_code, None, None, None
        
        # Ensure the DataFrame index is the default RangeIndex before converting to NumPy
        # Keep the original MultiIndex separately
        original_multi_index = features_group.index
        features_np = features_group.reset_index(drop=True).values.astype(np.float32)
        # Use targets_group index as well for consistency check
        targets_original_multi_index = targets_group.index 
        targets_np = targets_group.reset_index(drop=True).values.astype(np.float32)
        
        # --- Added Check: Verify input lengths BEFORE calling create_windows_for_stock ---
        if not (len(features_np) == len(targets_np) == len(original_multi_index)):
            print(f"股票 {stock_code}: 输入数组/索引长度不匹配! Features NP: {len(features_np)}, Targets NP: {len(targets_np)}, Index: {len(original_multi_index)}")
            # Also check original group lengths and index equality
            print(f"  Original lengths: Features Group: {len(features_group)}, Targets Group: {len(targets_group)}")
            if not features_group.index.equals(targets_group.index):
                print(f"  警告: 股票 {stock_code} 的特征和目标组索引不相等！")
            return stock_code, None, None, None
        # ---------------------------------------------------------------------------------
            
        # Call the core window creation logic (assuming create_windows_for_stock exists globally or is accessible)
        # We'll use the previously defined global create_windows_for_stock here
        try:
            features_flattened, aligned_targets, aligned_indices_tuples = create_windows_for_stock(
                features_np, 
                targets_np, 
                original_multi_index, # Pass the original MultiIndex from features
                lookback_days
            )
        except Exception as e:
            print(f"股票 {stock_code}: 调用 create_windows_for_stock 时出错: {e}")
            traceback.print_exc()
            return stock_code, None, None, None

        if features_flattened is None:
            # print(f"股票 {stock_code}: 未能创建窗口 (可能数据不足或长度不匹配)。") # Too verbose
            return stock_code, None, None, None

        # aligned_indices_tuples should already be a list of tuples from create_windows_for_stock
        return stock_code, features_flattened, aligned_targets, aligned_indices_tuples

    # --- Modified prepare_features for sequential processing ---
    def prepare_features(self, features_daily_df, targets_daily_series, batch_size=100):
        """
        分批准备窗口化特征和目标数据，使用生成器模式减少内存占用。
        默认每批处理100只股票。
        
        Args:
            features_daily_df (pd.DataFrame): 包含每日特征的 DataFrame
            targets_daily_series (pd.Series): 包含每日目标的 Series
            batch_size (int): 每批处理的股票数量，默认100只
        """
        print("\n开始分批准备窗口化特征和目标...")
        start_prep_time = time.time()
        initial_memory = get_memory_usage()
        print(f"初始内存使用: {initial_memory:.2f} MB")
        
        # 输入验证
        if features_daily_df.empty or targets_daily_series.empty:
            print("错误: 输入的每日特征或目标为空。")
            return
        
        if not isinstance(features_daily_df.index, pd.MultiIndex) or \
           not isinstance(targets_daily_series.index, pd.MultiIndex):
            print("错误: 特征和目标必须使用MultiIndex (DATE, S_INFO_WINDCODE)。")
            return
        
        if not features_daily_df.index.equals(targets_daily_series.index):
            print("错误: 特征和目标的索引不匹配。")
            return
        
        # 获取所有唯一的股票代码
        all_stock_codes = features_daily_df.index.get_level_values('S_INFO_WINDCODE').unique()
        num_stocks = len(all_stock_codes)
        num_batches = (num_stocks + batch_size - 1) // batch_size
        print(f"总股票数量: {num_stocks}，将分成 {num_batches} 批处理，每批 {batch_size} 只股票")
        
        # 分批处理股票
        for i in range(0, num_stocks, batch_size):
            batch_start_memory = get_memory_usage()
            batch_stock_codes = all_stock_codes[i:i + batch_size]
            current_batch_size = len(batch_stock_codes)
            print(f"\n开始处理第 {i//batch_size + 1}/{num_batches} 批，包含 {current_batch_size} 只股票")
            print(f"批次开始时内存使用: {batch_start_memory:.2f} MB")
            
            try:
                # 获取当前批次的特征和目标
                batch_features = features_daily_df[features_daily_df.index.get_level_values('S_INFO_WINDCODE').isin(batch_stock_codes)]
                batch_targets = targets_daily_series[targets_daily_series.index.get_level_values('S_INFO_WINDCODE').isin(batch_stock_codes)]
                
                # 验证批次数据
                if batch_features.empty or batch_targets.empty:
                    print(f"警告: 批次 {i//batch_size + 1} 的特征或目标为空，跳过此批次。")
                    continue
                
                if not batch_features.index.equals(batch_targets.index):
                    print(f"警告: 批次 {i//batch_size + 1} 的特征和目标索引不匹配，跳过此批次。")
                    continue
                
                # 处理当前批次
                batch_features_windowed = []
                batch_targets_windowed = []
                batch_indices = []
                
                for stock_code in tqdm(batch_stock_codes, desc=f"处理第 {i//batch_size + 1}/{num_batches} 批股票"):
                    try:
                        # 获取单只股票的数据
                        stock_features = batch_features.xs(stock_code, level='S_INFO_WINDCODE')
                        stock_targets = batch_targets.xs(stock_code, level='S_INFO_WINDCODE')
                        
                        # 验证股票数据
                        if stock_features.empty or stock_targets.empty:
                            continue
                        
                        if not stock_features.index.equals(stock_targets.index):
                            continue
                        
                        # 创建窗口
                        features_np = stock_features.values.astype(np.float32)
                        targets_np = stock_targets.values.astype(np.float32)
                        stock_indices = [(date, stock_code) for date in stock_features.index]
                        
                        # 使用create_windows_for_stock创建窗口
                        features_flattened, aligned_targets, aligned_indices = create_windows_for_stock(
                            features_np, targets_np, stock_indices, self.lookback_days
                        )
                        
                        if features_flattened is not None:
                            batch_features_windowed.append(features_flattened)
                            batch_targets_windowed.append(aligned_targets)
                            batch_indices.extend(aligned_indices)
                        
                        # 每处理5只股票检查一次内存
                        if len(batch_features_windowed) % 5 == 0:
                            current_memory = get_memory_usage()
                            if current_memory > batch_start_memory * 1.5:  # 内存增加超过50%
                                print(f"警告: 内存使用显著增加 - 当前: {current_memory:.2f} MB")
                                # 强制垃圾回收
                                gc.collect()
                        
                    except Exception as e:
                        print(f"处理股票 {stock_code} 时出错: {e}")
                        continue
                
                # 合并当前批次的结果
                if batch_features_windowed:
                    try:
                        # 转换为numpy数组
                        features_batch = np.concatenate(batch_features_windowed, axis=0)
                        targets_batch = np.concatenate(batch_targets_windowed, axis=0)
                        batch_index = pd.MultiIndex.from_tuples(batch_indices, names=['DATE', 'S_INFO_WINDCODE'])
                        
                        # 验证数据
                        if len(features_batch) != len(targets_batch) or len(features_batch) != len(batch_index):
                            print(f"警告: 批次 {i//batch_size + 1} 的数据长度不匹配，跳过此批次。")
                            continue
                        
                        # 创建DataFrame和Series
                        features_df = pd.DataFrame(features_batch, index=batch_index, dtype=np.float32)
                        targets_series = pd.Series(targets_batch, index=batch_index, dtype=np.float32)
                        
                        # 清理中间变量
                        del batch_features_windowed, batch_targets_windowed, batch_indices
                        del features_batch, targets_batch, batch_index
                        gc.collect()
                        
                        # 返回当前批次的结果
                        yield features_df, targets_series
                        
                    except Exception as e:
                        print(f"合并批次结果时出错: {e}")
                        continue
                
                # 清理当前批次的原始数据
                del batch_features, batch_targets
                gc.collect()
                
                # 打印内存使用情况
                current_memory = get_memory_usage()
                memory_change = current_memory - batch_start_memory
                print(f"批次处理完成。当前内存使用: {current_memory:.2f} MB (变化: {memory_change:+.2f} MB)")
                
            except Exception as e:
                print(f"处理批次时出错: {e}")
                continue
        
        end_prep_time = time.time()
        final_memory = get_memory_usage()
        memory_change = final_memory - initial_memory
        print(f"\n分批处理完成。总耗时: {end_prep_time - start_prep_time:.2f} 秒")
        print(f"最终内存使用: {final_memory:.2f} MB (总变化: {memory_change:+.2f} MB)")

    def train_models(self, features_train_generator, targets_train_generator, input_dim_flattened, batch_size=64, epochs=2, learning_rate=1e-4):
        """训练 DNN 和 Transformer 模型 (使用数据生成器)"""
        print("\n开始训练模型 (使用生成器)...")
        print(f"Batch Size: {batch_size}, Epochs: {epochs}, Learning Rate: {learning_rate}")
        print(f"接收到的展平特征维度: {input_dim_flattened}")
        
        if input_dim_flattened <= 0:
            print("错误: 无效的输入维度，无法训练模型。")
            return
        
        # 初始化模型
        criterion = nn.MSELoss()
        
        # DNN Model
        print("\n--- 初始化 DNN 模型 ---")
        self.model_dnn = IndexEnhancementModel(input_dim_flattened).to(self.device)
        print(f"DNN 模型已初始化并移至 {self.device}")
        optimizer_dnn = torch.optim.Adam(self.model_dnn.parameters(), lr=learning_rate)
        
        # Transformer Model
        print("\n--- 初始化 Transformer 模型 ---")
        self.model_transformer = TransformerForTimeSeries(input_dim=input_dim_flattened).to(self.device)
        print(f"Transformer 模型已初始化并移至 {self.device}")
        optimizer_transformer = torch.optim.Adam(self.model_transformer.parameters(), lr=learning_rate)
        
        # 训练循环
        for epoch in range(epochs):
            print(f"\n--- 开始 Epoch {epoch+1}/{epochs} ---")
            
            # 重置每个epoch的损失
            total_loss_dnn_epoch = 0
            total_loss_transformer_epoch = 0
            num_batches_processed_dnn = 0
            num_batches_processed_transformer = 0
            
            # 使用生成器获取数据
            for features_batch, targets_batch in features_train_generator:
                # DNN训练
                self.model_dnn.train()
                chunk_dataset = StockDataset(features_batch.values, targets_batch.values)
                chunk_dataloader = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=True, 
                                            pin_memory=True if self.device.type == 'cuda' else False)
                
                for batch_features, batch_targets in chunk_dataloader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    # 确保目标维度正确
                    if batch_targets.dim() == 1:
                        batch_targets = batch_targets.unsqueeze(1)
                    
                    outputs_dnn = self.model_dnn(batch_features)
                    # 确保输出维度与目标维度匹配
                    if outputs_dnn.dim() == 1:
                        outputs_dnn = outputs_dnn.unsqueeze(1)
                    
                    loss_dnn = criterion(outputs_dnn, batch_targets)
                    
                    optimizer_dnn.zero_grad()
                    loss_dnn.backward()
                    optimizer_dnn.step()
                    
                    total_loss_dnn_epoch += loss_dnn.item() * batch_features.size(0)
                    num_batches_processed_dnn += 1
                
                # Transformer训练
                self.model_transformer.train()
                for batch_features, batch_targets in chunk_dataloader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    # 确保目标维度正确
                    if batch_targets.dim() == 1:
                        batch_targets = batch_targets.unsqueeze(1)
                    
                    outputs_transformer = self.model_transformer(batch_features)
                    # 确保输出维度与目标维度匹配
                    if outputs_transformer.dim() == 1:
                        outputs_transformer = outputs_transformer.unsqueeze(1)
                    
                    loss_transformer = criterion(outputs_transformer, batch_targets)
                    
                    optimizer_transformer.zero_grad()
                    loss_transformer.backward()
                    optimizer_transformer.step()
                    
                    total_loss_transformer_epoch += loss_transformer.item() * batch_features.size(0)
                    num_batches_processed_transformer += 1
                
                # 清理内存
                del chunk_dataset, chunk_dataloader, features_batch, targets_batch
                clear_memory()
            
            # 计算平均损失
            avg_loss_dnn = total_loss_dnn_epoch / num_batches_processed_dnn if num_batches_processed_dnn > 0 else 0
            avg_loss_transformer = total_loss_transformer_epoch / num_batches_processed_transformer if num_batches_processed_transformer > 0 else 0
            
            print(f"DNN Epoch [{epoch+1}/{epochs}] 平均 Loss: {avg_loss_dnn:.4f}")
            print(f"Transformer Epoch [{epoch+1}/{epochs}] 平均 Loss: {avg_loss_transformer:.4f}")
        
        print("--- 所有模型训练完成 ---")

    def predict_returns(self,
                        features_val_scaled_np, # 接收完整的 NumPy 数组
                        targets_val_aligned_series, # 接收对齐的 Series 以获取索引
                        model_type='dnn'):
        """使用指定模型预测股票的下一期收益率 (使用数据生成器)"""
        print(f"\n开始使用 {model_type.upper()} 模型和生成器进行预测...")

        # Select the model
        if model_type == 'dnn':
            model = self.model_dnn
            if model is None: print("错误: DNN 模型未训练或未初始化。无法预测。"); return None
        elif model_type == 'transformer':
            model = self.model_transformer
            if model is None: print("错误: Transformer 模型未训练或未初始化。无法预测。"); return None
        else:
            print(f"错误: 未知的模型类型 '{model_type}'"); return None

        model.to(self.device)
        model.eval() 
        
        all_predictions_list = [] # List to hold numpy arrays
        all_indices_list = []   # List to hold pandas MultiIndex objects
        batch_size = 1024       # Prediction batch size within a chunk
        processed_chunks = 0

        # Create the generator instance inside the function
        print(f"Predict ({model_type.upper()}): 创建预测生成器实例 (使用数据副本)...")
        predict_generator = self.prepare_features(
            features_val_scaled_np, # Pass the features directly
            targets_val_aligned_series, # Pass the targets directly
            batch_size=batch_size
        )

        with torch.no_grad():
            progress_bar = tqdm(enumerate(predict_generator), desc=f"预测块 ({model_type.upper()})")
            for chunk_idx, (features_chunk, _) in progress_bar:
                if features_chunk.empty: continue

                processed_chunks += 1
                chunk_indices = features_chunk.index # Keep MultiIndex object
                features_np = features_chunk.values.astype(np.float32)
                chunk_predictions_batches = [] # Collect batch predictions for this chunk

                num_samples_in_chunk = features_np.shape[0]
                for i in range(0, num_samples_in_chunk, batch_size):
                    batch_features_np = features_np[i : i + batch_size]
                    batch_features_tensor = torch.FloatTensor(batch_features_np).to(self.device)
                    batch_preds = model(batch_features_tensor)
                    # 确保预测结果维度正确
                    if batch_preds.dim() == 1:
                        batch_preds = batch_preds.unsqueeze(1)
                    chunk_predictions_batches.append(batch_preds.cpu().numpy())
                
                if chunk_predictions_batches:
                    # Concatenate predictions for the current chunk
                    chunk_predictions_np = np.concatenate(chunk_predictions_batches, axis=0).squeeze()
                    all_predictions_list.append(chunk_predictions_np)
                    all_indices_list.append(chunk_indices) # Append the MultiIndex object
                
                del features_chunk, features_np, chunk_predictions_batches, chunk_indices
                clear_memory()
                progress_bar.set_postfix(chunks=f'{processed_chunks}')

        print(f"预测完成 ({model_type.upper()})，处理了 {processed_chunks} 个数据块。")

        if not all_predictions_list:
            print(f"错误 ({model_type.upper()}): 未能从生成器生成任何预测结果。")
            return None

        # Concatenate results from all chunks
        try:
            print(f"合并 {len(all_predictions_list)} 个块的预测结果...")
            # Concatenate numpy arrays
            final_predictions_np = np.concatenate(all_predictions_list, axis=0)
            
            # Correctly concatenate MultiIndex objects
            index_tuples = [idx_tuple for chunk_index in all_indices_list for idx_tuple in chunk_index.tolist()]
            if not index_tuples:
                 raise ValueError("未能从块中提取任何索引元组。")
            final_index = pd.MultiIndex.from_tuples(index_tuples, names=all_indices_list[0].names)

            if len(final_index) != len(final_predictions_np):
                 raise ValueError(f"合并后的索引长度 ({len(final_index)}) 与预测数量 ({len(final_predictions_np)}) 不匹配！")
                     
            predicted_returns_series = pd.Series(final_predictions_np, index=final_index, dtype=np.float32)
            
            # 确保预测结果与目标Series对齐
            if not predicted_returns_series.index.equals(targets_val_aligned_series.index):
                print(f"警告: 预测结果索引与目标Series索引不匹配，进行对齐...")
                predicted_returns_series = predicted_returns_series.reindex(targets_val_aligned_series.index)
            
            print(f"最终预测 Series 创建完成 ({model_type.upper()})，形状: {predicted_returns_series.shape}")
            return predicted_returns_series
        except Exception as e:
            print(f"错误 ({model_type.upper()}): 合并预测结果时出错: {e}")
            traceback.print_exc()
            return None

    def run_strategy(self, top_n=50, epochs=2, learning_rate=1e-4, transaction_cost_rate=None):
        """运行策略，包括数据划分、Scaler拟合、训练、预测和回测"""
        print("开始运行策略...")
        
        # --- Load, Preprocess ---
        if not self.load_data(): return
        if transaction_cost_rate is not None:
            self.transaction_cost_rate = transaction_cost_rate
            print(f"更新单边交易成本率: {self.transaction_cost_rate:.4f}")
        if not self.preprocess_data(): return

        # --- 提取完整的每日特征和目标 ---
        print("\n提取完整的每日特征和目标...")
        try:
            # 选择特征列
            s_dq_columns = [col for col in self.data.columns if col.startswith('S_DQ_')]
            s_fa_lagged_columns = [col for col in self.data.columns if col.startswith('S_FA_') and col.endswith('_lagged')]
            calculated_indicator_columns = ['RETURN', 'MA5', 'MA10', 'MA20', 'VOLUME_MA5', 'VOLUME_MA10', 'VOLATILITY']
            
            # 获取可用的指标列
            available_indicators = [col for col in calculated_indicator_columns if col in self.data.columns]
            potential_feature_columns = s_dq_columns + s_fa_lagged_columns + available_indicators
            potential_feature_columns = sorted(list(set(potential_feature_columns)))
            
            # 排除不需要的列
            exclude_columns = ['DATE', 'S_INFO_WINDCODE', 'S_DQ_TRADESTATUSCODE', 'S_DQ_ADJFACTOR'] + \
                             [col for col in self.data.columns if col.startswith('S_FA_') and not col.endswith('_lagged')]
            feature_columns = [col for col in potential_feature_columns if col not in exclude_columns]
            
            # 验证特征列
            valid_feature_columns = [col for col in feature_columns if col in self.data.columns]
            if not valid_feature_columns:
                print("错误: 未找到任何有效的潜在特征列。")
                return
            
            # 选择数值特征列
            numeric_feature_columns = self.data[valid_feature_columns].select_dtypes(include=np.number).columns.tolist()
            if not numeric_feature_columns:
                print("错误: 未能选择任何有效的每日数值特征列。")
                return
            
            print(f"选择了 {len(numeric_feature_columns)} 个每日数值特征列。")
            input_dim_flattened = self.lookback_days * len(numeric_feature_columns)
            print(f"计算得到的展平特征维度: {input_dim_flattened}")
            
            # 确保索引正确
            if not isinstance(self.data.index, pd.MultiIndex):
                if 'DATE' in self.data.columns and 'S_INFO_WINDCODE' in self.data.columns:
                    if self.data[['DATE', 'S_INFO_WINDCODE']].isnull().any().any():
                        print("错误: DATE 或 S_INFO_WINDCODE 列包含 NaN，无法设置为索引。")
                        return
                    print("Setting index to ['DATE', 'S_INFO_WINDCODE']...")
                    self.data = self.data.drop_duplicates(subset=['DATE', 'S_INFO_WINDCODE'])
                    self.data = self.data.set_index(['DATE', 'S_INFO_WINDCODE'])
                    print("Sorting index after setting...")
                    self.data = self.data.sort_index()
                else:
                    print("错误: DATE 或 S_INFO_WINDCODE 列未找到。")
                    return
            
            # 提取特征和目标
            print("Extracting daily features...")
            features_daily_df = self.data[numeric_feature_columns].astype(np.float32)
            
            print("Calculating future returns (target)...")
            if not self.data.index.is_monotonic_increasing:
                self.data = self.data.sort_index()
            targets_daily_series = self.data.groupby('S_INFO_WINDCODE', group_keys=False)['RETURN'].shift(-1).astype(np.float32)
            targets_daily_series.name = 'TARGET'
            
            # 对齐和清理NaN
            print("Aligning features and targets and cleaning NaNs...")
            combined_daily_data = pd.concat([features_daily_df, targets_daily_series], axis=1, join='outer')
            print(f"  Shape before dropna: {combined_daily_data.shape}")
            
            if 'TARGET' not in combined_daily_data.columns:
                print("错误: 'TARGET' 列未能成功合并。")
                return
            
            rows_before_dropna = len(combined_daily_data)
            combined_daily_data.dropna(inplace=True)
            rows_after_dropna = len(combined_daily_data)
            print(f"  Dropped {rows_before_dropna - rows_after_dropna} rows containing NaNs.")
            print(f"  Shape after dropna: {combined_daily_data.shape}")
            
            if combined_daily_data.empty:
                print("错误: 清理 NaN 后数据为空。")
                return
            
            # 分离特征和目标
            features_daily_df = combined_daily_data[numeric_feature_columns].astype(np.float32)
            targets_daily_series = combined_daily_data['TARGET'].astype(np.float32)
            
            if not features_daily_df.index.equals(targets_daily_series.index):
                print("错误: 清理后重新分离特征和目标时索引不匹配！")
                return
            
            print(f"清理和对齐完成。Cleaned Features Shape: {features_daily_df.shape}, Cleaned Targets Shape: {targets_daily_series.shape}")
            
            # --- 数据划分 ---
            print(f"\n按日期 {self.validation_start_date} 划分清理后的每日数据...")
            train_dates_mask = features_daily_df.index.get_level_values('DATE') < self.validation_start_date
            val_dates_mask = features_daily_df.index.get_level_values('DATE') >= self.validation_start_date
            
            features_daily_train_df = features_daily_df[train_dates_mask]
            targets_daily_train_series = targets_daily_series[train_dates_mask]
            features_daily_val_df = features_daily_df[val_dates_mask]
            targets_daily_val_series = targets_daily_series[val_dates_mask]
            
            if features_daily_train_df.empty or features_daily_val_df.empty:
                print("错误: 清理后的每日训练集或验证集为空。")
                return
                
            # 验证数据集的完整性
            print("\n验证数据集的完整性...")
            print(f"训练集大小: {features_daily_train_df.shape}, 验证集大小: {features_daily_val_df.shape}")
            
            # 验证特征列的一致性
            train_cols = set(features_daily_train_df.columns)
            val_cols = set(features_daily_val_df.columns)
            if train_cols != val_cols:
                print("错误: 训练集和验证集的特征列不匹配！")
                print(f"训练集特有列: {train_cols - val_cols}")
                print(f"验证集特有列: {val_cols - train_cols}")
                return
                
            # 创建训练集生成器
            print("\n创建训练集数据生成器...")
            train_generator = self.prepare_features(
                features_daily_train_df,
                targets_daily_train_series,
                batch_size=256  # 修改为256
            )
            
            # 创建验证集生成器
            print("\n创建验证集数据生成器...")
            val_generator = self.prepare_features(
                features_daily_val_df,
                targets_daily_val_series,
                batch_size=256  # 修改为256
            )
            
            if train_generator is None or val_generator is None:
                print("错误: 无法创建数据生成器。")
                return
                
            print("\n开始模型训练...")
            try:
                self.train_models(train_generator, targets_daily_train_series, 
                                input_dim_flattened, batch_size=256,  # 修改为256
                                epochs=epochs, learning_rate=learning_rate)
            except Exception as e:
                print(f"训练模型时出错: {str(e)}")
                traceback.print_exc()
                return
            
            # --- 预测和回测 ---
            print("\n开始预测验证集收益率...")
            try:
                # 验证集预测
                print("\n开始使用 DNN 模型进行预测...")
                predicted_returns_dnn_series = self.predict_returns(
                    features_daily_val_df,  # 直接传递DataFrame
                    targets_daily_val_series,  # 直接传递Series
                    model_type='dnn')
                print("\n开始使用 Transformer 模型进行预测...")
                predicted_returns_transformer_series = self.predict_returns(
                    features_daily_val_df,  # 直接传递DataFrame
                    targets_daily_val_series,  # 直接传递Series
                    model_type='transformer')
                if predicted_returns_dnn_series is None or predicted_returns_transformer_series is None:
                    print("错误: 一个或多个模型预测失败。无法继续进行回测。")
                    return
            except Exception as e:
                print(f"预测验证集收益率时出错: {str(e)}")
                traceback.print_exc()
                return
            # 回测
            print("\n开始生成投资组合并进行回测 (DNN vs Transformer vs Index)...")
            if predicted_returns_dnn_series is None or predicted_returns_transformer_series is None or targets_daily_val_series is None:
                print("错误: 预测结果或实际目标未能成功准备，无法进行回测。")
                return
            try:
                # 验证集回测
                self.generate_portfolio_and_backtest(
                    predicted_returns_dnn_series,
                    predicted_returns_transformer_series,
                    targets_daily_val_series,
                    top_n=top_n,
                    transaction_cost_rate=self.transaction_cost_rate
                )
            except Exception as e:
                print(f"生成投资组合或回测时出错: {str(e)}")
                traceback.print_exc()
                return
            print("\n策略运行完成。")
            
        except Exception as e:
            print(f"提取特征和目标时出错: {e}")
            traceback.print_exc()
            return

    def generate_portfolio_and_backtest(self, 
                                        predicted_returns_dnn_series,      # Series predictions
                                        predicted_returns_transformer_series, # Series predictions
                                        actual_returns_aligned_series, # Series with MultiIndex
                                        top_n=50, 
                                        transaction_cost_rate=0.001,
                                        is_train_set=False):
        """根据预测收益率生成 Top N 投资组合并进行回测 (DNN vs Transformer vs Index)"""

        # --- Input Validation & Preparation ---
        if not isinstance(actual_returns_aligned_series, pd.Series) or \
           not isinstance(actual_returns_aligned_series.index, pd.MultiIndex):
            print("错误: actual_returns_aligned_series 必须是带有 MultiIndex 的 Pandas Series。")
            return
        if len(predicted_returns_dnn_series) != len(actual_returns_aligned_series) or \
           len(predicted_returns_transformer_series) != len(actual_returns_aligned_series):
            print("错误: 预测数组长度与对齐的目标 Series 长度不匹配。")
            return

        if transaction_cost_rate is None:
            transaction_cost_rate = self.transaction_cost_rate

        print(f"\n开始回测 Top {top_n} 策略 (DNN vs Transformer, 每周调仓)...")
        print(f"使用单边交易成本率: {transaction_cost_rate:.4f}")

        # --- Data Preparation for Backtest Loop ---
        try:
            backtest_data = pd.DataFrame({
                'predicted_dnn': predicted_returns_dnn_series.values,
                'predicted_transformer': predicted_returns_transformer_series.values,
                'actual': actual_returns_aligned_series.values
            }, index=actual_returns_aligned_series.index)
        except Exception as e:
             print(f"创建回测 DataFrame 时出错: {e}")
             traceback.print_exc()
             return

        # 获取唯一的日期
        dates = backtest_data.index.get_level_values('DATE').unique().sort_values()
        if len(dates) == 0:
             print("错误：无法从回测数据中提取有效日期。")
             return
        print(f"回测时间范围: {dates.min()} 到 {dates.max()}，共 {len(dates)} 天")

        # --- Weekly Rebalancing Loop ---
        daily_portfolio_returns_dnn = []
        daily_portfolio_returns_transformer = []
        held_stock_codes_dnn = pd.Index([])
        held_stock_codes_transformer = pd.Index([])
        last_rebalance_week_marker = None

        for date in tqdm(dates, desc="周调仓回测进度 (DNN & Transformer)"):
            current_week_marker = (date.year, date.isocalendar().week)
            is_rebalancing_decision_day = False

            if last_rebalance_week_marker is None or current_week_marker != last_rebalance_week_marker:
                is_rebalancing_decision_day = True
                last_rebalance_week_marker = current_week_marker

            try:
                daily_data = backtest_data.loc[date]
                current_preds_dnn = daily_data['predicted_dnn']
                current_preds_transformer = daily_data['predicted_transformer']
                current_actuals = daily_data['actual']
            except KeyError:
                daily_portfolio_returns_dnn.append(0)
                daily_portfolio_returns_transformer.append(0)
                continue
            except Exception as e:
                 print(f"日期 {date}: 提取当日数据时出错: {e}")
                 daily_portfolio_returns_dnn.append(0)
                 daily_portfolio_returns_transformer.append(0)
                 continue

            # --- Calculate Daily Gross Returns ---
            daily_gross_return_dnn = 0
            if not held_stock_codes_dnn.empty:
                 available_held_dnn = held_stock_codes_dnn.intersection(daily_data.index)
                 if not available_held_dnn.empty:
                     selected_actual_returns_dnn = current_actuals.loc[available_held_dnn]
                     daily_gross_return_dnn = selected_actual_returns_dnn.mean()

            daily_gross_return_transformer = 0
            if not held_stock_codes_transformer.empty:
                 available_held_transformer = held_stock_codes_transformer.intersection(daily_data.index)
                 if not available_held_transformer.empty:
                     selected_actual_returns_transformer = current_actuals.loc[available_held_transformer]
                     daily_gross_return_transformer = selected_actual_returns_transformer.mean()

            # --- Apply Costs and Update Holdings ---
            daily_net_return_dnn = daily_gross_return_dnn
            daily_net_return_transformer = daily_gross_return_transformer
            turnover_cost_dnn = 0
            turnover_cost_transformer = 0

            if is_rebalancing_decision_day:
                new_held_stock_codes_dnn = pd.Index([])
                if not current_preds_dnn.empty:
                    sorted_preds_dnn = current_preds_dnn.sort_values(ascending=False)
                    top_stocks_dnn = sorted_preds_dnn.head(top_n)
                    new_held_stock_codes_dnn = top_stocks_dnn.index

                new_held_stock_codes_transformer = pd.Index([])
                if not current_preds_transformer.empty:
                    sorted_preds_transformer = current_preds_transformer.sort_values(ascending=False)
                    top_stocks_transformer = sorted_preds_transformer.head(top_n)
                    new_held_stock_codes_transformer = top_stocks_transformer.index

                turnover_dnn = len(held_stock_codes_dnn.difference(new_held_stock_codes_dnn)) + \
                               len(new_held_stock_codes_dnn.difference(held_stock_codes_dnn))
                turnover_transformer = len(held_stock_codes_transformer.difference(new_held_stock_codes_transformer)) + \
                                       len(new_held_stock_codes_transformer.difference(held_stock_codes_transformer))

                if top_n > 0:
                     turnover_cost_dnn = (turnover_dnn / top_n) * transaction_cost_rate if not held_stock_codes_dnn.empty else (len(new_held_stock_codes_dnn)/top_n) * transaction_cost_rate
                     turnover_cost_transformer = (turnover_transformer / top_n) * transaction_cost_rate if not held_stock_codes_transformer.empty else (len(new_held_stock_codes_transformer)/top_n) * transaction_cost_rate

                held_stock_codes_dnn = new_held_stock_codes_dnn
                held_stock_codes_transformer = new_held_stock_codes_transformer

            daily_net_return_dnn -= turnover_cost_dnn
            daily_net_return_transformer -= turnover_cost_transformer
            daily_portfolio_returns_dnn.append(daily_net_return_dnn)
            daily_portfolio_returns_transformer.append(daily_net_return_transformer)

            # --- Memory cleanup ---
            del daily_data, current_preds_dnn, current_preds_transformer, current_actuals
            if is_rebalancing_decision_day:
                 if 'sorted_preds_dnn' in locals(): del sorted_preds_dnn, top_stocks_dnn, new_held_stock_codes_dnn
                 if 'sorted_preds_transformer' in locals(): del sorted_preds_transformer, top_stocks_transformer, new_held_stock_codes_transformer

        # --- Performance Calculation & Plotting ---
        if len(daily_portfolio_returns_dnn) != len(dates):
             print(f"警告: DNN 日收益序列长度 ({len(daily_portfolio_returns_dnn)}) 与日期长度 ({len(dates)}) 不匹配!")
             dates_dnn = dates[:len(daily_portfolio_returns_dnn)]
        else: dates_dnn = dates
        if len(daily_portfolio_returns_transformer) != len(dates):
             print(f"警告: Transformer 日收益序列长度 ({len(daily_portfolio_returns_transformer)}) 与日期长度 ({len(dates)}) 不匹配!")
             dates_transformer = dates[:len(daily_portfolio_returns_transformer)]
        else: dates_transformer = dates

        portfolio_returns_dnn_series = pd.Series(daily_portfolio_returns_dnn, index=dates_dnn)
        portfolio_returns_transformer_series = pd.Series(daily_portfolio_returns_transformer, index=dates_transformer)
        cumulative_returns_dnn = (1 + portfolio_returns_dnn_series).cumprod() - 1
        cumulative_returns_transformer = (1 + portfolio_returns_transformer_series).cumprod() - 1

        print("准备基准指数数据...")
        index_file = os.path.join(self.data_dir, 'raw_csi300_index_prices.parquet')
        try:
             index_prices_full = pd.read_parquet(index_file)
             index_prices_full['DATE'] = pd.to_datetime(index_prices_full['TRADE_DT'])
             index_prices_full = index_prices_full.set_index('DATE').sort_index()
             index_prices_full['RETURN'] = index_prices_full['S_DQ_CLOSE'].pct_change()
             benchmark_returns = index_prices_full['RETURN'].reindex(dates).fillna(0)
        except Exception as e:
             print(f"加载或处理基准指数数据时出错: {e}. 无法计算基准表现。")
             benchmark_returns = pd.Series(0, index=dates)
        cumulative_benchmark_returns = (1 + benchmark_returns).cumprod() - 1

        metrics_dnn = self.calculate_metrics(portfolio_returns_dnn_series, benchmark_returns.reindex(dates_dnn).fillna(0))
        metrics_transformer = self.calculate_metrics(portfolio_returns_transformer_series, benchmark_returns.reindex(dates_transformer).fillna(0))
        metrics_benchmark = self.calculate_metrics(benchmark_returns, benchmark_returns)

        print("\n--- 回测性能指标 ---")
        print("指标        | DNN        | Transformer | 基准 (沪深300)")
        print("-----------------|------------|-------------|---------------")
        print(f"总回报 (%)   | {metrics_dnn['Total Return (%)']:.2f}     | {metrics_transformer['Total Return (%)']:.2f}      | {metrics_benchmark['Total Return (%)']:.2f}")
        print(f"年化回报 (%) | {metrics_dnn['Annualized Return (%)']:.2f}     | {metrics_transformer['Annualized Return (%)']:.2f}      | {metrics_benchmark['Annualized Return (%)']:.2f}")
        print(f"年化波动率(%)| {metrics_dnn['Annualized Volatility (%)']:.2f}     | {metrics_transformer['Annualized Volatility (%)']:.2f}      | {metrics_benchmark['Annualized Volatility (%)']:.2f}")
        print(f"夏普比率     | {metrics_dnn['Sharpe Ratio']:.2f}     | {metrics_transformer['Sharpe Ratio']:.2f}      | {metrics_benchmark['Sharpe Ratio']:.2f}")
        print(f"最大回撤 (%)  | {metrics_dnn['Max Drawdown (%)']:.2f}     | {metrics_transformer['Max Drawdown (%)']:.2f}      | {metrics_benchmark['Max Drawdown (%)']:.2f}")
        print(f"Beta         | {metrics_dnn['Beta']:.2f}     | {metrics_transformer['Beta']:.2f}      | {metrics_benchmark['Beta']:.2f}")
        print(f"Alpha (%)    | {metrics_dnn['Alpha (%)']:.2f}     | {metrics_transformer['Alpha (%)']:.2f}      | {metrics_benchmark['Alpha (%)']:.2f}")

        plt.figure(figsize=(14, 8))
        (cumulative_returns_dnn * 100).plot(label=f'DNN Top {top_n} (周调仓)')
        (cumulative_returns_transformer * 100).plot(label=f'Transformer Top {top_n} (周调仓)')
        (cumulative_benchmark_returns * 100).plot(label='沪深300指数 (基准)', linestyle='--')
        plt.title(f'模型累积收益率对比 (Top {top_n}, 成本率: {transaction_cost_rate:.2%})')
        plt.xlabel('日期')
        plt.ylabel('累积收益率 (%)')
        try:
             plt.rcParams['font.sans-serif'] = ['SimHei']
             plt.rcParams['axes.unicode_minus'] = False
        except Exception as font_e:
             print(f"设置中文字体失败: {font_e}. 图例可能显示不正确。")
        plt.legend()
        plt.grid(True)
        if not os.path.exists('plots'): os.makedirs('plots')
        plot_filename = f'plots/backtest_comparison_top{top_n}_cost{transaction_cost_rate:.4f}_non_generator.png'
        plt.savefig(plot_filename)
        print(f"回测结果图已保存至: {plot_filename}")
        # plt.show()

        self.plot_backtest_results(portfolio_returns_dnn_series, portfolio_returns_transformer_series, benchmark_returns, top_n, transaction_cost_rate, is_train_set=is_train_set)

    def plot_backtest_results(self, dnn_returns, transformer_returns, benchmark_returns, top_n, transaction_cost, is_train_set=False):
        """绘制回测结果对比图，is_train_set=True时为训练集区间"""
        # 计算累积收益率
        dnn_cumulative = (1 + dnn_returns).cumprod()
        transformer_cumulative = (1 + transformer_returns).cumprod()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        # 创建图形
        plt.figure(figsize=(12, 6))
        # 绘制累积收益率曲线
        plt.plot(dnn_cumulative.index, dnn_cumulative, label=f'DNN (Top {top_n})', linewidth=2)
        plt.plot(transformer_cumulative.index, transformer_cumulative, label=f'Transformer (Top {top_n})', linewidth=2)
        plt.plot(benchmark_cumulative.index, benchmark_cumulative, label='沪深300', linewidth=2)
        # 设置标题和标签
        if is_train_set:
            plt.title(f'训练集表现对比 (交易成本: {transaction_cost:.4f})')
            plot_path = f'plots/train_set_comparison_top{top_n}_cost{transaction_cost:.4f}_non_generator.png'
        else:
            plt.title(f'Top {top_n} 策略回测结果对比 (交易成本: {transaction_cost:.4f})')
            plot_path = f'plots/backtest_comparison_top{top_n}_cost{transaction_cost:.4f}_non_generator.png'
        plt.xlabel('日期')
        plt.ylabel('累积收益率')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_path)
        plt.close()

    def calculate_metrics(self, returns, benchmark_returns, risk_free_rate=0.0):
        """计算投资组合性能指标"""
        if returns.empty:
             return {'Total Return (%)': 0,'Annualized Return (%)': 0,'Annualized Volatility (%)': 0,'Sharpe Ratio': 0,'Max Drawdown (%)': 0,'Beta': 0,'Alpha (%)': 0}
        common_index = returns.index.intersection(benchmark_returns.index)
        returns = returns.loc[common_index]; benchmark_returns = benchmark_returns.loc[common_index]
        if returns.empty: return {'Total Return (%)': 0,'Annualized Return (%)': 0,'Annualized Volatility (%)': 0,'Sharpe Ratio': 0,'Max Drawdown (%)': 0,'Beta': 0,'Alpha (%)': 0}
        total_return = (1 + returns).prod() - 1
        days = (returns.index[-1] - returns.index[0]).days
        years = days / 365.25 if days > 0 else 1
        trading_days_per_year = 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        annualized_volatility = returns.std() * np.sqrt(trading_days_per_year)
        excess_returns = returns - (risk_free_rate / trading_days_per_year)
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(trading_days_per_year) if excess_returns.std() != 0 else 0
        cumulative = (1 + returns).cumprod(); peak = cumulative.cummax(); drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        df_for_cov = pd.DataFrame({'returns': returns, 'benchmark': benchmark_returns}).dropna()
        if len(df_for_cov) > 1:
            cov_matrix = df_for_cov.cov()
            beta = cov_matrix.loc['returns', 'benchmark'] / cov_matrix.loc['benchmark', 'benchmark'] if cov_matrix.loc['benchmark', 'benchmark'] != 0 else 0
            alpha_daily = returns.mean() - (risk_free_rate / trading_days_per_year) - beta * (benchmark_returns.mean() - (risk_free_rate / trading_days_per_year))
            alpha_annualized = alpha_daily * trading_days_per_year
        else: beta = 0; alpha_annualized = 0
        return {'Total Return (%)': total_return * 100,'Annualized Return (%)': annualized_return * 100,'Annualized Volatility (%)': annualized_volatility * 100,'Sharpe Ratio': sharpe_ratio,'Max Drawdown (%)': max_drawdown * 100,'Beta': beta,'Alpha (%)': alpha_annualized * 100}

    def calculate_portfolio_returns(self, predictions, actual_returns, top_n=50, lookback_window=60):
        """计算投资组合收益，使用基于预测收益率的权重分配"""
        portfolio_returns = []
        transaction_costs = []
        current_weights = None
        previous_weights = None
        
        for date in tqdm(predictions.index, desc="计算投资组合收益"):
            current_preds = predictions.loc[date]
            
            # 选择预测收益率最高的前N只股票
            sorted_preds = current_preds.sort_values(ascending=False)
            top_stocks = sorted_preds.head(top_n)
            
            if len(top_stocks) == 0:
                portfolio_returns.append(0)
                transaction_costs.append(0)
                continue
                
            # 基于预测收益率计算权重
            current_weights = top_stocks / top_stocks.sum()
            
            # 计算交易成本
            if previous_weights is not None:
                # 计算权重变化
                weight_changes = abs(current_weights - previous_weights)
                # 计算交易成本
                cost = weight_changes.sum() * self.transaction_cost_rate
            else:
                cost = 1.0 * self.transaction_cost_rate  # 首次建仓成本
            
            # 计算当日收益
            daily_return = (current_weights * actual_returns.loc[date, top_stocks.index]).sum()
            
            # 扣除交易成本
            net_return = daily_return - cost
            
            portfolio_returns.append(net_return)
            transaction_costs.append(cost)
            
            previous_weights = current_weights.copy()
        
        return pd.Series(portfolio_returns, index=predictions.index), pd.Series(transaction_costs, index=predictions.index)

def count_parameters(model):
    """计算模型的参数量"""
    return sum(p.numel() for p in model.parameters())

def print_model_parameters():
    """打印两个模型的参数量"""
    # 假设input_dim为100（实际值会根据特征数量变化）
    input_dim = 100
    
    # 创建DNN模型
    dnn_model = IndexEnhancementModel(input_dim)
    dnn_params = count_parameters(dnn_model)
    
    # 创建Transformer模型
    transformer_model = TransformerForTimeSeries(input_dim)
    transformer_params = count_parameters(transformer_model)
    
    print(f"\n模型参数量统计:")
    print(f"DNN模型总参数量: {dnn_params:,}")
    print(f"Transformer模型总参数量: {transformer_params:,}")
    print(f"参数量比例 (Transformer/DNN): {transformer_params/dnn_params:.2f}")

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='指数增强策略回测程序')
    parser.add_argument('--top_n', type=int, default=50,
                      help='选择排名前N的股票进行投资（默认：50）')
    parser.add_argument('--epochs', type=int, default=2,
                      help='模型训练轮数（默认：2）')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='学习率（默认：0.0001）')
    parser.add_argument('--transaction_cost', type=float, default=0.001,
                      help='单边交易成本率（默认：0.001）')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 打印模型参数量
    print_model_parameters()
    
    start_time = time.time()
    strategy = IndexEnhancementStrategy()
    # 运行策略，使用命令行参数
    strategy.run_strategy(
        top_n=args.top_n,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        transaction_cost_rate=args.transaction_cost
    )
    end_time = time.time()
    print(f"\n总运行时间: {end_time - start_time:.2f} 秒")