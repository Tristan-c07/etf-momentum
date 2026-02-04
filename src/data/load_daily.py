import numpy as np
import akshare as ak
import pandas as pd

from typing import List, Union
from datetime import datetime
import time


def load_daily(
    universe: Union[List[str], str],
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
    retry: int = 3,
    sleep: float = 0.5
) -> pd.DataFrame:
    # 处理单个代码的情况
    if isinstance(universe, str):
        universe = [universe]
    
    # 转换日期格式（去掉 '-'）
    start_date_fmt = start_date.replace('-', '')
    end_date_fmt = end_date.replace('-', '')
    
    all_data = []
    failed_symbols = []
    
    for symbol in universe:
        success = False
        
        for attempt in range(retry):
            try:
                # 提取纯数字代码（AkShare 只需要数字部分）
                code = symbol.split('.')[0]
                
                # 调用 AkShare API
                df = ak.fund_etf_hist_em(
                    symbol=code,
                    period="daily",
                    start_date=start_date_fmt,
                    end_date=end_date_fmt,
                    adjust=adjust
                )
                
                if df.empty:
                    print(f"Warning: {symbol} returned empty data")
                    break
                
                # 重命名列（AkShare 返回中文列名）
                column_mapping = {
                    '日期': 'date',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume',
                    '成交额': 'amount'
                }
                df = df.rename(columns=column_mapping)
                
                # 选择需要的列
                df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
                
                # 添加 symbol 列（保留原始格式）
                df['symbol'] = symbol
                
                # 转换数据类型
                df['date'] = pd.to_datetime(df['date'])
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                df['amount'] = df['amount'].astype(float)
                
                # 重新排列列顺序
                df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount']]
                
                all_data.append(df)
                success = True
                print(f"✓ {symbol}: {len(df)} records")
                
                # 避免请求过快
                time.sleep(sleep)
                break
                
            except Exception as e:
                if attempt < retry - 1:
                    print(f"Retry {attempt + 1}/{retry} for {symbol}: {e}")
                    time.sleep(1)
                else:
                    print(f"✗ Failed to load {symbol} after {retry} attempts: {e}")
                    failed_symbols.append(symbol)
    
    # 合并所有数据
    if not all_data:
        print("Error: No data loaded successfully")
        return pd.DataFrame(columns=['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount'])
    
    result = pd.concat(all_data, ignore_index=True)
    
    # 排序：先按 date，再按 symbol
    result = result.sort_values(['date', 'symbol']).reset_index(drop=True)
    
    # 打印摘要
    print(f"\n{'='*50}")
    print(f"Total symbols: {len(universe)}")
    print(f"Successfully loaded: {len(universe) - len(failed_symbols)}")
    print(f"Failed: {len(failed_symbols)}")
    if failed_symbols:
        print(f"Failed symbols: {failed_symbols}")
    print(f"Date range: {result['date'].min().date()} to {result['date'].max().date()}")
    print(f"Total records: {len(result)}")
    print(f"{'='*50}\n")
    
    return result


def load_daily_from_config(config_path: str, universe_path: str) -> pd.DataFrame:
    """
    从配置文件加载数据
    
    Parameters
    ----------
    config_path : str
        experiment.yaml 路径
    universe_path : str
        universe.yaml 路径
    
    Returns
    -------
    pd.DataFrame
        日线数据
    """
    import yaml
    
    # 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 读取资产池
    with open(universe_path, 'r', encoding='utf-8') as f:
        universe_config = yaml.safe_load(f)
    
    # 合并所有 ETF
    all_etfs = []
    for category in universe_config.values():
        if isinstance(category, list):
            all_etfs.extend(category)
    
    # 加载数据
    return load_daily(
        universe=all_etfs,
        start_date=config['start_date'],
        end_date=config['end_date']
    )


if __name__ == '__main__':
    # 测试代码
    test_symbols = ['510300.SH', '510500.SH', '159915.SZ']
    df = load_daily(test_symbols, '2024-01-01', '2024-12-31')
    
    print("\nData Preview:")
    print(df.head(10))
    print("\nData Info:")
    print(df.info())
    print("\nBasic Stats:")
    print(df.describe())
