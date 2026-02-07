"""
动量特征计算模块

提供动量信号计算和截面排名功能
"""

import pandas as pd
import numpy as np
from typing import Optional


def compute_momentum(
    df: pd.DataFrame,
    window: int = 20,
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    计算动量信号
    
    对每个 symbol 计算：mom = close / close.shift(window) - 1
    
    ⚠️ 关键：信号会自动 shift(1)，避免收盘价当日可用的前瞻偏差
    即：t日的动量信号 = (t-1日收盘价 / t-window-1日收盘价) - 1
    
    Parameters
    ----------
    df : pd.DataFrame
        日线数据，必须包含 ['date', 'symbol', price_col] 列
        要求已按 ['symbol', 'date'] 排序
    window : int, default 20
        动量计算窗口（交易日数）
    price_col : str, default 'close'
        用于计算动量的价格列名
    
    Returns
    -------
    pd.DataFrame
        在原始数据基础上新增 f'mom_{window}' 列
        格式：[..., mom_20]
    
    Examples
    --------
    >>> df = load_daily(['510300.SH'], '2024-01-01', '2024-12-31')
    >>> df = compute_momentum(df, window=20)
    >>> # t日的 mom_20 反映的是 t-1 到 t-1-20 的收益率
    
    Notes
    -----
    - 前 window+1 个交易日的动量为 NaN
    - shift(1) 确保 t 日信号使用 t-1 日及之前的数据
    - 适用于 t 日收盘后计算信号，t+1 日开盘执行交易的场景
    """
    # 检查必需列
    required_cols = ['date', 'symbol', price_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame 缺少必需列: {missing_cols}")
    
    # 复制数据避免修改原始DataFrame
    df = df.copy()
    
    # 确保按 symbol 和 date 排序
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # 按 symbol 分组计算动量
    # mom = close / close.shift(window) - 1
    df[f'mom_{window}'] = df.groupby('symbol')[price_col].apply(
        lambda x: x / x.shift(window) - 1
    ).values
    
    # ⚠️ 关键：信号 shift(1) 避免前瞻偏差
    # t 日的信号使用 t-1 日收盘价计算
    df[f'mom_{window}'] = df.groupby('symbol')[f'mom_{window}'].shift(1).values
    
    return df


def make_cross_section_rank(
    df: pd.DataFrame,
    signal_col: str,
    ascending: bool = True,
    method: str = 'average'
) -> pd.DataFrame:
    """
    在每个日期截面内对信号进行排名
    
    对每个 date，将所有 symbol 的信号值进行排名（1 到 N）
    
    Parameters
    ----------
    df : pd.DataFrame
        包含信号的数据，必须包含 ['date', 'symbol', signal_col] 列
    signal_col : str
        要排名的信号列名（如 'mom_20'）
    ascending : bool, default True
        True: 值越大排名越高（rank 越大）
        False: 值越小排名越高
    method : str, default 'average'
        排名方法，pandas.rank() 的 method 参数
        - 'average': 相同值取平均排名
        - 'min': 相同值取最小排名
        - 'max': 相同值取最大排名
        - 'first': 相同值按出现顺序排名
        - 'dense': 相同值取相同排名，排名连续
    
    Returns
    -------
    pd.DataFrame
        在原始数据基础上新增 f'{signal_col}_rank' 列
        格式：[..., signal_col, {signal_col}_rank]
        rank 范围: [1, N]，N 为该日期截面的 symbol 数量
    
    Examples
    --------
    >>> df = compute_momentum(df, window=20)
    >>> df = make_cross_section_rank(df, 'mom_20')
    >>> # 每个日期内，mom_20 越大，mom_20_rank 越大
    
    Notes
    -----
    - NaN 值不参与排名，其 rank 为 NaN
    - 排名在每个 date 截面内独立计算
    - ascending=True 表示值越大排名越高（适用于动量等信号）
    """
    # 检查必需列
    required_cols = ['date', 'symbol', signal_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame 缺少必需列: {missing_cols}")
    
    # 复制数据避免修改原始DataFrame
    df = df.copy()
    
    # 按日期分组，对信号进行排名
    rank_col = f'{signal_col}_rank'
    df[rank_col] = df.groupby('date')[signal_col].rank(
        ascending=ascending,
        method=method,
        na_option='keep'  # NaN 保持为 NaN
    )
    
    return df


def compute_momentum_features(
    df: pd.DataFrame,
    windows: list[int] = [20],
    add_rank: bool = True
) -> pd.DataFrame:
    """
    批量计算多个窗口的动量特征及其排名
    
    便捷函数，一次性计算多个动量窗口及其截面排名
    
    Parameters
    ----------
    df : pd.DataFrame
        日线数据，必须包含 ['date', 'symbol', 'close'] 列
    windows : list of int, default [20]
        动量窗口列表，如 [5, 10, 20, 60]
    add_rank : bool, default True
        是否添加截面排名列
    
    Returns
    -------
    pd.DataFrame
        添加了动量特征和排名的数据
        新增列：mom_{w}, mom_{w}_rank (如果 add_rank=True)
    
    Examples
    --------
    >>> df = load_daily(universe, '2020-01-01', '2024-12-31')
    >>> df = compute_momentum_features(df, windows=[20, 60])
    >>> # 新增列: mom_20, mom_20_rank, mom_60, mom_60_rank
    """
    df = df.copy()
    
    for window in windows:
        # 计算动量
        df = compute_momentum(df, window=window)
        
        # 计算排名
        if add_rank:
            signal_col = f'mom_{window}'
            df = make_cross_section_rank(df, signal_col)
    
    return df
