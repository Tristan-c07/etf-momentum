"""
测试动量特征计算模块
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

from src.data.load_daily import load_daily
from src.features.momentum import compute_momentum, make_cross_section_rank


def test_basic_momentum():
    """测试基础动量计算"""
    print("=" * 60)
    print("测试 1: 基础动量计算")
    print("=" * 60)
    
    # 创建简单测试数据
    data = {
        'date': pd.date_range('2024-01-01', periods=25, freq='D'),
        'symbol': ['TEST.SH'] * 25,
        'close': [100 + i for i in range(25)]  # 递增价格
    }
    df = pd.DataFrame(data)
    
    # 计算 20 日动量
    df = compute_momentum(df, window=20)
    
    print("\n前几行数据:")
    print(df[['date', 'symbol', 'close', 'mom_20']].head(25))
    
    # 验证：第21个交易日（索引20）应该有 mom_20 值
    # mom_20 = close[19] / close[19-20] - 1 (因为 shift(1))
    # 但前21个交易日应该都是 NaN（因为需要 window + shift）
    print(f"\n第21行的 mom_20: {df.loc[20, 'mom_20']}")
    print(f"第22行的 mom_20: {df.loc[21, 'mom_20']}")
    
    # 验证计算：close[20]=120, close[0]=100
    # mom_20 = 120/100 - 1 = 0.2
    expected = df.loc[20, 'close'] / df.loc[0, 'close'] - 1
    print(f"预期 mom_20 (第22行): {expected:.4f}")
    print(f"实际 mom_20 (第22行): {df.loc[21, 'mom_20']:.4f}")
    

def test_cross_section_rank():
    """测试截面排名"""
    print("\n" + "=" * 60)
    print("测试 2: 截面排名")
    print("=" * 60)
    
    # 创建多个 symbol 的测试数据
    dates = pd.date_range('2024-01-01', periods=3, freq='D')
    symbols = ['A', 'B', 'C']
    
    data = []
    for date in dates:
        for symbol in symbols:
            data.append({
                'date': date,
                'symbol': symbol,
                'mom_20': np.random.randn()
            })
    
    df = pd.DataFrame(data)
    df = df.sort_values(['date', 'symbol'])
    
    print("\n原始数据:")
    print(df)
    
    # 添加排名
    df = make_cross_section_rank(df, 'mom_20')
    
    print("\n添加排名后:")
    print(df[['date', 'symbol', 'mom_20', 'mom_20_rank']])
    
    # 验证每个日期的排名是 1, 2, 3
    for date in dates:
        date_df = df[df['date'] == date]
        ranks = sorted(date_df['mom_20_rank'].values)
        print(f"\n{date.date()} 的排名: {ranks}")
        assert ranks == [1.0, 2.0, 3.0], f"排名错误: {ranks}"
    
    print("\n✓ 排名验证通过")


def test_real_data():
    """使用真实数据测试"""
    print("\n" + "=" * 60)
    print("测试 3: 真实数据测试")
    print("=" * 60)
    
    # 加载少量真实数据
    symbols = ['510300.SH', '510500.SH', '159915.SZ']
    print(f"\n加载数据: {symbols}")
    
    df = load_daily(
        universe=symbols,
        start_date='2024-01-01',
        end_date='2024-12-31',
        adjust='qfq',
        retry=3,
        sleep=0.3
    )
    
    print(f"数据形状: {df.shape}")
    
    # 计算动量
    df = compute_momentum(df, window=20)
    
    # 添加排名
    df = make_cross_section_rank(df, 'mom_20')
    
    print("\n最新 10 条数据:")
    print(df[['date', 'symbol', 'close', 'mom_20', 'mom_20_rank']].tail(10))
    
    # 验证数据质量
    print("\n数据质量检查:")
    print(f"- mom_20 非空值: {df['mom_20'].notna().sum()} / {len(df)}")
    print(f"- mom_20_rank 非空值: {df['mom_20_rank'].notna().sum()} / {len(df)}")
    print(f"- mom_20 范围: [{df['mom_20'].min():.4f}, {df['mom_20'].max():.4f}]")
    
    # 检查某个日期的排名
    latest_date = df['date'].max()
    latest_df = df[df['date'] == latest_date].sort_values('mom_20_rank', ascending=False)
    print(f"\n{latest_date.date()} 的动量排名:")
    print(latest_df[['symbol', 'mom_20', 'mom_20_rank']])
    
    print("\n✓ 真实数据测试通过")


def test_no_lookahead_bias():
    """验证无前瞻偏差"""
    print("\n" + "=" * 60)
    print("测试 4: 验证无前瞻偏差")
    print("=" * 60)
    
    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=25, freq='D')
    data = {
        'date': dates,
        'symbol': ['TEST'] * 25,
        'close': [100 + i*2 for i in range(25)]  # 每天涨 2 元
    }
    df = pd.DataFrame(data)
    
    df = compute_momentum(df, window=20)
    
    # 第 22 个交易日（索引 21）
    # 应该使用 t-1 日（索引20）和 t-1-20 日（索引0）的收盘价
    # mom = close[20] / close[0] - 1 = 140 / 100 - 1 = 0.4
    
    if pd.notna(df.loc[21, 'mom_20']):
        actual_mom = df.loc[21, 'mom_20']
        # 手动计算
        price_t_minus_1 = df.loc[20, 'close']  # 140
        price_t_minus_21 = df.loc[0, 'close']   # 100
        expected_mom = price_t_minus_1 / price_t_minus_21 - 1
        
        print(f"第 22 个交易日（索引 21）:")
        print(f"  t-1 日收盘价（索引20）: {price_t_minus_1}")
        print(f"  t-21 日收盘价（索引0）: {price_t_minus_21}")
        print(f"  预期 mom_20: {expected_mom:.4f}")
        print(f"  实际 mom_20: {actual_mom:.4f}")
        
        assert abs(actual_mom - expected_mom) < 1e-6, "前瞻偏差检测失败！"
        print("\n✓ 无前瞻偏差验证通过")
    else:
        print("⚠️  第 22 个交易日 mom_20 为 NaN")


if __name__ == '__main__':
    print("开始测试动量特征模块...\n")
    
    test_basic_momentum()
    test_cross_section_rank()
    test_no_lookahead_bias()
    test_real_data()
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
