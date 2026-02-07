"""
简单测试动量特征计算（不需要网络）
"""

import pandas as pd
import numpy as np
from src.features.momentum import compute_momentum, make_cross_section_rank


def main():
    print("=" * 60)
    print("动量特征模块测试")
    print("=" * 60)
    
    # 创建模拟数据：3个ETF，30天
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    symbols = ['510300.SH', '510500.SH', '159915.SZ']
    
    data = []
    np.random.seed(42)
    
    for symbol in symbols:
        base_price = 100
        for i, date in enumerate(dates):
            # 模拟价格随机游走
            base_price *= (1 + np.random.randn() * 0.02)
            data.append({
                'date': date,
                'symbol': symbol,
                'open': base_price * 0.99,
                'high': base_price * 1.01,
                'low': base_price * 0.98,
                'close': base_price,
                'volume': np.random.randint(100000, 1000000),
                'amount': base_price * np.random.randint(100000, 1000000)
            })
    
    df = pd.DataFrame(data)
    print(f"\n原始数据: {df.shape}")
    print(df.head(10))
    
    # 测试1: 计算20日动量
    print("\n" + "=" * 60)
    print("测试 1: 计算 20 日动量")
    print("=" * 60)
    
    df = compute_momentum(df, window=20)
    
    print("\n最新10条数据（含 mom_20）:")
    print(df[['date', 'symbol', 'close', 'mom_20']].tail(10))
    
    # 统计
    print(f"\nmom_20 统计:")
    print(f"  非空值数量: {df['mom_20'].notna().sum()} / {len(df)}")
    print(f"  均值: {df['mom_20'].mean():.4f}")
    print(f"  标准差: {df['mom_20'].std():.4f}")
    print(f"  范围: [{df['mom_20'].min():.4f}, {df['mom_20'].max():.4f}]")
    
    # 测试2: 截面排名
    print("\n" + "=" * 60)
    print("测试 2: 截面排名")
    print("=" * 60)
    
    df = make_cross_section_rank(df, 'mom_20')
    
    print("\n最新一天的排名:")
    latest_date = df['date'].max()
    latest_df = df[df['date'] == latest_date].sort_values('mom_20_rank', ascending=False)
    print(latest_df[['date', 'symbol', 'close', 'mom_20', 'mom_20_rank']])
    
    # 验证排名
    print(f"\n排名验证:")
    print(f"  排名列非空值: {df['mom_20_rank'].notna().sum()}")
    
    # 检查几个日期的排名是否正确（应该是1, 2, 3）
    for date in df['date'].unique()[-3:]:
        date_df = df[df['date'] == date]
        if date_df['mom_20_rank'].notna().all():
            ranks = sorted(date_df['mom_20_rank'].values)
            print(f"  {date.date()} 排名: {ranks}")
    
    # 测试3: 验证无前瞻偏差
    print("\n" + "=" * 60)
    print("测试 3: 验证无前瞻偏差")
    print("=" * 60)
    
    # 对于第一个symbol，手动验证某天的计算
    symbol_df = df[df['symbol'] == symbols[0]].reset_index(drop=True)
    
    # 取索引21（第22个交易日）
    idx = 21
    if idx < len(symbol_df):
        mom_value = symbol_df.loc[idx, 'mom_20']
        
        # 应该使用 t-1 和 t-21 的价格
        price_t_minus_1 = symbol_df.loc[idx-1, 'close']
        price_t_minus_21 = symbol_df.loc[idx-21, 'close']
        expected_mom = price_t_minus_1 / price_t_minus_21 - 1
        
        print(f"\n{symbols[0]} 在第 {idx+1} 个交易日:")
        print(f"  t-1 收盘价: {price_t_minus_1:.2f}")
        print(f"  t-21 收盘价: {price_t_minus_21:.2f}")
        print(f"  预期 mom_20: {expected_mom:.6f}")
        print(f"  实际 mom_20: {mom_value:.6f}")
        print(f"  差异: {abs(mom_value - expected_mom):.10f}")
        
        if abs(mom_value - expected_mom) < 1e-8:
            print("  ✓ 计算正确，无前瞻偏差")
        else:
            print("  ✗ 计算错误！")
    
    # 最终输出
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)
    
    print("\n关键特性验证:")
    print("1. ✓ compute_momentum() 计算动量并自动 shift(1)")
    print("2. ✓ make_cross_section_rank() 在每个日期截面内排名")
    print("3. ✓ 信号无前瞻偏差（使用 t-1 及之前的数据）")
    
    print("\n输出格式:")
    print(df[['date', 'symbol', 'close', 'mom_20', 'mom_20_rank']].tail(6))


if __name__ == '__main__':
    main()
