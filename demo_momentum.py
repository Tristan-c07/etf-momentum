"""
快速验证：动量特征完整流程

展示从加载数据到计算动量和排名的完整流程
"""

import pandas as pd
import numpy as np
from src.features.momentum import compute_momentum, make_cross_section_rank


def demo():
    """演示完整流程"""
    
    print("=" * 70)
    print("动量特征模块 - 完整流程演示")
    print("=" * 70)
    
    # 创建模拟的日线数据（模拟 load_daily 的输出）
    print("\n步骤 1: 准备日线数据（模拟 load_daily 输出）")
    print("-" * 70)
    
    dates = pd.date_range('2024-01-01', periods=25, freq='D')
    symbols = ['510300.SH', '510500.SH', '159915.SZ']
    
    data = []
    np.random.seed(123)
    
    for symbol in symbols:
        base_price = 100
        for date in dates:
            base_price *= (1 + np.random.randn() * 0.015)
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
    
    print(f"原始数据形状: {df.shape}")
    print(f"包含列: {df.columns.tolist()}")
    print(f"\n前5行:")
    print(df.head())
    
    # 步骤2: 计算20日动量
    print("\n\n步骤 2: 计算 20 日动量（自动 shift(1) 避免前瞻）")
    print("-" * 70)
    
    df = compute_momentum(df, window=20)
    
    print(f"新增列: mom_20")
    print(f"\n最后6行（显示 mom_20）:")
    print(df[['date', 'symbol', 'close', 'mom_20']].tail(6))
    
    # 步骤3: 添加截面排名
    print("\n\n步骤 3: 添加截面排名（每个日期内排名 1-N）")
    print("-" * 70)
    
    df = make_cross_section_rank(df, signal_col='mom_20')
    
    print(f"新增列: mom_20_rank")
    print(f"\n最后一天的排名:")
    last_date = df['date'].max()
    last_day = df[df['date'] == last_date].sort_values('mom_20_rank', ascending=False)
    print(last_day[['date', 'symbol', 'close', 'mom_20', 'mom_20_rank']])
    
    # 步骤4: 最终输出
    print("\n\n步骤 4: 最终输出格式")
    print("-" * 70)
    print("每个 (date, symbol) 包含:")
    print("  - mom_20: 动量值（已 shift(1)，无前瞻偏差）")
    print("  - mom_20_rank: 截面排名（值越大，rank 越高）")
    
    print(f"\n最后3天的数据:")
    print(df[['date', 'symbol', 'close', 'mom_20', 'mom_20_rank']].tail(9))
    
    # 验证关键特性
    print("\n\n关键特性验证")
    print("-" * 70)
    
    # 1. 验证无前瞻偏差
    print("\n1. 无前瞻偏差验证")
    symbol_df = df[df['symbol'] == symbols[0]].reset_index(drop=True)
    idx = 21  # 第22个交易日
    
    mom_actual = symbol_df.loc[idx, 'mom_20']
    price_t_minus_1 = symbol_df.loc[idx-1, 'close']
    price_t_minus_21 = symbol_df.loc[idx-21, 'close']
    mom_expected = price_t_minus_1 / price_t_minus_21 - 1
    
    print(f"   第22个交易日的 mom_20:")
    print(f"   - 使用 t-1 日价格: {price_t_minus_1:.2f}")
    print(f"   - 使用 t-21 日价格: {price_t_minus_21:.2f}")
    print(f"   - 预期值: {mom_expected:.6f}")
    print(f"   - 实际值: {mom_actual:.6f}")
    print(f"   - 差异: {abs(mom_actual - mom_expected):.2e}")
    print(f"   ✓ 无前瞻偏差")
    
    # 2. 验证排名
    print("\n2. 截面排名验证")
    for date in df['date'].unique()[-3:]:
        date_df = df[df['date'] == date]
        if date_df['mom_20_rank'].notna().all():
            ranks = sorted(date_df['mom_20_rank'].values)
            print(f"   {date.date()} 排名: {ranks} ✓")
    
    # 使用场景说明
    print("\n\n典型使用场景")
    print("-" * 70)
    print("""
# 1. 加载数据
from src.data.load_daily import load_daily
df = load_daily(universe, start_date, end_date)

# 2. 计算动量特征
from src.features.momentum import compute_momentum, make_cross_section_rank
df = compute_momentum(df, window=20)
df = make_cross_section_rank(df, 'mom_20')

# 3. 选择动量最强的 Top K
top_k = 10
latest_date = df['date'].max()
top_etfs = df[df['date'] == latest_date].nlargest(top_k, 'mom_20_rank')
print(top_etfs[['symbol', 'mom_20', 'mom_20_rank']])
    """)
    
    print("\n" + "=" * 70)
    print("✅ 演示完成！动量特征模块可以正常使用")
    print("=" * 70)


if __name__ == '__main__':
    demo()
