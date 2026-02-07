# 动量特征使用示例

## 快速开始

```python
import pandas as pd
from src.data.load_daily import load_daily
from src.features.momentum import compute_momentum, make_cross_section_rank

# 1. 加载数据
df = load_daily(
    universe=['510300.SH', '510500.SH', '159915.SZ'],
    start_date='2020-01-01',
    end_date='2024-12-31',
    adjust='qfq'
)

# 2. 计算 20 日动量（自动 shift(1) 避免前瞻偏差）
df = compute_momentum(df, window=20)

# 3. 添加截面排名（每个日期内排名）
df = make_cross_section_rank(df, signal_col='mom_20')

# 4. 查看结果
print(df[['date', 'symbol', 'close', 'mom_20', 'mom_20_rank']].tail())
```

## 输出格式

给定日线数据表，输出每个 `(date, symbol)` 的：
- `mom_20`: 20日动量值（已 shift(1)，无前瞻偏差）
- `mom_20_rank`: 该日期截面内的排名（1到N，N为该日ETF数量）

示例输出：
```
         date     symbol   close    mom_20  mom_20_rank
0  2024-12-27  510300.SH   3.456    0.0523          3.0
1  2024-12-27  510500.SH   2.987    0.0312          2.0
2  2024-12-27  159915.SZ   1.876   -0.0156          1.0
```

## 核心特性

### 1. 无前瞻偏差

`compute_momentum()` 自动 shift(1)，确保：
- t 日的 `mom_20` = (t-1日收盘价 / t-21日收盘价) - 1
- t 日的信号只使用 t-1 及之前的数据
- 适用于"t日收盘后计算，t+1日开盘执行"的交易场景

```python
# 验证无前瞻偏差
symbol_df = df[df['symbol'] == '510300.SH'].reset_index(drop=True)

# 第22个交易日（索引21）的动量
idx = 21
mom = symbol_df.loc[idx, 'mom_20']
price_t_minus_1 = symbol_df.loc[idx-1, 'close']
price_t_minus_21 = symbol_df.loc[idx-21, 'close']
expected = price_t_minus_1 / price_t_minus_21 - 1

assert abs(mom - expected) < 1e-8  # ✓ 验证通过
```

### 2. 截面排名

`make_cross_section_rank()` 在每个日期内独立排名：
- 值越大，排名越高（ascending=True）
- 每个日期的排名范围：[1, N]，N为该日期的ETF数量
- NaN 值不参与排名

```python
# 查看某个日期的排名
date = '2024-12-31'
date_df = df[df['date'] == date].sort_values('mom_20_rank', ascending=False)
print(date_df[['symbol', 'mom_20', 'mom_20_rank']])

# 输出：
# symbol      mom_20  mom_20_rank
# 510300.SH   0.0523         3.0  <- 动量最强
# 510500.SH   0.0312         2.0
# 159915.SZ  -0.0156         1.0  <- 动量最弱
```

## 批量计算多窗口

使用 `compute_momentum_features()` 一次计算多个窗口：

```python
from src.features.momentum import compute_momentum_features

# 计算 5日、20日、60日动量及其排名
df = compute_momentum_features(
    df, 
    windows=[5, 20, 60],
    add_rank=True
)

# 新增列：
# - mom_5, mom_5_rank
# - mom_20, mom_20_rank  
# - mom_60, mom_60_rank
```

## 数据要求

输入 DataFrame 必须包含：
- `date`: 日期列（datetime64）
- `symbol`: 标的代码（str）
- `close`: 收盘价（float）

输入数据应已按 `['symbol', 'date']` 排序（函数会自动排序）。

## 常见问题

### Q1: 为什么前 21 个交易日的 mom_20 是 NaN？

A: 20日动量需要 20 个历史数据点，加上 shift(1)，所以前 21 个交易日无法计算。

### Q2: 如何理解 shift(1)？

A: 
- 不 shift：t 日信号使用 t 日收盘价 → ❌ 前瞻偏差（t 日收盘价在 t 日收盘后才知道）
- shift(1)：t 日信号使用 t-1 日收盘价 → ✓ 无偏差（t 日可以用 t-1 日已知数据）

### Q3: 排名的 ascending=True 是什么意思？

A: 
- `ascending=True`（默认）：值越大，rank 越高（适用于动量等"越大越好"的信号）
- `ascending=False`：值越小，rank 越高（适用于估值等"越小越好"的信号）

## 性能提示

- 对于大数据集，按 symbol 分组的计算已优化
- 建议先排序数据：`df.sort_values(['symbol', 'date'], inplace=True)`
- 截面排名使用 pandas 内置 rank()，性能良好
