# -*- coding: utf-8 -*-
"""
V2版本：ML风控 + 多因子动量 + 连续仓位调节
====================================================

核心升级（基于V1 + 第1问notebook ML部分融合）：

1. 【多因子动量选股】—— 不再只看单一MOM
   - ROC (多窗口)、RSI、MACD柱、EMA趋势差
   - 合成"复合动量得分"做截面排名

2. 【ML市场状态预测】—— 替代V1的MA硬规则
   - 特征：反转+状态切换类（借鉴notebook的make_features_reversal_regime）
   - 双模型集成：HuberRegressor + HistGradientBoosting
   - Walk-Forward OOS预测基准未来5日收益
   - 输出连续的"市场情绪得分"

3. 【连续仓位调节】—— 替代V1的三档离散切换
   - ML预测分 → 经Sigmoid映射到 [0,1] 的动量仓位比例
   - 动态平滑 + 变化率限幅，避免单次调仓过大
   - 仓位 = Sigmoid(score) × 基础权重
   - 剩余部分 → 避险标的(511880)

4. 【危机过滤器】—— 极端保护
   - 高波动+趋势下行 → 仓位额外衰减（与ML共同作用）
   - ATR跟踪止损保底

架构概览：
┌────────────────────────┐
│   每个调仓日（REBAL）    │
│                        │
│  1. ML模型预测市场状态   │
│     → market_score ∈ ℝ │
│                        │
│  2. Sigmoid映射连续仓位  │
│     → mom_pct ∈ [0,1]  │
│                        │
│  3. 多因子动量选股       │
│     → TopK ETF列表      │
│                        │
│  4. 最终权重 =           │
│     mom_pct × 等权TopK  │
│     + (1-mom_pct) × 防守│
│                        │
│  5. 危机过滤/止损        │
└────────────────────────┘

适用平台：聚宽(joinquant) 回测
"""
from jqdata import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor

# ========================================
# 参数区
# ========================================
UNIVERSE = [
    '510300.XSHG', '510050.XSHG', '510500.XSHG', '159919.XSHE', '159915.XSHE',
    '588000.XSHG', '588080.XSHG', '512100.XSHG', '512010.XSHG', '512760.XSHG',
    '512480.XSHG', '512800.XSHG', '512660.XSHG', '516160.XSHG', '515790.XSHG',
    '159941.XSHE', '518880.XSHG', '513050.XSHG', '513100.XSHG', '511880.XSHG',
]

BENCHMARK = '000300.XSHG'
DEFENSIVE = '511880.XSHG'      # 避险标的（银华日利）

# 选股参数
TOPK = 10
REBALANCE_EVERY = 5
MAX_WEIGHT = 0.15

# ========================================
# 多因子动量参数（V2新增）
# ========================================
# 各因子权重（合成复合动量得分）
FACTOR_WEIGHTS = {
    'mom_20':   0.25,   # 20日收益率动量
    'mom_60':   0.20,   # 60日收益率动量
    'roc_10':   0.15,   # 10日变动率
    'rsi_14':   0.15,   # RSI（标准化后使用）
    'macd_h':   0.15,   # MACD柱状图
    'trend':    0.10,   # EMA12-EMA26趋势差
}

# ========================================
# ML风控参数（V2新增）
# ========================================
USE_ML_RISK = True               # 启用ML风控

# Walk-Forward参数
ML_TRAIN_WINDOW = 252 * 3       # 3年滚动训练窗口
ML_RETRAIN_EVERY = 10           # 每10个交易日重新训练
ML_HORIZON = 5                  # 预测未来5日收益
ML_HALF_LIFE = 126              # 时间衰减半衰期
ML_TAIL_WEIGHT = 3.0            # 极端样本加权倍数
ML_TAIL_QUANTILE = 0.85         # 极端样本阈值分位数

# 集成权重
ENSEMBLE_LIN_WEIGHT = 0.45      # 线性模型权重
ENSEMBLE_TREE_WEIGHT = 0.55     # 树模型权重

# 连续仓位映射参数
SIGMOID_SCALE = 8.0             # Sigmoid缩放（越大越陡峭，越接近0/1）
SIGMOID_CENTER = 0.0            # Sigmoid中心点
POS_SMOOTH_ALPHA = 0.3          # 仓位EMA平滑系数（越小越平滑）
POS_MAX_CHANGE = 0.25           # 单次调仓最大变化幅度

# 最低/最高动量仓位
MIN_MOMENTUM_PCT = 0.0          # 最悲观时0%动量
MAX_MOMENTUM_PCT = 1.0          # 最乐观时100%动量

# 危机过滤参数
CRISIS_VOL_RATIO_THRESH = 1.3   # 波动率比值阈值（短/长 > 阈值视为高波动）
CRISIS_TREND_THRESH = -0.01     # 趋势阈值（EMA差/价格 < 阈值视为下行）
CRISIS_DAMPING = 0.5            # 危机时仓位衰减系数

# ATR止损
USE_ATR_STOP = True
ATR_STOP_MULTIPLIER = 3.0       # ATR倍数止损


# ========================================
# 初始化
# ========================================
def initialize(context):
    set_benchmark(BENCHMARK)
    set_option('use_real_price', True)
    log.set_level('order', 'error')

    set_commission(PerTrade(buy_cost=0.0003, sell_cost=0.0003, min_cost=5))
    set_slippage(FixedSlippage(0.0002))

    g.universe = list(UNIVERSE)
    g.day_count = 0
    g.last_target = []

    # V2新增：ML模型状态
    g.model_lin = None                # 线性模型
    g.model_tree = None               # 树模型
    g.last_train_day = -999           # 上次训练的day_count
    g.market_score = 0.0              # 当前市场情绪得分（原始值）
    g.current_mom_pct = 0.5           # 当前动量仓位比例（初始50%）

    # 止损追踪
    g.highest_portfolio_value = 0.0

    # 统计
    g.train_count = 0
    g.score_history = []

    log.info(f"=== V2策略初始化 ===")
    log.info(f"选股: TOPK={TOPK}, REBAL={REBALANCE_EVERY}")
    log.info(f"多因子: {list(FACTOR_WEIGHTS.keys())}")
    if USE_ML_RISK:
        log.info(f"ML风控: TRAIN={ML_TRAIN_WINDOW}d, RETRAIN={ML_RETRAIN_EVERY}d, HORIZON={ML_HORIZON}d")
        log.info(f"仓位映射: SIGMOID_SCALE={SIGMOID_SCALE}, SMOOTH={POS_SMOOTH_ALPHA}")
        log.info(f"危机过滤: VOL_RATIO>{CRISIS_VOL_RATIO_THRESH}, TREND<{CRISIS_TREND_THRESH}")

    run_daily(rebalance, time='09:35')


def before_trading_start(context):
    pass


def handle_data(context, data):
    pass


# ========================================
# 主调仓逻辑
# ========================================
def rebalance(context):
    g.day_count += 1
    if g.day_count % REBALANCE_EVERY != 1:
        return

    universe = [s for s in g.universe if is_tradable(context, s)]
    if not universe:
        log.warn(f"无可交易标的，跳过")
        return

    # ========================================
    # Step 1: ML市场状态预测 → 连续仓位
    # ========================================
    momentum_pct = 0.5  # 默认中性

    if USE_ML_RISK:
        # 训练/预测
        market_score = ml_predict_market(context)
        g.market_score = market_score

        # Sigmoid映射到 [0,1]
        raw_pct = sigmoid_map(market_score, scale=SIGMOID_SCALE, center=SIGMOID_CENTER)

        # 危机过滤：额外衰减
        crisis_factor = calc_crisis_factor(context)
        raw_pct = raw_pct * crisis_factor

        # 限制范围
        raw_pct = np.clip(raw_pct, MIN_MOMENTUM_PCT, MAX_MOMENTUM_PCT)

        # 平滑 + 限幅
        momentum_pct = smooth_position(raw_pct, g.current_mom_pct,
                                        alpha=POS_SMOOTH_ALPHA,
                                        max_change=POS_MAX_CHANGE)
        g.current_mom_pct = momentum_pct

        g.score_history.append((context.current_dt, market_score, momentum_pct, crisis_factor))

        log.info(f"ML得分={market_score:.4f} | 原始仓位={raw_pct:.2%} | "
                 f"平滑仓位={momentum_pct:.2%} | 危机因子={crisis_factor:.2f}")

    # ========================================
    # Step 2: 多因子复合动量选股
    # ========================================
    composite_mom = calc_composite_momentum(context, universe)
    if composite_mom.empty:
        log.info("无动量数据，跳过")
        return

    target = composite_mom.sort_values(ascending=False).head(TOPK).index.tolist()

    # ========================================
    # Step 3: 构建目标权重
    # ========================================
    if momentum_pct > 0.01:
        momentum_weights = make_weights_equal(target, max_weight=MAX_WEIGHT)
        momentum_weights = {s: w * momentum_pct for s, w in momentum_weights.items()}
    else:
        momentum_weights = {}

    defensive_pct = 1.0 - momentum_pct
    final_weights = momentum_weights.copy()
    if defensive_pct > 0.01 and is_tradable(context, DEFENSIVE):
        final_weights[DEFENSIVE] = defensive_pct

    # ========================================
    # Step 4: ATR止损检查
    # ========================================
    if USE_ATR_STOP:
        stop_triggered = check_atr_stop(context)
        if stop_triggered:
            log.info(f"⚠️ ATR止损触发，全部转防守")
            final_weights = {}
            if is_tradable(context, DEFENSIVE):
                final_weights[DEFENSIVE] = 1.0

    # ========================================
    # Step 5: 执行下单
    # ========================================
    apply_target_weights(context, final_weights)
    g.last_target = target


# ========================================
# ML市场状态预测模块
# ========================================
def ml_predict_market(context):
    """
    ML核心：用基准指数的技术指标预测未来5日收益
    Walk-Forward：每隔ML_RETRAIN_EVERY天重新训练

    返回：market_score（原始预测值，正=看多，负=看空）
    """
    # 需要足够的历史数据
    need_bars = ML_TRAIN_WINDOW + 100
    try:
        df = get_price(
            BENCHMARK,
            end_date=context.current_dt,
            count=need_bars,
            frequency='1d',
            fields=['open', 'high', 'low', 'close', 'volume'],
            panel=False
        )
    except Exception as e:
        log.warn(f"获取基准数据失败: {e}")
        return 0.0

    if df is None or df.empty or len(df) < ML_TRAIN_WINDOW + 50:
        return 0.0

    # 处理时间索引：如果时间在索引中则重置，否则使用time列
    if 'time' in df.columns:
        df = df.sort_values('time').reset_index(drop=True)
        df = df.rename(columns={'time': 'date'}).set_index('date')
    else:
        df = df.reset_index()
        if 'index' in df.columns:
            df = df.rename(columns={'index': 'date'}).set_index('date')
        df = df.sort_index()

    # 构建特征
    df_feat = make_ml_features(df)

    # 构建目标：未来h日收益
    df_feat['target'] = df_feat['close'].pct_change(ML_HORIZON).shift(-ML_HORIZON)

    # 去除NaN
    df_feat = df_feat.dropna(subset=['target'])

    # 选择数值特征列
    drop_cols = {'open', 'high', 'low', 'close', 'volume', 'target'}
    feature_cols = [c for c in df_feat.columns if c not in drop_cols
                    and df_feat[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    X = df_feat[feature_cols]
    y = df_feat['target']

    if len(X) < ML_TRAIN_WINDOW + 10:
        return 0.0

    # 判断是否需要重新训练
    should_retrain = (g.model_lin is None or
                      g.day_count - g.last_train_day >= ML_RETRAIN_EVERY)

    if should_retrain:
        # 用最近 ML_TRAIN_WINDOW 个样本训练
        X_train = X.iloc[-ML_TRAIN_WINDOW - ML_HORIZON:-ML_HORIZON]
        y_train = y.iloc[-ML_TRAIN_WINDOW - ML_HORIZON:-ML_HORIZON]

        if len(X_train) < 100:
            return 0.0

        # 手动填充缺失值（用中位数）
        X_train_filled = X_train.fillna(X_train.median())
        
        # 样本权重：时间衰减 + 极端加权
        sw = _make_sample_weight(y_train.values, ML_HALF_LIFE, ML_TAIL_WEIGHT, ML_TAIL_QUANTILE)

        # 训练双模型
        try:
            # 线性模型：标准化 + HuberRegressor
            g.scaler_lin = StandardScaler()
            X_train_scaled = g.scaler_lin.fit_transform(X_train_filled)
            g.model_lin = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=600)
            g.model_lin.fit(X_train_scaled, y_train, sample_weight=sw)

            # 树模型：GradientBoostingRegressor（兼容旧版sklearn）
            g.model_tree = GradientBoostingRegressor(
                max_depth=3, learning_rate=0.05, n_estimators=250,
                subsample=0.8, random_state=42
            )
            g.model_tree.fit(X_train_filled, y_train, sample_weight=sw)

            g.last_train_day = g.day_count
            g.train_count += 1
            log.info(f"🔧 ML模型重新训练 (第{g.train_count}次, 样本数={len(X_train)})")

        except Exception as e:
            log.warn(f"ML训练失败: {e}")
            return 0.0

    # 预测（用最新一行）
    try:
        X_latest = X.iloc[[-1]]
        # 手动填充缺失值（与训练时保持一致）
        X_latest_filled = X_latest.fillna(X_train.median())
        
        # 线性模型需要标准化
        X_latest_scaled = g.scaler_lin.transform(X_latest_filled)
        p_lin = g.model_lin.predict(X_latest_scaled)[0]
        
        # 树模型直接预测
        p_tree = g.model_tree.predict(X_latest_filled)[0]
        
        score = ENSEMBLE_LIN_WEIGHT * p_lin + ENSEMBLE_TREE_WEIGHT * p_tree
        return float(score)
    except Exception as e:
        log.warn(f"ML预测失败: {e}")
        return 0.0


def make_ml_features(df):
    """
    构建ML特征（借鉴notebook的make_features_reversal_regime）
    专注于：反转信号 + 状态切换 + 多尺度动量
    """
    df = df.copy().sort_index()

    # ------ 基础收益与多尺度动量 ------
    df['ret_1'] = df['close'].pct_change()
    for n in [2, 3, 5, 10, 20, 60, 120]:
        df[f'ret_{n}'] = df['close'].pct_change(n)

    # ------ 波动与状态 ------
    df['vol_20'] = df['ret_1'].rolling(20).std()
    df['vol_60'] = df['ret_1'].rolling(60).std()
    df['vol_ratio'] = df['vol_20'] / (df['vol_60'] + 1e-8)

    # ------ 成交量异常 ------
    if 'volume' in df.columns:
        vol_mean = df['volume'].rolling(20).mean()
        vol_std = df['volume'].rolling(20).std()
        df['volu_z_20'] = (df['volume'] - vol_mean) / (vol_std + 1e-8)
    else:
        df['volu_z_20'] = 0.0

    # ------ 均值偏离（Z-score）------
    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    df['z_price_20'] = (df['close'] - sma_20) / (std_20 + 1e-8)

    # ------ RSI ------
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi_14'] = 100 - 100 / (1 + rs)

    # ------ 布林带位置和宽度 ------
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_width = bb_upper - bb_lower
    df['bb_pos'] = (df['close'] - bb_lower) / (bb_width + 1e-8)
    df['bb_width'] = bb_width / (df['close'] + 1e-8)

    # ------ 趋势强度：EMA差 ------
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['trend_12_26'] = (ema_12 - ema_26) / (df['close'] + 1e-8)

    # ------ MACD柱 ------
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd_line - signal_line

    # ------ ADX近似（用DI+/DI-简化）------
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr_14 = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / (atr_14 + 1e-8)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr_14 + 1e-8)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    df['adx_14'] = dx.rolling(14).mean()
    df['atr_14'] = atr_14

    # ------ 日内振幅与开收收益 ------
    df['hl_range'] = (high - low) / (close + 1e-8)
    df['co_return'] = (close - df['open']) / (df['open'] + 1e-8)

    # ------ 拐点特征（差分）------
    for col in ['rsi_14', 'z_price_20', 'bb_pos', 'trend_12_26', 'vol_ratio', 'co_return']:
        if col in df.columns:
            df[f'd1_{col}'] = df[col].diff()
            df[f'd3_{col}'] = df[col].diff(3)

    # ------ 均线位置 ------
    sma_50 = df['close'].rolling(50).mean()
    sma_200 = df['close'].rolling(200).mean()
    df['price_vs_sma50'] = (df['close'] - sma_50) / (sma_50 + 1e-8)
    df['price_vs_sma200'] = (df['close'] - sma_200) / (sma_200 + 1e-8)
    df['sma50_vs_sma200'] = (sma_50 - sma_200) / (sma_200 + 1e-8)

    return df


def _make_sample_weight(y_values, half_life, tail_weight, tail_q):
    """时间衰减 × 极端收益加权"""
    n = len(y_values)
    # 时间衰减
    age = np.arange(n)[::-1]
    w_time = 0.5 ** (age / half_life)
    w_time = w_time / w_time.mean()

    # 极端收益加权
    abs_y = np.abs(y_values)
    finite = np.isfinite(abs_y)
    if finite.any():
        thr = np.percentile(abs_y[finite], tail_q * 100)  # percentile使用0-100，quantile使用0-1
    else:
        thr = 0.0
    w_tail = np.where(abs_y >= thr, tail_weight, 1.0)

    return w_time * w_tail


# ========================================
# 连续仓位映射与平滑
# ========================================
def sigmoid_map(score, scale=8.0, center=0.0):
    """
    将原始ML得分映射到 [0,1]
    score > 0 → 偏向1（看多）
    score < 0 → 偏向0（看空）
    """
    x = (score - center) * scale
    # 防溢出
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))


def smooth_position(target_pct, current_pct, alpha=0.3, max_change=0.25):
    """
    EMA平滑 + 最大变化限幅
    避免仓位剧烈跳变
    """
    # EMA平滑
    smoothed = alpha * target_pct + (1 - alpha) * current_pct

    # 限幅
    change = smoothed - current_pct
    if abs(change) > max_change:
        smoothed = current_pct + np.sign(change) * max_change

    return np.clip(smoothed, 0.0, 1.0)


# ========================================
# 危机过滤器
# ========================================
def calc_crisis_factor(context):
    """
    危机因子：当市场处于高波动+下行趋势时衰减仓位
    返回 [CRISIS_DAMPING, 1.0]
    """
    try:
        df = get_price(
            BENCHMARK,
            end_date=context.current_dt,
            count=70,
            frequency='1d',
            fields=['close'],
            panel=False
        )
    except:
        return 1.0

    if df is None or df.empty or len(df) < 65:
        return 1.0

    closes = df['close'].values
    ret_1d = np.diff(closes) / closes[:-1]

    # 短期波动率 vs 长期波动率
    vol_20 = np.std(ret_1d[-20:])
    vol_60 = np.std(ret_1d[-60:])
    vol_ratio = vol_20 / (vol_60 + 1e-8)

    # 趋势：EMA12 vs EMA26
    def _ema(arr, span):
        alpha = 2.0 / (span + 1)
        result = np.zeros_like(arr, dtype=float)
        result[0] = arr[0]
        for i in range(1, len(arr)):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
        return result

    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    trend = (ema12[-1] - ema26[-1]) / (closes[-1] + 1e-8)

    # 判定危机
    is_crisis = (vol_ratio > CRISIS_VOL_RATIO_THRESH) and (trend < CRISIS_TREND_THRESH)

    if is_crisis:
        # 连续衰减：危机越严重衰减越多
        severity = min((vol_ratio - CRISIS_VOL_RATIO_THRESH) / 0.5, 1.0)  # 归一化到[0,1]
        factor = 1.0 - severity * (1.0 - CRISIS_DAMPING)
        return max(factor, CRISIS_DAMPING)
    else:
        return 1.0


# ========================================
# ATR止损
# ========================================
def check_atr_stop(context):
    """
    组合层面ATR止损：
    若组合净值从高点回落超过 ATR_STOP_MULTIPLIER × 组合ATR → 触发止损
    """
    current_value = context.portfolio.total_value
    if current_value > g.highest_portfolio_value:
        g.highest_portfolio_value = current_value

    if g.highest_portfolio_value <= 0:
        return False

    drawdown = (g.highest_portfolio_value - current_value) / g.highest_portfolio_value

    # 简化ATR：用最近20日组合波动率近似
    try:
        df = get_price(
            BENCHMARK,
            end_date=context.current_dt,
            count=25,
            frequency='1d',
            fields=['close'],
            panel=False
        )
        if df is not None and len(df) >= 20:
            closes = df['close'].values
            ret = np.diff(closes) / closes[:-1]
            daily_vol = np.std(ret[-20:])
            atr_equiv = daily_vol * np.sqrt(5)  # 5日等效ATR
            threshold = ATR_STOP_MULTIPLIER * atr_equiv
            return drawdown > threshold
    except:
        pass

    # 备选：固定阈值
    return drawdown > 0.15


# ========================================
# 多因子复合动量
# ========================================
def calc_composite_momentum(context, universe):
    """
    计算复合动量得分：
    - mom_20: 20日收益率
    - mom_60: 60日收益率
    - roc_10: 10日ROC
    - rsi_14: RSI（标准化）
    - macd_h: MACD柱状图
    - trend:  EMA趋势差

    各因子截面标准化后加权合成
    """
    if not universe:
        return pd.Series(dtype=float)

    need = 70  # 需要至少70根K线
    try:
        df = get_price(
            universe,
            end_date=context.current_dt,
            count=need,
            frequency='1d',
            fields=['close'],
            panel=False
        )
    except Exception as e:
        log.error(f"get_price失败: {e}")
        return pd.Series(dtype=float)

    if df is None or df.empty:
        return pd.Series(dtype=float)

    # 处理时间列：确保time列存在
    if 'time' not in df.columns:
        df = df.reset_index()
        if 'index' in df.columns:
            df = df.rename(columns={'index': 'time'})
    
    df = df.sort_values(['code', 'time'])
    factor_data = {}

    for code, sub in df.groupby('code'):
        closes = sub['close'].values
        if len(closes) < need:
            continue

        # 使用 t-1 收盘价（避免前瞻偏差）
        c = closes[:-1]  # 去掉最新一根
        if len(c) < 62:
            continue

        factors = {}

        # mom_20
        if len(c) >= 21:
            factors['mom_20'] = c[-1] / c[-21] - 1.0

        # mom_60
        if len(c) >= 61:
            factors['mom_60'] = c[-1] / c[-61] - 1.0

        # roc_10
        if len(c) >= 11:
            factors['roc_10'] = (c[-1] - c[-11]) / (c[-11] + 1e-8)

        # rsi_14
        if len(c) >= 15:
            deltas = np.diff(c[-15:])
            gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
            loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
            rs = gain / (loss + 1e-8)
            factors['rsi_14'] = 100 - 100 / (1 + rs)

        # macd_h (MACD柱状图)
        if len(c) >= 35:
            ema12 = _calc_ema(c, 12)
            ema26 = _calc_ema(c, 26)
            macd_line = ema12 - ema26
            signal = _calc_ema(macd_line[-9:], 9) if len(macd_line) >= 9 else macd_line[-1:]
            factors['macd_h'] = macd_line[-1] - signal[-1]

        # trend (EMA12 - EMA26) / price
        if len(c) >= 27:
            ema12 = _calc_ema(c, 12)
            ema26 = _calc_ema(c, 26)
            factors['trend'] = (ema12[-1] - ema26[-1]) / (c[-1] + 1e-8)

        factor_data[code] = factors

    if not factor_data:
        return pd.Series(dtype=float)

    # 构建因子DataFrame
    factor_df = pd.DataFrame(factor_data).T

    # 截面Z-score标准化
    factor_z = (factor_df - factor_df.mean()) / (factor_df.std() + 1e-8)

    # 加权合成
    composite = pd.Series(0.0, index=factor_z.index)
    for factor_name, weight in FACTOR_WEIGHTS.items():
        if factor_name in factor_z.columns:
            composite += weight * factor_z[factor_name].fillna(0)

    return composite


def _calc_ema(arr, span):
    """手动计算EMA"""
    alpha = 2.0 / (span + 1)
    result = np.zeros(len(arr))
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


# ========================================
# 工具函数（复用V1）
# ========================================
def make_weights_equal(target_list, max_weight=0.15):
    """等权配置"""
    if not target_list:
        return {}
    w = 1.0 / len(target_list)
    w = min(w, max_weight)
    return {s: w for s in target_list}


def apply_target_weights(context, target_weights):
    """执行下单"""
    current = list(context.portfolio.positions.keys())
    target_set = set(target_weights.keys())

    # 卖出非目标
    for s in current:
        if s not in target_set:
            order_target(s, 0)

    # 买入/调仓目标
    total_value = context.portfolio.total_value
    for s, w in target_weights.items():
        if not is_tradable(context, s):
            continue

        target_value = total_value * w
        
        # 检查目标金额是否有效
        if not np.isfinite(target_value) or target_value <= 0:
            continue
            
        current_price = attribute_history(s, 1, '1d', ['close'])['close'][0]

        # 检查价格是否有效
        if not np.isfinite(current_price) or current_price <= 0:
            continue

        target_shares = int(target_value / current_price)
        target_shares = (target_shares // 100) * 100

        if target_shares < 100:
            continue

        order_target(s, target_shares)


def is_tradable(context, security):
    """可交易性检查"""
    try:
        df = attribute_history(security, 1, '1d', ['close'], skip_paused=True)
        if df is None or df.empty:
            return False
        return True
    except:
        return False


# ========================================
# V2 使用说明
# ========================================
"""
【V2核心升级 vs V1】

╔═══════════════════╦══════════════════════════╦══════════════════════════════════╗
║ 维度              ║ V1                       ║ V2                               ║
╠═══════════════════╬══════════════════════════╬══════════════════════════════════╣
║ 风控决策          ║ MA硬规则（趋势线穿越）   ║ ML双模型集成（Walk-Forward）     ║
║ 仓位管理          ║ 三档离散（100/50/0%）    ║ 连续Sigmoid映射 [0,1]            ║
║ 仓位平滑          ║ 冷却期 + 确认期          ║ EMA平滑 + 变化率限幅             ║
║ 选股因子          ║ 单因子MOM                ║ 6因子复合（MOM/ROC/RSI/MACD/趋势）║
║ 危机保护          ║ 趋势线穿越              ║ ML + 危机过滤器 + ATR止损          ║
║ 避险方式          ║ 三档切换到防守           ║ 连续调节避险标的权重              ║
╚═══════════════════╩══════════════════════════╩══════════════════════════════════╝

【参数调优建议】

1. ML模型参数：
   - ML_TRAIN_WINDOW: 504(2y) / 756(3y) / 1260(5y) — 越长越稳但响应慢
   - ML_RETRAIN_EVERY: 5 / 10 / 21 — 越小越及时但计算量大
   - ML_HORIZON: 3 / 5 / 10 — 预测周期

2. 仓位映射参数：
   - SIGMOID_SCALE: 5(平缓) / 8(中等) / 12(陡峭)
   - POS_SMOOTH_ALPHA: 0.2(很平滑) / 0.3(适中) / 0.5(快速响应)
   - POS_MAX_CHANGE: 0.15(保守) / 0.25(适中) / 0.4(激进)

3. 危机过滤参数：
   - CRISIS_VOL_RATIO_THRESH: 1.1(敏感) / 1.3(适中) / 1.5(迟钝)
   - CRISIS_DAMPING: 0.3(强衰减) / 0.5(中等) / 0.7(弱衰减)

4. 多因子权重：
   - 可按IC值调整各因子权重
   - 可增减因子（如加入波动率因子、偏度因子）

【预期效果 vs V1】
- MaxDD：通过连续调仓+ML预判，回撤更平滑（预期改善3-8%）
- Sharpe：多因子选股+智能风控，收益更稳（预期提升0.2-0.5）
- 换手率：平滑机制控制换手，降低交易成本
- 适应性：ML自动学习市场状态变化，无需人工调参

【注意事项】
1. 首次运行需等待ML_TRAIN_WINDOW天才能训练模型
2. 确保聚宽环境安装了scikit-learn（一般已预装）
3. 训练日志中监控ML得分分布是否合理（均值接近0）
4. 若模型频繁看空/看多，检查SIGMOID_SCALE是否合理
5. 危机过滤器参数需要根据回测结果微调
"""
