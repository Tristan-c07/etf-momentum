# -*- coding: utf-8 -*-
"""
V2风控参数扫描版
快速测试ML风控 + 多因子动量的各种参数组合

扫描维度：
1. SIGMOID_SCALE: 5 / 8 / 12    （仓位映射陡峭程度）
2. POS_SMOOTH_ALPHA: 0.2 / 0.3 / 0.5 （仓位平滑速度）
3. CRISIS_VOL_RATIO_THRESH: 1.1 / 1.3 / 1.5 （危机敏感度）
4. ML_TRAIN_WINDOW: 504 / 756    （ML训练窗口）
5. ML_RETRAIN_EVERY: 5 / 10      （ML重训练频率）

使用方法：在聚宽中复制本文件，修改下方"扫描区"参数运行回测
"""
from jqdata import *
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# ========================================
# 固定参数
# ========================================
UNIVERSE = [
    '510300.XSHG', '510050.XSHG', '510500.XSHG', '159919.XSHE', '159915.XSHE',
    '588000.XSHG', '588080.XSHG', '512100.XSHG', '512010.XSHG', '512760.XSHG',
    '512480.XSHG', '512800.XSHG', '512660.XSHG', '516160.XSHG', '515790.XSHG',
    '159941.XSHE', '518880.XSHG', '513050.XSHG', '513100.XSHG', '511880.XSHG',
]

BENCHMARK = '000300.XSHG'
DEFENSIVE = '511880.XSHG'

# 选股参数（固定）
TOPK = 5
REBALANCE_EVERY = 10
MAX_WEIGHT = 0.15

# 多因子权重（固定）
FACTOR_WEIGHTS = {
    'mom_20': 0.25, 'mom_60': 0.20, 'roc_10': 0.15,
    'rsi_14': 0.15, 'macd_h': 0.15, 'trend': 0.10,
}

# ========================================
# 👇 扫描区：每次改这里
# ========================================
USE_ML_RISK = True

# ML参数
ML_TRAIN_WINDOW = 756            # 👈 504 / 756
ML_RETRAIN_EVERY = 10            # 👈 5 / 10
ML_HORIZON = 5                   # 固定5日
ML_HALF_LIFE = 126
ML_TAIL_WEIGHT = 3.0
ML_TAIL_QUANTILE = 0.85

# 集成权重
ENSEMBLE_LIN_WEIGHT = 0.45
ENSEMBLE_TREE_WEIGHT = 0.55

# 仓位映射
SIGMOID_SCALE = 8.0              # 👈 5 / 8 / 12
SIGMOID_CENTER = 0.0
POS_SMOOTH_ALPHA = 0.3           # 👈 0.2 / 0.3 / 0.5
POS_MAX_CHANGE = 0.25            # 👈 0.15 / 0.25 / 0.40

MIN_MOMENTUM_PCT = 0.0
MAX_MOMENTUM_PCT = 1.0

# 危机过滤
CRISIS_VOL_RATIO_THRESH = 1.3    # 👈 1.1 / 1.3 / 1.5
CRISIS_TREND_THRESH = -0.01
CRISIS_DAMPING = 0.5             # 👈 0.3 / 0.5 / 0.7

# ATR止损
USE_ATR_STOP = True
ATR_STOP_MULTIPLIER = 3.0


# ========================================
# 建议扫描组合（12组）
# ========================================
"""
基础组（固定ML_TRAIN=756, ML_RETRAIN=10）：
1. SIGMOID=5,  SMOOTH=0.3, CRISIS=1.3  — 平缓映射
2. SIGMOID=8,  SMOOTH=0.3, CRISIS=1.3  — 基准
3. SIGMOID=12, SMOOTH=0.3, CRISIS=1.3  — 陡峭映射

平滑度扫描：
4. SIGMOID=8, SMOOTH=0.2, CRISIS=1.3   — 慢平滑
5. SIGMOID=8, SMOOTH=0.5, CRISIS=1.3   — 快响应

危机敏感度：
6. SIGMOID=8, SMOOTH=0.3, CRISIS=1.1   — 敏感
7. SIGMOID=8, SMOOTH=0.3, CRISIS=1.5   — 迟钝

ML窗口：
8. ML_TRAIN=504, SIGMOID=8, SMOOTH=0.3 — 短窗口
9. ML_TRAIN=756, ML_RETRAIN=5          — 高频重训练

危机衰减：
10. CRISIS_DAMPING=0.3                  — 强衰减
11. CRISIS_DAMPING=0.7                  — 弱衰减

变化限幅：
12. POS_MAX_CHANGE=0.15                 — 保守调仓
"""


def initialize(context):
    set_benchmark(BENCHMARK)
    set_option('use_real_price', True)
    log.set_level('order', 'error')

    set_commission(PerTrade(buy_cost=0.0003, sell_cost=0.0003, min_cost=5))
    set_slippage(FixedSlippage(0.0002))

    g.universe = list(UNIVERSE)
    g.day_count = 0
    g.last_target = []

    g.model_lin = None
    g.model_tree = None
    g.last_train_day = -999
    g.market_score = 0.0
    g.current_mom_pct = 0.5

    g.highest_portfolio_value = 0.0
    g.train_count = 0
    g.score_history = []
    g.pct_history = []

    log.info(f"=== V2扫描 ===")
    log.info(f"ML: TRAIN={ML_TRAIN_WINDOW}, RETRAIN={ML_RETRAIN_EVERY}")
    log.info(f"映射: SIGMOID={SIGMOID_SCALE}, SMOOTH={POS_SMOOTH_ALPHA}, MAX_CHG={POS_MAX_CHANGE}")
    log.info(f"危机: VOL_RATIO>{CRISIS_VOL_RATIO_THRESH}, DAMPING={CRISIS_DAMPING}")

    run_daily(rebalance, time='09:35')


def before_trading_start(context):
    pass


def handle_data(context, data):
    pass


def after_trading_end(context):
    """回测结束统计"""
    if context.current_dt.date() == context.run_params.end_date:
        log.info(f"=== V2扫描统计 ===")
        log.info(f"ML训练次数: {g.train_count}")
        if g.score_history:
            scores = [s[1] for s in g.score_history]
            pcts = [s[2] for s in g.score_history]
            log.info(f"ML得分: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")
            log.info(f"动量仓位: mean={np.mean(pcts):.2%}, min={np.min(pcts):.2%}, max={np.max(pcts):.2%}")


def rebalance(context):
    g.day_count += 1
    if g.day_count % REBALANCE_EVERY != 1:
        return

    universe = [s for s in g.universe if is_tradable(context, s)]
    if not universe:
        return

    momentum_pct = 0.5

    if USE_ML_RISK:
        market_score = ml_predict_market(context)
        g.market_score = market_score

        raw_pct = sigmoid_map(market_score, scale=SIGMOID_SCALE, center=SIGMOID_CENTER)
        crisis_factor = calc_crisis_factor(context)
        raw_pct = raw_pct * crisis_factor
        raw_pct = np.clip(raw_pct, MIN_MOMENTUM_PCT, MAX_MOMENTUM_PCT)

        momentum_pct = smooth_position(raw_pct, g.current_mom_pct,
                                        alpha=POS_SMOOTH_ALPHA,
                                        max_change=POS_MAX_CHANGE)
        g.current_mom_pct = momentum_pct
        g.score_history.append((context.current_dt, market_score, momentum_pct, crisis_factor))

    composite_mom = calc_composite_momentum(context, universe)
    if composite_mom.empty:
        return

    target = composite_mom.sort_values(ascending=False).head(TOPK).index.tolist()

    if momentum_pct > 0.01:
        momentum_weights = make_weights_equal(target, max_weight=MAX_WEIGHT)
        momentum_weights = {s: w * momentum_pct for s, w in momentum_weights.items()}
    else:
        momentum_weights = {}

    defensive_pct = 1.0 - momentum_pct
    final_weights = momentum_weights.copy()
    if defensive_pct > 0.01 and is_tradable(context, DEFENSIVE):
        final_weights[DEFENSIVE] = defensive_pct

    if USE_ATR_STOP:
        if check_atr_stop(context):
            final_weights = {}
            if is_tradable(context, DEFENSIVE):
                final_weights[DEFENSIVE] = 1.0

    apply_target_weights(context, final_weights)
    g.last_target = target


# ====================================================================
# 以下函数与V2_strategy.py完全相同，为扫描版自包含
# ====================================================================

def ml_predict_market(context):
    need_bars = ML_TRAIN_WINDOW + 100
    try:
        df = get_price(BENCHMARK, end_date=context.current_dt, count=need_bars,
                       frequency='1d', fields=['open', 'high', 'low', 'close', 'volume'], panel=False)
    except Exception as e:
        return 0.0

    if df is None or df.empty or len(df) < ML_TRAIN_WINDOW + 50:
        return 0.0

    df = df.sort_values('time').reset_index(drop=True)
    df = df.rename(columns={'time': 'date'}).set_index('date')

    df_feat = make_ml_features(df)
    df_feat['target'] = df_feat['close'].pct_change(ML_HORIZON).shift(-ML_HORIZON)
    df_feat = df_feat.dropna(subset=['target'])

    drop_cols = {'open', 'high', 'low', 'close', 'volume', 'target'}
    feature_cols = [c for c in df_feat.columns if c not in drop_cols
                    and df_feat[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    X = df_feat[feature_cols]
    y = df_feat['target']

    if len(X) < ML_TRAIN_WINDOW + 10:
        return 0.0

    should_retrain = (g.model_lin is None or g.day_count - g.last_train_day >= ML_RETRAIN_EVERY)

    if should_retrain:
        X_train = X.iloc[-ML_TRAIN_WINDOW - ML_HORIZON:-ML_HORIZON]
        y_train = y.iloc[-ML_TRAIN_WINDOW - ML_HORIZON:-ML_HORIZON]

        if len(X_train) < 100:
            return 0.0

        sw = _make_sample_weight(y_train.values, ML_HALF_LIFE, ML_TAIL_WEIGHT, ML_TAIL_QUANTILE)

        try:
            g.model_lin = Pipeline([
                ('imp', SimpleImputer(strategy='median')),
                ('sc', StandardScaler()),
                ('mdl', HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=600))
            ])
            g.model_lin.fit(X_train, y_train, mdl__sample_weight=sw)

            g.model_tree = Pipeline([
                ('imp', SimpleImputer(strategy='median')),
                ('mdl', HistGradientBoostingRegressor(
                    max_depth=3, learning_rate=0.05, max_iter=250,
                    l2_regularization=1e-3, random_state=42
                ))
            ])
            g.model_tree.fit(X_train, y_train, mdl__sample_weight=sw)

            g.last_train_day = g.day_count
            g.train_count += 1
        except:
            return 0.0

    try:
        X_latest = X.iloc[[-1]]
        p_lin = g.model_lin.predict(X_latest)[0]
        p_tree = g.model_tree.predict(X_latest)[0]
        return float(ENSEMBLE_LIN_WEIGHT * p_lin + ENSEMBLE_TREE_WEIGHT * p_tree)
    except:
        return 0.0


def make_ml_features(df):
    df = df.copy().sort_index()

    df['ret_1'] = df['close'].pct_change()
    for n in [2, 3, 5, 10, 20, 60, 120]:
        df[f'ret_{n}'] = df['close'].pct_change(n)

    df['vol_20'] = df['ret_1'].rolling(20).std()
    df['vol_60'] = df['ret_1'].rolling(60).std()
    df['vol_ratio'] = df['vol_20'] / (df['vol_60'] + 1e-8)

    if 'volume' in df.columns:
        vol_mean = df['volume'].rolling(20).mean()
        vol_std = df['volume'].rolling(20).std()
        df['volu_z_20'] = (df['volume'] - vol_mean) / (vol_std + 1e-8)
    else:
        df['volu_z_20'] = 0.0

    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    df['z_price_20'] = (df['close'] - sma_20) / (std_20 + 1e-8)

    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi_14'] = 100 - 100 / (1 + rs)

    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_width = bb_upper - bb_lower
    df['bb_pos'] = (df['close'] - bb_lower) / (bb_width + 1e-8)
    df['bb_width'] = bb_width / (df['close'] + 1e-8)

    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['trend_12_26'] = (ema_12 - ema_26) / (df['close'] + 1e-8)

    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd_line - signal_line

    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / (atr_14 + 1e-8)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr_14 + 1e-8)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    df['adx_14'] = dx.rolling(14).mean()
    df['atr_14'] = atr_14

    df['hl_range'] = (high - low) / (close + 1e-8)
    df['co_return'] = (close - df['open']) / (df['open'] + 1e-8)

    for col in ['rsi_14', 'z_price_20', 'bb_pos', 'trend_12_26', 'vol_ratio', 'co_return']:
        if col in df.columns:
            df[f'd1_{col}'] = df[col].diff()
            df[f'd3_{col}'] = df[col].diff(3)

    sma_50 = df['close'].rolling(50).mean()
    sma_200 = df['close'].rolling(200).mean()
    df['price_vs_sma50'] = (df['close'] - sma_50) / (sma_50 + 1e-8)
    df['price_vs_sma200'] = (df['close'] - sma_200) / (sma_200 + 1e-8)
    df['sma50_vs_sma200'] = (sma_50 - sma_200) / (sma_200 + 1e-8)

    return df


def _make_sample_weight(y_values, half_life, tail_weight, tail_q):
    n = len(y_values)
    age = np.arange(n)[::-1]
    w_time = 0.5 ** (age / half_life)
    w_time = w_time / w_time.mean()
    abs_y = np.abs(y_values)
    finite = np.isfinite(abs_y)
    thr = np.quantile(abs_y[finite], tail_q) if finite.any() else 0.0
    w_tail = np.where(abs_y >= thr, tail_weight, 1.0)
    return w_time * w_tail


def sigmoid_map(score, scale=8.0, center=0.0):
    x = np.clip((score - center) * scale, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))


def smooth_position(target_pct, current_pct, alpha=0.3, max_change=0.25):
    smoothed = alpha * target_pct + (1 - alpha) * current_pct
    change = smoothed - current_pct
    if abs(change) > max_change:
        smoothed = current_pct + np.sign(change) * max_change
    return np.clip(smoothed, 0.0, 1.0)


def calc_crisis_factor(context):
    try:
        df = get_price(BENCHMARK, end_date=context.current_dt, count=70,
                       frequency='1d', fields=['close'], panel=False)
    except:
        return 1.0

    if df is None or df.empty or len(df) < 65:
        return 1.0

    closes = df['close'].values
    ret_1d = np.diff(closes) / closes[:-1]

    vol_20 = np.std(ret_1d[-20:])
    vol_60 = np.std(ret_1d[-60:])
    vol_ratio = vol_20 / (vol_60 + 1e-8)

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

    is_crisis = (vol_ratio > CRISIS_VOL_RATIO_THRESH) and (trend < CRISIS_TREND_THRESH)

    if is_crisis:
        severity = min((vol_ratio - CRISIS_VOL_RATIO_THRESH) / 0.5, 1.0)
        factor = 1.0 - severity * (1.0 - CRISIS_DAMPING)
        return max(factor, CRISIS_DAMPING)
    else:
        return 1.0


def check_atr_stop(context):
    current_value = context.portfolio.total_value
    if current_value > g.highest_portfolio_value:
        g.highest_portfolio_value = current_value
    if g.highest_portfolio_value <= 0:
        return False
    drawdown = (g.highest_portfolio_value - current_value) / g.highest_portfolio_value
    try:
        df = get_price(BENCHMARK, end_date=context.current_dt, count=25,
                       frequency='1d', fields=['close'], panel=False)
        if df is not None and len(df) >= 20:
            closes = df['close'].values
            ret = np.diff(closes) / closes[:-1]
            daily_vol = np.std(ret[-20:])
            atr_equiv = daily_vol * np.sqrt(5)
            threshold = ATR_STOP_MULTIPLIER * atr_equiv
            return drawdown > threshold
    except:
        pass
    return drawdown > 0.15


def calc_composite_momentum(context, universe):
    if not universe:
        return pd.Series(dtype=float)

    need = 70
    try:
        df = get_price(universe, end_date=context.current_dt, count=need,
                       frequency='1d', fields=['close'], panel=False)
    except:
        return pd.Series(dtype=float)

    if df is None or df.empty:
        return pd.Series(dtype=float)

    df = df.sort_values(['code', 'time'])
    factor_data = {}

    for code, sub in df.groupby('code'):
        closes = sub['close'].values
        if len(closes) < need:
            continue
        c = closes[:-1]
        if len(c) < 62:
            continue

        factors = {}
        if len(c) >= 21: factors['mom_20'] = c[-1] / c[-21] - 1.0
        if len(c) >= 61: factors['mom_60'] = c[-1] / c[-61] - 1.0
        if len(c) >= 11: factors['roc_10'] = (c[-1] - c[-11]) / (c[-11] + 1e-8)
        if len(c) >= 15:
            deltas = np.diff(c[-15:])
            gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
            loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
            factors['rsi_14'] = 100 - 100 / (1 + gain / (loss + 1e-8))
        if len(c) >= 35:
            ema12 = _calc_ema(c, 12)
            ema26 = _calc_ema(c, 26)
            macd_line = ema12 - ema26
            signal = _calc_ema(macd_line[-9:], 9) if len(macd_line) >= 9 else macd_line[-1:]
            factors['macd_h'] = macd_line[-1] - signal[-1]
        if len(c) >= 27:
            ema12 = _calc_ema(c, 12)
            ema26 = _calc_ema(c, 26)
            factors['trend'] = (ema12[-1] - ema26[-1]) / (c[-1] + 1e-8)

        factor_data[code] = factors

    if not factor_data:
        return pd.Series(dtype=float)

    factor_df = pd.DataFrame(factor_data).T
    factor_z = (factor_df - factor_df.mean()) / (factor_df.std() + 1e-8)

    composite = pd.Series(0.0, index=factor_z.index)
    for factor_name, weight in FACTOR_WEIGHTS.items():
        if factor_name in factor_z.columns:
            composite += weight * factor_z[factor_name].fillna(0)

    return composite


def _calc_ema(arr, span):
    alpha = 2.0 / (span + 1)
    result = np.zeros(len(arr))
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def make_weights_equal(target_list, max_weight=0.15):
    if not target_list:
        return {}
    w = min(1.0 / len(target_list), max_weight)
    return {s: w for s in target_list}


def apply_target_weights(context, target_weights):
    current = list(context.portfolio.positions.keys())
    target_set = set(target_weights.keys())
    for s in current:
        if s not in target_set:
            order_target(s, 0)
    total_value = context.portfolio.total_value
    for s, w in target_weights.items():
        if not is_tradable(context, s):
            continue
        target_value = total_value * w
        current_price = attribute_history(s, 1, '1d', ['close'])['close'][0]
        if current_price <= 0:
            continue
        target_shares = (int(target_value / current_price) // 100) * 100
        if target_shares < 100:
            continue
        order_target(s, target_shares)


def is_tradable(context, security):
    try:
        df = attribute_history(security, 1, '1d', ['close'], skip_paused=True)
        return df is not None and not df.empty
    except:
        return False
