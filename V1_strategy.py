# -*- coding: utf-8 -*-
"""
V1版本：三档风控 + 冷却期 + 灵活趋势过滤
基于V0的改进，加入更智能的风险管理
"""
from jqdata import *
import numpy as np
import pandas as pd

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
DEFENSIVE = '511880.XSHG'

# 动量参数
MOM_WINDOW = 20
TOPK = 10
REBALANCE_EVERY = 5
MAX_WEIGHT = 0.15

# ========================================
# 风控参数（V1新增）
# ========================================
USE_RISK_CONTROL = True          # 启用风险控制

# 趋势线参数（扫描范围：120/150/200）
TREND_MA = 150                   # 长期趋势均线
CONFIRM_MA = 20                  # 短期确认均线
THRESHOLD_PCT = 0.005            # 触发阈值（0.5%）

# 冷却期参数
RISK_OFF_MIN_DAYS = 10           # 触发risk-off后最少持有天数
RISK_ON_CONFIRM_DAYS = 3         # 重新risk-on需连续确认天数

# 三档仓位
RISK_ON_MOMENTUM_PCT = 1.0       # risk-on：100%动量
RISK_NEUTRAL_MOMENTUM_PCT = 0.5  # risk-neutral：50%动量+50%防守
RISK_OFF_MOMENTUM_PCT = 0.0      # risk-off：0%动量（100%防守）


def initialize(context):
    set_benchmark(BENCHMARK)
    set_option('use_real_price', True)
    log.set_level('order', 'error')
    
    set_commission(PerTrade(buy_cost=0.0003, sell_cost=0.0003, min_cost=5))
    set_slippage(FixedSlippage(0.0002))
    
    g.universe = list(UNIVERSE)
    g.day_count = 0
    g.last_target = []
    
    # V1新增：风控状态跟踪
    g.risk_mode = 'on'               # 当前风控模式：'on'/'neutral'/'off'
    g.risk_off_enter_date = None     # 进入risk-off的日期
    g.risk_on_confirm_count = 0      # risk-on连续确认计数
    
    log.info(f"=== V1策略初始化 ===")
    log.info(f"动量: WINDOW={MOM_WINDOW}, TOPK={TOPK}, REBAL={REBALANCE_EVERY}")
    if USE_RISK_CONTROL:
        log.info(f"风控: TREND_MA={TREND_MA}, CONFIRM_MA={CONFIRM_MA}, THRESHOLD={THRESHOLD_PCT:.2%}")
        log.info(f"冷却: OFF_MIN={RISK_OFF_MIN_DAYS}天, ON_CONFIRM={RISK_ON_CONFIRM_DAYS}天")
    
    run_daily(rebalance, time='09:35')


def before_trading_start(context):
    pass


def handle_data(context, data):
    pass


def rebalance(context):
    g.day_count += 1
    if g.day_count % REBALANCE_EVERY != 1:
        return
    
    universe = [s for s in g.universe if is_tradable(context, s)]
    if not universe:
        log.warn(f"No tradable securities. Skip.")
        return
    
    # ========================================
    # V1核心：三档风控逻辑
    # ========================================
    momentum_pct = 1.0  # 默认100%动量仓位
    
    if USE_RISK_CONTROL:
        risk_signal, risk_mode = calc_risk_control(context)
        momentum_pct = get_momentum_allocation(risk_mode)
        
        log.info(f"风控状态: {g.risk_mode} -> {risk_mode} | 动量仓位: {momentum_pct:.0%}")
        g.risk_mode = risk_mode
    
    # ========================================
    # 计算动量并选股
    # ========================================
    mom = calc_momentum(context, universe, window=MOM_WINDOW)
    if mom.empty:
        log.info("No momentum data; skip rebalance.")
        return
    
    target = mom.sort_values(ascending=False).head(TOPK).index.tolist()
    
    # ========================================
    # 构建目标权重（考虑风控仓位）
    # ========================================
    if momentum_pct > 0:
        # 有动量仓位：分配给TopK
        momentum_weights = make_weights_equal(target, max_weight=MAX_WEIGHT)
        # 按比例缩放
        momentum_weights = {s: w * momentum_pct for s, w in momentum_weights.items()}
    else:
        momentum_weights = {}
    
    # 防守资产权重
    defensive_pct = 1.0 - momentum_pct
    if defensive_pct > 0 and is_tradable(context, DEFENSIVE):
        final_weights = momentum_weights.copy()
        final_weights[DEFENSIVE] = defensive_pct
    else:
        final_weights = momentum_weights
    
    apply_target_weights(context, final_weights)
    g.last_target = target


def calc_risk_control(context):
    """
    V1核心函数：三档风控 + 冷却期
    
    返回：
        risk_signal: bool，是否触发风险信号
        risk_mode: str，目标风控模式 ('on'/'neutral'/'off')
    """
    # 1. 获取市场数据
    need_bars = max(TREND_MA, CONFIRM_MA) + 10
    try:
        df = get_price(
            BENCHMARK, 
            end_date=context.current_dt, 
            count=need_bars, 
            frequency='1d', 
            fields=['close'], 
            panel=False
        )
    except Exception as e:
        log.warn(f"获取基准数据失败: {e}")
        return False, 'on'
    
    if df is None or df.empty or len(df) < need_bars:
        return False, 'on'
    
    closes = df['close'].values
    
    # 2. 计算趋势指标
    current_price = closes[-2]  # 用昨天收盘（避免未来函数）
    ma_trend = np.mean(closes[-TREND_MA-1:-1])  # 长期趋势线
    ma_confirm = np.mean(closes[-CONFIRM_MA-1:-1])  # 短期确认线
    
    # 3. 判断风险信号
    # 条件1：价格跌破趋势线（带阈值）
    threshold = ma_trend * (1 - THRESHOLD_PCT)
    below_trend = current_price < threshold
    
    # 条件2：短期确认线也跌破（可选，更保守）
    confirm_weak = ma_confirm < ma_trend
    
    # 综合判断
    risk_signal = below_trend and confirm_weak
    
    # 4. 状态机逻辑（考虑冷却期）
    current_mode = g.risk_mode
    target_mode = current_mode  # 默认保持当前状态
    
    if current_mode == 'on':
        # 从risk-on状态
        if risk_signal:
            # 触发风险 -> 先切换到neutral（缓冲）
            target_mode = 'neutral'
            log.info(f"⚠️ 触发风险信号: 价格={current_price:.2f}, MA{TREND_MA}={ma_trend:.2f}")
    
    elif current_mode == 'neutral':
        # 从neutral状态
        if risk_signal:
            # 风险持续 -> 切换到risk-off
            target_mode = 'off'
            g.risk_off_enter_date = context.current_dt
            g.risk_on_confirm_count = 0
            log.info(f"🛑 进入防守模式")
        else:
            # 风险解除 -> 回到risk-on
            target_mode = 'on'
            g.risk_on_confirm_count = 0
            log.info(f"✅ 风险解除，恢复动量")
    
    elif current_mode == 'off':
        # 从risk-off状态（需要冷却期和确认期）
        days_in_off = (context.current_dt - g.risk_off_enter_date).days
        
        if days_in_off < RISK_OFF_MIN_DAYS:
            # 冷却期未满，强制保持risk-off
            target_mode = 'off'
        else:
            # 冷却期已过，检查是否可以恢复
            if not risk_signal:
                # 风险解除
                g.risk_on_confirm_count += 1
                if g.risk_on_confirm_count >= RISK_ON_CONFIRM_DAYS:
                    # 连续确认足够天数 -> 恢复到neutral
                    target_mode = 'neutral'
                    g.risk_on_confirm_count = 0
                    log.info(f"🔄 风险解除确认，切换到中性仓位 (确认{g.risk_on_confirm_count}天)")
                else:
                    target_mode = 'off'
                    log.info(f"⏳ 风险解除确认中... ({g.risk_on_confirm_count}/{RISK_ON_CONFIRM_DAYS}天)")
            else:
                # 风险仍在，重置确认计数
                g.risk_on_confirm_count = 0
                target_mode = 'off'
    
    return risk_signal, target_mode


def get_momentum_allocation(risk_mode):
    """根据风控模式返回动量仓位比例"""
    if risk_mode == 'on':
        return RISK_ON_MOMENTUM_PCT
    elif risk_mode == 'neutral':
        return RISK_NEUTRAL_MOMENTUM_PCT
    else:  # 'off'
        return RISK_OFF_MOMENTUM_PCT


def calc_momentum(context, universe, window=20):
    """计算动量（与V0相同）"""
    if not universe:
        return pd.Series(dtype=float)
    
    need = window + 2
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
        log.error(f"get_price failed: {str(e)}")
        return pd.Series(dtype=float)
    
    if df is None or df.empty:
        return pd.Series(dtype=float)
    
    df = df.sort_values(['code', 'time'])
    out = {}
    for code, sub in df.groupby('code'):
        closes = sub['close'].values
        if len(closes) < need:
            continue
        if len(closes) >= 2:
            c1 = closes[-2]
            c0_idx = min(-2 - window, -len(closes))
            c0 = closes[c0_idx]
            if c0 > 0:
                out[code] = c1 / c0 - 1.0
    
    return pd.Series(out)


def make_weights_equal(target_list, max_weight=0.15):
    """等权配置"""
    if not target_list:
        return {}
    w = 1.0 / len(target_list)
    w = min(w, max_weight)
    return {s: w for s in target_list}


def apply_target_weights(context, target_weights):
    """执行下单"""
    # 卖出非目标
    current = list(context.portfolio.positions.keys())
    target_set = set(target_weights.keys())
    
    for s in current:
        if s not in target_set:
            order_target(s, 0)
    
    # 买入/调仓目标
    total_value = context.portfolio.total_value
    for s, w in target_weights.items():
        if not is_tradable(context, s):
            continue
        
        target_value = total_value * w
        current_price = attribute_history(s, 1, '1d', ['close'])['close'][0]
        
        if current_price <= 0:
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
    except Exception as e:
        return False


# ========================================
# V1 使用说明
# ========================================
"""
【V1核心改进】

1. 三档仓位（替代二元开关）
   - risk-on：100%动量组合
   - risk-neutral：50%动量 + 50%防守（缓冲状态）
   - risk-off：100%防守/现金（完全防守）

2. 冷却期机制（避免频繁切换）
   - 进入risk-off后，最少持有RISK_OFF_MIN_DAYS天
   - 重新恢复需连续RISK_ON_CONFIRM_DAYS天确认
   - 有效降低震荡市的反复切换成本

3. 灵活趋势过滤器
   - 长期趋势线：MA(TREND_MA)
   - 短期确认线：MA(CONFIRM_MA)
   - 触发阈值：避免贴线抖动
   
4. 状态机设计
   on -> (触发风险) -> neutral -> (风险持续) -> off
   off -> (冷却期满 + 连续确认) -> neutral -> (风险解除) -> on

【参数扫描建议】（6组）

趋势线扫描：
1. TREND_MA=120, CONFIRM_MA=20, THRESHOLD_PCT=0.005
2. TREND_MA=150, CONFIRM_MA=20, THRESHOLD_PCT=0.005
3. TREND_MA=200, CONFIRM_MA=20, THRESHOLD_PCT=0.005

确认线变化：
4. TREND_MA=150, CONFIRM_MA=50, THRESHOLD_PCT=0.005
5. TREND_MA=150, CONFIRM_MA=20, THRESHOLD_PCT=0.010  (阈值加倍)
6. TREND_MA=150, CONFIRM_MA=20, THRESHOLD_PCT=0.002  (阈值减半)

冷却期调整（可选）：
- RISK_OFF_MIN_DAYS: 5 / 10 / 15天
- RISK_ON_CONFIRM_DAYS: 2 / 3 / 5天

【对比指标】
与V0对比，重点关注：
- MaxDD（最大回撤）：期望降低5-10个百分点
- Sharpe：期望提升0.3-0.5
- Calmar比率（年化收益/最大回撤）：更优
- 切换次数：控制在合理范围（<10次/年）

【预期效果】
- 2015年股灾：提前切换到防守，避开大跌
- 2020年疫情：快速进入防守，后续恢复
- 震荡市：neutral状态提供缓冲，减少损失
- 牛市：及时恢复到full动量，不错过趋势

【实盘注意】
1. 首次运行建议保守参数（MA200 + 长冷却期）
2. 监控状态切换日志，检查是否合理
3. 记录每次切换的时机和后续表现
4. 1-2个季度后根据实际效果调参
"""
