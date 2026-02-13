# -*- coding: utf-8 -*-
"""
V1风控参数扫描版
快速测试不同趋势线和冷却期参数
"""
from jqdata import *
import numpy as np
import pandas as pd

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

# 动量参数（建议先用V0最优参数）
MOM_WINDOW = 60         # 从V0扫描结果中选择
TOPK = 10
REBALANCE_EVERY = 10
MAX_WEIGHT = 0.15

# ========================================
# 风控参数扫描区（每次改这里）
# ========================================
USE_RISK_CONTROL = True

# 趋势线参数（扫描：120/150/200）
TREND_MA = 150                   # 👈 改这个：120/150/200
CONFIRM_MA = 20                  # 👈 改这个：20/50
THRESHOLD_PCT = 0.005            # 👈 改这个：0.002/0.005/0.010

# 冷却期参数（可选扫描）
RISK_OFF_MIN_DAYS = 10           # 👈 改这个：5/10/15
RISK_ON_CONFIRM_DAYS = 3         # 👈 改这个：2/3/5

# 三档仓位（固定）
RISK_ON_MOMENTUM_PCT = 1.0
RISK_NEUTRAL_MOMENTUM_PCT = 0.5
RISK_OFF_MOMENTUM_PCT = 0.0


def initialize(context):
    set_benchmark(BENCHMARK)
    set_option('use_real_price', True)
    log.set_level('order', 'error')
    
    set_commission(PerTrade(buy_cost=0.0003, sell_cost=0.0003, min_cost=5))
    set_slippage(FixedSlippage(0.0002))
    
    g.universe = list(UNIVERSE)
    g.day_count = 0
    g.last_target = []
    
    # 风控状态
    g.risk_mode = 'on'
    g.risk_off_enter_date = None
    g.risk_on_confirm_count = 0
    
    # 统计信息
    g.mode_switches = []  # 记录模式切换
    
    log.info(f"=== V1参数扫描 ===")
    log.info(f"动量: WINDOW={MOM_WINDOW}, TOPK={TOPK}, REBAL={REBALANCE_EVERY}")
    log.info(f"风控: TREND_MA={TREND_MA}, CONFIRM_MA={CONFIRM_MA}, THRESHOLD={THRESHOLD_PCT:.2%}")
    log.info(f"冷却: OFF_MIN={RISK_OFF_MIN_DAYS}, ON_CONFIRM={RISK_ON_CONFIRM_DAYS}")
    
    run_daily(rebalance, time='09:35')


def before_trading_start(context):
    pass


def handle_data(context, data):
    pass


def after_trading_end(context):
    """回测结束统计"""
    if context.current_dt.date() == context.run_params.end_date:
        log.info(f"=== 风控统计 ===")
        log.info(f"模式切换次数: {len(g.mode_switches)}")
        if g.mode_switches:
            for i, (date, from_mode, to_mode) in enumerate(g.mode_switches[:10]):  # 只显示前10次
                log.info(f"  {i+1}. {date.date()}: {from_mode} -> {to_mode}")


def rebalance(context):
    g.day_count += 1
    if g.day_count % REBALANCE_EVERY != 1:
        return
    
    universe = [s for s in g.universe if is_tradable(context, s)]
    if not universe:
        return
    
    # 风控逻辑
    momentum_pct = 1.0
    
    if USE_RISK_CONTROL:
        risk_signal, risk_mode = calc_risk_control(context)
        
        # 记录模式切换
        if risk_mode != g.risk_mode:
            g.mode_switches.append((context.current_dt, g.risk_mode, risk_mode))
            log.info(f"风控切换: {g.risk_mode} -> {risk_mode}")
        
        momentum_pct = get_momentum_allocation(risk_mode)
        g.risk_mode = risk_mode
    
    # 计算动量
    mom = calc_momentum(context, universe, window=MOM_WINDOW)
    if mom.empty:
        return
    
    target = mom.sort_values(ascending=False).head(TOPK).index.tolist()
    
    # 构建权重
    if momentum_pct > 0:
        momentum_weights = make_weights_equal(target, max_weight=MAX_WEIGHT)
        momentum_weights = {s: w * momentum_pct for s, w in momentum_weights.items()}
    else:
        momentum_weights = {}
    
    defensive_pct = 1.0 - momentum_pct
    if defensive_pct > 0 and is_tradable(context, DEFENSIVE):
        final_weights = momentum_weights.copy()
        final_weights[DEFENSIVE] = defensive_pct
    else:
        final_weights = momentum_weights
    
    apply_target_weights(context, final_weights)


def calc_risk_control(context):
    """风控逻辑（同V1_risk_control.py）"""
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
        return False, 'on'
    
    if df is None or df.empty or len(df) < need_bars:
        return False, 'on'
    
    closes = df['close'].values
    current_price = closes[-2]
    ma_trend = np.mean(closes[-TREND_MA-1:-1])
    ma_confirm = np.mean(closes[-CONFIRM_MA-1:-1])
    
    threshold = ma_trend * (1 - THRESHOLD_PCT)
    below_trend = current_price < threshold
    confirm_weak = ma_confirm < ma_trend
    risk_signal = below_trend and confirm_weak
    
    current_mode = g.risk_mode
    target_mode = current_mode
    
    if current_mode == 'on':
        if risk_signal:
            target_mode = 'neutral'
    
    elif current_mode == 'neutral':
        if risk_signal:
            target_mode = 'off'
            g.risk_off_enter_date = context.current_dt
            g.risk_on_confirm_count = 0
        else:
            target_mode = 'on'
            g.risk_on_confirm_count = 0
    
    elif current_mode == 'off':
        days_in_off = (context.current_dt - g.risk_off_enter_date).days
        
        if days_in_off < RISK_OFF_MIN_DAYS:
            target_mode = 'off'
        else:
            if not risk_signal:
                g.risk_on_confirm_count += 1
                if g.risk_on_confirm_count >= RISK_ON_CONFIRM_DAYS:
                    target_mode = 'neutral'
                    g.risk_on_confirm_count = 0
                else:
                    target_mode = 'off'
            else:
                g.risk_on_confirm_count = 0
                target_mode = 'off'
    
    return risk_signal, target_mode


def get_momentum_allocation(risk_mode):
    """动量仓位分配"""
    if risk_mode == 'on':
        return RISK_ON_MOMENTUM_PCT
    elif risk_mode == 'neutral':
        return RISK_NEUTRAL_MOMENTUM_PCT
    else:
        return RISK_OFF_MOMENTUM_PCT


def calc_momentum(context, universe, window=20):
    """动量计算"""
    if not universe:
        return pd.Series(dtype=float)
    
    need = window + 2
    try:
        df = get_price(universe, end_date=context.current_dt, count=need, 
                      frequency='1d', fields=['close'], panel=False)
    except:
        return pd.Series(dtype=float)
    
    if df is None or df.empty:
        return pd.Series(dtype=float)
    
    df = df.sort_values(['code', 'time'])
    out = {}
    for code, sub in df.groupby('code'):
        closes = sub['close'].values
        if len(closes) >= 2:
            c1 = closes[-2]
            c0_idx = min(-2 - window, -len(closes))
            c0 = closes[c0_idx]
            if c0 > 0:
                out[code] = c1 / c0 - 1.0
    
    return pd.Series(out)


def make_weights_equal(target_list, max_weight=0.15):
    if not target_list:
        return {}
    w = 1.0 / len(target_list)
    w = min(w, max_weight)
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
        
        target_shares = int(target_value / current_price)
        target_shares = (target_shares // 100) * 100
        if target_shares < 100:
            continue
        
        order_target(s, target_shares)


def is_tradable(context, security):
    try:
        df = attribute_history(security, 1, '1d', ['close'], skip_paused=True)
        return df is not None and not df.empty
    except:
        return False


# ========================================
# 参数扫描指南
# ========================================
"""
【扫描方案：2步走】

Step 1：趋势线扫描（6组）
─────────────────────────
固定：RISK_OFF_MIN_DAYS=10, RISK_ON_CONFIRM_DAYS=3

组合1：TREND_MA=120, CONFIRM_MA=20, THRESHOLD=0.005
组合2：TREND_MA=150, CONFIRM_MA=20, THRESHOLD=0.005  👈 推荐基准
组合3：TREND_MA=200, CONFIRM_MA=20, THRESHOLD=0.005
组合4：TREND_MA=150, CONFIRM_MA=50, THRESHOLD=0.005
组合5：TREND_MA=150, CONFIRM_MA=20, THRESHOLD=0.010
组合6：TREND_MA=150, CONFIRM_MA=20, THRESHOLD=0.002

Step 2：冷却期扫描（可选，3组）
─────────────────────────
选择Step1最优趋势线参数，然后调整冷却期：

组合A：RISK_OFF_MIN_DAYS=5,  RISK_ON_CONFIRM_DAYS=2
组合B：RISK_OFF_MIN_DAYS=10, RISK_ON_CONFIRM_DAYS=3  👈 推荐基准
组合C：RISK_OFF_MIN_DAYS=15, RISK_ON_CONFIRM_DAYS=5

【记录表格】
─────────────────────────────────────────────────────
参数组 | TREND | CONFIRM | THRESH | CAGR | Sharpe | MaxDD | Switches
─────────────────────────────────────────────────────────────────────
1      | 120   | 20      | 0.5%   | ___% | ___    | ___% | ___次
2      | 150   | 20      | 0.5%   | ___% | ___    | ___% | ___次
3      | 200   | 20      | 0.5%   | ___% | ___    | ___% | ___次
4      | 150   | 50      | 0.5%   | ___% | ___    | ___% | ___次
5      | 150   | 20      | 1.0%   | ___% | ___    | ___% | ___次
6      | 150   | 20      | 0.2%   | ___% | ___    | ___% | ___次
─────────────────────────────────────────────────────────────────────

【关键指标】
1. Sharpe vs V0：期望提升0.3-0.5
2. MaxDD vs V0：期望降低5-10个百分点
3. Switches：控制在5-15次/5年（太多=成本高，太少=不敏感）
4. Calmar（CAGR/MaxDD）：综合性价比

【预期规律】
- TREND_MA越长：反应越慢，切换次数越少，可能错过顶部
- TREND_MA越短：反应越快，但可能误触发
- THRESHOLD越大：更保守，减少误触发
- CONFIRM_MA=50：双重确认，更稳但滞后

【最优组合预判】
TREND_MA=150, CONFIRM_MA=20, THRESHOLD=0.005
+ RISK_OFF_MIN_DAYS=10, RISK_ON_CONFIRM_DAYS=3

这是平衡灵敏度和稳定性的"黄金组合"。

【对比基准】
一定要和V0_param_scan的最优参数对比：
- 如果Sharpe提升 + MaxDD降低 -> 风控有效
- 如果Sharpe降低 + MaxDD降低 -> 过度保守
- 如果Sharpe降低 + MaxDD升高 -> 风控失效，调参

【时间估计】
6组趋势线扫描 + 3组冷却期扫描 = 9次回测
每次3分钟，约30分钟完成。
"""
