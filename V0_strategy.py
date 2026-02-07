# -*- coding: utf-8 -*-
from jqdata import *
import numpy as np
import pandas as pd

# =========================
# 参数区（你最常改的都在这）
# =========================
UNIVERSE = [
    '510300.XSHG', '510050.XSHG', '510500.XSHG', '159919.XSHE', '159915.XSHE',
    '588000.XSHG', '588080.XSHG', '512100.XSHG', '512010.XSHG', '512760.XSHG',
    '512480.XSHG', '512800.XSHG', '512660.XSHG', '516160.XSHG', '515790.XSHG',
    '159941.XSHE', '518880.XSHG', '513050.XSHG', '513100.XSHG', '511880.XSHG',
]

BENCHMARK = '000300.XSHG'     # 沪深300指数
DEFENSIVE = '511880.XSHG'     # 防守资产（先占位：货基/短债/现金类ETF，按你平台可交易的替换）

MOM_WINDOW = 20               # 动量窗口：20日
TOPK = 10                     # 选前K
REBALANCE_EVERY = 5           # 每5个交易日调仓（周频）
MAX_WEIGHT = 0.15             # 单票权重上限
COST_BPS = 5.0                # 交易成本（这里只用于回测引擎自带成本时；JQ下单成本靠set_commission更可信）

USE_RISK_OFF = False          # True: 开启风险开关（建议V1再开）
USE_ML_OVERLAY = False        # True: 开启ML overlay（建议V2再开）


# ==============
# 框架函数
# ==============
def initialize(context):
    set_benchmark(BENCHMARK)
    set_option('use_real_price', True)
    log.set_level('order', 'error')

    # 手续费/滑点：按你实际账户调（这里给一个保守默认）
    set_commission(PerTrade(buy_cost=0.0003, sell_cost=0.0003, min_cost=5))
    set_slippage(FixedSlippage(0.0002))

    g.universe = list(UNIVERSE)
    g.day_count = 0
    g.last_target = []

    run_daily(rebalance, time='09:35')  # 每天固定时刻检查是否到调仓日


def before_trading_start(context):
    pass


def handle_data(context, data):
    pass


# ==============
# 核心逻辑
# ==============
def rebalance(context):
    g.day_count += 1
    if g.day_count % REBALANCE_EVERY != 1:
        return  # 只在“周频节点”调仓（第1、6、11...个交易日）

    # 先打印调试信息
    log.info(f"Rebalance on {context.current_dt}, checking {len(g.universe)} securities")
    
    universe = [s for s in g.universe if is_tradable(context, s)]    
    # 检查可交易标的是否为空
    if not universe:
        log.warn(f"No tradable securities on {context.current_dt}. Skip rebalance.")
        log.info(f"Original universe: {g.universe[:5]}...")  # 打印前几个看看
        return
    # 1) 风险开关（可选，默认关闭）
    risk_off = False
    if USE_RISK_OFF:
        risk_off = calc_risk_off(context)

    if risk_off:
        target = [DEFENSIVE] if is_tradable(context, DEFENSIVE) else []
        weights = make_weights_equal(target, max_weight=1.0)
        apply_target_weights(context, weights)
        g.last_target = target
        return

    # 2) 计算动量并选TopK
    mom = calc_momentum(context, universe, window=MOM_WINDOW)
    if mom.empty:
        log.info("No momentum data; skip rebalance.")
        return

    target = mom.sort_values(ascending=False).head(TOPK).index.tolist()

    # 3) ML overlay（可选，默认关闭）：在候选池里二次排序/加权
    weights = make_weights_equal(target, max_weight=MAX_WEIGHT)
    if USE_ML_OVERLAY:
        weights = ml_overlay_weights(context, target, base_weights=weights)

    apply_target_weights(context, weights)
    g.last_target = target


def calc_momentum(context, universe, window=20):
    """
    用 t-1 的数据生成信号：取到window+2天数据，避免对齐问题。
    """
    # 先检查universe是否为空
    if not universe:
        log.warn("Universe is empty in calc_momentum")
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
        log.error(f"get_price failed: {str(e)}. Universe size: {len(universe)}")
        return pd.Series(dtype=float)
    
    if df is None or df.empty:
        log.warn(f"get_price returned empty data for {len(universe)} securities")
        return pd.Series(dtype=float)

    # df: date, code, close
    df = df.sort_values(['code', 'time'])
    out = {}
    for code, sub in df.groupby('code'):
        closes = sub['close'].values
        if len(closes) < need:
            log.debug(f"{code}: insufficient data, got {len(closes)} need {need}")
            continue
        # 用"昨天收盘"形成信号：mom_t = close_{t-1}/close_{t-1-window}-1
        if len(closes) >= 2:  # 确保有至少2个数据点
            c1 = closes[-2]
            c0_idx = min(-2 - window, -len(closes))  # 防止索引越界
            c0 = closes[c0_idx]
            if c0 > 0:
                out[code] = c1 / c0 - 1.0
    
    log.info(f"calc_momentum: processed {len(universe)} securities, got {len(out)} valid signals")
    return pd.Series(out)


def make_weights_equal(target_list, max_weight=0.15):
    if not target_list:
        return {}
    w = 1.0 / len(target_list)
    w = min(w, max_weight)
    # 如果因max_weight导致总和<1，则剩余留现金（更稳）
    return {s: w for s in target_list}


def apply_target_weights(context, target_weights):
    """
    先清仓不在目标里的，再按目标权重下单。
    """
    orders_placed = []
    
    # 先卖出非目标
    current = list(context.portfolio.positions.keys())
    target_set = set(target_weights.keys())
    for s in current:
        if s not in target_set:
            order = order_target(s, 0)
            if order:
                orders_placed.append(f"SELL {s}: 清仓")

    # 再买入/调仓目标
    total_value = context.portfolio.total_value
    for s, w in target_weights.items():
        # 下单前再次检查可交易性
        if not is_tradable(context, s):
            log.warn(f"Skip {s}: not tradable at order time")
            continue
        
        target_value = total_value * w
        current_price = attribute_history(s, 1, '1d', ['close'])['close'][0]
        
        if current_price <= 0:
            log.warn(f"Skip {s}: invalid price {current_price}")
            continue
        
        # 计算目标股数，向下取整到100的倍数
        target_shares = int(target_value / current_price)
        target_shares = (target_shares // 100) * 100  # 取整到100的倍数
        
        if target_shares < 100:
            log.warn(f"Skip {s}: target shares {target_shares} < 100")
            continue
        
        order = order_target(s, target_shares)
        if order:
            orders_placed.append(f"BUY {s}: {target_shares}股 (目标权重{w:.2%})")
    
    # 打印交易汇总
    if orders_placed:
        log.info(f"=== 交易执行 ({len(orders_placed)}笔) ===")
        for order_info in orders_placed:
            log.info(f"  {order_info}")
    else:
        log.info("本次调仓无交易")


def is_tradable(context, security):
    """
    简化版：只检查最基本的可交易性
    ETF通常流动性好，不需要过度过滤
    """
    try:
        # 使用 attribute_history 检查最近是否有数据（更可靠）
        df = attribute_history(security, 1, '1d', ['close'], skip_paused=True)
        if df is None or df.empty:
            log.debug(f"{security} no recent data")
            return False
        return True
    except Exception as e:
        log.debug(f"{security} check failed: {str(e)}")
        return False


# ==============
# 可选模块：risk-off（V1启用）
# ==============
def calc_risk_off(context):
    """
    示例：用沪深300的MA200趋势做risk-on/off。
    你也可以换成你 notebook 里的 trend_12_26 + vol_ratio 危机过滤。
    """
    idx = BENCHMARK
    df = get_price(idx, end_date=context.current_dt, count=220, frequency='1d', fields=['close'], panel=False)
    if df is None or df.empty:
        return False
    close = df['close'].values
    if len(close) < 210:
        return False
    ma200 = np.mean(close[-201:-1])  # 昨天为止的200日均线
    last = close[-2]
    return last < ma200


# ==============
# 可选模块：ML overlay（V2启用）
# ==============
def ml_overlay_weights(context, target_list, base_weights):
    """
    预留接口：输入候选池与base权重 -> 输出最终权重
    你明天接入：用技术指标/GBDT 得到每个ETF的score，再二次排序或按score加权。
    """
    # TODO: scores = load_or_predict_scores(context, target_list)
    # TODO: reweight by scores with cap MAX_WEIGHT, normalize
    return base_weights


# ====================================================================
#                        策略设计思路详解
# ====================================================================
"""
【策略概述】
这是一个基于动量因子的ETF轮动策略，采用分层模块化设计，从V0→V1→V2逐步演进。

═══════════════════════════════════════════════════════════════════
一、参数配置区（UNIVERSE、BENCHMARK等）
═══════════════════════════════════════════════════════════════════

UNIVERSE（标的池）：
    - 包含20个流动性好的ETF，覆盖宽基、行业、海外等
    - 选择标准：规模大、流动性好、跟踪误差小
    - 可根据实际平台可交易品种调整

BENCHMARK（基准）：
    - 000300.XSHG（沪深300指数）
    - 用于业绩比较和风险评估

DEFENSIVE（防守资产）：
    - 511880.XSHG（货币基金ETF）
    - 用于风险开关触发时的避险配置
    - 可替换为短债ETF或国债ETF

核心参数：
    - MOM_WINDOW=20：动量计算窗口（20日收益率）
      * 经典技术分析中，20日代表约1个月的交易日
      * 捕捉短期趋势，避免过度滞后
    
    - TOPK=10：从候选池选出动量最强的前10个
      * 集中度适中，既保证收益弹性又分散风险
      * 约占总池50%，避免过度集中或过度分散
    
    - REBALANCE_EVERY=5：每5个交易日调仓（周频）
      * 平衡交易成本与信号跟踪
      * 避免日频调仓的高成本，也避免月频的信号延迟
    
    - MAX_WEIGHT=0.15：单票权重上限15%
      * 防止单一标的过度集中
      * 即使选10个，也允许等权配置（1/10=10% < 15%）
    
    - COST_BPS=5.0：交易成本5个基点（0.05%）
      * 保守估计，包含佣金、滑点、印花税
      * 实际成本通过set_commission配置

策略开关（V0默认关闭，后续版本启用）：
    - USE_RISK_OFF=False：风险开关（V1启用）
    - USE_ML_OVERLAY=False：机器学习增强（V2启用）

═══════════════════════════════════════════════════════════════════
二、框架函数（initialize、before_trading_start等）
═══════════════════════════════════════════════════════════════════

initialize(context)：
    目的：策略初始化，设置全局参数
    
    关键配置：
    1. set_benchmark()：设置业绩基准
    2. set_option('use_real_price', True)：使用真实价格（非复权价）
       - 确保下单价格与实际市场一致
    
    3. log.set_level('order', 'error')：减少订单日志噪音
       - 只记录错误信息，避免大量INFO淹没关键信息
    
    4. set_commission()：设置交易成本
       - PerTrade：按每笔交易收费
       - buy_cost=0.0003：买入费率0.03%
       - sell_cost=0.0003：卖出费率0.03%
       - min_cost=5：最低手续费5元
    
    5. set_slippage()：设置滑点
       - FixedSlippage(0.0002)：固定滑点0.02%
       - 模拟市场冲击成本
    
    全局变量：
    - g.universe：标的池列表（copy避免被修改）
    - g.day_count：交易日计数器（用于周频调仓判断）
    - g.last_target：上次持仓标的（用于对比变化）
    
    定时任务：
    - run_daily(rebalance, time='09:35')
      * 每天9:35执行rebalance函数
      * 开盘5分钟后，价格相对稳定
      * 在rebalance内部判断是否真正调仓

before_trading_start(context)：
    - 目前为空，预留接口
    - 可在此更新全局变量、检查持仓等

handle_data(context, data)：
    - 目前为空，预留接口
    - 实时回调函数，不适合低频策略

═══════════════════════════════════════════════════════════════════
三、核心逻辑（rebalance、calc_momentum等）
═══════════════════════════════════════════════════════════════════

【rebalance(context)】调仓主流程
────────────────────────────────────
执行流程：
1. 日计数器+1，判断是否到调仓日
   - day_count % REBALANCE_EVERY != 1：跳过
   - 第1、6、11、16...天执行调仓（周频）

2. 过滤可交易标的
   - universe = [s for s in g.universe if is_tradable(s)]
   - 剔除停牌、数据缺失等不可交易标的
   - 如果全部过滤掉，记录警告并返回

3. 风险开关检查（V0关闭）
   - 如果USE_RISK_OFF=True且触发风险信号
   - 全部切换到防守资产（货基/短债）
   - 等待风险信号解除

4. 计算动量并选TopK
   - mom = calc_momentum(context, universe, window=20)
   - 返回每个标的的20日动量值
   - 降序排序，取前TOPK个
   - 这是策略的核心Alpha来源

5. 构建权重（等权配置）
   - weights = make_weights_equal(target, max_weight=0.15)
   - 每个标的权重 = 1/TOPK = 1/10 = 10%
   - 受MAX_WEIGHT限制，确保不过度集中

6. ML增强（V0关闭）
   - 如果USE_ML_OVERLAY=True
   - 用机器学习模型对TopK进一步优化权重
   - 可能将某些标的权重提升到15%，其他降低

7. 执行下单
   - apply_target_weights(context, weights)
   - 先卖出不在目标池的持仓
   - 再买入/调整目标池的持仓

【calc_momentum(context, universe, window)】动量计算
────────────────────────────────────
目的：计算每个标的的动量信号

参数：
- universe：候选标的列表
- window：回溯窗口（默认20日）

实现细节：
1. 数据获取
   - need = window + 2：多取2天数据
   - 原因：用t-1的数据生成信号（避免未来函数）
   - get_price(..., count=need, panel=False)
   - panel=False返回长格式DataFrame（时间+标的+价格）

2. 异常处理
   - try-except捕获get_price失败
   - 空数据检查：if df is None or df.empty
   - 返回空Series，触发上层的"skip rebalance"

3. 逐个标的计算动量
   - 按code分组：df.groupby('code')
   - 检查数据长度：if len(closes) < need: continue
   - 避免数据不足导致计算错误
   
4. 动量公式
   - c1 = closes[-2]：昨天收盘价（t-1）
   - c0 = closes[-2-window]：window天前的收盘价（t-1-window）
   - momentum = c1/c0 - 1
   - 防止索引越界：c0_idx = min(-2-window, -len(closes))

5. 日志记录
   - 记录处理了多少标的，得到多少有效信号
   - 方便调试和监控

动量因子的逻辑：
- 动量效应（Momentum Effect）：过去表现好的资产，未来一段时间继续表现好
- 学术基础：Jegadeesh & Titman (1993)
- 适用于趋势性市场，捕捉资金流向和情绪传导

【make_weights_equal(target_list, max_weight)】权重构建
────────────────────────────────────
目的：将候选标的分配等权重

逻辑：
1. 空列表检查：if not target_list: return {}
2. 计算等权：w = 1.0 / len(target_list)
3. 权重上限：w = min(w, max_weight)
   - 例如：10个标的，等权10%，不触及15%上限
   - 如果只选5个，等权20%，会被限制到15%
4. 留现金策略：如果因max_weight导致总和<1，剩余留现金
   - 更保守，避免过度集中风险

【apply_target_weights(context, target_weights)】执行下单
────────────────────────────────────
目的：根据目标权重调整实际持仓

两阶段执行：
1. 清仓阶段
   - 遍历当前持仓：context.portfolio.positions.keys()
   - 如果不在目标池：order_target(s, 0) 清仓
   - 记录：orders_placed.append("SELL ...")

2. 建仓/调仓阶段
   - 遍历目标权重字典
   - 再次检查可交易性：is_tradable(context, s)
     * 防止动量计算后到下单前出现停牌
   
   - 获取当前价格：attribute_history(s, 1, '1d', ['close'])
   
   - 计算目标股数：
     * target_value = total_value * w
     * target_shares = int(target_value / current_price)
     * target_shares = (target_shares // 100) * 100
     * 关键：向下取整到100的倍数（ETF最小交易单位）
   
   - 最小交易量检查：
     * if target_shares < 100: skip
     * 避免"开仓数量不能小于100"错误
   
   - 下单：order_target(s, target_shares)
     * 按股数下单（而非按金额）
     * 聚宽会自动处理买卖方向
   
   - 记录：orders_placed.append("BUY ...")

3. 交易日志
   - 汇总打印所有交易
   - 格式：股票代码、股数、目标权重
   - 方便复盘和监控

为什么用order_target而非order_target_value？
- order_target_value：按金额下单，系统自动计算股数
- 但可能产生非100整数倍的股数，导致下单失败
- order_target：按股数下单，我们手动取整到100倍数
- 更可控，避免舍入错误

【is_tradable(context, security)】可交易性检查
────────────────────────────────────
目的：判断标的当前是否可以交易

实现：
- attribute_history(security, 1, '1d', ['close'], skip_paused=True)
- 获取最近1天的收盘价数据
- skip_paused=True：自动跳过停牌标的
- 如果返回空数据，说明停牌或数据缺失

优势：
- 比get_current_data()更可靠
  * get_current_data()在开盘早期可能数据不全
  * attribute_history使用历史数据，更稳定
- 自动处理停牌
- 简洁高效，适合ETF（流动性好）

为什么不检查涨跌停/成交量？
- ETF通常流动性极好，很少涨跌停
- 过度过滤可能误杀正常标的
- 保持简单，减少边界case

═══════════════════════════════════════════════════════════════════
四、可选模块（risk-off、ML overlay）
═══════════════════════════════════════════════════════════════════

【calc_risk_off(context)】风险开关（V1启用）
────────────────────────────────────
目的：在市场危机时切换到防守资产

当前实现（示例）：
- 用沪深300指数的MA200趋势判断
- if 当前价格 < MA200: risk_off = True
- 逻辑：跌破长期均线 = 市场进入下跌趋势

更复杂的实现（可选）：
- 结合volatility ratio（波动率异常检测）
- 结合多个指数的趋势（A股、港股、美股）
- 结合VIX恐慌指数（如果有数据）

触发后的行为：
- 清仓所有动量标的
- 全仓买入DEFENSIVE（货基/短债）
- 等待风险信号解除再恢复动量策略

优势：
- 避开系统性风险（如2015年股灾、2020年疫情暴跌）
- 保护已实现收益
- 代价：可能错过反弹初期

【ml_overlay_weights(context, target_list, base_weights)】ML增强（V2启用）
────────────────────────────────────
目的：用机器学习对动量信号二次优化

输入：
- target_list：动量TopK候选池（10个）
- base_weights：基础权重（等权10%）

处理流程（预留接口）：
1. 提取技术指标特征
   - RSI、MACD、布林带等
   - 成交量特征
   - 波动率特征

2. 预测未来收益/风险
   - 用训练好的GBDT/XGBoost模型
   - 输入特征 -> 输出预测score

3. 根据score重新分配权重
   - score高的标的：权重提升到15%
   - score低的标的：权重降低到5%
   - 保持权重和=1，cap=MAX_WEIGHT

输出：
- 优化后的权重字典

优势：
- 在动量基础上叠加更多信息
- 可能提升夏普比率
- 降低回撤

风险：
- 过拟合：模型在训练集表现好，实盘差
- 数据泄露：用到未来数据
- 需要持续维护和监控

═══════════════════════════════════════════════════════════════════
五、策略迭代路径
═══════════════════════════════════════════════════════════════════

V0（当前版本）：
✓ 纯动量因子
✓ 等权配置
✓ 周频调仓
✓ 固定参数
- 优势：简单、可解释性强
- 目标：验证动量因子在ETF上的有效性

V1（下一步）：
+ 启用风险开关（risk-off）
+ 动态调整仓位（满仓 vs 半仓 vs 空仓）
+ 引入多周期动量（5日、20日、60日）
+ 参数优化（最优窗口、TopK数量）
- 优势：降低回撤，提升风险调整收益
- 目标：夏普比率 > 1.5

V2（最终版）：
+ 启用ML overlay
+ 多因子融合（动量、价值、质量）
+ 自适应仓位管理
+ 在线学习（模型持续更新）
- 优势：捕捉非线性关系
- 目标：夏普比率 > 2.0，最大回撤 < 15%

═══════════════════════════════════════════════════════════════════
六、风险提示
═══════════════════════════════════════════════════════════════════

1. 动量失效风险
   - 市场从趋势性转向震荡时，动量策略表现差
   - 频繁调仓导致高成本、低收益

2. 流动性风险
   - 某些小规模ETF可能流动性不足
   - 大资金可能冲击价格

3. 交易成本
   - 周频调仓，年化约10次完整调仓
   - 单边成本0.05%，双边0.1%，年化约1%
   - 需要确保超额收益 > 成本

4. 模型风险
   - 历史表现不代表未来
   - 参数基于历史优化，可能过拟合

5. 黑天鹅事件
   - 极端市场（如2015、2020）可能导致大幅回撤
   - 风险开关可部分缓解，但不能完全避免

═══════════════════════════════════════════════════════════════════
七、实盘建议
═══════════════════════════════════════════════════════════════════

1. 小资金测试
   - 先用少量资金（如10万）运行1-3个月
   - 验证信号质量、交易成本、滑点等

2. 监控指标
   - 持仓集中度
   - 换手率
   - 跟踪误差
   - 夏普比率

3. 定期复盘
   - 每月检查交易记录
   - 分析盈亏来源
   - 是否需要调参或优化

4. 风险控制
   - 设置止损线（如回撤 > 20%暂停）
   - 预留应急资金
   - 不要all-in单一策略

5. 合规性
   - 确保标的可交易（有些ETF有投资者门槛）
   - 了解税务影响
   - 遵守平台规则
"""
