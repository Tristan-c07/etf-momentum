# src/make_universe.py
import pandas as pd

# 你可以改这里：20–40只
SYMBOLS = [
    "510300","510500","510050","159915","159919","588000",
    "512100","512690","512800","512000","512010","512170",
    "512480","512290","512660","512400","515000","515050",
    "515030","515880","516160","516970","159995","159928",
]

def try_get_names_from_jq(symbols):
    """
    使用 jqdatasdk 拉名称（推荐）。如果你不用聚宽数据源，直接在下面返回None即可。
    """
    try:
        from jqdatasdk import auth, get_security_info
        # 方式1：环境变量登录（推荐）
        # setx JQ_USER "xxx" ; setx JQ_PASS "yyy"
        import os
        user = os.getenv("JQ_USER")
        pwd = os.getenv("JQ_PASS")
        if user and pwd:
            auth(user, pwd)

        names = []
        for s in symbols:
            # 聚宽 ETF 通常是 XSHE 或 XSHG，简单规则：以1/3开头多为深，5多为沪；不准就两边都试
            candidates = [f"{s}.XSHG", f"{s}.XSHE"]
            name = None
            for c in candidates:
                try:
                    info = get_security_info(c)
                    if info is not None:
                        name = info.display_name
                        break
                except Exception:
                    pass
            names.append(name)
        if all(n is None for n in names):
            return None
        return names
    except Exception:
        return None

def main():
    names = try_get_names_from_jq(SYMBOLS)
    if names is None:
        # 兜底：先用 symbol 当 name，确保 universe.csv 先生成
        names = SYMBOLS[:]

    df = pd.DataFrame({"symbol": SYMBOLS, "name": names})
    df.to_csv("universe.csv", index=False, encoding="utf-8-sig")
    print("Wrote universe.csv", df.shape)

if __name__ == "__main__":
    main()
