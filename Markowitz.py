import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import argparse
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# === Project Setup and Data Download ===

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

start = "2019-01-01"
end = "2024-04-01"

# Download all tickers in one go, using threads for speed
df = (
    yf.download(
        assets,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
    )["Adj Close"]
)
df_returns = df.pct_change().fillna(0)


# === Problem 1: Equal Weight Portfolio ===

class EqualWeightPortfolio:
    def __init__(self, exclude):
        self.exclude = exclude

    def calculate_weights(self):
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        # equal weight: 1/n for each asset
        n_assets = len(assets)
        equal_w = 1 / n_assets
        self.portfolio_weights.loc[:, assets] = equal_w

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        self.calculate_weights()
        self.calculate_portfolio_returns()
        return self.portfolio_weights, self.portfolio_returns


# === Problem 2: Risk Parity Portfolio ===

class RiskParityPortfolio:
    def __init__(self, exclude, lookback=50):
        self.exclude = exclude
        self.lookback = lookback

    def calculate_weights(self):
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        # 從第 lookback 日開始，每天都算一次權重
        for i in range(self.lookback, len(df)):
            date = df.index[i]
            # 1. 取出前 lookback 天的報酬
            window = df_returns[assets].iloc[i - self.lookback : i]
            # 2. 計算各資產波動度（標準差）
            vol = window.std()
            # 3. 取波動度的反比
            inv_vol = 1.0 / vol
            # 4. 正規化，總和為 1
            w = inv_vol / inv_vol.sum()
            # 5. 指派到該日期的權重欄位
            self.portfolio_weights.loc[date, assets] = w.values

        # 向前填補缺失的日期，並把 SPY 欄位（被排除的欄）設為 0
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

# === Problem 3: Mean-Variance Portfolio ===

class MeanVariancePortfolio:
    def __init__(self, exclude, lookback=50, gamma=0):
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        # 從 lookback 日開始，每天都算一次最適化權重
        for i in range(self.lookback, len(df)):
            date = df.index[i]
            window = df_returns[assets].iloc[i - self.lookback : i]
            w = self.mv_opt(window, self.gamma)
            self.portfolio_weights.loc[date, assets] = w

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def mv_opt(self, R_n, gamma):
        # R_n: lookback 天的回報 DF，只含 assets 欄位
        Sigma = R_n.cov().values    # 共變異數矩陣
        mu = R_n.mean().values      # 平均報酬向量
        n = len(mu)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("DualReductions", 0)
            env.start()

            with gp.Model(env=env) as model:
                # 1. 定義決策變數 w_i ∈ [0,1]
                w = model.addMVar(shape=n, lb=0.0, ub=1.0, name="w")
                # 2. 風險調整後報酬目標：μᵀw − γ/2 · wᵀΣw
                obj = mu @ w - (gamma / 2) * (w @ Sigma @ w)
                model.setObjective(obj, gp.GRB.MAXIMIZE)
                # 3. 長短倉限制：sum(w) = 1
                model.addConstr(w.sum() == 1, name="budget")
                # 4. 優化
                model.optimize()

                if model.status in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
                    return w.X.tolist()
                else:
                    # 若無解或失敗，就退回等權
                    return [1 / n] * n

    def calculate_portfolio_returns(self):
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        self.calculate_weights()
        self.calculate_portfolio_returns()
        return self.portfolio_weights, self.portfolio_returns



# === Main ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Homework 3 Part 1")
    parser.add_argument("--score", action="append", help="Score for assignment")
    parser.add_argument("--allocation", action="append", help="Plot allocation")
    parser.add_argument("--performance", action="append", help="Plot performance")
    parser.add_argument("--report", action="append", help="Report metrics")
    args = parser.parse_args()

    # === Scoring ===
    if args.score:
        judge = AssignmentJudge()

        if "eqw" in args.score:
            w_eqw, _ = EqualWeightPortfolio("SPY").get_results()
            judge.check_eqw(w_eqw)

        if "rp" in args.score:
            w_rp, _ = RiskParityPortfolio("SPY").get_results()
            judge.check_rp(w_rp)

        if "mv" in args.score:
            params = [
                dict(exclude="SPY"),
                dict(exclude="SPY", gamma=100),
                dict(exclude="SPY", lookback=100),
                dict(exclude="SPY", lookback=100, gamma=100),
            ]
            mv_weights = []
            for p in params:
                w_mv, _ = MeanVariancePortfolio(**p).get_results()
                mv_weights.append(w_mv)
            judge.check_mv(mv_weights)

        if "all" in args.score:
            total = 0
            # re-run each to accumulate
            w_eqw, _ = EqualWeightPortfolio("SPY").get_results()
            total += judge.check_eqw(w_eqw)
            w_rp, _ = RiskParityPortfolio("SPY").get_results()
            total += judge.check_rp(w_rp)
            mv_weights = []
            for p in params:
                w_mv, _ = MeanVariancePortfolio(**p).get_results()
                mv_weights.append(w_mv)
            total += judge.check_mv(mv_weights)
            print(f"==> total Score = {total} <==")

    # === Visualization and Reports ===
    if args.allocation or args.performance or args.report:
        from Helper import Helper  # assume Helper in separate file or define above
        helper = Helper()

        if args.allocation:
            if "eqw" in args.allocation:
                helper.plot_eqw_allocation()
            if "rp" in args.allocation:
                helper.plot_rp_allocation()
            if "mv" in args.allocation:
                helper.plot_mean_variance_allocation()

        if args.performance:
            helper.plot_mean_variance_portfolio_performance()

        if args.report:
            helper.plot_report_metrics()
