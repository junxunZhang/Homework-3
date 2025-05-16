import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import argparse
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# === Project Setup ===
assets = [
    "SPY","XLB","XLC","XLE","XLF","XLI",
    "XLK","XLP","XLRE","XLU","XLV","XLY",
]
start = "2019-01-01"
end   = "2024-04-01"

# === Data Download (一次多檔下載) ===
raw = yf.download(
    assets,
    start=start,
    end=end,
    auto_adjust=False,
    threads=True,
)
# 取得調整後收盤價
df = raw["Adj Close"]
# 計算日報酬
df_returns = df.pct_change().fillna(0)


# === Problem 1: Equal Weight Portfolio ===
class EqualWeightPortfolio:
    def __init__(self, exclude):
        self.exclude = exclude

    def calculate_weights(self):
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)
        n_assets = len(assets)
        w_equal = 1.0 / n_assets
        self.portfolio_weights.loc[:, assets] = w_equal
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

        for i in range(self.lookback, len(df)):
            date = df.index[i]
            window = df_returns[assets].iloc[i - self.lookback : i]
            vol = window.std()
            inv_vol = 1.0 / vol
            w = inv_vol / inv_vol.sum()
            self.portfolio_weights.loc[date, assets] = w.values

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


# === Problem 3: Mean-Variance Portfolio ===
class MeanVariancePortfolio:
    def __init__(self, exclude, lookback=50, gamma=0):
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        for i in range(self.lookback, len(df)):
            date = df.index[i]
            window = df_returns[assets].iloc[i - self.lookback : i]
            w = self.mv_opt(window, self.gamma)
            self.portfolio_weights.loc[date, assets] = w

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def mv_opt(self, R_n, gamma):
        Sigma = R_n.cov().values
        mu    = R_n.mean().values
        n     = len(mu)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("DualReductions", 0)
            env.start()
            with gp.Model(env=env) as model:
                w = model.addMVar(shape=n, lb=0.0, ub=1.0, name="w")
                obj = mu @ w - (gamma / 2) * (w @ Sigma @ w)
                model.setObjective(obj, gp.GRB.MAXIMIZE)
                model.addConstr(w.sum() == 1, name="budget")
                model.optimize()

                if model.status in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
                    return w.X.tolist()
                else:
                    return [1.0 / n] * n

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


# === Assignment Judge and Main ===
class AssignmentJudge:
    def __init__(self):
        self.eqw_path  = "./Answer/eqw.pkl"
        self.rp_path   = "./Answer/rp.pkl"
        self.mv_paths  = [
            "./Answer/mv_list_0.pkl",
            "./Answer/mv_list_1.pkl",
            "./Answer/mv_list_2.pkl",
            "./Answer/mv_list_3.pkl",
        ]

    def _compare(self, a, b, tol=0.01):
        if a.shape != b.shape or not a.index.equals(b.index) or not a.columns.equals(b.columns):
            return False
        for col in a.columns:
            if np.issubdtype(a[col].dtype, np.number):
                if not np.allclose(a[col], b[col], atol=tol): return False
            else:
                if not (a[col] == b[col]).all(): return False
        return True

    def check_eqw(self, w):
        ans = pd.read_pickle(self.eqw_path)
        if self._compare(ans, w):
            print("Problem 1 Complete - Get 20 Points"); return 20
        print("Problem 1 Fail"); return 0

    def check_rp(self, w):
        ans = pd.read_pickle(self.rp_path)
        if self._compare(ans, w):
            print("Problem 2 Complete - Get 20 Points"); return 20
        print("Problem 2 Fail"); return 0

    def check_mv(self, w_list):
        ans_list = [pd.read_pickle(p) for p in self.mv_paths]
        ok = all(self._compare(a, w) for a, w in zip(ans_list, w_list))
        if ok:
            print("Problem 3 Complete - Get 30 points"); return 30
        print("Problem 3 Fail"); return 0

    def check_all(self):
        total = 0
        total += self.check_eqw(w_eqw)
        total += self.check_rp(w_rp)
        total += self.check_mv(w_mv_list)
        print(f"==> total Score = {total} <==")
        return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Homework 3 Part 1")
    parser.add_argument("--score", action="append", help="Score for assignment")
    args = parser.parse_args()

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
            w_mv_list = []
            for p in params:
                w, _ = MeanVariancePortfolio(**p).get_results()
                w_mv_list.append(w)
            judge.check_mv(w_mv_list)

        if "all" in args.score:
            # 再次跑一次累計所有分數
            w_eqw, _ = EqualWeightPortfolio("SPY").get_results()
            s = judge.check_eqw(w_eqw)
            w_rp, _ = RiskParityPortfolio("SPY").get_results()
            s += judge.check_rp(w_rp)
            w_mv_list = []
            for p in params:
                w, _ = MeanVariancePortfolio(**p).get_results()
                w_mv_list.append(w)
            s += judge.check_mv(w_mv_list)
            print(f"==> total Score = {s} <==")
