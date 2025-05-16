import yfinance as yf
import numpy as np
import pandas as pd
import gurobipy as gp
import argparse
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# === Config ===
assets = [
    "SPY","XLB","XLC","XLE","XLF","XLI",
    "XLK","XLP","XLRE","XLU","XLV","XLY"
]
start = "2019-01-01"
end   = "2024-04-01"

# === Download data ===
raw = yf.download(assets, start=start, end=end, auto_adjust=False, threads=True)
df = raw["Adj Close"]
df_returns = df.pct_change().fillna(0)


# === Problem 1: Equal Weight Portfolio ===
class EqualWeightPortfolio:
    def __init__(self, exclude):
        self.exclude = exclude

    def calculate_weights(self):
        # 所有除 benchmark 外的資產
        assets_ = [c for c in df.columns if c != self.exclude]
        n = len(assets_)
        w = 1.0 / n

        # 建立完整欄位的 weights DataFrame
        W = pd.DataFrame(0.0, index=df.index, columns=df.columns)
        W.loc[:, assets_] = w
        self.portfolio_weights = W

    def calculate_portfolio_returns(self):
        self.calculate_weights()
        assets_ = [c for c in df.columns if c != self.exclude]

        # 用 weights 跟 returns 相乘再加總
        pr = df_returns.copy()
        pr["Portfolio"] = (
            pr[assets_]
            .multiply(self.portfolio_weights[assets_], axis=1)
            .sum(axis=1)
        )
        self.portfolio_returns = pr

    def get_results(self):
        self.calculate_portfolio_returns()
        return self.portfolio_weights, self.portfolio_returns


# === Problem 2: Risk Parity Portfolio ===
class RiskParityPortfolio:
    def __init__(self, exclude, lookback=50):
        self.exclude = exclude
        self.lookback = lookback

    def calculate_weights(self):
        assets_ = [c for c in df.columns if c != self.exclude]
        W = pd.DataFrame(0.0, index=df.index, columns=df.columns)

        for i in range(self.lookback, len(df)):
            date = df.index[i]
            window = df_returns[assets_].iloc[i - self.lookback : i]
            inv_vol = 1.0 / window.std()
            w = inv_vol / inv_vol.sum()
            W.loc[date, assets_] = w.values

        self.portfolio_weights = W

    def calculate_portfolio_returns(self):
        self.calculate_weights()
        assets_ = [c for c in df.columns if c != self.exclude]

        pr = df_returns.copy()
        pr["Portfolio"] = (
            pr[assets_]
            .multiply(self.portfolio_weights[assets_], axis=1)
            .sum(axis=1)
        )
        self.portfolio_returns = pr

    def get_results(self):
        self.calculate_portfolio_returns()
        return self.portfolio_weights, self.portfolio_returns


# === Problem 3: Mean-Variance Portfolio ===
class MeanVariancePortfolio:
    def __init__(self, exclude, lookback=50, gamma=0):
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        assets_ = [c for c in df.columns if c != self.exclude]
        W = pd.DataFrame(0.0, index=df.index, columns=df.columns)

        for i in range(self.lookback, len(df)):
            window = df_returns[assets_].iloc[i - self.lookback : i]
            Sigma = window.cov().values
            mu = window.mean().values
            n = len(mu)

            with gp.Env(empty=True) as env:
                env.setParam("OutputFlag", 0)
                env.setParam("DualReductions", 0)
                env.start()
                model = gp.Model(env=env)
                w = model.addMVar(shape=n, lb=0.0, ub=1.0, name="w")
                # 目標函數 μᵀw − γ/2·wᵀΣw
                model.setObjective(mu @ w - (self.gamma / 2) * (w @ Sigma @ w), gp.GRB.MAXIMIZE)
                model.addConstr(w.sum() == 1, name="budget")
                model.optimize()

                if model.status in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
                    sol = w.X
                else:
                    sol = np.ones(n) / n

            W.iloc[i][assets_] = sol

        self.portfolio_weights = W

    def calculate_portfolio_returns(self):
        self.calculate_weights()
        assets_ = [c for c in df.columns if c != self.exclude]

        pr = df_returns.copy()
        pr["Portfolio"] = (
            pr[assets_]
            .multiply(self.portfolio_weights[assets_], axis=1)
            .sum(axis=1)
        )
        self.portfolio_returns = pr

    def get_results(self):
        self.calculate_portfolio_returns()
        return self.portfolio_weights, self.portfolio_returns


# === Assignment Judge ===
class AssignmentJudge:
    def __init__(self):
        self.eqw_path = "./Answer/eqw.pkl"
        self.rp_path  = "./Answer/rp.pkl"
        self.mv_paths = [f"./Answer/mv_list_{i}.pkl" for i in range(4)]

    def _compare(self, a, b, tol=1e-2):
        if a.shape != b.shape or not a.index.equals(b.index) or not a.columns.equals(b.columns):
            return False
        for c in a.columns:
            if np.issubdtype(a[c].dtype, np.number):
                if not np.allclose(a[c], b[c], atol=tol):
                    return False
            else:
                if not (a[c] == b[c]).all():
                    return False
        return True

    def check_eqw(self, w):
        ans = pd.read_pickle(self.eqw_path)
        if self._compare(ans, w):
            print("Problem 1 Complete - Get 20 Points")
            return 20
        print("Problem 1 Fail")
        return 0

    def check_rp(self, w):
        ans = pd.read_pickle(self.rp_path)
        if self._compare(ans, w):
            print("Problem 2 Complete - Get 20 Points")
            return 20
        print("Problem 2 Fail")
        return 0

    def check_mv(self, wl):
        ans = [pd.read_pickle(p) for p in self.mv_paths]
        if all(self._compare(a, w) for a, w in zip(ans, wl)):
            print("Problem 3 Complete - Get 30 points")
            return 30
        print("Problem 3 Fail")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Homework 3 Part 1")
    parser.add_argument("--score", action="append", help="Score for assignment")
    args = parser.parse_args()

    # Mean-Variance 的 4 種組合
    params = [
        {"exclude":"SPY"},
        {"exclude":"SPY", "gamma":100},
        {"exclude":"SPY", "lookback":100},
        {"exclude":"SPY", "lookback":100, "gamma":100},
    ]

    if args.score:
        judge = AssignmentJudge()
        total = 0

        if "eqw" in args.score:
            w, _ = EqualWeightPortfolio("SPY").get_results()
            judge.check_eqw(w)

        if "rp" in args.score:
            w, _ = RiskParityPortfolio("SPY").get_results()
            judge.check_rp(w)

        if "mv" in args.score:
            wl = [MeanVariancePortfolio(**p).get_results()[0] for p in params]
            judge.check_mv(wl)

        if "all" in args.score:
            total += judge.check_eqw(EqualWeightPortfolio("SPY").get_results()[0])
            total += judge.check_rp(RiskParityPortfolio("SPY").get_results()[0])
            wl = [MeanVariancePortfolio(**p).get_results()[0] for p in params]
            total += judge.check_mv(wl)
            print(f"==> total Score = {total} <==")
