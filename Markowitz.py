import yfinance as yf
import numpy as np
import pandas as pd
import gurobipy as gp
import argparse
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# 1. CONFIG
assets = ["SPY","XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY"]
start = "2019-01-01"
end   = "2024-04-01"

# 2. DATA
raw = yf.download(assets, start=start, end=end, auto_adjust=False, threads=True)
df = raw["Adj Close"]
df_returns = df.pct_change().fillna(0)

# 3. PROBLEM 1: EQUAL WEIGHT
class EqualWeightPortfolio:
    def __init__(self, exclude): self.exclude=exclude
    def calculate_weights(self):
        assets_ = df.columns[df.columns!=self.exclude]
        n = len(assets_)
        w = 1.0/n
        self.portfolio_weights = pd.DataFrame(w, index=df.index, columns=assets_)
        # fill SPY with 0
        self.portfolio_weights["SPY"]=0
    def calculate_portfolio_returns(self):
        self.calculate_weights()
        self.portfolio_returns = df_returns.copy()
        self.portfolio_returns["Portfolio"] = (self.portfolio_returns[df.columns!=self.exclude]
                                              .mul(self.portfolio_weights[df.columns!=self.exclude])
                                              .sum(axis=1))
    def get_results(self):
        self.calculate_portfolio_returns()
        return self.portfolio_weights, self.portfolio_returns

# 4. PROBLEM 2: RISK PARITY
class RiskParityPortfolio:
    def __init__(self, exclude, lookback=50):
        self.exclude=exclude; self.lookback=lookback
    def calculate_weights(self):
        assets_ = df.columns[df.columns!=self.exclude]
        W = pd.DataFrame(0.0, index=df.index, columns=assets_)
        for i in range(self.lookback, len(df)):
            window = df_returns[assets_].iloc[i-self.lookback:i]
            inv_vol = 1.0/window.std()
            w = inv_vol/inv_vol.sum()
            W.iloc[i]=w.values
        W["SPY"]=0
        self.portfolio_weights=W
    def calculate_portfolio_returns(self):
        self.calculate_weights()
        self.portfolio_returns = df_returns.copy()
        self.portfolio_returns["Portfolio"] = (self.portfolio_returns[df.columns!=self.exclude]
                                              .mul(self.portfolio_weights[df.columns!=self.exclude])
                                              .sum(axis=1))
    def get_results(self):
        self.calculate_portfolio_returns()
        return self.portfolio_weights, self.portfolio_returns

# 5. PROBLEM 3: MEAN-VARIANCE
class MeanVariancePortfolio:
    def __init__(self, exclude, lookback=50, gamma=0):
        self.exclude=exclude; self.lookback=lookback; self.gamma=gamma
    def calculate_weights(self):
        assets_ = df.columns[df.columns!=self.exclude]
        W = pd.DataFrame(0.0, index=df.index, columns=assets_)
        for i in range(self.lookback, len(df)):
            R = df_returns[assets_].iloc[i-self.lookback:i]
            Sigma=R.cov().values; mu=R.mean().values; n=len(mu)
            with gp.Env(empty=True) as env:
                env.setParam("OutputFlag",0); env.start()
                m=gp.Model(env=env)
                w=m.addMVar(n, lb=0, ub=1, name="w")
                m.setObjective(mu@w - (self.gamma/2)*(w@Sigma@w), gp.GRB.MAXIMIZE)
                m.addConstr(w.sum()==1)
                m.optimize()
                if m.status in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
                    sol=w.X
                else:
                    sol=np.ones(n)/n
            W.iloc[i]=sol
        W["SPY"]=0
        self.portfolio_weights=W
    def calculate_portfolio_returns(self):
        self.calculate_weights()
        self.portfolio_returns=df_returns.copy()
        self.portfolio_returns["Portfolio"]=(self.portfolio_returns[df.columns!=self.exclude]
                                             .mul(self.portfolio_weights[df.columns!=self.exclude])
                                             .sum(axis=1))
    def get_results(self):
        self.calculate_portfolio_returns()
        return self.portfolio_weights, self.portfolio_returns

# 6. JUDGE
class AssignmentJudge:
    def __init__(self):
        self.eqw_path="./Answer/eqw.pkl"
        self.rp_path="./Answer/rp.pkl"
        self.mv_paths=[f"./Answer/mv_list_{i}.pkl" for i in range(4)]
    def _cmp(self,a,b,tol=1e-2):
        if a.shape!=b.shape or not a.index.equals(b.index) or not a.columns.equals(b.columns):
            return False
        for c in a.columns:
            if np.issubdtype(a[c].dtype, np.number):
                if not np.allclose(a[c],b[c],atol=tol): return False
            else:
                if not (a[c]==b[c]).all(): return False
        return True
    def check_eqw(self,w):
        ans=pd.read_pickle(self.eqw_path)
        if self._cmp(ans,w): print("Problem 1 Complete - Get 20 Points"); return 20
        print("Problem 1 Fail"); return 0
    def check_rp(self,w):
        ans=pd.read_pickle(self.rp_path)
        if self._cmp(ans,w): print("Problem 2 Complete - Get 20 Points"); return 20
        print("Problem 2 Fail"); return 0
    def check_mv(self,wl):
        ans=[pd.read_pickle(p) for p in self.mv_paths]
        if all(self._cmp(a,w) for a,w in zip(ans,wl)):
            print("Problem 3 Complete - Get 30 points"); return 30
        print("Problem 3 Fail"); return 0

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--score",action="append")
    args=p.parse_args()
    # PARAMS for MV
    params=[{"exclude":"SPY"},{"exclude":"SPY","gamma":100},
            {"exclude":"SPY","lookback":100},{"exclude":"SPY","lookback":100,"gamma":100}]
    if args.score:
        judge=AssignmentJudge()
        total=0
        if "eqw" in args.score:
            w,_=EqualWeightPortfolio("SPY").get_results(); judge.check_eqw(w)
        if "rp" in args.score:
            w,_=RiskParityPortfolio("SPY").get_results(); judge.check_rp(w)
        if "mv" in args.score:
            wl=[MeanVariancePortfolio(**p).get_results()[0] for p in params]
            judge.check_mv(wl)
        if "all" in args.score:
            total+=judge.check_eqw( EqualWeightPortfolio("SPY").get_results()[0] )
            total+=judge.check_rp(  RiskParityPortfolio("SPY").get_results()[0] )
            wl=[MeanVariancePortfolio(**p).get_results()[0] for p in params]
            total+=judge.check_mv(wl)
            print(f"==> total Score = {total} <==")
