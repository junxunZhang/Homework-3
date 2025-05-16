import yfinance as yf
import numpy as np
import pandas as pd
import quantstats as qs
import argparse
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

assets=["SPY","XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY"]
Bdf=yf.download(assets,start="2012-01-01",end="2024-04-01",auto_adjust=False,threads=True)["Adj Close"]
df=Bdf.loc["2019-01-01":"2024-04-01"]

class MyPortfolio:
    def __init__(self,price,exclude,lookback=50,gamma=0):
        self.price=price; self.returns=price.pct_change().fillna(0)
        self.exclude,self.lookback,self.gamma=exclude,lookback,gamma
    def calculate_weights(self):
        assets_=self.price.columns[self.price.columns!=self.exclude]
        W=pd.DataFrame(0.0,index=self.price.index,columns=self.price.columns)
        # EXAMPLE STRATEGY: momentum rank of last 20d returns, top 3 equally weighted
        mom = self.price.pct_change(20).iloc[-1]
        top3 = mom[assets_].nlargest(3).index
        for d in self.price.index:
            W.loc[d,top3]=1/3
        self.portfolio_weights=W
    def calculate_portfolio_returns(self):
        self.calculate_weights()
        R=self.returns.copy()
        assets_=self.price.columns[self.price.columns!=self.exclude]
        R["Portfolio"]=(R[assets_].mul(self.portfolio_weights[assets_]).sum(axis=1))
        self.portfolio_returns=R
    def get_results(self):
        self.calculate_portfolio_returns()
        return self.portfolio_weights,self.portfolio_returns

class AssignmentJudge:
    def __init__(self):
        self.mp=MyPortfolio(df,"SPY").get_results()
        self.Bmp=MyPortfolio(Bdf,"SPY").get_results()
    def report_sharpe(self,price,strategy):
        dfbl=pd.DataFrame({"SPY":price.pct_change().fillna(0)["SPY"],
                           "MP":strategy[1]["Portfolio"]})
        return qs.stats.sharpe(dfbl)
    def check_one(self):
        sr=self.report_sharpe(df,self.mp)
        if sr[1]>1:print("Problem 4.1 Success - Get 15 points"); return 15
        print("Problem 4.1 Fail"); return 0
    def check_spy(self):
        sr_full=self.report_sharpe(Bdf,self.Bmp)
        if sr_full[1]>sr_full[0]:print("Problem 4.2 Success - Get 15 points"); return 15
        print("Problem 4.2 Fail"); return 0

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--score",action="append"); args=p.parse_args()
    if args.score:
        if "one" in args.score: AssignmentJudge().check_one()
        if "spy" in args.score: AssignmentJudge().check_spy()
        if "all" in args.score:
            s=AssignmentJudge().check_one()+AssignmentJudge().check_spy()
            print(f"==> total Score = {s} <==")
