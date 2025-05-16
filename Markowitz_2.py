"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

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

# Download price history for 2012–2024
Bdf = (
    yf.download(
        assets,
        start="2012-01-01",
        end="2024-04-01",
        auto_adjust=False,
        threads=True,
        group_by="ticker",
    )["Adj Close"]
)

# Use only 2019–2024 for df
df = Bdf.loc["2019-01-01":"2024-04-01"]


"""
Strategy Creation

Create your own strategy in MyPortfolio.calculate_weights.
"""

class MyPortfolio:
    """
    You can modify the init signature but keep `price` and `exclude`.
    """
    def __init__(self, price: pd.DataFrame, exclude: str, lookback=50, gamma=0):
        self.price = price
        # compute daily returns
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # exclude the benchmark column
        assets = self.price.columns[self.price.columns != self.exclude]

        # prepare weights DataFrame
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index,
            columns=self.price.columns,
            dtype=float
        )

        """
        TODO: Complete Task 4 Below
        Example: pick the single asset (excluding SPY) with
        the highest Sharpe ratio over the entire period,
        then allocate 100% to it.
        """

        # 1. compute Sharpe per asset: mean/std * sqrt(252)
        asset_returns = self.returns[assets]
        sharpe = (asset_returns.mean() / asset_returns.std()) * np.sqrt(252)
        # 2. choose the asset with highest Sharpe
        best_asset = sharpe.idxmax()
        # 3. allocate 100% to that asset on every date
        self.portfolio_weights.loc[:, assets] = 0.0
        self.portfolio_weights.loc[:, best_asset] = 1.0

        """
        TODO: Complete Task 4 Above
        """

        # fill forward then zeros for exclude column
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # apply weights to daily returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # ensure both weights and returns exist
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()
        return self.portfolio_weights, self.portfolio_returns


"""
Assignment Judge

Checks Sharpe > 1 and Sharpe > SPY.
"""

class AssignmentJudge:
    def __init__(self):
        # compute both 2019-2024 and 2012-2024 results
        self.mp = MyPortfolio(df, "SPY").get_results()
        self.Bmp = MyPortfolio(Bdf, "SPY").get_results()

    def plot_performance(self, price, strategy):
        _, ax = plt.subplots()
        returns = price.pct_change().fillna(0)
        (1 + returns["SPY"]).cumprod().plot(ax=ax, label="SPY")
        (1 + strategy[1]["Portfolio"]).cumprod().plot(ax=ax, label="MyPortfolio")
        ax.set_title("Cumulative Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.legend()
        plt.show()

    def plot_allocation(self, df_weights):
        dfw = df_weights.fillna(0).ffill()
        dfw[dfw < 0] = 0  # long only
        _, ax = plt.subplots()
        dfw.plot.area(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Allocation")
        ax.set_title("Asset Allocation Over Time")
        plt.show()

    def report_metrics(self, price, strategy, show=False):
        df_bl = pd.DataFrame({
            "SPY": price.pct_change().fillna(0)["SPY"],
            "MP": pd.to_numeric(strategy[1]["Portfolio"], errors="coerce"),
        })
        sharpe_series = qs.stats.sharpe(df_bl)
        if show:
            qs.reports.metrics(df_bl, mode="full", display=True)
        return sharpe_series

    def check_sharp_ratio_greater_than_one(self):
        # ensure no leverage
        if (self.mp[0].sum(axis=1) <= 1.01).all():
            sr = self.report_metrics(df, self.mp)
            if sr[1] > 1:
                print("Problem 4.1 Success - Get 15 points")
                return 15
        else:
            print("Portfolio Position Exceeds 1. No Leverage.")
        print("Problem 4.1 Fail")
        return 0

    def check_sharp_ratio_greater_than_spy(self):
        if (self.Bmp[0].sum(axis=1) <= 1.01).all():
            sr = self.report_metrics(Bdf, self.Bmp)
            if sr[1] > sr[0]:
                print("Problem 4.2 Success - Get 15 points")
                return 15
        else:
            print("Portfolio Position Exceeds 1. No Leverage.")
        print("Problem 4.2 Fail")
        return 0

    def check_all_answer(self):
        total = 0
        total += self.check_sharp_ratio_greater_than_one()
        total += self.check_sharp_ratio_greater_than_spy()
        print(f"==> total Score = {total} <==")
        return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 2"
    )
    parser.add_argument("--score", action="append", help="Score for assignment")
    parser.add_argument("--allocation", action="append", help="Allocation plot")
    parser.add_argument("--performance", action="append", help="Performance plot")
    parser.add_argument("--report", action="append", help="Evaluation report")
    args = parser.parse_args()

    judge = AssignmentJudge()

    if args.score:
        if "one" in args.score:
            judge.check_sharp_ratio_greater_than_one()
        if "spy" in args.score:
            judge.check_sharp_ratio_greater_than_spy()
        if "all" in args.score:
            judge.check_all_answer()

    if args.allocation:
        if "mp" in args.allocation:
            judge.plot_allocation(judge.mp[0])
        if "bmp" in args.allocation:
            judge.plot_allocation(judge.Bmp[0])

    if args.performance:
        if "mp" in args.performance:
            judge.plot_performance(df, judge.mp)
        if "bmp" in args.performance:
            judge.plot_performance(Bdf, judge.Bmp)

    if args.report:
        if "mp" in args.report:
            judge.report_metrics(df, judge.mp, show=True)
        if "bmp" in args.report:
            judge.report_metrics(Bdf, judge.Bmp, show=True)
