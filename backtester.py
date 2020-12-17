import pandas_datareader as dr
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA

df = dr.data.get_data_yahoo('QQQ', start='2000-01-01', end='2020-11-01')

class SmaCross(Strategy):
    len_fast = 50
    len_slow = 100
    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, self.len_fast)
        self.ma2 = self.I(SMA, price, self.len_slow)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()


bt = Backtest(df, SmaCross, commission=.002,
              exclusive_orders=True)
stats = bt.run()
print('------Initial parameters------')
print(stats)
print(stats.tail())

opt = bt.optimize(len_fast=range(1, 50, 5),
                  len_slow=range(50, 200, 5),
                  maximize='Sharpe Ratio',
                  constraint=lambda p: p.len_fast < p.len_slow)
print('------Optimized parameters------')
print(opt)
print(opt.tail())

bt.plot()