import tushare as ts
import time
from datahelper.mysql import sqlHelper

def get_daily(self, ts_code='', trade_date='', start_date='', end_date=''):
    for _ in range(3):
      try:
        if trade_date:
            df = self.pro.daily(ts_code=ts_code, trade_date=trade_date)
        else:
            df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
      except Exception as e:
         time.sleep(1)
      else:
         return df




if __name__ == '__main__':
    ts.set_token('022c40d6b1fe52d51cb97fd5eae1db5faedd1875b1eafc0fb076e365')
    pro = ts.pro_api()
    # helper = sqlHelper('localhost', 3306, 'stock', 'root', '123456')
    # count,stocks = helper.queryAll('select ts_code from stock_basic where list_status=%s',['L'])
    # for stock in stocks:
    #     df = pro.daily(ts_code=stock[0], start_date='20100101')
    df = ts.pro_bar(ts_code='000001.SZ', adj='hfq', start_date='20100101',ma=['ma5'])
    # df2 = ts.get_hist_data()
    print(df)