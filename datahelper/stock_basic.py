import tushare as ts
from datahelper.mysql import sqlHelper


helper = sqlHelper('localhost',3306, 'stock', 'root', '123456')
ts.set_token('022c40d6b1fe52d51cb97fd5eae1db5faedd1875b1eafc0fb076e365')
pro = ts.pro_api()
data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,market,list_status,list_date')
sql = 'insert into stock_basic(ts_code,symbol,name,area,industry,market,list_status,list_date) values(%s,%s,%s,%s,%s,%s,%s,%s)'
n = 0
flag = 500
while(n*flag < len(data)):
    slices = data[n*flag:(n+1)*flag]
    helper.updateMany(sql,slices.values.tolist())
    n = n+1
