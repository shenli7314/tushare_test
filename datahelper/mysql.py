import pymysql

class sqlHelper:

    def __init__(self,host,port,db,user,pwd):

        self.host=host
        self.port=port
        self.db=db
        self.user=user
        self.pwd=pwd
        self.charset='utf8'


    def connection(self):
        self.conn=pymysql.connect(host=self.host,port=self.port,db=self.db,user=self.user,
        password=self.pwd,charset=self.charset)
        self.cursor=self.conn.cursor()
        return'OK'

#创建查询一条数据的方法

    def queryOne(self,sql,params):
        #首先连接数据库获取游标
        try:
            self.connection()
            #执行sql语句
            count=self.cursor.execute(sql,params)
            #返回执行结果
            res=self.cursor.fetchone()
            return count,res

        except Exception as ex:
            print('失败,失败信息是:',ex)
        finally:
            #调用关闭资源的方法
            self.closes()

    #创建查询多条数据的方法

    def queryAll(self,sql,params):

        try:
        #连接数据库
            self.connection()
            #执行sql语句
            count=self.cursor.execute(sql,params)
            #获取结果
            res=self.cursor.fetchall()
            return count,res
        except Exception as e:
            print('查询失败失败信息是：',e)
        finally:
            #调用关闭资源的方法
            self.closes()
    #创建修改数据库的方法

    def update(self,sql,params):

        try:
            #连接数据库
            self.connection()
            #执行sql语句
            count=self.cursor.execute(sql,params)
            #把事物(更新的信息写入数据库)
            self.conn.commit()
            return count
        except Exception as e:
            print('查询失败失败信息是：',e)
            #如果更新失败数据库就返回到更新之前的数据
            self.conn.rollback()
        finally:
            #调用关闭资源的方法
            self.closes()

    def updateMany(self,sqls,params):

        try:
            #连接数据库
            self.connection()
            #执行sql语句
            count=self.cursor.executemany(sqls,params)
            #把事物(更新的信息写入数据库)
            self.conn.commit()
            return count
        except Exception as e:
            print('查询失败失败信息是：',e)
            #如果更新失败数据库就返回到更新之前的数据
            self.conn.rollback()
        finally:
            #调用关闭资源的方法
            self.closes()

    #定义关闭资源的方法
    def closes(self):
        if self.cursor!=None:
            self.cursor.close()
        if self.conn!=None:
            self.conn.close()

if __name__ == '__main__':

    # 创建连接对象  host,db,user,pwd
    helper = sqlHelper('10.240.5.5',31005, 'epc', 'native', 'native')
    # print(helper.connection())
    # a,s = helper.queryAll('select * from tb_inform_fix_case where language=%s',['zh_CN'])
    # with open("../data/fix_case.txt", mode='a', encoding='utf-8') as ff:
    #     for item in s:
    #         item = [str(tups).replace('\t','') for tups in item]
    #         ff.write('\t'.join(item)+'\n')

    a = helper.update('update tb_inform_fix_case set pre_label = %s where id = %s',[1,1291295008728686592])
    print(a)
    # a,s = helper.queryAll('select * from t_stumessage',[])
    # print(s)
