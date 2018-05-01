# -*- coding:utf8 -*-
import os
import sys
import csv
import pandas as pd
from pandas import Series,DataFrame
import socket
import platform
import numpy as np
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn import  feature_selection
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件
# path_train = "../demo.csv"
# path_test = "../demo.csv"
startlno = 0
endlno = 300
dumpcvs = './model/dump' + str(startlno) + '_' + str(endlno)
dumptar = dumpcvs + '.gz'
cmd = 'gzip ' + dumpcvs
pay = 0.3 #赔付率阈值
start_col = 'hstd'

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def cal_ratio(df,col,pre):
    df_temp1 = df[['TERMINALNO',col,'TRIP_ID']].groupby(['TERMINALNO',col]).count().reset_index()
    df_temp1.rename(columns={'TRIP_ID':'count1'}, inplace = True)
    df_temp2 = df[['TERMINALNO','TRIP_ID']].groupby(['TERMINALNO']).count().reset_index()
    df_temp2.rename(columns={'TRIP_ID':'count2'}, inplace = True)
    df_result = pd.merge(df_temp1, df_temp2, how='left', on=['TERMINALNO'])
    df_result['ratio'] = df_result['count1']/df_result['count2']
    df_temp1 = df_result[['TERMINALNO',col,'ratio']]
    df_temp2 = pd.get_dummies(df_temp1[col], prefix= pre)
    df_temp1 = pd.concat([df_temp1, df_temp2], axis=1)
    startcol = df_temp2.columns.values[0]
    df_temp1.loc[:,startcol:] = df_temp1.loc[:,startcol:].mul(df_temp1['ratio'],axis=0)
    df_temp1 = df_temp1.groupby('TERMINALNO').sum().reset_index()
    return df_temp1

def hourms(df,pre): #按车主进行小时级别的统计计算:mean,std
    lable1 = pre + '_hmean'
    lable2 = pre + '_hstd'
    df_temp = df.groupby(['TERMINALNO']).agg({'hour':{lable1:'mean',lable2:'std'}})
    levels = df_temp.columns.levels
    labels = df_temp.columns.labels
    df_temp.columns = levels[1][labels[1]]
#     df_temp = df_am.reset_index()
    return df_temp

def makefeature(df):
    df['ftime'] = df['TIME'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
    df['ftime'] = pd.to_datetime(df['ftime'])
    df['hour'] = df['ftime'].dt.hour.astype(np.uint8)
    # print("lineno:", sys._getframe().f_lineno)
    df['day'] = df['ftime'].dt.day.astype(np.uint8)
    df['weekday'] = df['ftime'].dt.weekday.astype(np.uint8)
    df['month'] = df['ftime'].dt.month.astype(np.uint8)
    # print("lineno:", sys._getframe().f_lineno)

    df = df.fillna(0)

    df_nam = hourms(df[(df['hour'] < 12) & (df['weekday'] < 5)],'nam').reset_index()
    df_wam = hourms(df[(df['hour'] < 12) & (df['weekday'] >= 5)],'wam').reset_index()
    df_npm = hourms(df[(df['hour'] >= 12) & (df['weekday'] < 5)],'npm').reset_index()
    df_wpm = hourms(df[(df['hour'] >= 12) & (df['weekday'] >= 5)],'wpm').reset_index()

    df_nhour = cal_ratio(df[df['weekday'] < 5],'hour','nth')
    df_whour = cal_ratio(df[df['weekday'] >= 5],'hour','wth')
    df_nhour = df_nhour.drop(['hour','ratio'], axis=1)
    df_whour = df_whour.drop(['hour','ratio'], axis=1)
    df_nhour['nhfreq'] = df_nhour.loc[:,'nth_0':][df_nhour>0].count(axis=1) #24小时出行分布
    df_whour['whfreq'] = df_whour.loc[:,'wth_0':][df_whour>0].count(axis=1)

    # df_hour = cal_ratio(df, 'hour', 'th')
    # df_hour = df_hour.drop(['ratio'], axis=1)
    # df_hour['hfreq'] = df_hour.loc[:,'th_00':][df_hour>0].count(axis=1)
    # t = df.groupby(['TERMINALNO','month','day','hour']).agg({'SPEED':'count'}).reset_index()
    # t = t.groupby(['TERMINALNO','month','day']).agg({'hour':'count'}).reset_index()
    # t = t.groupby(['TERMINALNO','month']).agg({'day':'count','hour':'mean'}).reset_index()
    # t = t.groupby(['TERMINALNO']).agg({'day':'mean','hour':'mean'}).reset_index()
    # df_hour[['dayspm','hourspd']] = t[['day','hour']]
    # df_main = pd.merge(df_hour, df_speed, how='left', on=['TERMINALNO'])

    df_weekday = cal_ratio(df, 'weekday', 'tw')
    df_weekday = df_weekday.drop(['weekday', 'ratio'], axis=1)
    df_month = cal_ratio(df, 'month', 'tm')
    df_month = df_month.drop(['month', 'ratio'], axis=1)

    df_timespan = df.groupby(['TERMINALNO','month','day','hour']).agg({'SPEED':'count'}).reset_index()
    df_timespan = df_timespan.groupby(['TERMINALNO','month','day']).agg({'hour':'count'}).reset_index()
    df_timespan = df_timespan.groupby(['TERMINALNO','month']).agg({'day':'count','hour':'mean'}).reset_index()
    df_timespan = df_timespan.groupby(['TERMINALNO']).agg({'day':'mean','hour':'mean'}).reset_index()
    df_timespan.rename(columns={'day':'dayspm','hour':'hourspd'}, inplace = True)
    

    # df_hour['thc1'] = df_hour.loc[:, 'th_00':'th_04'].sum(axis=1)  # 凌晨：0-4
    # df_hour['thc2'] = df_hour.loc[:, ['th_08', 'th_09', 'th_17', 'th_18']].sum(axis=1)  # 上下班高峰：8-9,17-18
    # df_hour['thc3'] = df_hour['thc1'] + df_hour['thc2']
    # df_hour['thc3'] = df_hour['thc3'].apply(lambda x: 1 - x)  # 其他时间
    # df_weekday['twc1'] = df_weekday.loc[:, ['tw_1', 'tw_5']].sum(axis=1)  # 上下班高峰:周一，周五
    # df_weekday['twc2'] = df_weekday.loc[:, ['tw_0', 'tw_6']].sum(axis=1)  # 双休日
    # df_weekday['twc3'] = df_weekday['twc1'] + df_weekday['twc2']
    # df_weekday['twc3'] = df_weekday['twc3'].apply(lambda x: 1 - x)  # 其他时间
    # df_hour = df_hour.loc[:, ['TERMINALNO', 'thc1', 'thc2', 'thc3']]
    # df_weekday = df_weekday.loc[:, ['TERMINALNO', 'twc1', 'twc2', 'twc3']]

    df_hight = df.groupby(['TERMINALNO'])['HEIGHT'].agg({'hstd':np.std,'hmean':np.mean}).reset_index()
    df_speed = df.groupby(['TERMINALNO'])['SPEED'].agg({'sstd':np.std,'smean':np.mean}).reset_index()
    # df_main = pd.merge(df_month, df_weekday, how='left', on=['TERMINALNO'])
    
    df_main = pd.merge(df_hight, df_speed, how='left', on=['TERMINALNO'])
    df_main = pd.merge(df_main, df_month, how='left', on=['TERMINALNO'])
    df_main = pd.merge(df_main, df_weekday, how='left', on=['TERMINALNO'])
    df_main = pd.merge(df_main, df_nam, how='left', on=['TERMINALNO'])
    df_main = pd.merge(df_main, df_wam, how='left', on=['TERMINALNO'])
    df_main = pd.merge(df_main, df_npm, how='left', on=['TERMINALNO'])
    df_main = pd.merge(df_main, df_wpm, how='left', on=['TERMINALNO'])
    df_main = pd.merge(df_main, df_nhour, how='left', on=['TERMINALNO'])
    df_main = pd.merge(df_main, df_whour, how='left', on=['TERMINALNO'])
    df_main = pd.merge(df_main, df_timespan, how='left', on=['TERMINALNO'])
    df_main = df_main.fillna(0)

    return df_main

def predicate(X_train,y_train,X_test):
    # fs = feature_selection.SelectPercentile(feature_selection.f_regression, percentile=20)
    # X_train = fs.fit_transform(X_tain, y_tain)
    # X_test = fs.transform(X_test)
    # print(np.isfinite(X_train).all())
    Scaler = StandardScaler()
    X_train = Scaler.fit_transform(X_train)
    X_test = Scaler.transform(X_test)
    params = {'n_estimators': 500, 'max_depth': 4,'learning_rate': 0.01, 'loss': 'ls'}
    model = GradientBoostingRegressor(**params)
    # model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)

def output(df):
    temp = df[['TERMINALNO', 'Y']].sort_values(by='Y', axis=0, ascending=True)
    temp.rename(columns={'TERMINALNO': 'Id', 'Y': 'Pred'}, inplace=True)
    temp.to_csv('./model/output.csv', index=False)


def process():
    # print("lineno:", sys._getframe().f_lineno)
    df = pd.read_csv(path_train, dtype=dict(TERMINALNO=np.uint16, TIME=np.uint32, TRIP_ID=np.uint16,
                                                  LONGITUDE=np.float32, LATITUDE=np.float32, DIRECTION=np.int16,
                                                  HEIGHT=np.float32, SPEED=np.float32, CALLSTATE=np.uint8,
                                                  Y=np.float16))
    # print("lineno:",sys._getframe().f_lineno)
    df_tain = df[['TERMINALNO', 'Y', 'TRIP_ID']].groupby(['TERMINALNO', 'Y']).count().reset_index()
    df_tain.rename(columns={'TRIP_ID': 'counts'}, inplace=True)

    assert isinstance(df, object)
    df_main = makefeature(df)
    df_tain = pd.merge(df_tain, df_main, how='left', on=['TERMINALNO'])
    # print(df_tain[['hfreq','dayspm','hourspd','Y']][df_tain['Y'] > 1].sort_values(by = 'Y',axis = 0,ascending = True) )
    # print(df_tain[['hmean','smean','hfreq','dayspm','hourspd','Y']][df_tain['Y'] > 1].mean())
    # print(df_tain[['hmean','smean','hfreq','dayspm','hourspd','Y']][df_tain['Y'] <=0.3].mean())
    # return
    # t = df_tain.loc[startlno:endlno, : ][df_tain["Y"] > 0].sort_values(by='Y', axis=0, ascending=False)
    # zip(t)


    df = pd.read_csv(path_test, dtype=dict(TERMINALNO=np.uint16, TIME=np.uint32, TRIP_ID=np.uint16,
                                                  LONGITUDE=np.float32, LATITUDE=np.float32, DIRECTION=np.int16,
                                                  HEIGHT=np.float32, SPEED=np.float32, CALLSTATE=np.uint8))

    df_test = df[['TERMINALNO', 'TRIP_ID']].groupby(['TERMINALNO']).count().reset_index()

    assert isinstance(df, object)
    df_main = makefeature(df)
    df_test = pd.merge(df_test, df_main, how='left', on=['TERMINALNO'])
    df_test["Y"] = predicate(df_tain.loc[:, start_col:],df_tain['Y'],df_test.loc[:, start_col:])
    output(df_test)


def zip(df):
    os.system('rm -rf ./model/*')
    # df_outer = df.loc[startlno:endlno, : ]
    df.to_csv(dumpcvs, index=False)
    ret = os.system(cmd)
    if ret != 0:
        print("\n tar shell error!\n")
    else:
        with open(dumptar, 'rb') as f:
            print(f.read())

def detection():
    print(platform.uname())
    print(os.getcwd())
    # print('\ncurrent dir:%s , files: %s' % (os.getcwd(),os.listdir('./model')))
    print('\nother dir :%s' % os.listdir(os.path.abspath('..')))
    print(socket.getaddrinfo(socket.gethostname(),None))
    # addrs = socket.getaddrinfo(socket.gethostname(),None)
    # for item in addrs:
    #     print(item)
    # socket.setdefaulttimeout(1)
    # for p in range(1, 10000):
    #     portScanner('127.0.0.1', p)


if __name__ == "__main__":
    # 程序入口
    process()
    # detection()