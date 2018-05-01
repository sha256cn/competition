# -*- coding:utf8 -*-
import pandas as pd
import sys
from pandas import Series,DataFrame
import numpy as np
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn import  feature_selection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import xgboost as xgb
from xgboost import plot_importance
from sklearn.preprocessing import Imputer
import lightgbm as lgb
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件
pay = 0.3 #赔付率阈值
speedcut = [0,5,30,80,120,200] #速度分段
scl = ["1", "2", "3", "4", "5"] #速度分段标签
fcl = {'tfc' : 7, 'sfc' : 2, 'hfc' : 2,'lfc' : 2} #特征选择参数
fs = {'p1':['tfc1','tfc2','tfc3','hfc2'],
      'p2':['tfc7','tfc2','tfc3','hfc2'],
      'p3':['tfc1','tfc2','tfc3','sfc2'],
      'p4':['tfc7','tfc2','tfc3','sfc2'],
      'p5':['tfc1','tfc2','tfc3','hfc2','lfc2'],
      'p6':['tfc7','tfc2','tfc3','hfc2','lfc2'],
      'p7':['tfc1','tfc2','tfc3','sfc2','lfc2'],
      'p8':['tfc7','tfc2','tfc3','sfc2','lfc2']}#特征方案
# fs = {'p1' : ['tfc7','tfc6','tfc3','tfc4','tfc5','hfc1','sfc1','hfc2','sfc2','lfc2']}
istrain = 1
mn = 'gbr'
dfn = 'df_tfc1'
hci = 0.75
tci = 0.75
lci = 0.75
sci = 0.75

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def subcol(x1,x2):
    v1 = x1.split('_')
    v2 = x2.split('_')
    ret = 0
    if v1[0] == v2[0]:
        ret = int(v1[1]) - int(v2[1])
    return ret

def data_transformation(df):
    df['ftime'] = df['TIME'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(x)))
    df['ftime'] = pd.to_datetime(df['ftime'])
    df['hour'] = df['ftime'].dt.hour.astype(np.uint8)
    df['weekday'] = df['ftime'].dt.weekday.astype(np.uint8)
    df['day'] = df['ftime'].dt.day.astype(np.uint8)
    df['month'] = df['ftime'].dt.month.astype(np.uint8)
    df['rs'] = np.rint(df['SPEED']/10).astype(np.uint8)
    df = df[(df['month'] != 10) | ((df['month'] == 10) & (df['day'] > 9))]#排除国庆长假
    # df['ctime'] = df['TIME'].map(lambda x : str(x))
    # df['cday'] = df['day'].map(lambda x : str(x))
    # df['tshift'] = df['ctime'].shift(1)
    # df['dshift'] = df['cday'].shift(1)
    # df['ctime'] = df['cday'].str.cat(df['ctime'],sep='_')
    # df['tshift'] = df['dshift'].str.cat(df['tshift'],sep='_')
    # df.loc[:0,'tshift'] = '2_1'
    # df['tsub'] = df.apply(lambda x : subcol(x['ctime'],x['tshift']), axis=1)

def tuning(X_train,y_train):
    param = {
        'n_estimators': range(30, 50, 2),
        'max_depth': range(2, 7, 1)
    }
    if mn == 'gbr':
        X_train.fillna(0, inplace = True)
        params = {'n_estimators': 500, 'max_depth': 4,'learning_rate': 0.01, 'loss': 'ls'}
        model = GradientBoostingRegressor(**params)
    else:
        model = xgb.XGBRegressor(learning_rate=0.01,n_estimators=500, max_depth=4, silent=True, objective='reg:gamma')
    # clf = GridSearchCV(estimator = model, param_grid = param, scoring='r2', cv=10)
    # clf.fit(X_train, y_train)
    # print(clf.grid_scores_, clf.best_params_, clf.best_score_)

    percentiles = range(1, 100, 2)
    results = []
    X_train.fillna(0, inplace = True)
    for i in percentiles:
        fs = feature_selection.SelectPercentile(feature_selection.f_regression, percentile = i)
        X_train_fs = fs.fit_transform(X_train, y_train)
        scores = cross_val_score(model, X_train_fs, y_train, cv=5)
        results = np.append(results, scores.mean())
    print(results)
    opt = np.where(results == results.max())[0]
    print(percentiles[int(opt)])
    

def predicate(X_train,y_train,X_test):
    if mn == 'gbr':
        X_train.fillna(0, inplace = True)
        X_test.fillna(0, inplace = True)
        params = {'n_estimators': 500, 'max_depth': 4,'learning_rate': 0.01, 'loss': 'ls'}
        model = GradientBoostingRegressor(**params)
    elif mn == 'lgb':
        model = lgb.LGBMRegressor(objective='regression',num_leaves=31,learning_rate=0.01,n_estimators=500)
    else:
        model = xgb.XGBRegressor(max_depth=4, learning_rate=0.01, n_estimators=500, silent=True, objective='reg:linear')
        # model = xgb.XGBRegressor(learning_rate=0.1, max_depth=2, silent=True, objective='reg:linear')        
    model.fit(X_train, y_train)
    if mn != 'xgboost': 
        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(-feature_importance)
        feature_name = X_train.columns
        print(feature_name[sorted_idx[0]]+"|"+feature_name[sorted_idx[1]]+"|"+feature_name[sorted_idx[2]]+"|"+feature_name[sorted_idx[3]]+"|"+feature_name[sorted_idx[4]])
    # return cross_val_score(model,X_train,y_train,scoring = ginival, cv=10)
    return model.predict(X_test)

def output(df):
    temp = df[['TERMINALNO', 'Y']].sort_values(by='Y', axis=0, ascending=True)
    temp.rename(columns={'TERMINALNO': 'Id', 'Y': 'Pred'}, inplace=True)
    temp.to_csv('./model/output.csv', index=False)

def cal_ratio(df,col,pre):
#     df = df_train
#     col = 'CALLSTATE'
#     pre = 'cs'
    df_temp1 = df[['TERMINALNO',col,'TRIP_ID']].groupby(['TERMINALNO',col]).count().reset_index()
    df_temp1.rename(columns={'TRIP_ID':'count1'}, inplace = True)
    df_temp2 = df[['TERMINALNO','TRIP_ID']].groupby(['TERMINALNO']).count().reset_index()
    df_temp2.rename(columns={'TRIP_ID':'count2'}, inplace = True)
    df_result = pd.merge(df_temp1, df_temp2, how='left', on=['TERMINALNO'])
    df_result['ratio'] = df_result['count1']/df_result['count2']
    df_temp1 = df_result[['TERMINALNO',col,'ratio']]
    df_temp2 = pd.get_dummies(df_temp1[col], prefix= pre)
    df_temp1 = pd.concat([df_temp1, df_temp2], axis=1)
    t = df_temp2.loc[:0,:].count(axis=1)
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

def hourspd(df,pre):
    hourspd = pre + 'hourspd'
    df_timespan = df.groupby(['TERMINALNO','month','day','hour']).agg({'SPEED':'count'}).reset_index()
    df_timespan = df_timespan.groupby(['TERMINALNO','month','day']).agg({'hour':'count'}).reset_index()
    df_timespan = df_timespan.groupby(['TERMINALNO','month']).agg({'day':'count','hour':'mean'}).reset_index()
    t_group = df_timespan.groupby(['TERMINALNO'])
    df_timespan = df_timespan.groupby(['TERMINALNO'])['day'].agg({'dayspm':'mean'}).reset_index()
    t = t_group.apply(lambda x: np.average(x['hour'],weights = x['day'])).reset_index(drop=True)
    df_timespan[hourspd] = t
    df_timespan = df_timespan.drop(['dayspm'], axis=1)
    return df_timespan

def dayspm(df,pre):
    dayspm = pre + 'dayspm'
    df_timespan = df.groupby(['TERMINALNO','month','day','hour']).agg({'SPEED':'count'}).reset_index()
    df_timespan = df_timespan.groupby(['TERMINALNO','month','day']).agg({'hour':'count'}).reset_index()
    df_timespan = df_timespan.groupby(['TERMINALNO','month']).agg({'day':'count','hour':'mean'}).reset_index()
    df_timespan = df_timespan.groupby(['TERMINALNO'])['day'].agg({dayspm:'mean'}).reset_index()
    return df_timespan

def sfc1(df,df_base): # 构造速度特征:均值及标准差,返回车主ID,标准差,均值,极值
    pre = sys._getframe().f_code.co_name + '_'
    lable1 = pre + 'sstd'
    lable2 = pre + 'smean'
    lable3 = pre + 'smax'
    lable4 = pre + 'smin'
    df_speed = df[df['SPEED'] > 0 ].groupby(['TERMINALNO'])['SPEED'].agg({lable1:np.std,lable2:np.mean,lable3:np.max,lable4:np.min}).reset_index()
    sfc1 = pd.merge(df_base, df_speed, how='left', on=['TERMINALNO'])
    return sfc1

def sfc2(df,df_base):#按周计算速度的置信区间上下限
    pre = sys._getframe().f_code.co_name + '_s'
    df_sfc2 = weekcal(df[df['SPEED'] > 0 ],df_base,'SPEED',pre,sci)
    return df_sfc2

def hfc1(df,df_base): # 构造海拔特征:最大及最小值
    pre = sys._getframe().f_code.co_name + '_'
    lable1 = pre + 'hmax'
    lable2 = pre + 'hmin'
    lable3 = pre + 'hmean'
    lable4 = pre + 'hstd'
    df_hight = df.groupby(['TERMINALNO'])['HEIGHT'].agg({lable1:max,lable2:min,lable3:np.mean,lable4:np.std}).reset_index()
    hfc1 = pd.merge(df_base, df_hight, how='left', on=['TERMINALNO'])
    return hfc1

def weekcal(df,df_base,col,pre,ci):#按周计算极值及置信区间
    lable1 = pre + 'ncil'
    lable2 = pre + 'ncih'
    lable3 = pre + 'nmax'
    lable4 = pre + 'nmin'
    lable5 = pre + 'wcil'
    lable6 = pre + 'wcih'
    lable7 = pre + 'wmax'
    lable8 = pre + 'wmin'
    lable9 = pre + 'nmean'
    lable10 = pre + 'wmean'
    lable11 = pre + 'ncsub'
    lable12 = pre + 'wcsub'
    lable13 = pre + 'nstd'
    lable14 = pre + 'wstd'

    t1 = df[df['weekday'] < 5].groupby(['TERMINALNO'])[col].agg({lable3: np.max,lable4:np.min,lable9:np.mean,lable13:np.std}).reset_index()
    t2 = df[df['weekday'] >= 5].groupby(['TERMINALNO'])[col].agg({lable7: np.max,lable8:np.min,lable10:np.mean,lable14:np.std}).reset_index()
    df_temp = pd.merge(df_base, t1, how='left', on=['TERMINALNO'])
    df_temp = pd.merge(df_temp, t2, how='left', on=['TERMINALNO'])
    df_temp[lable1] = df[df['weekday']<5].groupby(['TERMINALNO'])[col].apply(lambda x: stats.norm.interval(ci,np.mean(x),np.std(x))[0])
    df_temp[lable2] = df[df['weekday']<5].groupby(['TERMINALNO'])[col].apply(lambda x: stats.norm.interval(ci,np.mean(x),np.std(x))[1]) 
    df_temp[lable5] = df[df['weekday']>=5].groupby(['TERMINALNO'])[col].apply(lambda x: stats.norm.interval(ci,np.mean(x),np.std(x))[0])
    df_temp[lable6] = df[df['weekday']>=5].groupby(['TERMINALNO'])[col].apply(lambda x: stats.norm.interval(ci,np.mean(x),np.std(x))[1])
    # df_temp.fillna(0,inplace = True)
    # df_temp[lable1] = df_temp[lable1].apply(lambda x: int(x)).astype(np.uint8)
    # df_temp[lable2] = df_temp[lable2].apply(lambda x: int(x)).astype(np.uint8)
    # df_temp[lable3] = df_temp[lable3].apply(lambda x: int(x)).astype(np.uint8)
    # df_temp[lable4] = df_temp[lable4].apply(lambda x: int(x)).astype(np.uint8)
    # df_temp[lable5] = df_temp[lable5].apply(lambda x: int(x)).astype(np.uint8)
    # df_temp[lable6] = df_temp[lable6].apply(lambda x: int(x)).astype(np.uint8)
    # df_temp[lable7] = df_temp[lable7].apply(lambda x: int(x)).astype(np.uint8)
    # df_temp[lable8] = df_temp[lable8].apply(lambda x: int(x)).astype(np.uint8)
    df_temp[lable11] = df_temp[lable2] -df_temp[lable1]
    df_temp[lable12] = df_temp[lable6] -df_temp[lable5]
    # df_temp = df_temp.drop([lable1,lable2,lable5,lable6], axis=1)
    return df_temp


def hfc2(df,df_base):#按周计算海拔的置信区间上下限
    pre = sys._getframe().f_code.co_name + '_h'
    df_hfc2 = weekcal(df,df_base,'HEIGHT',pre,hci)
    return df_hfc2

def tfc1(df,df_base): #构造时间特征:平时及周末上下午出行时间段特征：mean,std,mode
    pre = sys._getframe().f_code.co_name + '_'
    df_nh = hourms(df[(df['weekday'] < 5)],pre +'nh').reset_index()
    df_wh = hourms(df[(df['weekday'] >= 5)],pre +'wh').reset_index()
    df_tfc1 = pd.merge(df_base, df_nh, how='left', on=['TERMINALNO'])
    df_tfc1 = pd.merge(df_tfc1, df_wh, how='left', on=['TERMINALNO'])
    # df_nam = hourms(df[(df['hour'] < 12) & (df['weekday'] < 5)],pre +'nam').reset_index()
    # df_wam = hourms(df[(df['hour'] < 12) & (df['weekday'] >= 5)],pre +'wam').reset_index()
    # df_npm = hourms(df[(df['hour'] >= 12) & (df['weekday'] < 5)],pre +'npm').reset_index()
    # df_wpm = hourms(df[(df['hour'] >= 12) & (df['weekday'] >= 5)],pre +'wpm').reset_index()
    # df_tfc1 = pd.merge(df_base, df_nam, how='left', on=['TERMINALNO'])
    # df_tfc1 = pd.merge(df_tfc1, df_npm, how='left', on=['TERMINALNO'])
    # df_tfc1 = pd.merge(df_tfc1, df_wam, how='left', on=['TERMINALNO'])
    # df_tfc1 = pd.merge(df_tfc1, df_wpm, how='left', on=['TERMINALNO'])
    return df_tfc1



def tfc2(df,df_base): #构造时间特征:平时及周末24小时内出行频度，即24小时内有几次高频出行
    pre = sys._getframe().f_code.co_name + '_'
    label1 = pre + 'nth'
    label2 = pre + 'wth'
    nhfreq = pre + 'nhfreq'
    whfreq = pre + 'whfreq'
    df_nhour = cal_ratio(df[df['weekday'] < 5],'hour',label1)
    df_whour = cal_ratio(df[df['weekday'] >= 5],'hour',label2)
    df_nhour = df_nhour.drop(['hour','ratio'], axis=1)
    df_whour = df_whour.drop(['hour','ratio'], axis=1)
    label1 = label1 + '_0'
    label2 = label2 + '_0'
    df_nhour[nhfreq] = df_nhour.loc[:,label1:][df_nhour>0].count(axis=1)
    df_whour[whfreq] = df_whour.loc[:,label2:][df_whour>0].count(axis=1)
    t = df_nhour.loc[:,label1:].apply(lambda x: x >= 1/x[nhfreq],axis=1)
    df_nhour[nhfreq] = t[t == True].count(axis=1)
    t = df_whour.loc[:,label2:].apply(lambda x: x >= 1/x[whfreq],axis=1)
    df_whour[whfreq] = t[t == True].count(axis=1)
    df_tfc2 = pd.merge(df_base, df_nhour, how='left', on=['TERMINALNO'])
    df_tfc2 = pd.merge(df_tfc2, df_whour, how='left', on=['TERMINALNO'])
    # df_tfc2 = df_tfc2[['TERMINALNO','tfc2_nhfreq','tfc2_whfreq']]
    # print(df_tfc2.columns)
    return df_tfc2

def tfc3(df,df_base): #构造时间特征:出行时长，每天的出行时长，每月的出行天数
    pre = sys._getframe().f_code.co_name + '_'
    pren = pre + 'n'
    df_timespan = hourspd(df[df['weekday'] < 5],pren)
    df_tfc3 = pd.merge(df_base, df_timespan, how='left', on=['TERMINALNO'])
    prew = pre + 'w'
    df_timespan = hourspd(df[df['weekday'] >= 5],prew)
    df_tfc3 = pd.merge(df_tfc3, df_timespan, how='left', on=['TERMINALNO'])
    df_timespan = dayspm(df,pre)
    df_tfc3 = pd.merge(df_tfc3, df_timespan, how='left', on=['TERMINALNO'])
    return df_tfc3

def tfc4(df,df_base):#构造时间特征:按周统计出行时间段特征：mean,std,mode
    pre = sys._getframe().f_code.co_name + '_'
    pre = pre + 'tw'
    df_weekday = cal_ratio(df,'weekday',pre)
    df_weekday = df_weekday.drop(['weekday','ratio'], axis=1)
    df_tfc4 = pd.merge(df_base, df_weekday, how='left', on=['TERMINALNO'])
    return df_tfc4

def tfc5(df,df_base):#构造时间特征:按月统计出行时间段特征：mean,std,mode
    pre = sys._getframe().f_code.co_name + '_'
    pre = pre + 'tm'
    df_month = cal_ratio(df,'month',pre)
    df_month = df_month.drop(['month','ratio'], axis=1)
    df_tfc5 = pd.merge(df_base, df_month, how='left', on=['TERMINALNO'])
    return df_tfc5

def tfc6(df,df_base): #构造时间特征:24小时内出行频度，即24小时内有几次高频出行
    pre = sys._getframe().f_code.co_name + '_'
    label1 = pre + 'th'
    hfreq = pre + 'hfreq'
    df_hour = cal_ratio(df,'hour',label1)
    df_hour = df_hour.drop(['hour','ratio'], axis=1)
    label1 = label1 + '_0'
    df_hour[hfreq] = df_hour.loc[:,label1:][df_hour>0].count(axis=1)
    t = df_hour.loc[:,label1:].apply(lambda x: x >= 1/x[hfreq],axis=1)
    df_hour[hfreq] = t[t == True].count(axis=1)
    df_tfc6 = pd.merge(df_base, df_hour, how='left', on=['TERMINALNO'])
    # print(df_tfc6.columns)
    return df_tfc6

def tfc7(df,df_base):#构造时间特征:平时及周末出行时间置信区间
    pre = sys._getframe().f_code.co_name + '_h'
    df_tfc7 = weekcal(df,df_base,'hour',pre,tci)
    return df_tfc7

def lfc1(df,df_base):#构造经纬度特征:最大及最小之差
    pre = sys._getframe().f_code.co_name + '_'
    lable1 = pre + 'losub'
    lable2 = pre + 'lasub'
    t = df.groupby(['TERMINALNO']).agg({'LONGITUDE':{'lomax':max,'lomin':min},'LATITUDE':{'lamax':max,'lamin':min}})
    levels = t.columns.levels
    labels = t.columns.labels
    t.columns = levels[1][labels[1]]
    t = t.reset_index()
    t[lable1] = t['lomax'] -t['lomin']
    t[lable2] = t['lamax'] -t['lamin']
    t = t.drop(['lomax','lomin','lamax','lamin'], axis=1)
    lfc1 = pd.merge(df_base, t, how='left', on=['TERMINALNO'])
    return lfc1

def lfc2(df,df_base): #构造经纬度特征:以经纬度置信区间为边长,估算行车面积
    pre = sys._getframe().f_code.co_name + '_'
    lable1 = pre + 'narea'
    lable2 = pre + 'warea'
    t = df_base.copy()
    t[lable1] = df[df['weekday'] < 5].groupby(['TERMINALNO']).apply(
        lambda x: (stats.norm.interval(lci,np.mean(x['LONGITUDE']),np.std(x['LONGITUDE']))[1] -
        stats.norm.interval(lci,np.mean(x['LONGITUDE']),np.std(x['LONGITUDE']))[0]) * 
        (stats.norm.interval(lci,np.mean(x['LATITUDE']),np.std(x['LATITUDE']))[1] -
        stats.norm.interval(lci,np.mean(x['LATITUDE']),np.std(x['LATITUDE']))[0]))
    t[lable2] = df[df['weekday'] >= 5].groupby(['TERMINALNO']).apply(
        lambda x: (stats.norm.interval(lci,np.mean(x['LONGITUDE']),np.std(x['LONGITUDE']))[1] -
        stats.norm.interval(lci,np.mean(x['LONGITUDE']),np.std(x['LONGITUDE']))[0]) * 
        (stats.norm.interval(lci,np.mean(x['LATITUDE']),np.std(x['LATITUDE']))[1] -
        stats.norm.interval(lci,np.mean(x['LATITUDE']),np.std(x['LATITUDE']))[0]))
    t.reset_index()
    lfc2 = pd.merge(df_base, t, how='left', on=['TERMINALNO'])
    return lfc2

def train(X,y):
    # print(X.columns.tolist)
    # Scaler = StandardScaler()
    # Xs = Scaler.fit_transform(X)
    # X.fillna(0, inplace = True)
    # Xs = Scaler.fit_transform(X)
    # k = 18
    # fsk = len(X.columns.tolist())
    # if( fsk < 18 ):
    #     k = fsk
    # fs = feature_selection.SelectKBest(feature_selection.f_regression, k)
    # Xs = fs.fit_transform(X,y)
    # print(X.columns)
    t = pd.concat([X,y], axis=1)
    df_pay = t[t['Y']>0]
    df_npay = t[t['Y']<=0]
    X_train1, X_test1, y_train1, y_test1 = train_test_split(df_pay.iloc[:,:-1], df_pay['Y'], random_state=11, test_size= 0.15)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(df_npay.iloc[:,:-1], df_npay['Y'], random_state=20, test_size= 0.15)
    X_train = pd.concat([X_train1,X_train2], axis=0)
    X_test = pd.concat([X_test1,X_test2], axis=0)
    y_train = pd.concat([y_train1,y_train2],axis=0)
    y_test = pd.concat([y_test1,y_test2],axis=0)
    # print(X_train.columns)
    pred = predicate(X_train, y_train, X_test)
    print(mn + 'r2_score:'+ str(r2_score(y_test,pred)))
    print(mn + ' gini:'+str(gini(y_test,pred)))
    


def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses
    giniSum -= (len(actual) + 1) / 2
    return giniSum / len(actual)

def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)

def ginival(estimator,X, y):
    pred = estimator.predict(X)
    actual = y
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses
    giniSum -= (len(actual) + 1) / 2
    return giniSum / len(actual)

def makefeature(fcl,df,df_base,df_owner):
    i = 1
    df_temp = df_owner
    for key,value in fcl.items():
        while i <= value:
            func = key + str(i)
            dfn = 'df_' + func
            func = globals().get(key + str(i))
            dfn = func(df,df_base)
            df_temp = pd.merge(df_temp, dfn, how='left', on=['TERMINALNO']) 
            i = i + 1
        i = 1
    return df_temp

def makefeature1(fs,df,df_base,df_owner):
    df_temp = df_owner
    for index,value in fs.items():
        for i,v in enumerate(value):
            dfn = 'df_' + v
            func = globals().get(v)
            dfn = func(df,df_base)
            df_temp = pd.merge(df_temp, dfn, how='left', on=['TERMINALNO'])
    # print(df_temp.columns)
    return df_temp

def featureselected(fs,df_base,df_feature): 
    # print(fs)
    first = False
    starti = 0
    endi = -1
    t = df_base
    colums = list(df_feature.columns)
    for index,value in fs.items():
        for i,v in enumerate(value):
            for i1,v1 in enumerate(colums):
                if v1.startswith(ｖ):
                    if first == False:
                        starti=i1
                        first = True
                else:
                    if first == True and endi == -1:
                        endi = i1
            if endi == -1:
                endi = i1 + 1
            t = pd.concat([t,df_feature.iloc[:,starti : endi]], axis=1)
            first = False
            starti = 0
            endi = -1
    return t

def score(fs,df_base,df_feature):
    first = False
    starti = 0
    endi = -1
    t = df_base
    # print(df_feature.columns.tolist)
    colums = list(df_feature.columns)
    for index,value in fs.items():
        for i,v in enumerate(value):
            for i1,v1 in enumerate(colums):
                if v1.startswith(ｖ):
                    if first == False:
                        starti=i1
                        first = True
                        # print('starti'+str(starti))
                        # print('endi'+str(endi))
                else:
                    if first == True and endi == -1:
                        endi = i1
                        # print('endi'+str(endi))
            if endi == -1:
                endi = i1 + 1
                # print('endi'+str(endi))
            t = pd.concat([t,df_feature.iloc[:,starti : endi]], axis=1)
            # print(t.columns)
            first = False
            starti = 0
            endi = -1
        print("feature:" + ','.join(value))
        train(t.iloc[:,1:],df_feature['Y'])
        # tuning(t.iloc[:,1:],df_feature['Y'])
        t = df_base

def process():
    # print("lineno:", sys._getframe().f_lineno)
    df = pd.read_csv(path_train, dtype=dict(TERMINALNO=np.uint16, TIME=np.uint32, TRIP_ID=np.uint16,
                                                  LONGITUDE=np.float32, LATITUDE=np.float32, DIRECTION=np.int16,
                                                  HEIGHT=np.float32, SPEED=np.float32, CALLSTATE=np.uint8,
                                                  Y=np.float16))
    # df = df.sort_values(by = ['TERMINALNO','TIME'],axis = 0).reset_index(drop=True)
    data_transformation(df)
    # df = df.drop(['ctime','tshift','cday','dshift'], axis=1)
    df_owner = df[['TERMINALNO','Y','TRIP_ID']].groupby(['TERMINALNO','Y']).count().reset_index()
    # print(df_owner.shape[0],df_pay.shape[0])
    df_owner.rename(columns={'TRIP_ID':'counts'}, inplace = True)
    df_base = df_owner.iloc[:,:1]
    print("hci"+str(hci)+",tci"+str(tci)+",lci"+str(lci)+",sci"+str(sci))
    if istrain == 1:
        df_feature = makefeature(fcl,df,df_base,df_owner)
        score(fs,df_base,df_feature)    
    else:
        f = {'p1':['tfc7','tfc2','tfc3','hfc2','sfc1']}
        df_feature = makefeature1(f,df,df_base,df_owner)
        t = featureselected( f ,df_base,df_feature)
        X_train = t.iloc[:,1:]
        y_train = df_feature.iloc[:,1]
        df = pd.read_csv(path_test, dtype=dict(TERMINALNO=np.uint16, TIME=np.uint32, TRIP_ID=np.uint16,
                                                    LONGITUDE=np.float32, LATITUDE=np.float32, DIRECTION=np.int16,
                                                    HEIGHT=np.float32, SPEED=np.float32, CALLSTATE=np.uint8))
        # df = df.sort_values(by = ['TERMINALNO','TIME'],axis = 0).reset_index(drop=True)
        data_transformation(df)
        df_owner = df[['TERMINALNO', 'TRIP_ID']].groupby(['TERMINALNO']).count().reset_index()
        df_owner.rename(columns={'TRIP_ID':'counts'}, inplace = True)
        df_base = df_owner.iloc[:,:1]
        df_feature = makefeature(fcl,df,df_base,df_owner)
        t = featureselected(f,df_base,df_feature)
        X_test = t.iloc[:,1:]
        X_train.fillna(0, inplace = True)
        X_test.fillna(0, inplace = True)
        df_feature["Y"] = predicate(X_train,y_train,X_test)
        output(df_feature)

if __name__ == "__main__":
    # 程序入口
    process()
    # detection()
