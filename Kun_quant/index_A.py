# 模型A
import pandas as pd 
import numpy as np
import tqdm
import random
from sklearn.linear_model import LinearRegression
import lightgbm as lgb


def invoke(data1,id_list) -> pd.DataFrame():
    #data1: pd.DataFrame, 数据来自于 main chart price.parquet
    #id_list: T当日不同的id 值, 用于结果数据的预测
    price_df = data1
    stock_pool = price_df["security_id"].unique()  # 票池 
    date_list = price_df["date_id"].unique()  # 日期
    # price特征转为字典，每个字段格式为df，index是stock，columns是date
    price_dict = {}
    for key in tqdm.tqdm(list(price_df.columns)[3:]):
        price_dict[key] = pd.DataFrame(index=stock_pool, columns=date_list, data=price_df.pivot(index='security_id', columns='date_id', values=key).fillna(np.nan))
    # 模型A只能用交易行情表数据，即price_dict
    mom1 = price_dict["adj_Close"]/price_dict["adj_Close"].shift(1,axis=1) - 1
    vol_change = price_dict["adj_Volume"]/price_dict["adj_Volume"].shift(1,axis=1) - 1
    c2o = price_dict["adj_Open"]/price_dict["adj_Close"].shift(1,axis=1) - 1
    go_high = price_dict["adj_High"]/price_dict["adj_Close"].shift(1,axis=1) - 1
    go_low = price_dict["adj_Low"]/price_dict["adj_Close"].shift(1,axis=1) - 1
    rr_demean = demean_abs(price_dict["adj_High"]/price_dict["adj_Low"]-1)
    valc = ma(price_dict["adj_Close"]*price_dict["adj_Volume"],5)/ma(price_dict["adj_Close"]*price_dict["adj_Volume"],20) - 1

    # 相对市场均值的放量情况
    vol_diff = (price_dict["adj_Volume"]-price_dict["adj_Volume"].shift(1,axis=1))/ma(price_dict["adj_Volume"],120)
    vol_diff = -demean_abs(vol_diff)

    # 偏度和峰度的组合
    close_skew = price_dict["adj_Close"].rolling(20,min_periods=1,axis=1).skew()
    close_kur = price_dict["adj_Close"].rolling(20,min_periods=1,axis=1).kurt()
    fac_sk = (-rank_neut(close_skew)*(mom1>0)-rank_neut(close_kur)*(mom1<0)).replace(0,np.nan)

    # 收盘价分布不对称性
    close = price_dict["adj_Close"]
    fac_nonsym = (close.rolling(20,min_periods=1,axis=1).quantile(0.6)-close.rolling(20,min_periods=1,axis=1).quantile(0.2))/close.rolling(20,min_periods=1,axis=1).std()
    fac_nonsym = regression(fac_nonsym,[mom1],stock_pool,date_list)

    # 纯量价表特征
    price_feature_dict = {
        "mom1":-mom1,
        "vol_diff":vol_diff,
        "go_hl":-(go_high*(mom1>0).astype("int")-go_high*(mom1<0).astype("int")).replace(0,np.nan),
        "rr_demean":ewma(-rr_demean,5),
        "valc":-cal_dret([-valc],ret_today=mom1),
        "fac_sk":fac_sk,
        "fac_nonsym":fac_nonsym
    }

    X_df= generate_X(price_feature_dict,id_list=id_list)
    
    #此处添加模型预测，修改下面的预测y
    #模型载入
    bst = lgb.Booster(model_file='/home/mw/project/modelA-gbdt.txt')  # init model
    pred_y = bst.predict(X_df.values)# 预测的y

    data ={'id':id_list,'y':pred_y}
    data =pd.DataFrame(data)
    return data

def generate_X(feature_dict,id_list):
    # 生成特征集X，索引用id
    X_df = pd.concat([df.stack(dropna=False) for df in feature_dict.values()], axis=1)
    X_df.index = list(map(lambda x:str(x[1])+'d'+str(x[0]),X_df.index.to_numpy()))
    X_df.columns = list(feature_dict.keys())
    X_df = X_df.loc[id_list]
    return X_df

def to_1(df):
    # 归1化
    return (df - df.min())/(df.max()-df.min())

def standard(df):
    # 标准化
    return (df-df.mean())/df.std()

def softmax(df):
    return 1/(1+np.exp(-df))

def deal_nan(series):
    # 如果series全nan，则添加一个非nan值
    if series.count()==0:
        series.iloc[0] = 0
    return series

def demean_abs(df):
    # 距离市场均值的距离
    return (df-df.mean()).abs()

def cal_dret(fac_list, ret_today, ret_qt=0.95, min_periods=1, n=5):
    # 距离筛选框架，有优化因子作用
    # ret_today：收益率df，fac_list：[df1,df2,...] 因子df的列表(作为分量)，ret_qt：收益率分位数，n和min_periods：rolling的参数
    sign_df = (ret_today > ret_today.quantile(ret_qt)).astype('int')  
    fac = sign_df - sign_df
    for i in range(len(fac_list)):
        comp_point = (fac_list[i] * sign_df).rolling(n, min_periods=min_periods, axis=1).mean()
        fac = fac + (fac_list[i] - comp_point)**2
    fac = np.sqrt(fac)
    return fac

def ma(df,n):
    # n日滑动平均
    return df.rolling(n,min_periods=1,axis=1).mean()


def ewma(df,n):
    # n日指数滑动平均
    return df.ewm(n,min_periods=1,axis=1).mean()

def df_mean(df_list):
    # 多个df的均值，输入：df的列表
    return sum(df_list)/len(df_list)

def rank_neut(df):
    # rank中性
    return (df.rank(pct=True) - 0.5).fillna(0)

def normal_neut(df):
    # 标准化中性
    return (df - df.mean()).fillna(0)

def size_neut(df,price_dict):
    # 市值中性，市值用开盘价估计
    mask = df.notna().astype('int').replace(0, np.nan)
    FloatMarketValue = price_dict['adj_FloatShares']*price_dict['adj_Open']
    size_wgt = (FloatMarketValue * mask) / (FloatMarketValue * mask).sum()
    return (df - (df * size_wgt).sum()).fillna(0)

def process_factor(factor_df,stock_pool,date_list):
    df = pd.DataFrame(index=stock_pool, columns=date_list, data=factor_df)
    return df.fillna(np.nan)

def regression(y,x_list,stock_pool,date_list,relu=False):
    # y关于x_list做回归，取残差
    fac_raw = process_factor(y,stock_pool,date_list)
    if len(x_list) > 1:
        X = pd.concat([process_factor(fac,stock_pool,date_list).fillna(0).stack() for fac in x_list],axis=1)
    else:
        X = np.array(x_list[0].fillna(0).stack()).reshape(-1, 1)
    y = y.fillna(0).stack()
    model = LinearRegression().fit(X, y)
    if relu:
        fac_reg = sum([relu_number(model.coef_[i])*x_list[i].fillna(0) for i in range(len(x_list))])
    else:
        fac_reg = sum([model.coef_[i]*x_list[i].fillna(0) for i in range(len(x_list))])
    return fac_raw - process_factor(fac_reg,stock_pool,date_list)

def getCorrFac(fac1, fac2, window, stock_pool):
    # 计算fac1和fac2的时序相关性
    fac = {}
    for ticker in tqdm.tqdm(stock_pool):
        fac[ticker] = fac1.loc[ticker].rolling(window, min_periods=3).corr(fac2.loc[ticker])
    return pd.DataFrame(fac).T

def relu_number(number):
    return number if number > 0 else 0 

# demo
# def invoke(data1,id_list) -> pd.DataFrame():
#     #data1: pd.DataFrame, 数据来自于 main chart price.parquet
#     #id_list: T当日不同的id 值, 用于结果数据的预测
#     y = [random.random()for i in range(len(id_list))]
    
#     data ={'id':id_list,'y':y}
#     data =pd.DataFrame(data)

#     return data