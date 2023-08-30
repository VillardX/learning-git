# 模型A
import pandas as pd 
import numpy as np
import tqdm
import random
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import json


def invoke(data1, data2, data3,id_list) -> pd.DataFrame():
    #data1: pd.DataFrame, data from main chart price.parquet
    #data2: pd.DataFrame, data from forum_a.parquet
    #data3: pd.DataFrame, data from forum_b.parquet
    #id_list: unique date_id value ranged from past date to recent date (based on data1)

    price_df = data1
    forum_a = data2
    forum_b = data3
    stock_pool = price_df["security_id"].unique()  # 票池 
    date_list = list(set(price_df["date_id"].values) | set(forum_a["date_id"].values) | set(forum_b["date_id"].values))  # 日期

    # 论坛a特征转为字典，每个字段格式为df，index是stock，columns是date
    fa_dict = {}
    for key in tqdm.tqdm(list(forum_a.columns)[3:]):
        fa_dict[key] = pd.DataFrame(index=stock_pool, columns=date_list, data=forum_a.pivot(index='security_id', columns='date_id', values=key).fillna(np.nan))

    # 论坛b特征转为字典，每个字段格式为df，index是stock，columns是date
    # 类型2：与目标投资标的高相关的长文
    forum_b_2 = forum_b[forum_b["relevant_type"]==2]
    fb2_dict = {}
    for key in tqdm.tqdm(list(forum_b_2.columns)[4:]):
        fb2_dict[key] = pd.DataFrame(index=stock_pool, columns=date_list, data=forum_b_2.pivot(index='security_id', columns='date_id', values=key).fillna(np.nan))

    # 类型4：单独提及到目标投资标的的短贴
    forum_b_4 = forum_b[forum_b["relevant_type"]==4]
    fb4_dict = {}
    for key in tqdm.tqdm(list(forum_b_4.columns)[4:]):
        fb4_dict[key] = pd.DataFrame(index=stock_pool, columns=date_list, data=forum_b_4.pivot(index='security_id', columns='date_id', values=key).fillna(np.nan))
    
    # price特征转为字典，每个字段格式为df，index是stock，columns是date
    price_dict = {}
    for key in tqdm.tqdm(list(price_df.columns)[3:]):
        price_dict[key] = pd.DataFrame(index=stock_pool, columns=date_list, data=price_df.pivot(index='security_id', columns='date_id', values=key).fillna(np.nan))
    
    fb_dict = {key:((fb2_dict[key]+1e-3).fillna(0)+(fb4_dict[key]+1e-3).fillna(0)*2).replace(0,np.nan) for key in fb2_dict.keys()}
   
    # 模型B
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
    close_skew = price_dict["adj_Close"].T.rolling(20,min_periods=1).skew().T
    close_kur = price_dict["adj_Close"].T.rolling(20,min_periods=1).kurt().T
    fac_sk = (-rank_neut(close_skew)*(mom1>0)-rank_neut(close_kur)*(mom1<0)).replace(0,np.nan)

    # 收盘价分布不对称性
    close = price_dict["adj_Close"]
    fac_nonsym = (close.T.rolling(20,min_periods=1).quantile(0.6).T-close.T.rolling(20,min_periods=1).quantile(0.2).T)/close.T.rolling(20,min_periods=1).std().T
    fac_nonsym = regression(fac_nonsym,[mom1])

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
    
    fa_rolling_dict = {key+"_rolling5":ma(fa_dict[key],5) for key in fa_dict.keys()}
    fb_rolling_dict = {key+"_rolling5":ma(fb_dict[key],5) for key in fb_dict.keys()}
    fa_total_dict = fa_dict.copy()
    fa_total_dict.update(fa_rolling_dict)
    fb_total_dict = fb_dict.copy()
    fb_total_dict.update(fb_rolling_dict)

    fa_index = [
        'regi_senti_score_div', 'regi_senti_score_log', 'senti_score_div',
        'senti_score_log', 'posts_all_unregi_sum_rolling5',
        'unregi_senti_score_log_rolling5', 'regi_senti_score_div_rolling5',
        'regi_senti_score_log_rolling5', 'senti_score_div_rolling5',
        'senti_score_log_rolling5'
    ]

    fb_index = [
        'fav_neg_sum', 'retweet_neg_sum', 'fav_neu_sum', 'retweet_neu_sum',
        'reply_neu_sum', 'like_pos_sum', 'fav_pos_sum', 'retweet_pos_sum',
        'reply_pos_sum', 'post_pos_sum', 'uid_pos', 'user_avg_barage_pos',
        'fav_all_sum', 'retweet_all_sum', 'post_all_sum', 'uid_all',
        'user_avg_barage_all', 'fav_neg_sum_rolling5',
        'retweet_neg_sum_rolling5', 'retweet_neu_sum_rolling5',
        'fav_pos_sum_rolling5', 'retweet_pos_sum_rolling5',
        'fav_all_sum_rolling5', 'retweet_all_sum_rolling5',
        'post_all_sum_rolling5', 'uid_all_rolling5'
    ]
    
    fa_dret_feature_dict = {key+"_a_dret":cal_dret([softmax(fa_total_dict[key])],ret_today=mom1) for key in fa_index}
    fb_dret_feature_dict = {key+"_b_dret":cal_dret([softmax(fb_total_dict[key])],ret_today=mom1) for key in fb_index}
    
    # 特征字典，便于后面拼X_df
    feature_dict = fa_dret_feature_dict.copy()
    feature_dict.update(fb_dret_feature_dict)
    feature_dict.update(price_feature_dict)
    
    X_df1= generate_X(feature_dict,id_list=id_list)

    forum_a["id"] = get_id(forum_a["date_id"],forum_a["security_id"])
    forum_b["id"] = get_id(forum_b["date_id"],forum_b["security_id"])

    id_all = set(forum_b["id"])|set(forum_a["id"])|set(id_list)

    # A表对齐id，缺失值填0
    forum_a_used = forum_a.drop(columns=['date_id','security_id']).set_index("id")
    forum_a_used = pd.DataFrame(forum_a_used,index=id_all,columns=forum_a_used.columns)
    forum_a_used = forum_a_used.fillna(0)

    # B表按relevant_type分成两部分
    fb_all = {}
    for key, value in forum_b.groupby("relevant_type"):
        fb_all[key]=value

    # B表对齐，缺失值填0
    fb_all_used = {}
    for key in fb_all.keys():
        forum_b_used = fb_all[key].drop(columns=['date_id','security_id','relevant_type']).set_index("id")
        forum_b_used = pd.DataFrame(forum_b_used,index=id_all,columns=forum_b_used.columns)
        fb_all_used[key] = forum_b_used.fillna(0)

    # 通过softmax映射至[0,1]
    forum_b_used = fb_all_used[2].loc[id_list] + fb_all_used[4].loc[id_list]
    
    a_rolling10_features = [
        'uids_neg',
        'uids_all',
        'reads_neg_unregi_sum',
        'replies_neg_unregi_sum',
        'posts_neg_unregi_sum',
        'reads_neu_unregi_sum',
        'replies_neu_unregi_sum',
        'posts_neu_unregi_sum',
        'reads_pos_unregi_sum',
        'replies_pos_unregi_sum',
        'posts_pos_unregi_sum',
        'reads_all_unregi_sum',
        'replies_all_unregi_sum',
        'posts_all_unregi_sum',
        'unregi_senti_score_log'
    ]

    b_rolling10_features = [
        'post_neg_sum', 'uid_neg', 'post_all_sum', 'uid_all'
    ]

    a_roll_used = forum_a_used.loc[:,a_rolling10_features]
    a_features = a_roll_used.loc[id_list]/df_rolling(a_roll_used,10,shape=True,method="mean",quantile=0.5,id_all=id_all).loc[id_list].replace(0,np.nan)
    b_roll_used = forum_b_used.loc[:,b_rolling10_features]
    b_features = b_roll_used.loc[id_list]/df_rolling(b_roll_used,10,shape=True,method="mean",quantile=0.5,id_all=id_all).loc[id_list].replace(0,np.nan)

    a_roll_columns = ["fa_"+name+"_div_roll10" for name in a_features.columns]
    b_roll_columns = ["fb_"+name+"_div_roll10" for name in b_features.columns]
    a_features.columns = a_roll_columns
    b_features.columns = b_roll_columns

    a_features["uid_pos_strength"]=(forum_a_used["uids_pos"]-forum_a_used["uids_neg"])/forum_a_used["uids_all"].replace(0,np.nan)

    # 生成X_df
    X_df = pd.concat([X_df1,a_features,b_features],axis=1)
    X_df = X_df.replace([np.inf, -np.inf, "", np.nan], 0)
    # '''
    # 此处添加模型预测，修改下面的预测y
    # '''
    
    #集成
    with open('modelB_dict.json','r') as fm:
        model_dict = json.load(fm)
    
    top3_name = list(model_dict.keys())
    top3_score = list(model_dict.values())
    top3_weight = np.exp(top3_score) / np.exp(top3_score).sum()
    top3_weight = top3_weight.tolist()
    
        
    # #分base筛选
    # gbdt_dic={}
    # rf_dic={}
    # dart_dic={}
    
    # for e_model_name in model_dict.keys():
    #     if '-dart-' in e_model_name:
    #         dart_dic[e_model_name] = model_dict[e_model_name]
    #     elif '-rf-' in e_model_name:
    #         rf_dic[e_model_name] = model_dict[e_model_name]
    #     elif '-gbdt-' in e_model_name:
    #         gbdt_dic[e_model_name] = model_dict[e_model_name]

    # dart_top3 = sorted(dart_dic.items(), key = lambda x:x[1],reverse=True)[:]
    # rf_top3 = sorted(rf_dic.items(),key = lambda x:x[1],reverse=True)[:]
    # gbdt_top3 = sorted(gbdt_dic.items(),key = lambda x:x[1],reverse=True)[:]
    
    # top3_name = [e[0] for e in dart_top3] + [e[0] for e in rf_top3] + [e[0] for e in gbdt_top3]
    # top3_score = [model_dict[e] for e in top3_name]
    # top3_weight = np.exp(top3_score) / np.exp(top3_score).sum()
    # top3_weight = top3_weight.tolist()
    
    y = 0#初始化
    for e_model_name,e_weight in zip(top3_name,top3_weight):
        bst = lgb.Booster(model_file=e_model_name)  # init model
        pred_y = bst.predict(X_df.values)# 预测的y
        y += pred_y*e_weight
    
    # y = pd.Series(0,index=X_df.index)
    data ={'id':id_list,'y':y}
    data =pd.DataFrame(data)
    return data


def get_id(date_id_srs,security_id_srs):
    return date_id_srs.astype("str") + "d" +security_id_srs.astype("str")

def df_rolling(df,n,id_all,shape=True,method="mean",quantile=0.5):
    # df所有特征n日时序平均或标准差或分位数，shape表示是否按id_all格式化；df可以是srs
    df = pd.DataFrame(df)
    df["date_id"] = pd.Series(df.index).apply(lambda x:x.split("d")[0]).values
    df["security_id"] = pd.Series(df.index).apply(lambda x:x.split("d")[1]).values
    if method=="mean":
        df_res = df.set_index("date_id").groupby("security_id").rolling(n,min_periods=1).mean().reset_index()
    elif method=="std":
        df_res = df.set_index("date_id").groupby("security_id").rolling(n,min_periods=1).std().reset_index()
    elif method=="quantile":
        df_res = df.set_index("date_id").groupby("security_id").rolling(n,min_periods=1).quantile(quantile).reset_index()
    else:
        print("Wrong method.")
        return None
    if "id" not in df_res.columns:
        df_res["id"] = get_id(df_res["date_id"],df_res["security_id"])
    df_res = df_res.drop(columns=['date_id','security_id']).set_index("id").fillna(0)
    if shape:
        df_res = pd.DataFrame(df_res,index=id_all,columns=df_res.columns).fillna(0)
    return df_res

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
        comp_point = (fac_list[i] * sign_df).T.rolling(n, min_periods=min_periods).mean().T
        fac = fac + (fac_list[i] - comp_point)**2
    fac = np.sqrt(fac)
    return fac

def ma(df,n):
    # n日滑动平均
    return df.T.rolling(n,min_periods=1).mean().T

def ewma(df,n):
    # n日指数滑动平均
    return df.T.ewm(n,min_periods=1).mean().T

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