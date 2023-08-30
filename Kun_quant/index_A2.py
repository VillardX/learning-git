# 模型A
import pandas as pd 
import numpy as np
import tqdm
import random
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import json

def invoke(dataA, dataB, id_list) -> pd.DataFrame():
    forum_a = dataA
    forum_b = dataB
    forum_a["id"] = get_id(forum_a["date_id"],forum_a["security_id"])
    forum_b["id"] = get_id(forum_b["date_id"],forum_b["security_id"])

    # A、B和id_list的所有id，这里id_list是
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

    fa_features = softmax(forum_a_used.loc[id_list])  # A表特征
    fb_features = softmax(fb_all_used[2].loc[id_list])+softmax(fb_all_used[4].loc[id_list])
    fa_features.rename({
        'senti_conform':'fa_senti_conform',
        'senti_score_div':'fa_senti_score_div',
        'senti_score_log':'fa_senti_score_log'
    },inplace=True,axis=1)
    fb_features.rename({
        'senti_conform':'fb_senti_conform',
        'senti_score_div':'fb_senti_score_div',
        'senti_score_log':'fb_senti_score_log'
    },inplace=True,axis=1)
    # 生成X_df
    X_df = pd.concat([fa_features,fb_features],axis=1).loc[id_list]

    # '''
    # 此处添加模型预测，修改下面的预测y
    # '''
    
    # coef = {'mom1': 0.5104290803752711,'vol_diff': 0.0004935505241692537,'go_hl': -0.06096307747547033,'rr_demean': 0.9102674676772314,
    #         'valc': 0.03943196404135161,'fac_sk': 0.047715724516717246,'fac_nonsym': 0.010118186196143778}
    # y = pd.Series(0,index=X_df.index)
    # for key in coef.keys():
    #     y = y + X_df.loc[:,key].fillna(0)*coef[key]
    
    #集成
    with open('model_dict.json','r') as fm:
        model_dict = json.load(fm)
        
    #分base筛选
    gbdt_dic={}
    rf_dic={}
    dart_dic={}
    
    for e_model_name in model_dict.keys():
        if '-dart-' in e_model_name:
            dart_dic[e_model_name] = model_dict[e_model_name]
        elif '-rf-' in e_model_name:
            rf_dic[e_model_name] = model_dict[e_model_name]
        elif '-gbdt-' in e_model_name:
            gbdt_dic[e_model_name] = model_dict[e_model_name]

    dart_top3 = sorted(dart_dic.items(), key = lambda x:x[1],reverse=True)[:3]
    rf_top3 = sorted(rf_dic.items(),key = lambda x:x[1],reverse=True)[:3]
    gbdt_top3 = sorted(gbdt_dic.items(),key = lambda x:x[1],reverse=True)[:3]
    
    top3_name = [e[0] for e in dart_top3] + [e[0] for e in rf_top3] + [e[0] for e in gbdt_top3]
    top3_score = [model_dict[e] for e in top3_name]
    top3_weight = np.exp(top3_score) / np.exp(top3_score).sum()
    top3_weight = top3_weight.tolist()
    
    y = 0#初始化
    for e_model_name,e_weight in zip(top3_name,top3_weight):
        bst = lgb.Booster(model_file=e_model_name)  # init model
        pred_y = bst.predict(X_df.values)# 预测的y
        y += pred_y*e_weight
    
    data ={'id':id_list,'y':y}        
    # data ={'id':id_list,'y':y.replace(0,np.nan).values}  
    data =pd.DataFrame(data)
    return data

def get_id(date_id_srs,security_id_srs):
    return date_id_srs.astype("str") + "d" +security_id_srs.astype("str")

def softmax(df):
    return 1/(1+np.exp(-df))

def cos2pi(df):
    return np.cos(2*np.pi*df)

def sin2pi(df):
    return np.sin(2*np.pi*df)

def df_rolling(df,n,id_all,shape=True,method="mean",quantile=0.5):
    # df所有特征n日时序平均或标准差或分位数，shape表示是否按id_all格式化
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
    df_res = df_res.drop(columns=['date_id','security_id','relevant_type']).set_index("id").fillna(0)
    if shape:
        df_res = pd.DataFrame(df_res,index=id_all,columns=df_res.columns).fillna(0)
    return df_res
