import pandas as pd 
import numpy as np
import tqdm
import os
import gc
import xgboost as xgb
import lightgbm as lgb
import pickle
from sklearn.model_selection import train_test_split
import json 

with open('../temp/X_df.pkl','r') as f:
    X_df = pd.read_pickle(f)
    
    
    
price_df = pd.read_parquet('../input/train_8857/price.parquet')
y_df = price_df['y(label)']
y_df.index = price_df.id
X_train, X_test, Y_train, Y_test = train_test_split(X_df, y_df.fillna(0), test_size=0.1, random_state=2)


#自定义metric
def corr(preds, train_data):
    '''
    :param preds:  array, 预测值
    :param train_data:  lgb Dataset, lgb的传入数据集
    :return:
    '''
    labels=train_data.get_label()
    cor = pd.Series(preds).corr(pd.Series(labels.reshape(-1)))
    ###  返回 (评估指标名称, 评估计算值, 是否评估值越大模型性能越好)
    return 'correlation', cor, True

#自定义metric
def corr4combine(preds, y_test):
    '''
    用于评估当前模型在val上的分数
    :param preds:  array, 预测值
        :return:
    '''
    cor = pd.Series(preds).corr(pd.Series((y_test.values.reshape(-1))))
    ###  返回 (评估指标名称, 评估计算值, 是否评估值越大模型性能越好)
    return cor


#自定义metric
def avg_date_corr(preds, train_data):
    '''
    :param preds:  array, 预测值
    :param train_data:  lgb Dataset, lgb的传入数据集，free_raw_data设为False，所以label还有对应的id
    :return:
    '''
    labels=train_data.get_label()#一个np.array
    idx = train_data.get_data().index#索引，类似于965d30256，d前为日，d后为标的代码
    
    tmp_df = pd.DataFrame()
    tmp_df['idx'] = idx
    tmp_df['y'] = labels
    tmp_df['pred'] = preds

    tmp_df['date'] = tmp_df['idx'].apply(lambda x: x.split('d')[0])
    tmp_df = tmp_df.drop(columns='idx')
    
    each_corr = tmp_df.groupby('date').apply(lambda x: x['pred'].corr(x['y']))#分date计算所有标的实际收益率与预测收益率的相关系数,id为date
    avg_date_corr = each_corr.mean()

    # cor = pd.Series(preds).corr(pd.Series(labels.reshape(-1)))
    ###  返回 (评估指标名称, 评估计算值, 是否评估值越大模型性能越好)
    # return 'correlation', cor, True
    return 'avg_date_correlation',avg_date_corr,True

def avg_date_corr4combine(preds, y_test):
    '''
    用于评估当前模型在val上的分数
    :param preds:  array, 预测值
        :return:
    '''
    idx = y_test.index#索引，类似于965d30256，d前为日，d后为标的代码
    tmp_df = pd.DataFrame()
    tmp_df['idx'] = idx
    tmp_df['y'] = y_test.values
    tmp_df['pred'] = preds

    tmp_df['date'] = tmp_df['idx'].apply(lambda x: x.split('d')[0])
    tmp_df = tmp_df.drop(columns='idx')
    each_corr = tmp_df.groupby('date').apply(lambda x: x['pred'].corr(x['y']))#分date计算所有标的实际收益率与预测收益率的相关系数,id为date
    avg_date_corr = each_corr.mean()
    return avg_date_corr


def mkdir_model_dict(dir_name='modelB_dict.json'):
    '''
        判断是否有model_dict.json，如果没有，构建一个，并返回，否则读取并返回
    '''
    if not os.path.exists(dir_name):
        print('文件名{}不存在，故创建一个'.format(dir_name))
        return {}
    else:
        with open(dir_name,'r') as fm:
            model_dict = json.load(fm)
        return model_dict

#初始化字典
model_dict = mkdir_model_dict()


train_data = lgb.Dataset(X_train, label=Y_train)
val_data = lgb.Dataset(X_test,label=Y_test)

param_list = {
    'objective': 'regression',#后面的参数都改成列表
    'verbose':1,
    'num_leaves': 31,   
    'seed':42,
    'early_stopping_round':200,#20
    'boosting':['gbdt','dart','rf'],
    'num_iterations':[5],#[2000,1000,500]
    'eta':[1e-2],#[5e-3,1e-2,2e-2]  
    'max_depth':[7],#[4,5,6]
    'num_leaves':[64],#[8,16,32]
    'bagging_fraction':[0.8],#[0.8,0.9,1]
    'bagging_freq':[10],#[5,10,20]
    'feature_fraction':[0.8],#[0.8,0.9,0.95]
}

#先根据param_list生成当前要训练模型的超参数
for e_boosting in param_list['boosting']:
    for e_num_iterations,e_eta in zip(param_list['num_iterations'],param_list['eta']):
        for e_maxd,e_maxnleafs in zip(param_list['max_depth'],param_list['num_leaves']):
            for e_bfrac,e_bfreq,e_ffrac in zip(param_list['bagging_fraction'],param_list['bagging_freq'],param_list['feature_fraction']):
                tmp_name = 'modelA-{}-niter{}-eta{}-maxdep{}-maxleaf{}-bfrac{}-bfreq{}-ffrac{}.txt'.format(e_boosting,e_num_iterations,e_eta,e_maxd,e_maxnleafs,e_bfrac,e_bfreq,e_ffrac)
                if tmp_name in model_dict.keys():
                    print('模型{}已训练过，故跳过'.format(tmp_name))
                    continue#再check当前model_dict是否包含上述超参数的模型，如果有就跳过，没有就训练
                p = {'objective': 'regression',#后面的参数都改成列表
                    'verbose':1,
                    'num_leaves': 31,   
                    'seed':42,
                    'early_stopping_round':50,#20轮不再进展那就早停
                    'boosting':e_boosting,
                    'num_iterations':e_num_iterations,
                    'eta':e_eta,  
                    'max_depth':e_maxd,
                    'num_leaves':e_maxnleafs,
                    'bagging_fraction':e_bfrac,
                    'bagging_freq':e_bfreq,
                    'feature_fraction':e_ffrac,
                    } 
                try:#bagging_frac=1时rf会报错，那就跳过
                    tmp_model = lgb.train(p, train_data, feval=corr,valid_sets=[train_data,val_data])#data要设置一下
                    # train_score,val_score#出存模型以及效果
                    each_preds = tmp_model.predict(X_test)
                    each_cor = avg_date_corr4combine(each_preds,Y_test)#在测试集上的性能
                    print('结束')
                    model_dict[tmp_name] = each_cor#加入模型字典
                    tmp_model.save_model(tmp_name)#储存模型
                    # 储存json
                    with open('modelB_dict.json', 'w') as f:
                        json.dump(model_dict, f)  # 会在目录下生成一个.json的文件，文件内容是dict数据转成的json数据
                except:
                    print('模型{}报错，故跳过'.format(tmp_name))
                    continue