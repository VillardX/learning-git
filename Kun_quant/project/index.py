import pandas as pd
import random



def invoke(data1,id_list) -> pd.DataFrame():
    #data1: pd.DataFrame, 数据来自于 main chart price.parquet
    #id_list: T当日不同的id 值, 用于结果数据的预测
    y = [random.random()for i in range(len(id_list))]
    
    data ={'id':id_list,'y':y}
    data =pd.DataFrame(data)

    return data
