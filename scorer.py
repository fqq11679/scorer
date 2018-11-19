import numpy as np
import pandas as pd
from datetime import datetime, date
from sklearn import model_selection

# read the market data
df = pd.read_csv('/root/stock_data/market_data/m001.csv')

for i in range(2,42):
 print(i)
 df_tmp = pd.read_csv('/root/stock_data/market_data/m'+"%03d"%i+'.csv')
 df = df.append(df_tmp)

Market_train_df = df.drop(["Unnamed: 0"],axis=1)
market_train_df = Market_train_df

# read the news data
df = pd.read_csv('/home/share_data/news_data/n001.csv')

for i in range(2,84):
 print(i)
 df_tmp = pd.read_csv('/home/share_data/news_data/n'+"%03d"%i+'.csv')
 df = df.append(df_tmp)

print(df)
for i in range(84,95):
 print(i)
 df_tmp = pd.read_csv('/home/share_data/news_data2/n'+"%03d"%i+'.csv')
 df = df.append(df_tmp)

news_train = df.drop(["Unnamed: 0"],axis=1)

market_train_df['time'] = pd.to_datetime(market_train_df['time'], errors='coerce')
news_train['time'] = pd.to_datetime(news_train['time'], errors='coerce')

# cite sources
import time
import datetime
import gc
gc.enable()

# multiply the values by the relevance
def update_news(df):
    df['sentimentNegative'] = df['sentimentNegative'] * df['relevance']
    df['sentimentNeutral'] = df['sentimentNeutral'] * df['relevance']
    df['sentimentPositive'] = df['sentimentPositive'] * df['relevance']
    return df
news_train = update_news(news_train)

# 
import warnings
import datetime
warnings.filterwarnings(action ='ignore',category = DeprecationWarning)

print(news_train.time)
news_train['time'] = news_train['time'].dt.floor('d')

cols = ['sentimentNegative','sentimentNeutral','sentimentPositive']
def get_news_train(raw_data,days = 2):
    news_last = pd.DataFrame()
    for i in range(days):
        cur_train = raw_data[cols]
        cur_train['time'] = raw_data['time'] + datetime.timedelta(days = i,hours=22)
        cur_train['key'] = cur_train['time'].astype(str)+ raw_data['assetName'].astype(str)
        news_last = pd.concat([news_last, cur_train[['key'] + cols]])
        print("after concat the shape is:",news_last.shape)
        news_last = news_last.groupby('key').sum()
        news_last['key'] = news_last.index.values
        print("the result shape is:",news_last.shape)
        del cur_train
        gc.collect()
    del news_last['key']
    return news_last
news_last = get_news_train(news_train)
print(news_last.shape)
print(news_last.head())
print(news_last.dtypes)

# 
market_train_df['key'] = market_train_df['time'].astype(str) + market_train_df['assetName'].astype(str)
market_train_df = market_train_df.join(news_last,on = 'key',how='left')
print(market_train_df['sentimentNeutral'].isnull().value_counts())
market_train_df

#
def fill_in(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64" or data[i].dtype == "float32"):
            data[i] = data[i].fillna(data[i].mean())
        else:
            pass
    return data

market_train_df = fill_in(market_train_df)

#
def data_other(market_train_df):
    market_train_df.time = market_train_df.time.dt.date
    lbl = {k: v for v, k in enumerate(market_train_df['assetCode'].unique())}
    market_train_df['assetCodeT'] = market_train_df['assetCode'].map(lbl)
    
    #market_train_df = market_train_df.dropna(axis=0)
    
    return market_train_df

market_train_df = data_other(market_train_df)

#
market_train_df = market_train_df.loc[market_train_df['time']>=date(2009, 1, 1)]

#
green = market_train_df.returnsOpenNextMktres10 > 0
green = green.values

#
fcol = [c for c in market_train_df if c not in ['time', 'assetCode', 'assetName', 'returnsOpenNextMktres10', 'universe', 
                                             'key']]

#
X = market_train_df[fcol].values
mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
rng = maxs - mins
X = 1 - ((maxs - X) / rng)

#
all_10day = market_train_df.returnsOpenNextMktres10.values

#
X_train, X_test, green_train, green_test, all_train, all_test = model_selection.train_test_split(
    X, green, all_10day, test_size=0.20, random_state=59)

#
import lightgbm as lgb
train_data = lgb.Dataset(X_train, label=green_train.astype(int))
test_data = lgb.Dataset(X_test, label=green_test.astype(int))

#
# these are tuned params I found
x_1 = [0.19000424246380565, 2452, 212, 328, 202]
x_2 = [0.19016805202090095, 2583, 213, 312, 220]

params_1 = {
        'task': 'train',
        'boosting_type': 'dart',
        'objective': 'binary',
        'learning_rate': x_1[0],
        'num_leaves': x_1[1],
        'min_data_in_leaf': x_1[2],
        'num_iteration': x_1[3],
        'max_bin': x_1[4],
        'verbose': 1
    }

params_2 = {
        'task': 'train',
        'boosting_type': 'dart',
        'objective': 'binary',
        'learning_rate': x_2[0],
        'num_leaves': x_2[1],
        'min_data_in_leaf': x_2[2],
        'num_iteration': x_2[3],
        'max_bin': x_2[4],
        'verbose': 1
    }


gbm_1 = lgb.train(params_1,
        train_data,
        num_boost_round=100,
        valid_sets=test_data,
        early_stopping_rounds=5)
        
gbm_2 = lgb.train(params_2,
        train_data,
        num_boost_round=100,
        valid_sets=test_data,
        early_stopping_rounds=5)

# read the groundtruth data
df = pd.read_csv('/root/whole_data/full_data1.csv')
for i in range(2,54):
 print("reading groundtruth")
 print(i)
 df_tmp = pd.read_csv('/root/whole_data/full_data'+"%d"%i+'.csv')
 df = df.append(df_tmp)  
Groundtruth = df.drop(["Unnamed: 0"],axis=1)
gcol = ['time', 'returnsOpenNextMktres10', 'universe']
groundtruth = Groundtruth[gcol]
groundtruth['time'] = pd.to_datetime(groundtruth['time'], errors='coerce')
print(groundtruth)

# read observation data
# read the market data
df = pd.read_csv('/home/share_data/obs_data/mkt_obs_data1.csv')

for i in range(2,13):
 print(i)
 df_tmp = pd.read_csv('/home/share_data/obs_data/mkt_obs_data'+"%d"%i+'.csv')
 df = df.append(df_tmp)

Market_obs_df = df.drop(["Unnamed: 0"],axis=1)
market_obs_df = Market_obs_df 

#read the news data
df = pd.read_csv('/home/share_data/obs_data/news_obs_df1.csv')

for i in range(2,24):
 print(i)
 df_tmp = pd.read_csv('/home/share_data/obs_data/news_obs_df'+"%d"%i+'.csv')
 df = df.append(df_tmp)

News_obs_df = df.drop(["Unnamed: 0"],axis=1)
news_obs_df = News_obs_df

##process the obs data
news_obs_df = update_news(news_obs_df)

market_obs_df['time'] = pd.to_datetime(market_obs_df['time'], errors='coerce')
news_obs_df['time'] = pd.to_datetime(news_obs_df['time'], errors='coerce')


news_obs_df['time'] = news_obs_df['time'].dt.floor('d')
news_last = get_news_train(news_obs_df)
market_obs_df['key'] = market_obs_df['time'].astype(str) + market_obs_df['assetName'].astype(str)
market_obs_df = market_obs_df.join(news_last,on = 'key',how='left')
market_obs_df = fill_in(market_obs_df)
market_obs_df = data_other(market_obs_df)
X_live = market_obs_df[fcol].values
X_live = 1 - ((maxs - X_live) / rng)
lp = (gbm_1.predict(X_live) + gbm_2.predict(X_live))/2
confidence = lp
confidence = (confidence-confidence.min())/(confidence.max()-confidence.min())

for i in range(len(confidence)):
   if confidence[i] < 0.47:
      confidence[i] = -1.0
   elif confidence[i] > 0.52:
      confidence[i] = 1.0
   else:
      confidence[i] = 0.0

print(confidence)
print(len(confidence))

groundtruth = groundtruth[(groundtruth['time'] >= '2017-01-01')]
print(len(groundtruth.time.values))

groundtruth['yhat'] = confidence
#
#groundtruth['yhat'] = 1.0
groundtruth['returns'] = groundtruth.yhat * groundtruth.returnsOpenNextMktres10

groundtruth = groundtruth[groundtruth['universe'] == 1.0]
#stage 1
#groundtruth = groundtruth[(groundtruth['time'] < '2018-07-01')]
#print(max(groundtruth.time.values))
#
day_returns = groundtruth.groupby('time').returns.sum()
score = day_returns.mean() / day_returns.std()
print(score)


