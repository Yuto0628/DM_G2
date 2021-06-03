import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from sklearn.model_selection import train_test_split
pd.set_option('display.float_format', lambda x:'%.5f' % x)
import numpy as np
from sklearn import svm,linear_model 


# データタイプを指定
types_dict_train = {'train_id':'int64', 'item_condition_id':'int8', 'price':'float64', 'shipping':'int8'}
types_dict_test = {'test_id':'int64', 'item_condition_id':'int8', 'shipping':'int8'}


# tsvファイルからPandas DataFrameへ読み込み(冒頭2万行)
train = pd.read_csv('~/Downloads/mercari-price-suggestion-challenge//train.tsv', delimiter='\t', low_memory=True, dtype=types_dict_train, nrows=20000)
# test = pd.read_csv('test.tsv', delimiter='\t', low_memory=True, dtype=types_dict_test, nrows=2000)


#文字列をカテゴリ型に変換し、数値に変換する
train['name'] = train['name'].astype('category').cat.codes
train['item_description'] = train['item_description'].astype('category').cat.codes
train['brand_name'] = train['brand_name'].astype('category').cat.codes
train['category_name'] = train['category_name'].astype('category').cat.codes


# priceを目的変数に設定
y = train['price']
x = train.drop(columns='price')


# 半分を訓練用データに、残り半分をテスト用データにわける
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, shuffle=False)


# ランダムフォレスト
m = RandomForestRegressor(n_jobs=-1, min_samples_leaf=5, n_estimators=200)
m.fit(x_train, y_train)
print("ランダムフォレスト: "+ str(m.score(x_train, y_train)))

# リッジ回帰
clf_ridge= linear_model.Ridge(alpha=1.0)
clf_ridge.fit(x_train, y_train)
print("リッジ回帰: "+ str(clf_ridge.score(x_train, y_train)))

# 線形回帰
reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
print("線形回帰: " + str(reg.score(x_train, y_train)))


