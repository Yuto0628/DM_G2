import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from sklearn.model_selection import train_test_split
pd.set_option('display.float_format', lambda x:'%.5f' % x)
import numpy as np

#pushのテスト aaaaaaaaaaaaaaaaa

# データタイプを指定
types_dict_train = {'train_id':'int64', 'item_condition_id':'int8', 'price':'float64', 'shipping':'int8'}
types_dict_test = {'test_id':'int64', 'item_condition_id':'int8', 'shipping':'int8'}
 
# tsvファイルからPandas DataFrameへ読み込み
train = pd.read_csv('~/Downloads/mercari-price-suggestion-challenge//train.tsv', delimiter='\t', low_memory=True, dtype=types_dict_train)
test = pd.read_csv('~/Downloads/mercari-price-suggestion-challenge//test.tsv', delimiter='\t', low_memory=True, dtype=types_dict_test)

mini_train = train[train['train_id'] < 10000]
mini_test = train[(train['train_id'] >= 10000) & (train['train_id'] < 20000)]

#print(mini_train.head())
#print(mini_test.head())
print(mini_train.shape)
print(mini_test.shape)

# mini_trainのカテゴリ名、商品説明、投稿タイトル、ブランド名のデータタイプを「category」へ変換する
mini_train.category_name = train.category_name.astype('category')
mini_train.item_description = train.item_description.astype('category')
mini_train.name = train.name.astype('category')
mini_train.brand_name = train.brand_name.astype('category')

# 両方のセットへ「is_train」のカラムを追加
# 1 = trainのデータ、0 = testデータ
mini_train['is_train'] = 1
mini_test['is_train'] = 0

# mini_trainのprice(価格）以外のデータをmini_testのprice以外のデータと連結
mini_train_test_combine = pd.concat([mini_train.drop(['price'], axis=1), mini_test.drop(['price'], axis=1)], axis=0 )

# mini_train_test_combineの文字列のデータタイプを「category」へ変換
mini_train_test_combine.category_name = mini_train_test_combine.category_name.astype('category')
mini_train_test_combine.item_description = mini_train_test_combine.item_description.astype('category')
mini_train_test_combine.name = mini_train_test_combine.name.astype('category')
mini_train_test_combine.brand_name = mini_train_test_combine.brand_name.astype('category')

# combinedDataの文字列を「.cat.codes」で数値へ変換する
mini_train_test_combine.name = mini_train_test_combine.name.cat.codes
mini_train_test_combine.category_name = mini_train_test_combine.category_name.cat.codes
mini_train_test_combine.brand_name = mini_train_test_combine.brand_name.cat.codes
mini_train_test_combine.item_description = mini_train_test_combine.item_description.cat.codes

# 「is_train」のフラグでcombineからtestとtrainへ切り分ける
df_mini_test = mini_train_test_combine.loc[mini_train_test_combine['is_train'] == 0]
df_mini_train = mini_train_test_combine.loc[mini_train_test_combine['is_train'] == 1]

# 「is_train」をtrainとtestのデータフレームから落とす
df_mini_test = df_mini_test.drop(['is_train'], axis=1)
df_mini_train = df_mini_train.drop(['is_train'], axis=1)

# サイズの確認をしておきましょう
print(df_mini_test.shape, df_mini_train.shape)

# df_trainへprice（価格）を戻す
df_mini_train['price'] = mini_train.price
#df_mini_test['price'] = mini_test.price

# price（価格）をlog関数で処理
df_mini_train['price'] = df_mini_train['price'].apply(lambda x: np.log(x) if x>0 else x)
#f_mini_test['price'] = df_mini_test['price'].apply(lambda x: np.log(x) if x>0 else x)

# x ＝ price以外の全ての値、y = price（ターゲット）で切り分ける
x_mini_train, y_mini_train = df_mini_train.drop(['price'], axis=1), df_mini_train.price

# モデルの作成
m = RandomForestRegressor(n_jobs=-1, min_samples_leaf=5, n_estimators=200)
m.fit(x_mini_train, y_mini_train)

# スコアを表示
print(m.score(x_mini_train, y_mini_train))

# 作成したランダムフォレストのモデル「m」に「df_test」を入れて予測する
preds = m.predict(df_mini_test)

# Numpy配列からpandasシリーズへ変換
preds = pd.Series(np.exp(preds))

#testのprice(価格)の予測値と実際のtestのprice(価格)を連結
pred_test_price_combine = pd.concat([mini_test.price, preds], axis=1)

pred_test_price_combine['pred_price'] = preds
pred_test_price_combine['real_price'] = mini_test.price
pred_test_price_combine['result'] = (pred_test_price_combine['pred_price'] == pred_test_price_combine['real_price'])

print(pred_test_price_combine)

'''
#pred_test_price_combineの列名を最初の名前から順にpred_price(予測値),real_price(実価格)に変更
pred_test_price_combine = pred_test_price_combine.set_axis(['pred_price', 'real_price'], axis=1)

#pred_test_price_combineにpred_priceとreal_priceを比較して同じかどうか(trueかfalse)を格納した列(result)を追加
pred_test_price_combine['result'] = (pred_test_price_combine['pred_price'] == pred_test_price_combine['real_price'])

#result(predとrealの比較)をint型にキャスト
pred_test_price_combine['result'] = pred_test_price_combine['result'].astype(int)

accuracy = pred_test_price_combine['result'].sum()/10000

print(accuracy)
print(pred_test_price_combine)
'''

"""
# trainとtestのデータフレームの冒頭5行を表示させる
# print(train.head())
# print(test.head())
 
# trainとtestのサイズを確認
#print(train.shape)
#print(test.shape)


def display_all(df):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns", 1000):
            display(df)
 
# trainの基本統計量を表示
# display_all(train.describe(include='all').transpose())


# trainのカテゴリ名、商品説明、投稿タイトル、ブランド名のデータタイプを「category」へ変換する
train.category_name = train.category_name.astype('category')
train.item_description = train.item_description.astype('category')
train.name = train.name.astype('category')
train.brand_name = train.brand_name.astype('category')
 
# dtypesで念のためデータ形式を確認しましょう
train.dtypes, test.dtypes


# trainの中のユニークな値を確認する
#train.apply(lambda x: x.nunique())
 
# testの中のユニークな値を確認する
#test.apply(lambda x: x.nunique())


# trainの欠損データの個数と%を確認
#train.isnull().sum(),train.isnull().sum()/train.shape[0]
 
# testの欠損データの個数と%を確認
##test.isnull().sum(),test.isnull().sum()/test.shape[0]


# trainとtestのidカラム名を変更する
train = train.rename(columns = {'train_id':'id'})
test = test.rename(columns = {'test_id':'id'})
 
# 両方のセットへ「is_train」のカラムを追加
# 1 = trainのデータ、0 = testデータ
train['is_train'] = 1
test['is_train'] = 0
 

# trainのprice(価格）以外のデータをtestと連結
train_test_combine = pd.concat([train.drop(['price'], axis=1),test],axis=0)
 
# 念のためデータの中身を表示させましょう
#train_test_combine.head()


# train_test_combineの文字列のデータタイプを「category」へ変換
train_test_combine.category_name = train_test_combine.category_name.astype('category')
train_test_combine.item_description = train_test_combine.item_description.astype('category')
train_test_combine.name = train_test_combine.name.astype('category')
train_test_combine.brand_name = train_test_combine.brand_name.astype('category')
 
# combinedDataの文字列を「.cat.codes」で数値へ変換する
train_test_combine.name = train_test_combine.name.cat.codes
train_test_combine.category_name = train_test_combine.category_name.cat.codes
train_test_combine.brand_name = train_test_combine.brand_name.cat.codes
train_test_combine.item_description = train_test_combine.item_description.cat.codes
 
# データの中身とデータ形式を表示して確認しましょう
#train_test_combine.head()
#train_test_combine.dtypes


# 「is_train」のフラグでcombineからtestとtrainへ切り分ける
df_test = train_test_combine.loc[train_test_combine['is_train'] == 0]
df_train = train_test_combine.loc[train_test_combine['is_train'] == 1]
 
# 「is_train」をtrainとtestのデータフレームから落とす
df_test = df_test.drop(['is_train'], axis=1)
df_train = df_train.drop(['is_train'], axis=1)
 
# サイズの確認をしておきましょう
#print(df_test.shape, df_train.shape)
 
#((693359, 7), (1482535, 7))


# df_trainへprice（価格）を戻す
df_train['price'] = train.price
 
# price（価格）をlog関数で処理
df_train['price'] = df_train['price'].apply(lambda x: np.log(x) if x>0 else x)
 
# df_trainを表示して確認
# df_train.head()


# x ＝ price以外の全ての値、y = price（ターゲット）で切り分ける
x_train, y_train = df_train.drop(['price'], axis=1), df_train.price
 
# モデルの作成
m = RandomForestRegressor(n_jobs=-1, min_samples_leaf=5, n_estimators=200)
m.fit(x_train, y_train)
 
# スコアを表示
print(m.score(x_train, y_train))


# 作成したランダムフォレストのモデル「m」に「df_test」を入れて予測する
preds = m.predict(df_test)
 
# 予測値 predsをnp.exp()で処理
np.exp(preds)
 
# Numpy配列からpandasシリーズへ変換
preds = pd.Series(np.exp(preds))
 
# テストデータのIDと予測値を連結
submit = pd.concat([df_test.id, preds], axis=1)
 
# カラム名をメルカリの提出指定の名前をつける
submit.columns = ['test_id', 'price']
 
# 提出ファイルとしてCSVへ書き出し
submit.to_csv('submit_rf_base.csv', index=False)
"""

