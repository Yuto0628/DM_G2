import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from sklearn.model_selection import train_test_split
pd.set_option('display.float_format', lambda x:'%.5f' % x)
import numpy as np
from matplotlib import pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
import nltk


# データタイプを指定
types_dict_train = {'train_id':'int64', 'item_condition_id':'int8', 'price':'float64', 'shipping':'int8'}
types_dict_test = {'test_id':'int64', 'item_condition_id':'int8', 'shipping':'int8'}
 
# tsvファイルからPandas DataFrameへ読み込み
dataset = pd.read_csv('~/Downloads/mercari-price-suggestion-challenge//train.tsv', delimiter='\t', low_memory=True, dtype=types_dict_train)


#----今回直したい処理--一旦stemminfの確認のためにitem_descriptionだけを使用する----------------------
dataset = dataset.drop(["train_id", "category_name", "name", "brand_name"], axis=1)


def noun_stem_analyzer(string):
    st = nltk.stem.lancaster.LancasterStemmer()
    return [st.stem(word) for word, pos in nltk.pos_tag(
        nltk.word_tokenize(string)) if pos == "NN" or pos == "NNP"]


vect = CountVectorizer(analyzer=noun_stem_analyzer, min_df=0.25, max_df=0.75, stop_words="english")

vect.fit(mini_data.item_description)

dtm = vect.transform(mini_data.item_description)
dtm = dtm.toarray()

pd_dtm = pd.DataFrame(dtm, columns=vect.get_feature_names())

print(pd_dtm.head())

#-------------------------------------------------------------------------------------

# price（価格）をlog関数で処理
dataset['price'] = dataset['price'].apply(lambda x: np.log(x) if x>0 else x)

#trainとtestに分割
train, test = train_test_split(dataset, test_size=0.3, random_state=0)

# x ＝ price以外の全ての値、y = price（ターゲット）で切り分ける
x_test, y_test = test.drop(['price'], axis=1), test.price
x_train, y_train = train.drop(['price'], axis=1), train.price

# モデルの作成
m = RandomForestRegressor(n_jobs=-1, min_samples_leaf=5, n_estimators=200)
m.fit(x_train, y_train)

# スコアを表示
print(m.score(x_train, y_train))

# 作成したランダムフォレストのモデル「m」に「df_test」を入れて予測する
preds = m.predict(x_test)

#test
print('test:平均絶対誤差=', metrics.mean_absolute_error(y_test, preds))
print('score:テストのスコア=', m.score(x_test, y_test))

#実際のデータと予測データをplotする
#--------------------------------

# グラフエリアを設定し、散布図を描く。
plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds)

# yの最大値、最小値を計算する。
y_max = np.max(y_test)
y_min = np.min(y_test)

# y_testの最大値、最小値を計算する。
preds_max = np.max(preds)
preds_min = np.min(preds)

# 全てのプロットが収まるようにするには、yとpredict_y両方のうち
# 最も小さい値、最も大きい値を縦軸横軸の範囲にすればいい。
axis_max = max(y_max, preds_max)
axis_min = min(y_min, preds_min)

plt.xlim(axis_min, axis_max)
plt.ylim(axis_min, axis_max)

plt.xlabel('y_test')
plt.ylabel('preds')

plt.show()