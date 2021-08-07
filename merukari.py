import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from sklearn.model_selection import train_test_split
pd.set_option('display.float_format', lambda x:'%.5f' % x)
import numpy as np
from matplotlib import pyplot as plt

from sklearn.feature_extraction import text
import nltk

#文書から名詞と固有名詞のみ取り出してステミング
def noun_stem_analyzer(string):
    string.replace('/', ' ')
    st = nltk.stem.lancaster.LancasterStemmer()
    return [st.stem(word) for word, pos in nltk.pos_tag(
        nltk.word_tokenize(string)) if pos == "NN" or pos == "NNP"]



def features_convert_BoW(data_frame, min_df, max_df):
    '''
    特徴量をBoWに変換する関数

    params:
    ---
    data_frame: pd.DataFrame(dtype=string)
        BoWに変換したい特徴量1列分。

    min_df: float
        出現頻度の下限値。BoWに変換した際にmin_df以下の出現頻度の単語をベクトルから省く。

    min_df: float
        出現頻度の上限値。BoWに変換した際にmin_df以上の出現頻度の単語をベクトルから省く。

    return:
    ---
    pd_dtm: pd.DataFrame
        BoWに変換したDataFrame
    '''

    stopword = text.ENGLISH_STOP_WORDS.union('/')

    vect = text.CountVectorizer(analyzer=noun_stem_analyzer, min_df=min_df, max_df=max_df, stop_words=stopword)

    data_frame = data_frame.fillna(' ')

    vect.fit(data_frame)

    dtm = vect.transform(data_frame)
    dtm = dtm.toarray()

    pd_dtm = pd.DataFrame(dtm, columns=vect.get_feature_names())

    return pd_dtm


#予測データと実際のデータをplotする
def plot_graph(pred_prices, real_prices):
    # グラフエリアを設定し、散布図を描く。
    plt.figure(figsize=(6, 6))
    plt.scatter(real_prices, pred_prices)

    # 実際値の最大値、最小値を計算する。
    y_max = np.max(real_prices)
    y_min = np.min(real_prices)

    # 予測値の最大値、最小値を計算する。
    preds_max = np.max(pred_prices)
    preds_min = np.min(pred_prices)

    # 全てのプロットが収まるようにするには、yとpredict_y両方のうち
    # 最も小さい値、最も大きい値を縦軸横軸の範囲にすればいい。
    axis_max = max(y_max, preds_max)
    axis_min = min(y_min, preds_min)

    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)

    plt.xlabel('real_price')
    plt.ylabel('pred_price')

    plt.show()


if __name__ == "__main__":
    # データタイプを指定
    types_dict_train = {'train_id':'int64', 'item_condition_id':'int8', 'price':'float64', 'shipping':'int8'}
    
    # tsvファイルからPandas DataFrameへ読み込み
    dataset = pd.read_csv("./train.tsv", delimiter='\t', low_memory=True, dtype=types_dict_train)

    #規模を小さくして動作を確認したい時用
    #dataset = dataset[dataset['train_id'] < 20000]

    #---前処理-------------------------------------------------------------------------------
    
    #BoWを用いた文字列の処理
    pd_name = features_convert_BoW(data_frame=dataset.name, min_df=0.01, max_df=0.99)
    pd_category_name = features_convert_BoW(data_frame=dataset.category_name, min_df=0.01, max_df=0.99)
    pd_brand_name = features_convert_BoW(data_frame=dataset.brand_name, min_df=0.01, max_df=0.99)
    pd_item_description = features_convert_BoW(dataset.item_description, 0.01, 0.99)

    #文字列をそのままユニークな数値に変換した処理
    '''
    pd_name2 = dataset.name.astype('category')
    pd_name2 = pd_name2.cat.codes
    pd_category_name2 = dataset.category_name.astype('category')
    pd_category_name2 = pd_category_name2.cat.codes
    pd_brand_name2 = dataset.brand_name.astype('category')
    pd_brand_name2 = pd_brand_name2.cat.codes
    pd_item_description2 = dataset.item_description.astype('category')
    pd_item_description2 = pd_item_description2.cat.codes
    '''

    #変換した各特徴量を元のデータセットに結合
    dataset = dataset.drop(["train_id", "name", "category_name", "brand_name", "item_description"], axis=1)
    BoW_dataset = pd.concat([dataset, pd_name, pd_category_name, pd_brand_name, pd_item_description], axis=1)
    #str2num_dataset = pd.concat([dataset, pd_name2, pd_category_name2, pd_brand_name2, pd_item_description2], axis=1)

    # price（価格）をlog関数で処理
    BoW_dataset['price'] = BoW_dataset['price'].apply(lambda x: np.log(x) if x>0 else x)
    #str2num_dataset['price'] = str2num_dataset['price'].apply(lambda x: np.log(x) if x>0 else x)

    #print(dataset.head())
    
    #---学習--------------------------------------------------------------------------------

    #trainとtestに分割
    train, test = train_test_split(BoW_dataset, test_size=0.3, random_state=0)
    #train, test = train_test_split(str2num_dataset, test_size=0.3, random_state=0)

    # x ＝ price以外の全ての値、y = price（ターゲット）で切り分ける
    x_test, y_test = test.drop(['price'], axis=1), test.price
    x_train, y_train = train.drop(['price'], axis=1), train.price

    # モデルの作成
    m = RandomForestRegressor(n_jobs=-1, min_samples_leaf=5, n_estimators=200)
    m.fit(x_train, y_train)

    # 作成したランダムフォレストのモデル「m」に「df_test」を入れて予測する
    preds = m.predict(x_test)

    # スコアを表示
    print('score:トレーニングのスコア=', m.score(x_train, y_train))
    print('score:テストのスコア=', m.score(x_test, y_test))

    #実際のデータと予測データをplotする
    plot_graph(preds, y_test)