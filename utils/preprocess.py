
def preprocess(df):
    # 欠損値処理
    df['fare'] = df['fare'].fillna(df['fare'].median())
    df['age'] = df['age'].fillna(df['age'].median())
    df['embarked'] = df['embarked'].fillna('S')
    # カテゴリ変数の変換
    df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)
    df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    return df
