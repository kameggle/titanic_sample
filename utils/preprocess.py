def preprocess(df):
    # 欠損値処理
    df['age'] = df['age'].fillna(df['age'].median())
    df['embarked'] = df['embarked'].fillna('S')
    df = conv_categorical_var(df)

    return df


def conv_categorical_var(df):
    df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0).astype(int)
    df['embarked'] = df['embarked'].map(
        {'S': 0, 'C': 1, 'Q': 2}).astype(int)

    return df


def pick_NaNdata(df):
    df['embarked'] = df['embarked'].fillna('S')
    pick_data = df[df.isnull().any(axis=1)]

    return pick_data


def drop_NaNdata(df):
    df['embarked'] = df['embarked'].fillna('S')
    fill_data = df.dropna(axis=0)

    return fill_data
