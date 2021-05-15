import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split

import utils.preprocess as prep

DATA_PATH = './dataset/train.tsv'
SAVE_PATH = './test_dataset/'
PICK_NAME = 'agepred_test.tsv'
FILL_NAME = 'agepred_train.tsv'
TEST_SIZE = 0.2
RANDOM_STATE = 87


def prepare_data(df):
    os.makedirs(SAVE_PATH, exist_ok=True)
    pick_data = prep.pick_NaNdata(df)
    pick_data.to_csv(SAVE_PATH + PICK_NAME, sep='\t', index=False)

    fill_data = prep.drop_NaNdata(df)
    fill_data.to_csv(SAVE_PATH + FILL_NAME, sep='\t', index=False)

    return fill_data


def train(fill_data):
    df = prep.conv_categorical_var(fill_data)
    train_X = df.drop('age', axis=1)
    train_y = df.age.astype(int)
    train_X, test_X, train_y, test_y = train_test_split(
        train_X, train_y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model = LR()
    model = model.fit(train_X, train_y)
    pred = model.predict(test_X)
    score = model.score(test_X, test_y)
    score = round(score, 3)
    save_predgraph(test_y, pred, score)

    return model, score


def save_predgraph(test_y, pred, score):
    test_y_list = []
    for i in test_y:
        test_y_list.append(i)
    max_num = len(test_y_list)

    plt.plot(range(0, max_num), test_y_list, color='red', marker='o')
    plt.plot(range(0, max_num), pred, color='blue', marker='o')
    plt.savefig('{}{}.png'.format(SAVE_PATH, score))


def save_model(model, score):
    os.makedirs(SAVE_PATH, exist_ok=True)
    model_name = '{}{}_model'.format(SAVE_PATH, score)
    with open('{}.pickle'.format(model_name), mode='wb') as file:
        pickle.dump(model, file)


def main():
    df = pd.read_table(DATA_PATH)
    fill_data = prepare_data(df)
    model, score = train(fill_data)
    save_model(model, score)


if __name__ == '__main__':
    main()
