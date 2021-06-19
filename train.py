import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split

from utils.preprocess import preprocess


DATA_PATH = './dataset/train.tsv'
SAVE_PATH = './models/'
TEST_SIZE = 0.3
RANDOM_STATE = 123


def train(df):
    train_X = df.drop('survived', axis=1)
    train_y = df.survived
    train_X, test_X, train_y, test_y = train_test_split(
        train_X, train_y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model = RandomForestClassifier(random_state=0)
    model = model.fit(train_X, train_y)
    pred = model.predict(test_X)
    fpr, tpr, thresholds = roc_curve(test_y, pred, pos_label=1)
    auc_num = round(auc(fpr, tpr), 3)
    score = round(accuracy_score(pred, test_y), 3)
    print('auc: {}\nscore: {}'.format(auc_num, score))
    plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' % auc_num)

    return model, score


def save_model(model, score):
    os.makedirs(SAVE_PATH, exist_ok=True)
    model_name = '{}{}_model'.format(SAVE_PATH, score)
    with open('{}.pickle'.format(model_name), mode='wb') as file:
        pickle.dump(model, file)

    return model_name


def save_plotimage(model_name):
    plt.grid()
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.savefig('{}.png'.format(model_name))


def main():
    df = pd.read_table(DATA_PATH)
    df = preprocess(df)
    model, score = train(df)
    model_name = save_model(model, score)
    save_plotimage(model_name)


if __name__ == '__main__':
    main()
