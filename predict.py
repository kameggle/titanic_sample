from datetime import datetime
import os
import pandas as pd
import pickle

import click

from utils.preprocess import preprocess

DATA_PATH = './dataset/test.tsv'
MODEL_DIR = './models/'
SAVE_DIR = './submit/'


def load_model(model_name):
    with open(MODEL_DIR + model_name, mode='rb') as file:
        model = pickle.load(file)

    return model


def prediction(model, df):
    ids = df['id']
    results = model.predict(df)

    predict_list = []
    for i in zip(ids, results):
        predict_list.append(i)

    return predict_list


def make_submit_tsv(predict_list):
    now = datetime.now()
    dt_txt = '{0:%m%d%H%M}'.format(now)

    os.makedirs(SAVE_DIR, exist_ok=True)
    submit_name = '{}{}_submit.tsv'.format(SAVE_DIR, dt_txt)
    submit_frame = pd.DataFrame(predict_list, columns=predict_list[0])
    submit_frame.to_csv(submit_name, sep='\t', header=None, index=False)


@click.command()
@click.option('--model_name', '-n', type=str, default='model.pickle')
def main(model_name):
    df = pd.read_table(DATA_PATH)
    df = preprocess(df)
    model = load_model(model_name)
    predict_list = prediction(model, df)
    make_submit_tsv(predict_list)


if __name__ == '__main__':
    main()
