## notice

for Signate 102: https://signate.jp/competitions/102

## readme

- please put signate_DL_datasets in ./dataset/

### using venv

- `python3 -m venv .titanic` in titanic_sample
- `source .titanic/bin/activate`
- `pip3 install --upgrade pip`
- `pip3 install -r requirements.txt`
- run `train.py` to make model
- run `predict.py` to make predict_file

## directry

```
titanic_sample
├── README.md
├── dataset
│   ├── sample_submit.tsv
│   ├── test.tsv
│   └── train.tsv
├── train.py
├── predict.py
└── utils
    └── preprocess.py
```
