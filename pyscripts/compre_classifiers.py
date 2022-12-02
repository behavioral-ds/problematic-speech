import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import pandas as pd
import numpy as np
from helpers import prediction_helper
from tqdm import tqdm

res = {model: dict() for model in ['rf', 'svm', 'xgboost', 'bert']}
for model in tqdm(['rf', 'svm', 'xgboost', 'bert']):
    train_data = pd.read_csv(f'./data/posting_iterations/postings_texts_iteration_1.csv')
    off_topic_data_text = []

    NO_PARAMS_SAMPLES = 500
    topics = ['2019-20 Australian bushfire season', 'Covid-19', 'vaccination', 'climate change']

    for topic_label in tqdm(topics, leave=False):
        # for model in models:
        labeled_text = train_data.loc[train_data.topic_label == topic_label]
        neg_text = train_data[~train_data.text.isin(labeled_text.text)]
        res[model][topic_label] = prediction_helper.topic_predictor(labeled_text.text.to_list(), [], known_negative_text=neg_text.text.to_list() + off_topic_data_text, n_iter=NO_PARAMS_SAMPLES, model_name=model, evaluation=True, cv=True)
