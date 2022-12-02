import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import pandas as pd
from helpers import utils
import dill as pickle
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
topics = ['vaccination', 'climate change', 'Covid-19', '2019-20 Australian bushfire season']

compute_total = []
topic_classifier = None
def test_on_data(classifier, true_text, neg_text):
    predictions = classifier.al.predict(true_text + neg_text)
    true_labels = [1 for _, _ in enumerate(true_text)] + [0 for _, _ in enumerate(neg_text)]
    acc = accuracy_score(predictions, true_labels)
    f1 = f1_score(predictions, true_labels)
    return acc, f1

for iter_no in tqdm(range(1, 8)):
    topic_classifier = pickle.load(file=open(f'data/classifier_iterations/topic_classifier_iteration_{iter_no}.pkl', 'rb'))

    test_data = pd.read_csv('data/postings_texts_reserved_test.csv')
    test_off_topic_data_text = list(pd.read_csv('data/off_topic_postings_texts_reserved_test.csv').text.unique())
    test_data = utils.filter_opinions(test_data)
    t = tqdm(topics, leave=False)
    for topic in t:
        t.set_description(topic)
        true_text = test_data.loc[test_data.topic_label == topic].text.to_list()
        neg_text = test_data[~test_data.text.isin(true_text)].text.to_list() + test_off_topic_data_text
        true_text = list(set(true_text))
        neg_text = list(set(neg_text))
        acc, f1 = test_on_data(topic_classifier[topic], true_text, neg_text)
        compute_total.append({'type': 'test', 'acc': acc, 'f1': f1, 'positives': len(true_text), 'topic': topic, 'iter_no': iter_no})

df = pd.DataFrame(compute_total)
df.to_csv('data/evaluation_at_batches.csv')
