import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import pandas as pd
from helpers import prediction_helper
from helpers import utils
from functools import partial
### the wikibase_helper is used to interact with our self-hosted wikibase
### not included in the repository
from helpers import wikibase_helper as wh

NO_PARAMS_SAMPLES=100
iter_no = 8 # current iteration
print('This is iteration {}'.format(iter_no))

############# data processing ####################
#######  load all data
unlabeled_data = pd.read_csv('data/all_postings.csv').dropna()
unlabeled_data = utils.filter_text_by_words(unlabeled_data)
unlabeled_data = unlabeled_data.groupby('text', as_index=False)

#######  load labeled data
# pulling new labbelled data from wikibase
data = wh.collect_new_labelled_dataset()
data = utils.filter_text_by_words(data)
# pulling off-topic data
off_topic_data = wh.collect_off_topic_dataset()
off_topic_data = utils.filter_text_by_words(off_topic_data)

###### determine the data that is not labeled yet
unlabeled_data = unlabeled_data[~unlabeled_data.text.isin(data.text)&~unlabeled_data.text.isin(off_topic_data.text)]

test_data, test_off_topic_data = wh.load_test_data()

########### building topic classifiers ###########
if os.path.exists(f'data/classifier_iterations/topic_classifier_iteration_{iter_no}.pkl'):
    topic_classifiers = pickle.load(open(f'data/classifier_iterations/topic_classifier_iteration_{iter_no}.pkl', 'rb'))
else:
    topic_classifiers = dict()
topics_to_refit = data.topic_label.unique()
for topic_label in data.topic_label.unique():
    if topic_label not in topic_classifiers or topic_label in topics_to_refit:
        print('building topic classifier for {}'.format(topic_label))
        labeled_text = data.loc[data.topic_label == topic_label]
        neg_text = data[~data.text.isin(labeled_text.text)]
        topic_classifiers[topic_label] = prediction_helper.topic_predictor(labeled_text.text.to_list(), unlabeled_data.text.to_list(), known_negative_text=neg_text.text.to_list() + off_topic_data.text.to_list(), n_iter=NO_PARAMS_SAMPLES, model_name='bert', evaluation=False, cv = False)
        pickle.dump(topic_classifiers, file=open(f'data/classifier_iterations/topic_classifier_iteration_{iter_no}.pkl', 'wb'))


########### building opinion classifiers #########
if os.path.exists(f'data/classifier_iterations/opinion_classifier_iteration_{iter_no}.pkl'):
    opinion_classifiers = pickle.load(open(f'data/classifier_iterations/opinion_classifier_iteration_{iter_no}.pkl', 'rb'))
else:
    opinion_classifiers = dict()
for topic_label in data.topic_label.unique():
    if topic_label not in opinion_classifiers:
        print('building opinion classifier for {}'.format(topic_label))
        selected_data = data.loc[data.topic_label == topic_label].groupby('text', as_index=False).agg({'opinion_label': set})
        opinion_classifiers[topic_label] = prediction_helper.opinion_predictor(selected_data.text, selected_data.opinion_label, n_iter=NO_PARAMS_SAMPLES, model_name='bert', evaluation=False)
        pickle.dump(opinion_classifiers, file=open(f'data/classifier_iterations/opinion_classifier_iteration_{iter_no}.pkl', 'wb'))

############ prediction phase ################
predicted_data = []
for topic_label in data.topic_label.unique():
    top_text = topic_classifiers[topic_label].return_top_likely_unlabeled()
    predicted_opinions = opinion_classifiers[topic_label].get_likely_labels(top_text[0])
    selected_data = [[text, topic_label, predicted_opinions[i], top_text[1][i]] for i, text in enumerate(top_text[0])]
    predicted_data.extend(selected_data)
predicted_data = pd.DataFrame(predicted_data, columns = ['text', 'topic_label', 'opinion_label', 'type'])

predicted_data = predicted_data.merge(unlabeled_data, how='left', on='text')

predicted_data = predicted_data.groupby('text', as_index=False).agg({'topic_label': set, 'opinion_label': utils.combine_lists, 'post_url': utils.combine_lists, 'type': set})


############ upload predictions to wikibase ##########
print('prepare to upload predicted labels to wikibase')

wikibase_helper = wh()
entity_ids = []
print('Start to upload to wikibase')
for text, _, opinion_labels, url, _ in predicted_data.iloc:
    entity_id = wikibase_helper.create_predicted_posting(text, list(url)[0])
    entity_ids.append(entity_id)
predicted_data['entity_id'] = entity_ids
print('Done! Waiting for people to label the data and conduct the next iteration!')