import numpy as np
import sklearn.ensemble
import sklearn.metrics
import random
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold, KFold,cross_validate
from sklearn.utils import shuffle
from modAL.models import ActiveLearner
from . import utils
from copy import deepcopy
from helpers import RobertaClassifier
import xgboost

def get_model(model_name, seed=10):
    if model_name == 'rf':
        vectorizer = utils.get_vectorizer()
        rf = sklearn.ensemble.RandomForestClassifier(random_state=seed)
        param_grid = {"classifier__n_estimators": [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                      "classifier__max_depth": [int(x) for x in np.linspace(10, 110, num = 11)],
                      "classifier__min_samples_split": [2, 5, 10, 15, 100],
                      'classifier__min_samples_leaf': [1, 2, 5, 10]}
        pipeline = Pipeline([('features', vectorizer), ('classifier', rf)])
    elif model_name == 'bert':
        param_grid = {'classifier__epochs': [5]}
        cls = RobertaClassifier()
        pipeline = Pipeline([('classifier', cls)])
    elif model_name == 'bert_output_only':
        param_grid = {'classifier__epochs': [5]}
        cls = RobertaClassifier(train_output_layer_only=True)
        pipeline = Pipeline([('classifier', cls)])
    elif model_name == 'svm':
        vectorizer = utils.get_vectorizer()
        cls = sklearn.svm.SVC(random_state=seed, probability=True)
        param_grid = [
            {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
            {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
        ]
        pipeline = Pipeline([('features', vectorizer), ('classifier', cls)])
    elif model_name == 'xgboost':
        vectorizer = utils.get_vectorizer()
        cls = xgboost.XGBClassifier(n_jobs=1, eval_metric='logloss', use_label_encoder=False)
        param_grid = {
            "classifier__n_estimators": [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
            'classifier__min_child_weight': [1, 5, 10],
            'classifier__gamma': [0.5, 1, 1.5, 2, 5],
            'classifier__subsample': [0.6, 0.8, 1.0],
            'classifier__colsample_bytree': [0.6, 0.8, 1.0],
            'classifier__max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
            'classifier__learning_rate': [1e-3, 1e-2, 1e-1]
        }
        pipeline = Pipeline([('features', vectorizer), ('classifier', cls)])
    else:
        raise NotImplementedError()

    return pipeline, param_grid

class topic_predictor:

    def __init__(self, labeled_text, unlabeled_text, model_name='rf', known_negative_text=[], seed=100, n_iter=100, n_job=-1, evaluation=True, cv=True):
        random.seed(seed)
        labeled_text, unlabeled_text, known_negative_text = list(set(labeled_text)), list(set(unlabeled_text)), list(set(known_negative_text))
        self.unlabeled_text = unlabeled_text
        self.X = labeled_text + known_negative_text
        self.Y = np.concatenate([np.ones(len(labeled_text), dtype=np.int), np.zeros(len(self.X)-len(labeled_text), dtype=np.int)])

        self.pipeline, param_grid = get_model(model_name, seed)
        scoring = {'acc': 'accuracy', 'f1': 'f1_macro'}
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        if 'bert' in model_name:
            print('is BERT model, updating n_job number')
            n_job = 1
            n_iter = 3
        self.gs = RandomizedSearchCV(self.pipeline, n_iter = n_iter, param_distributions=param_grid, cv=inner_cv, scoring=scoring, refit='acc', verbose=1, n_jobs=n_job, random_state=seed)

        if evaluation:
            print('Start computing nested CV...')
            scoring = {'acc': 'accuracy', 'f1': 'f1_weighted'}
            outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            nested_scores = cross_validate(self.gs, X=self.X, y=self.Y, cv=outer_cv, scoring=scoring)
            self.best_scores = {metric: nested_scores['test_'+metric].mean() for metric in scoring}

        print('Fitting the final model....')
        self.X, self.Y = shuffle(self.X, self.Y, random_state=seed)
        if cv:
            self.gs.fit(self.X, self.Y)
            self.al = ActiveLearner(estimator=deepcopy(self.gs.best_estimator_))
        else:
            self.al = ActiveLearner(estimator=self.pipeline)
            self.al.fit(self.X, self.Y)
            self.al.fit(self.X, self.Y)
        print('Done fitting...')

    def return_important_features(self, top_no=50):
        importances = self.al.estimator['classifier'].feature_importances_
        indices = np.argsort(importances)[::-1]

        # top 50 important features
        return [self.al.estimator['features'].get_feature_names()[indices[i]] for i in range(top_no)]

    def return_top_likely_unlabeled(self, n=25):
        probs = self.al.predict_proba(self.unlabeled_text)
        pos_probs = probs[:, 1]
        indices = np.argsort(pos_probs)[::-1]
        top_confident_instances = [self.unlabeled_text[indices[i]] for i in range(round(n*0.4))]

        # from active learners
        top_uncertain_instances = self.al.query(self.unlabeled_text, n_instances=round(n*0.4))[1]

        # random samples
        reversed_sampl_no=0
        random_samples = random.sample(set(self.unlabeled_text) - set(top_confident_instances) - set(top_uncertain_instances), round(n*0.2) + reversed_sampl_no)

        return [top_confident_instances+top_uncertain_instances+random_samples,
                ['top_confident' for i in range(len(top_confident_instances))] + ['active' for i in range(len(top_uncertain_instances))] + ['random' for i in range(len(random_samples) - reversed_sampl_no)] + ['reserved' for i in range(reversed_sampl_no)]]

class opinion_predictor:

    def __init__(self, text, labels, model_name='rf', seed=100, n_iter=100, evaluation=True, cv=True):
        self.mlb = MultiLabelBinarizer()
        self.Y = self.mlb.fit_transform(labels)
        print("there are {} different opinion labels, each has {} samples".format(len(self.mlb.classes_), self.Y.sum(0)))
        self.X = text

        self.pipeline, param_grid = get_model(model_name, seed)
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        scoring = {'acc': 'accuracy', 'f1': 'f1_macro'}
        self.gs = RandomizedSearchCV(self.pipeline, n_iter = n_iter, param_distributions=param_grid, scoring=scoring, cv=inner_cv, refit=False, verbose=2)

        if evaluation:
            print('Start computing nested CV...')
            outer_cv = KFold(n_splits=5, shuffle=True, random_state=seed)
            nested_score = cross_val_score(self.gs, X=self.X, y=self.Y, cv=outer_cv)
            self.best_score = nested_score.mean()
        else:
            print('Fitting the final model....')
            self.X, self.Y = shuffle(self.X, self.Y, random_state=seed)
            if cv:
                self.gs.fit(self.X, self.Y)
                self.al = ActiveLearner(estimator=deepcopy(self.gs.best_estimator_))
            else:
                self.pipeline['classifier'].epochs = 50
                self.al = ActiveLearner(estimator=self.pipeline)
                self.al.fit(self.X, self.Y)
            print('Done fitting...')

    def get_likely_labels(self, text):
        predicted_probs = self.al.predict_proba(text)
        threshhold = 0.5
        return [[self.mlb.classes_[j] for j in range(len(predicted_probs[0])) if predicted_probs[i][j] >= threshhold] for i in range(len(predicted_probs))]
