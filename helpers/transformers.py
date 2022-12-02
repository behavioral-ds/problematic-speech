import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from simpletransformers.classification import ClassificationModel, ClassificationArgs, MultiLabelClassificationModel, MultiLabelClassificationArgs
import pandas as pd
from scipy.special import softmax
import os

class RobertaClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, lr=1e-3, epochs=5, train_output_layer_only=False, model_path=None, gpu=None):
        super().__init__()
        self.lr = lr
        self.epochs = epochs
        self.train_output_layer_only = train_output_layer_only
        self.fitted = False
        self.gpu = gpu
        if model_path is not None:
            if 'opinions' in model_path:
                self.model_ = MultiLabelClassificationModel('roberta', model_path)
            elif 'topics' in model_path:
                self.model_ = ClassificationModel('roberta', model_path)
            else:
                raise ValueError('Unknown model type!')
            self.fitted = True

    def fit(self, X, y):
        if self.fitted:
            print('Fitting already finished...')
            return self
        # Create a ClassificationModel
        if len(y.shape) > 1:
            self.multilabel = True
            model_args = MultiLabelClassificationArgs(overwrite_output_dir=True, silent=False, save_model_every_epoch=False, use_early_stopping=True,
                                                      evaluate_during_training=True, early_stopping_consider_epochs=True, num_train_epochs=self.epochs,
                                                      early_stopping_patience=2, evaluate_during_training_verbose=True)
            self.model_ = MultiLabelClassificationModel("roberta", "roberta-base", args=model_args, num_labels=len(y[0]))
        else:
            self.multilabel = False
            model_args = ClassificationArgs(num_train_epochs=self.epochs, overwrite_output_dir=True,  n_gpu=1, silent=False, use_early_stopping=False,
                                            evaluate_during_training=False, early_stopping_consider_epochs=True, save_model_every_epoch=False, early_stopping_patience=2)
            if self.train_output_layer_only:
                model_args.train_custom_parameters_only = True
                model_args.custom_parameter_groups = [
                    {
                        'params': ['classifier.dense.weight', 'classifier.dense.bias', 'classifier.out_proj.weight', 'classifier.out_proj.bias'],
                        'lr': self.lr
                    }
                ]
            self.model_ = ClassificationModel(
                "roberta", "roberta-base", args=model_args
            )

        df = pd.DataFrame([[text, label] for text, label in zip(X, y)], columns=['text', 'labels'])
        res = self.model_.train_model(df)
        self.fitted = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        if isinstance(X, pd.core.series.Series):
            X = X.to_list()
        return self.model_.predict(X)[0]

    def predict_proba(self, X):
        check_is_fitted(self)
        if isinstance(X, pd.core.series.Series):
            X = X.to_list()
        if self.multilabel:
            return self.model_.predict(X)[1]
        else:
            return softmax(self.model_.predict(X)[1], axis=1)
