import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder


class OrdinalFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_enc_ft=['person_education']):
        self.ordinal_enc_ft = ordinal_enc_ft

    def fit(self, X, y=None):
        # No fitting operation needed, just return self
        return self

    def transform(self, X):
        if set(self.ordinal_enc_ft).issubset(X.columns):
            ordinal_enc = OrdinalEncoder()
            X[self.ordinal_enc_ft] = ordinal_enc.fit_transform(
                X[self.ordinal_enc_ft])
            return X
        else:
            print("Education level is not in the dataframe")
            return X


class SkewnessHandler(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_skewness=['person_income', 'loan_int_rate']):
        self.feat_with_skewness = feat_with_skewness

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if set(self.feat_with_skewness).issubset(X.columns):
            X.loc[:, self.feat_with_skewness] = np.cbrt(
                X[self.feat_with_skewness])
            return X
        else:
            print("One or more features are not in the dataframe skewness")
            return X


class BinningNumToYN(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_num_enc=['previous_loan_defaults_on_file']):
        self.feat_with_num_enc = feat_with_num_enc
        self.encoder = LabelEncoder()

    def fit(self, X, y=None):
        for ft in self.feat_with_num_enc:
            self.encoder.fit(X[ft].fillna('No').astype(str).unique())
        return self

    def transform(self, X):
        X = X.copy()
        for ft in self.feat_with_num_enc:
            if ft in X.columns:
                X[ft] = X[ft].fillna('No').astype(str)
                X.loc[:, ft] = self.encoder.transform(X[ft])
            else:
                raise ValueError(f"{ft} not found in DataFrame columns.")

        X[self.feat_with_num_enc] = X[self.feat_with_num_enc].astype(float)
        return X


class OneHotWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_enc_ft=['person_gender', 'person_home_ownership', 'loan_intent']):
        self.one_hot_enc_ft = one_hot_enc_ft

    def fit(self, X, y=None):
        # No fitting operation needed, just return self
        return self

    def transform(self, X):
        if set(self.one_hot_enc_ft).issubset(X.columns):
            one_hot_enc = OneHotEncoder(sparse_output=False, drop='first')
            one_hot_encoded = one_hot_enc.fit_transform(X[self.one_hot_enc_ft])
            feat_names_one_hot_enc = one_hot_enc.get_feature_names_out(
                self.one_hot_enc_ft)

            one_hot_enc_df = pd.DataFrame(
                one_hot_encoded, columns=feat_names_one_hot_enc, index=X.index)
            X = X.drop(columns=self.one_hot_enc_ft).reset_index(drop=True)
            X = pd.concat([X, one_hot_enc_df], axis=1)
            return X
        else:
            print("One or more features are not in the dataframe for one-hot encoding")
            return X
