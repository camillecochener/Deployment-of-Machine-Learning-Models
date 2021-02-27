from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import preprocessors as pp
import config


titanic_pipe = Pipeline(
    # complete with the list of steps from the preprocessors file
    # and the list of variables from the config
    [
        ('extract_firstletter',
        pp.ExtractFirstLetter(variables=config.CABIN)),

        ('add_missing_indicator_num',
        pp.MissingIndicator(variables=config.NUMERICAL_VARS_WITH_NA)),

        ('numerical_imputer',
        pp.NumericalImputer(variables=config.NUMERICAL_VARS_WITH_NA)),

        ('categorical_imputer',
        pp.CategoricalImputer(variables=config.CATEGORICAL_VARS)),

        ('rare_label_encoder',
        pp.RareLabelCategoricalEncoder(tol=0.05, variables=config.CATEGORICAL_VARS)),

        ('categorical_encoder',
        pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),

        ('scaler', StandardScaler()),
        ('logistic_regression', LogisticRegression(C=0.0005, random_state=0))
    ]
    )