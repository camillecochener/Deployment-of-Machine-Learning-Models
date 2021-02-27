# ====   PATHS ===================

TRAINING_DATA_FILE = "titanic.csv"
PIPELINE_NAME = 'logistic_regression.pkl'


# ======= FEATURE GROUPS =============

TARGET = 'survived'

CATEGORICAL_VARS = ['sex', 'cabin', 'embarked', 'title']

NUMERICAL_VARS_WITH_NA = ['age', 'fare']

NUMERICAL_VARS = ['pclass', 'age', 'sibsp', 'parch', 'fare']

CABIN = 'cabin'

FEATURES = ['sex', 'cabin', 'embarked', 'title', 'pclass', 'age', 'sibsp', 'parch', 'fare']