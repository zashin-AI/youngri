# 머신 러닝 돌리자 ~
# 아주 다양한 모델로

# 라이브러리 불러오기
import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
import datetime 
import sklearn

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, GridSearchCV
###
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

from mlxtend.classifier import StackingCVClassifier
import shap
###
from tensorflow.keras.models import Sequential, load_model, Model
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, hamming_loss, hinge_loss, log_loss, mean_squared_error, auc
import pickle  
import warnings
warnings.filterwarnings('ignore')
# ------------------------------------------------------------------
start_now = datetime.datetime.now()
# ------------------------------------------------------------------

# 2. Set up script parameters

RANDOM_SEED = 2021
PROBAS = True
FOLDS = 3
N_ESTIMATORS = 1000

# -----------------------------------------------------------------

# 데이터 불러오기
X = np.load('C:/nmb/nmb_data/npy/0428_5s_silence_denoise/project_total_npy/total_data.npy')
y = np.load('C:/nmb/nmb_data/npy/0428_5s_silence_denoise/project_total_npy/total_label.npy')
X = X[:2000]
y = y[:2000]
X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])

print (f'X:{X.shape} y: {y.shape} \n')
# X:(4536, 110336) y: (4536,) 

# -----------------------------------------------------------------
# 전처리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = RANDOM_SEED)

print (f'X_train:{X_train.shape} y_train: {y_train.shape}')
print (f'X_test:{X_test.shape} y_test: {y_test.shape}')
# X_train:(3628, 110336) y_train: (3628,)
# X_test:(908, 110336) y_test: (908,)

# -----------------------------------------------------------------
# 5. Declare list of classifiers (level 1) for testing
lgb_params = {
    'metric': 'binary_logloss',
    'n_estimators': N_ESTIMATORS,
    'objective': 'binary',
    'random_state': RANDOM_SEED,
    'learning_rate': 0.01,
    'min_child_samples': 150,
    'reg_alpha': 3e-5,
    'reg_lambda': 9e-2,
    'num_leaves': 20,
    'max_depth': 16,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'subsample_freq': 2,
    'max_bin': 240,
}

ctb_params = {
    'bootstrap_type': 'Poisson',
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss',
    'random_seed': RANDOM_SEED,
    'task_type': 'GPU',
    'max_depth': 8,
    'learning_rate': 0.01,
    'n_estimators': N_ESTIMATORS,
    'max_bin': 280,
    'min_data_in_leaf': 64,
    'l2_leaf_reg': 0.01,
    'subsample': 0.8
}

rf_params = {
    'max_depth': 15,
    'min_samples_leaf': 8,
    'random_state': RANDOM_SEED
}

xgb_params = {
    'n_jobs' : -1,
    'use_label_encoder' : False,
    'n_estimators' : 10000, 
    'tree_method' : 'gpu_hist',
    # 'predictor' : 'gpu_predictor'
    'predictor' : 'cpu_predictor',
    'gpu_id' : 0
}

# I do not make any hyperparameter optimization - just taken as they are - here is room for improvement
cl1 = KNeighborsClassifier(n_neighbors = 1)
cl2 = RandomForestClassifier(**rf_params)
cl3 = GaussianNB()
cl4 = DecisionTreeClassifier(max_depth = 5)
cl5 = CatBoostClassifier(**ctb_params, verbose = None, logging_level = 'Silent')
cl6 = LGBMClassifier(**lgb_params)
cl7 = XGBClassifier(**xgb_params)

# I used some hyperparameter search (ExtraTrees - Genetic search)
cl8 = ExtraTreesClassifier(bootstrap=False, criterion='entropy', max_features=0.55, min_samples_leaf=8, min_samples_split=4, n_estimators=100) # Optimized using TPOT
cl9 = MLPClassifier(activation = "relu", alpha = 0.1, hidden_layer_sizes = (10,10,10),
                            learning_rate = "constant", max_iter = 2000, random_state = RANDOM_SEED)

# ------------------------------------------------------------------
# 6. Declare meta-classifier (level 2) and stack them

# For this test I use Logistic Regression as a meta-classifier but you can ... take end experiment something else ...
mlr = LogisticRegression()

# "Ensemble learning works best when the base models are not correlated. 
# For instance, you can train different models such as linear models, decision trees, and neural nets on different datasets or features. 
# The less correlated the base models, the better." (https://neptune.ai/blog/ensemble-learning-guide)


scl = StackingCVClassifier(classifiers= [cl1, cl2, cl3, cl4, cl5, cl6, cl7,cl8],
                            meta_classifier = mlr, # use meta-classifier
                            use_probas = PROBAS,   # use_probas = True/False
                            random_state = RANDOM_SEED)

# Number of classifiers used (define it according to classifiers used)
NUM_CLAS = 9 # classifiers (l1) + stacked (meta-classifier) 

# Classifiers for experiment + stacking (meta-classifier)
classifiers = {
"KNeighborsClassifier" : cl1,
"RandomForest": cl2,
"GaussianNB" : cl3,
"DecisionTreeClassifier" : cl4,
"CatBoost": cl5,
"LGBM": cl6,
"XGBClassifier" : cl7,
"ExtraTrees": cl8,
"Stacked": scl}

# ------------------------------------------------------------------
# 8. Train classifiers

# This step could take some time .... it depends on classifiers you use .... So make a coffe or meditate ... 

print(">>>> Training started <<<<")
for key in classifiers:
    classifier = classifiers[key]
    scores = model_selection.cross_val_score(classifier, X_train, y_train, cv = FOLDS, scoring='accuracy')
    print("[%s] - accuracy: %0.2f " % (key, scores.mean()))
    classifier.fit(X_train, y_train)
    
    # Save classifier for prediction 
    classifiers[key] = classifier

