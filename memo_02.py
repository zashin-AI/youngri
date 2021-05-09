import numpy as np
import datetime 
import librosa
import sklearn
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, GridSearchCV
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
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
###
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, hamming_loss, hinge_loss, log_loss, mean_squared_error, auc
import pickle  
import warnings
warnings.filterwarnings('ignore')
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


# ---------------------------------------------------------------------------------------------------

pred_pathAudio = 'C:/nmb/nmb_data/predict/0509_5s_predict/total_predict'
files = librosa.util.find_files(pred_pathAudio, ext=['wav'])
files = np.asarray(files)
print(files)
for file in files:
    file.split('\\')[-1].split('.')[0]
