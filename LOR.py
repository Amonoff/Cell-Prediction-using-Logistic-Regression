import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import log_loss
cell_samples = pd.read_csv('ml6.csv')
cell_samples.head(10)
