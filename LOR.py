#import the necessary libraries. Note that the data has been saved on my computer
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

#explore the code
cell_samples.describe()
cell_samples.dtypes

#check if the data has missing values
cell_samples.isnull()
missing_data = cell_samples.isnull()

for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print('')
    
 #data processing and cleaning
cell_samples=cell_samples[pd.to_numeric(cell_samples['BareNuc'], errors='coerce').notnull()]
cell_samples['BareNuc']=cell_samples['BareNuc'].astype('int64')

X = np.asarray(cell_samples[['Clump', 'UnifSize', 
                             'UnifShape', 'BareNuc', 
                             'BlandChrom', 'NormNucl', 'Mit']])

Y = np.asarray(cell_samples['Class'])
X[0:5] , Y[0:5]

#we now standardize the data
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

#splitting the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 1)
X_train[0:5] , Y_train[0:5]

#fit and preict using our model
LOR = LogisticRegression(C = 0.01, solver ='liblinear').fit(X_train, Y_train)
cell_prediction = LOR.predict(X_test)
cell_prediction[0:5]

#calculate the probabilities of each class occurence
cell_prediction_prob = LOR.predict_proba(X_test)
cell_prediction_prob[0:5]

#low log loss and  a high jaccard score indicates the model is good
log_loss(cell_prediction, cell_prediction_prob) , jaccard_similarity_score(Y_test , cell_prediction)

#plot a confusion matrix
def plot_confusion_matrix(cm,classes,cmap = plt.cm.Greens,
                         normalize = False,
                         title = 'Confusion Matrix'):
    if normalize:
        cm = cm.astype(float) / cm.sum(axis = 1)
        print('Confusionm Matrix with Normalization')
    else:
        print('Confusionm Matrix without Normalization ')
        
    print(cm)
    
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j],fmt),
                 horizontalalignment = 'center',
                 color = 'white' if cm[i,j] > thresh else 'black')
    
    plt.tight_layout
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')  
    
cnf_matrix = confusion_matrix(Y_test, cell_prediction, labels =[4,2])
np.set_printoptions(precision = 2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes = ['Malignant(4)', 'Benign(2)'],
                      normalize = False, title = 'Confusion Matrix')

#calculate classification_report so as to get the precision and recall of our model
# precision 
'''is a measure of the accuracy provided that a class label has been predicted'''
#recall is the true positive rate
print(classification_report(Y_test, cell_prediction))
