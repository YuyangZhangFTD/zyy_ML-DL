from sklearn import metrics
from sklearn import learning_curve
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Ridge

# Classification        
# all metrics all in sklearn.metric
#   accuracy            metrics.accuracy_score
#   average_precision   metrics.average_precision_score
#   f1                  metrics.f1_score
#   f1_micro        	metrics.f1_score	
#   f1_macro        	metrics.f1_score	
#   f1_weighted     	metrics.f1_score	
#   f1_samples      	metrics.f1_score	
#   log_loss        	metrics.log_loss	
#   precision       	metrics.precision_score	
#   recall              metrics.recall_score	
#   roc_auc             metrics.roc_auc_score	 
# Clustering	 	 
#   adjusted_rand_score metrics.adjusted_rand_score	 
# Regression	 	 
#   mean_absolute_error metrics.mean_absolute_error	 
#   mean_squared_error  metrics.mean_squared_error	 
#   median_absolute_error   metrics.median_absolute_error	 
#   r2                  metrics.r2_score


# Classification metrics
# Accuracy score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
metrics.accuracy_score(y_true, y_pred)                  # percent of correct samples
metrics.accuracy_score(y_true, y_pred, normalize=False) # number of correct samples
# Classification report
y_true = [0, 1, 2, 2, 0]
y_pred = [0, 0, 2, 2, 0]
target_names = ['class 0', 'class 1', 'class 2']
metrics.classification_report(y_true, y_pred, target_names=target_names)
# Hamming loss
y_pred = [1, 2, 3, 4]
y_true = [2, 2, 3, 4]
metrics.hamming_loss(y_true, y_pred)    # Hamming loss = 1 - accuracy
# Precision, recall and F-measures
y_pred = [0, 1, 0, 0]
y_true = [0, 1, 0, 1]
metrics.precision_score(y_true, y_pred)
metrics.recall_score(y_true, y_pred)
metrics.f1_score(y_true, y_pred)  
metrics.fbeta_score(y_true, y_pred, beta=0.5)       
metrics.fbeta_score(y_true, y_pred, beta=1)          
metrics.fbeta_score(y_true, y_pred, beta=2)         
metrics.precision_recall_fscore_support(y_true, y_pred, beta=0.5)
# precision_recall_curve  
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
precision, recall, threshold = metrics.precision_recall_curve(y_true, y_scores)
metrics.average_precision_score(y_true, y_scores)  
# Log loss
y_true = [0, 0, 1, 1]
y_pred = [[.9, .1], [.8, .2], [.3, .7], [.01, .99]]
metrics.log_loss(y_true, y_pred) 
# Receiver operating characteristic (ROC)
y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
# Aera under roc (AOC)
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
metrics.roc_auc_score(y_true, y_scores)

# There are a lot of loss metrics for ranking
# They can be found at http://sklearn.lzjqsdd.com/modules/model_evaluation.html
# We don't discuss here


# Regression metrics
# Explained variance score
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
metrics.explained_variance_score(y_true, y_pred)  
y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
metrics.explained_variance_score(y_true, y_pred, multioutput='raw_values')
metrics.explained_variance_score(y_true, y_pred, multioutput=[0.3, 0.7])
# Mean absolute error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
metrics.mean_absolute_error(y_true, y_pred)
y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
metrics.mean_absolute_error(y_true, y_pred)
metrics.mean_absolute_error(y_true, y_pred, multioutput='raw_values')
metrics.mean_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7])
# Mean squared error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
metrics.mean_squared_error(y_true, y_pred)
y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
metrics.mean_squared_error(y_true, y_pred)  


# Curve
np.random.seed(0)
iris = load_iris()
X, y = iris.data, iris.target
indices = np.arange(y.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]
# Validation curve
train_scores, valid_scores = validation_curve.validation_curve(Ridge(), X, y, "alpha",
# Learning curve
train_sizes, train_scores, valid_scores = validation_curve.learning_curve(SVC(kernel='linear'), X, y, train_sizes=[50, 80, 110], cv=5)


