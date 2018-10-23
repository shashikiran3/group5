# copyright: yueshi@usc.edu
import pandas as pd 
import hashlib
import os 
from utils import logger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pdb
from sklearn.feature_selection import SelectFromModel
from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

from utils import logger
#def lassoSelection(X,y,)

def lassoSelection(X_train, y_train, n):
	'''
	Lasso feature selection.  Select n features. 
	'''
	#lasso feature selection
	#print (X_train)
	clf = LassoCV()
	sfm = SelectFromModel(clf, threshold=0)
	sfm.fit(X_train, y_train)
	X_transform = sfm.transform(X_train)
	n_features = X_transform.shape[1]
	
	# print(n_features)
	while n_features > n:
		sfm.threshold += 0.01
		X_transform = sfm.transform(X_train)
		n_features = X_transform.shape[1]
	features = [index for index,value in enumerate(sfm.get_support()) if value == True  ]
	logger.info("selected features are {}".format(features))
	return features


def specificity_score(y_true, y_predict):
	'''
	true_negative rate
	'''
	true_negative = len([index for index,pair in enumerate(zip(y_true,y_predict)) if pair[0]==pair[1] and pair[0]==0])
	real_negative = len([i for i in y_true if i==0])
	return true_negative / real_negative 

def model_fit_predict(X_train,X_test,y_train,y_test):

	np.random.seed(2018)
	from sklearn.linear_model import LogisticRegression
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.ensemble import GradientBoostingClassifier
	from sklearn.ensemble import ExtraTreesClassifier
	from sklearn.svm import SVC
	from sklearn.metrics import precision_score
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import f1_score
	from sklearn.metrics import recall_score
	from sklearn.neural_network import MLPClassifier
	models = {
		'LogisticRegression': LogisticRegression(),
		'ExtraTreesClassifier': ExtraTreesClassifier(),
		'RandomForestClassifier': RandomForestClassifier(),
    		#'AdaBoostClassifier': AdaBoostClassifier(),
    		'GradientBoostingClassifier': GradientBoostingClassifier(),
    		'SVC': SVC(),
		'MLPClassifier':MLPClassifier()
	}
	tuned_parameters = {
		'LogisticRegression':{'C': [1, 10]},
		'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
		'RandomForestClassifier': { 'n_estimators': [16, 32] },
    		#'AdaBoostClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.0001, 0.005] },
    		'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.1, 0.8] },
    		'SVC': {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
		'MLPClassifier':{'activation':['relu'],'alpha':[1e-5],'hidden_layer_sizes':[(20,10)],'solver':['lbfgs']},
	}
	
	# tuned_parameters = {
		# 'LogisticRegression':{'C': np.logspace(1, 10)},
	# #	'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
	# #	'RandomForestClassifier': { 'n_estimators': [16, 32] },
    # #	'AdaBoostClassifier': { 'n_estimators': [16, 32] },
    # #	'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
    # #	'SVC': {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
	# }
	#np.array([5])

	scores= {}
	for key in models:
		clf = GridSearchCV(models[key], tuned_parameters[key], scoring=None,  refit=True, cv=10)
		clf.fit(X_train,y_train)
		y_test_predict = clf.predict(X_test)
		precision = precision_score(y_test, y_test_predict,average='macro')
		accuracy = accuracy_score(y_test, y_test_predict)
		f1 = f1_score(y_test, y_test_predict,average='macro')
		recall = recall_score(y_test, y_test_predict,average='macro')
		specificity = specificity_score(y_test, y_test_predict,)
		scores[key] = [precision,accuracy,f1,recall,specificity]
	#print(scores)
	return scores

 def plotROC(X_train,y_train,X_test,y_test):
	y = label_binarize(y_test, classes=range(28))
	classifier = OneVsRestClassifier(AdaBoostClassifier())
	y_score = classifier.fit(X_train, y_train).decision_function(X_test)

	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(28):
		fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	plt.figure()
	plt.plot(fpr["micro"], tpr["micro"],
			 # label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]))
	for i in range(28):
		# plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))

	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic of multi-class')
	plt.legend(loc="lower right")
	plt.show()

def draw(scores):
	'''
	draw scores.
	'''
	import matplotlib.pyplot as plt
	logger.info("scores are {}".format(scores))
	ax = plt.subplot(111)
	precisions = []
	accuracies =[]
	f1_scores = []
	recalls = []
	categories = []
	specificities = []
	N = len(scores)
	ind = np.arange(N)  # set the x locations for the groups
	width = 0.1        # the width of the bars
	for key in scores:
		categories.append(key)
		precisions.append(scores[key][0])
		accuracies.append(scores[key][1])
		f1_scores.append(scores[key][2])
		recalls.append(scores[key][3])
		specificities.append(scores[key][4])

	precision_bar = ax.bar(ind, precisions,width=0.1,color='b',align='center')
	accuracy_bar = ax.bar(ind+1*width, accuracies,width=0.1,color='g',align='center')
	f1_bar = ax.bar(ind+2*width, f1_scores,width=0.1,color='r',align='center')
	recall_bar = ax.bar(ind+3*width, recalls,width=0.1,color='y',align='center')
	specificity_bar = ax.bar(ind+4*width,specificities,width=0.1,color='purple',align='center')

	print(categories)
	ax.set_xticks(np.arange(N))
	ax.set_xticklabels(categories)
	ax.legend((precision_bar[0], accuracy_bar[0],f1_bar[0],recall_bar[0],specificity_bar[0]), ('precision', 'accuracy','f1','sensitivity','specificity'))
	ax.grid()
	plt.show()

if __name__ == '__main__':

	data_dir ="/Users/lenovo/Downloads/GDCfinal/data_new/"

	data_file = data_dir + "miRNA_matrix.csv"

	df = pd.read_csv(data_file)
	# print(df)
	# pdb.set_trace()
	#labels = {'Normal':0}
	#count = 1;
	y_data = df.pop('label').values
	df.pop('file_id')

	columns =df.columns
	#print (columns)
	X_data = df.values
	
	# split the data to train and test set
	X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=0)
	

	#standardize the data.
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	# check the distribution of tumor and normal sampels in traing and test data set.
	#logger.info("Percentage of tumor cases in training set is {}".format(sum(y_train)/len(y_train)))
	#logger.info("Percentage of tumor cases in test set is {}".format(sum(y_test)/len(y_test)))
	
	n = 25
	feaures_columns = lassoSelection(X_train, y_train, n)
	
	#pdb.set_trace()


	scores = model_fit_predict(X_train[:,feaures_columns],X_test[:,feaures_columns],y_train,y_test)
	plotROC(X_train,y_train,X_test,y_test)
	
	
	
	
	draw(scores)
	#lasso cross validation
	# lassoreg = Lasso(random_state=0)
	# alphas = np.logspace(-4, -0.5, 30)
	# tuned_parameters = [{'alpha': alphas}]
	# n_fold = 10
	# clf = GridSearchCV(lassoreg,tuned_parameters,cv=10, refit = False)
	# clf.fit(X_train,y_train)
