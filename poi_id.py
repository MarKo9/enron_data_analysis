#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pprint
import numpy
from sklearn import cross_validation
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn import grid_search, svm
from sklearn.feature_selection import SelectKBest, f_classif   
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import classification_report, precision_recall_curve
from tester import test_classifier, dump_classifier_and_data
from sklearn.cross_validation import train_test_split, cross_val_score, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary','to_messages','deferral_payments',
                  'total_payments','exercised_stock_options',
                  'bonus','restricted_stock','shared_receipt_with_poi',
                  'restricted_stock_deferred','total_stock_value',
                  'total_stock_value','expenses','loan_advances','from_messages',
                  'other','from_this_person_to_poi',
                  'director_fees', 'deferred_income','long_term_incentive',
                  'from_poi_to_this_person','fraction_from_poi_1','fraction_to_poi_1' ] 

len(features_list)                  

"""
## only the original features

features_list = ['poi','salary','to_messages','deferral_payments',
                  'total_payments','exercised_stock_options',
                  'bonus','restricted_stock','shared_receipt_with_poi',
                  'restricted_stock_deferred','total_stock_value',
                  'total_stock_value','expenses','loan_advances','from_messages',
                  'other','from_this_person_to_poi',
                  'director_fees', 'deferred_income','long_term_incentive',
                  'from_poi_to_this_person' ] 
                  
len(features_list_orig)                  
"""                  
                

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)






###>>>>>>>>>>>>>>>>>>> Task 2: Remove outliers
## find outliers
key_list = [k for k in data_dict.keys() if data_dict[k]["salary"] != 
'NaN' and data_dict[k]["salary"] > 1000000 and data_dict[k]["bonus"] > 5000000]

# Print the key values to find the outliers
print key_list      
     
     
# delete outlier

data_dict.pop("TOTAL", 0 )



# If a data has zero value for all features, it is not useful, so should be removed as outlier

# Find out all the finance features that we need to calculate NaN percentage for
all_ftrs = data_dict['LAY KENNETH L'].keys()

email_ftrs = ['to_messages', 'shared_receipt_with_poi', 'from_messages',
              'from_this_person_to_poi', 'email_address', 'from_poi_to_this_person']

all_ftrs = [ftr for ftr in all_ftrs if ftr not in email_ftrs]

# Identify the data points:

zero_keys = []
for key in data_dict:
    n = 0
    for ftr in all_ftrs:
        if data_dict[key][ftr] == 'NaN':
            n += 1
    if n == len(all_ftrs) - 1: # excluding the 'poi' key
        zero_keys.append(key)
print("\nData Points that Have NaN's for All Features:")
print zero_keys, '\n'  # 'LOCKHART EUGENE E'

# Now remove them
for key in zero_keys:
    data_dict.pop(key, 0)



      
######>>>>>>>>>>>>>>>>>>> Task 3: Create new feature(s)



def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    if all(x != "NaN" for x in (poi_messages, all_messages)) and all(x != 0 for x in (poi_messages, all_messages)) :

        fraction = float(poi_messages) / float(all_messages)
    else:
        fraction = 0


    return fraction
# create a new dict for new features
submit_dict = {}

for name in data_dict:
    # create the "fraction_from_poi_1" feature
    data_point = data_dict[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi_1 = computeFraction( from_poi_to_this_person, to_messages )
    # add the new feature in the database    
    data_point["fraction_from_poi_1"] = fraction_from_poi_1

    # create the "fraction_to_poi_1" feature
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi_1 = computeFraction( from_this_person_to_poi, from_messages )
    # add the new feature in the database
    data_point["fraction_to_poi_1"] = fraction_to_poi_1
    
    # add the new features in the empty dictionary 
    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi_1,
                       "from_this_person_to_poi":fraction_to_poi_1}    
    
pprint.pprint(submit_dict)


new_data_dict = data_dict.copy()




# test print
"""
a = 0
for key in data_dict:
    a +=1
    if a < 16:
        pprint.pprint(data_dict[key]["fraction_from_poi_1"])

"""
      
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)      
      
## feature scaling with MinMaxScaler




#### siple data split on test and trainning 
#features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.2, random_state=42)

#### cross validation


# Set up cross validator (will be used for tuning all classifiers)
#cv = cross_validation.StratifiedShuffleSplit(labels, 1000, random_state = 42)

for train_idx, test_idx in cross_validation.StratifiedShuffleSplit(labels, 1000, random_state = 42): 
    features_train = []
    features_test  = []
    targets_train   = []
    targets_test    = []

    features_train = [features[ii] for ii in train_idx]
    labels_train = [labels[ii] for ii in train_idx]
    features_test = [features[jj] for jj in test_idx]
    labels_test = [labels[jj] for jj in test_idx]






#### ++++++++++++++++++++++  (from here)  working preprocess ++++++++++++++++++++++



####  ------------------------------  test DecisionTreeClassifier() ------------------
"""
# with PCA Accuracy: 0.80240	Precision: 0.27910	Recall: 0.30450	F1: 0.29125	F2: 0.29906
# no PCA Accuracy: 0.81607	Precision: 0.26958	Recall: 0.22200	F1: 0.24349	F2: 0.23012
parameters = { 'min_samples_split':[1, 10],'random_state':[0,5]} 
svr = tree.DecisionTreeClassifier()
algo = grid_search.GridSearchCV(svr, parameters)

algo = tree.DecisionTreeClassifier()
# no PCA, on GridsearchCV: Accuracy: 0.81013	Precision: 0.28167	Recall: 0.27350	F1: 0.27752
"""
####  ------------------------------  test MinMaxScaler() ------------------



####  ------------------------------  test GaussianNB() ------------------
# with PCA gs tuned for precision: 0.82553	Precision: 0.34505	Recall: 0.34350	F1: 0.34427
# with PCA  gs tuned for recall: Accuracy: 0.78527	Precision: 0.26672	Recall: 0.34900	F1: 0.30236
# NO PCA gs tuned for precision: Accuracy: 0.84013	Precision: 0.36481	Recall: 0.26850	F1: 0.30933	
###(winner) NO PCA  tuned for recall: Accuracy: 0.80913	Precision: 0.31280	Recall: 0.36050	F1: 0.33496 (KBest up to 17) (result 17)
# NO PCA  tuned for recall: Accuracy: 0.66987	Precision: 0.20445	Recall: 0.51050	F1: 0.29196	F2: 0.39287 (KBest up to 22) (result 21)
algo = GaussianNB()
# No Scale
#Accuracy: 0.83893, Precision: 0.37963, Recall: 0.32800, F1: 0.35193, F2: 0.33717

####  ------------------------------  test GaussianNB() ------------------


####  ------------------------------  test KNeighborsClassifier() ------------------
"""
# NO PCA: Accuracy: 0.86407	Precision: 0.46910	Recall: 0.14800	F1: 0.22501	F2: 0.17147
# WITH PCA: Accuracy: 0.85713	Precision: 0.36832	Recall: 0.10000	F1: 0.15729	F2: 0.11705
parameters = [{'weights': ['uniform', 'distance'], 'n_neighbors': [4,5,6], 'metric': ['minkowski','euclidean'], 'p':[1,2,3]}]
svr = KNeighborsClassifier()
algo = grid_search.GridSearchCV(svr, parameters)

algo = KNeighborsClassifier()
# NO GS NO PCA with Accuracy: 0.86407	Precision: 0.46910	Recall: 0.14800	F1: 0.22501	
"""
####  ------------------------------  test KNeighborsClassifier() ------------------



# scaler for KNN
#select = MinMaxScaler()

select = SelectKBest(f_classif)
# scaler
scaler = preprocessing.StandardScaler()
# PCA
pca = PCA()
# selectKBest

cv = StratifiedShuffleSplit(labels_train, test_size = 0.5,n_iter =  50, random_state = 42)


#pipeline_object= Pipeline(steps= [("SKB", select), ("trees", algo)])
#pipeline_object= Pipeline(steps= [("scaler", scaler), ("PCA", pca),("SKB", select), ("trees", algo)])

# for knn also include scale
# NO PCA (NO scale)
# pipeline_object= Pipeline(steps= [("SKB", select), ("trees", algo)])
# NO PCA (scale for KNN)
pipeline_object= Pipeline(steps= [("scaler", scaler), ("SKB", select), ("trees", algo)])
# WITH PCA
# pipeline_object= Pipeline(steps= [("scaler", scaler), ("PCA", pca),("SKB", select), ("trees", algo)])

# when pca in place
"""
pca_params = {"PCA__n_components":[17,18,19,20,21,22], "PCA__whiten": [True]}
kbest_params = {"SKB__k":[2, 4, 6, 8,10,12,14,15,16,17]}

pca_params.update(kbest_params)

gs = GridSearchCV(
    pipeline_object,
    pca_params,
    verbose=0,
    scoring = 'recall',
    cv=cv
)

gs.fit(features, labels)
"""

# when pca NOT in place
kbest_params = {"SKB__k":[2, 4, 6, 8,10,12,14,15,16,17]}
#kbest_params = {"SKB__k":[2, 4, 6, 8,10,12,14,15,16,17,18,19,20,21,22]}



gs = GridSearchCV(
    pipeline_object,
    kbest_params,
    verbose=0,
    scoring = 'recall',
    cv=cv
)

gs.fit(features, labels)


 #This is the model that passes to tester.py
clf = gs.best_estimator_

print " "
print "Optimal Model - by Grid Search"
print clf
print " "

best_parameters = gs.best_estimator_.get_params()

print " "
print "Best Parameters- by Grid Search"
print best_parameters
print " "

labels_pred = gs.predict(features_train)

# Print Results  (will print the Grid Search score)
print "Grid Search Classification report:" 
print " "
print classification_report(labels_train, labels_pred)
print ' ' 

# Print Results  (will print the tester.py score)
print "tester.py Classification report:" 
print " "
test_classifier(clf, my_dataset, features_list)
print " "



## get the scores for the features included

features_k= gs.best_params_['SKB__k']
SKB_k=SelectKBest(f_classif, k=features_k)
SKB_k.fit_transform(features_train, labels_train) 
feature_scores = SKB_k.scores_
features_selected=[features_list[i+1]for i in SKB_k.get_support(indices=True)]
features_scores_selected=[feature_scores[i]for i in SKB_k.get_support(indices=True)]
print ' '
print 'Selected Features', features_selected
print 'Feature Scores', features_scores_selected

len(features_selected) 
len(features_scores_selected)

### generate files for testing

dump_classifier_and_data(clf, my_dataset, features_list)

