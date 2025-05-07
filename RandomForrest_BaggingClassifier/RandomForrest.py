"""
This program uses random forests and linear classifiers to classify precipitation types.
Note: Uses train test split to determine the model score on data it did not train on

https://www.kaggle.com/datasets/nikhil7280/weather-type-classification
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC


df = pd.read_csv("weather_classification_data.csv")
df.dropna(inplace=True)
df['Cloud Cover'] = df['Cloud Cover'].replace({'clear':0, 'partly cloudy':1, 'overcast':2, 'cloudy':3})
df['Season'] = df['Season'].replace({'Winter':0, 'Spring':1, 'Summer':2, 'Autumn':3})

#Split data
y = df.iloc[:, -1].to_numpy()
X = df.iloc[:, 0:7].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Gridsearch for the optimal max depth of random forests
clf = RandomForestClassifier()
parameters = {"max_depth": range(3,15)}
grid_search = GridSearchCV(clf, param_grid = parameters, cv=5)
grid_search.fit(X_train, y_train)
score_df = pd.DataFrame(grid_search.cv_results_)
print(score_df[["param_max_depth", "mean_test_score", "rank_test_score"]])
maxD = grid_search.best_params_["max_depth"]
clf = RandomForestClassifier(max_depth=maxD, oob_score=True, n_jobs=-1)
clf.fit(X_train, y_train)

print(f"RF Score (Train): {clf.score(X_train, y_train):.3f}")
print(f"RF Score (Test): {clf.score(X_test, y_test):.3f}")
print(f"OOB Score: {clf.oob_score_:.3f}")

#Show the results of the grid searched random forest
cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()

importance = pd.DataFrame(clf.feature_importances_, index=df.columns[0:7])
importance.plot.bar()
plt.show()

#Bagging Classifier, using linear classifier maintains ~90% accuracy, and runs in reasonable time
clfLin = LinearSVC(C = 1.0)
bag_clf = BaggingClassifier(clfLin)
parameters_Bagging = {"max_features":np.linspace(0.1,1, num=10), "n_estimators":range(1,20)}
grid_search_bagging = GridSearchCV(bag_clf, parameters_Bagging, cv=5)
grid_search_bagging.fit(X_train,y_train)
mFeatures = grid_search_bagging.best_params_["max_features"]
nEstimators = grid_search_bagging.best_params_["n_estimators"]
bag_clf = BaggingClassifier(clfLin, max_features=mFeatures,n_estimators=nEstimators,
                            oob_score=True, n_jobs=-1)
bag_clf.fit(X_train, y_train)

cm = confusion_matrix(y_test, bag_clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=bag_clf.classes_)
disp_cm.plot()
plt.show()

#TODO can change score from default accuracy to Precision/Recall for misclassification info
print(f"Bagging Score (Train): {bag_clf.score(X_train, y_train):.3f}")
print(f"Bagging Score (Test): {bag_clf.score(X_test, y_test):.3f}")



