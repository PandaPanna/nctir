 
#1.Import library
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import classification_report
from prettytable import PrettyTable
from sklearn import metrics
import numpy as np
from IPython.display import Image
from sklearn import preprocessing

#2.Generate the training set
# Asset name should be lower case 
asset = "bench"


# Generate training dataset
# Read data and combine them as training data set
df_train = pd.read_excel('/content/drive/My Drive/NCTIR/{}/Train.xlsx'.format(asset))
print(df_train.columns)
# The positive samples vs negative sample
train_pos = df_train[df_train["TARGET"]==1]
train_neg = df_train[df_train["TARGET"]==0]
print("Positive sample: ", len(train_pos), "Negative sample: ", len(train_neg))

df_train.head()


# Print the positive and negative sample
color = ["lightblue", "coral"]
counts  = df_train['TARGET'].value_counts()
print(counts)
sns.barplot(counts.index, counts.values, palette=color)
plt.title('Imbalanced training set')
plt.ylabel('Number of Samples', fontsize=12)
plt.xlabel('Target Label', fontsize=12)
plt.show()
df_train = df_train.fillna(-1)

# If samples are not enough, let's do resampling
train_pos = train_pos.sample(len(train_neg), replace=True)
df_train = pd.concat([train_pos, train_neg])
counts  = df_train['TARGET'].value_counts()

sns.barplot(counts.index, counts.values, palette=color)
plt.title('Balanced training set')
plt.ylabel('Number of Samples', fontsize=12)
plt.xlabel('Target Label', fontsize=12)
plt.show()

# Replace the missing data with '-1'
df_train = df_train.fillna(-1)

# Choose dropped parameters
dropped_parameters = ["Feature_Type", "ID","autocad_layer_desc", "autocad_source_filename"]
train = df_train.drop(dropped_parameters, axis = 1)

# Label ecoding or hot ecoding 
train = pd.get_dummies(train, prefix_sep="_", columns=["COLOR"])
train = pd.get_dummies(train, prefix_sep="_", columns=["ORIGINAL_COLOR"])
train = pd.get_dummies(train, prefix_sep="_", columns=["ORIGINAL_ENTITY_TYPE"])
train = pd.get_dummies(train, prefix_sep="_", columns=["LINETYPE"])

train = train.fillna(-1)
X = train.iloc[:, 1:]
y = train.iloc[:,:1]


print("Target, (rows, columns)")
print("TARGET  = 1, ", train[train["TARGET"]==1].shape) 
print("TARGET  = 0, ", train[train["TARGET"]==0].shape) 


train[train['TARGET']==1].head()

# The features
print(len(X.columns), "\n",X.columns)


#3.Visualise the training data distribution
# Plot categorical data distribution 
def plot_data(df):
  """plot the binary data"""
  non = []
  for variable in df.columns[0:]:
    uniques = df[variable].unique()
    uniques = sorted(list(uniques))
    variable_len = len(uniques)
    if variable_len < 4:
      fig, axs = plt.subplots(1, 2)
      fig.suptitle(variable)
      for target in range(2):
        counts = []
        for unique in uniques:
          number = df[(df[variable]==unique) & (df["TARGET"]==target)].count()["TARGET"]
          counts.append(number)
        colors = ["lightcoral", "lightblue", "grey", "gold"]
        axs[target].pie(counts, labels=uniques, colors=colors, startangle=90, autopct='%.1f%%')
        axs[target].set_title("Target == {}".format(target))
      plt.show()
    else:
      non.append(variable)
  print("non category data are: ", non)
        

plot_data(train)


#4.Generate the test set
# Generate test data
df_test = pd.read_excel('/content/drive/My Drive/NCTIR/{}/Test.xlsx'.format(asset))
print(df_test.TARGET.value_counts())
df_test = df_test.fillna(-1)



# Drop parameters
test = df_test.drop(dropped_parameters, axis = 1)
# Label ecoding or hot ecoding 
test = pd.get_dummies(test, prefix_sep="_", columns=["COLOR"])
test = pd.get_dummies(test, prefix_sep="_", columns=["ORIGINAL_COLOR"])
test = pd.get_dummies(test, prefix_sep="_", columns=["ORIGINAL_ENTITY_TYPE"])
test= pd.get_dummies(test, prefix_sep="_", columns=["LINETYPE"])

# For test data, the more features are created during the hot ecoding
test = test.loc[:,train.columns]
test = test.fillna(-1)
print("Target, (rows, columns)")
print("TARGET  = 1, ", test[test["TARGET"]==1].shape) 
print("TARGET  = 0, ", test[test["TARGET"]==0].shape) 

X_test = test.iloc[:, 1:]
y_test = test.iloc[:,:1]
print(X_test.columns)


counts  = df_test['TARGET'].value_counts()
print(counts)
sns.barplot(counts.index, counts.values, palette=color)
plt.title('Test set')
plt.ylabel('Number of Samples', fontsize=12)
plt.xlabel('Target Label', fontsize=12)
plt.show()

#5.Evaluation functions
#5.1Single algorithm evaluation

# Evaluate model

def test_model(model, X, y, test_X, test_y):
  """Input model name, show the ROC curve and metrix"""
  model.fit(X, y)
  y_predict = model.predict(test_X)
  print("%s: %f" % (str(model), metrics.accuracy_score(y_test, y_predict)))
  print(metrics.classification_report(y_test, y_predict))
  y_predict_proba = model.predict_proba(test_X)
  fpr, tpr, _ = metrics.roc_curve(y_test, y_predict_proba[:,1])
  roc_auc = metrics.auc(fpr, tpr)
  
  plt.figure()
  lw = 2
  plt.plot(fpr, tpr, color='darkorange',
           lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic curve')
  plt.legend(loc="lower right")
  plt.show()
  # Print contigency matrix
  print("Contigency matrix")

  metrix = metrics.confusion_matrix(y_test, y_predict)
  
  table = PrettyTable()
  table.title = 'Results for prediction'
  table.field_names = [" ", 'Predicted 0 ', 'Predicted 1', "Total"]
  table.add_row(["Real 0", metrix[0][0], metrix[0][1], sum(metrix[0])])
  table.add_row(["Real 1", metrix[1][0],metrix[1][1], sum(metrix[1])])
  table.add_row(["-------------", "-------------", "-------------", "-------------"])
  table.add_row(["Total: ", metrix[0][0]+ metrix[1][0], metrix[0][1]+metrix[1][1],sum(metrix[0])+sum(metrix[1])])
  print(table)
  tn, fp, fn, tp = metrix.ravel()
  FP =   fp/(fp+tn)
  FN =   fn/(tp+fn)
  TP =   tp/(fn+tp)
  TN =   tn/(tn+fp)
  precision = tp/(tp+fp)
  accuracy = metrics.accuracy_score(y_test, y_predict)
  print("False Positive Rate: ", "%.3f" % FP)
  print("False Negative Rate: ", "%.3f" % FN)
  print("True Negative Rate (Specificity): ", "%.3f" % TN)
  print("True Positive Rate (Recall): ", "%.3f" % TP)
  print("Precision: ", "%.3f" % precision)
  print("Accuracy:", "%.3f" % accuracy)
  print("Cross validation accuracy score: " + "{:.3f}".format(get_cv_acc(model, X, y, X_test, y_test)))
  class_names=[0,1] # name  of classes
  fig, ax = plt.subplots()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names)
  plt.yticks(tick_marks, class_names)
  # create heatmap
  sns.heatmap(pd.DataFrame(metrix), annot=True, cmap="YlGnBu" ,fmt='g')
  ax.xaxis.set_label_position("top")
  plt.tight_layout()
  plt.title('Confusion matrix', y=1.1)
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  
#5.2Cross validation
# Build the k-fold cross-validator
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
def get_cv_acc(model, X, y, X_test, y_test, k=3):
  """Print cross validation accuracy score"""
  kfold = KFold(n_splits=k, random_state=7)
  X_total = pd.concat([X, X_test])
  y_total = pd.concat([y, y_test])
  result = cross_val_score(model, X_total, y_total, cv=kfold, scoring='accuracy')
  return result.mean()

#5.3Multiple model evaluation function

# Define a function to compare multiple models
import os.path



def test_multi_model(asset, models, X, y, test_X, test_y):
  """Input model name, show the ROC curve and metrix"""
  table = PrettyTable()
  table.reversesort = True
  table.title = 'Results are: ' 
  heads = ["Algorithm","False positive Rate", "False negative rate","True positive rate","True negative rate","Accuracy", "CV_Accuracy","Precision", "F1 score"]
  table.field_names = heads
  for model_name in models.keys():
    print(model_name)
    heads.append(model_name)
    model = models[model_name]
    model.fit(X, y)
    y_predict = model.predict(test_X)
    metrix = metrics.confusion_matrix(y_test, y_predict)
    tn, fp, fn, tp = metrix.ravel()
    FP =   fp/(fp+tn)
    FN =   fn/(tp+fn)
    TP =   tp/(fn+tp)
    TN =   tn/(tn+fp)
    precision = tp/(tp+fp)
    accuracy = metrics.accuracy_score(y_test, y_predict)
    cv_acc = get_cv_acc(model, X, y, X_test, y_test)
    row = [model_name]
    row.append("{:.3f}".format(FP))
    row.append("{:.3f}".format(FN))
    row.append("{:.3f}".format(TP))
    row.append("{:.3f}".format(TN))
    row.append("{:.3f}".format(accuracy))
    row.append("{:.3f}".format(cv_acc))
    row.append("{:.3f}".format(precision))
    f1_score = 2*((precision*TP)/(precision+TP))
    row.append("{:.3f}".format(f1_score))
    table.add_row(row)
  data = table.get_string(sortby="F1 score")
  print(data)
  # Write into txt file
  save_path = "/content/drive/My Drive/NCTIR/{}".format(asset)
  with open(os.path.join(save_path, 'accuracy.txt'), "w") as f:
    f.write(data)
    
#6.Evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier

# Add model into the dictionary to compare the result

models = {"LogistcRegression": LogisticRegression(), 
          "KNN n=3": neighbors.KNeighborsClassifier(n_neighbors=3), 
          "KNN n=5": neighbors.KNeighborsClassifier(n_neighbors=5), 
          "KNN n=10": neighbors.KNeighborsClassifier(n_neighbors=10), 
          "DecisionTree":DecisionTreeClassifier(),
          "RandomForest n=5": RandomForestClassifier(n_estimators=5,  random_state= 0),
          "RandomForest n=10": RandomForestClassifier(n_estimators=10,  random_state= 0),
          "NaiveBayes":GaussianNB(),
          "SVM (kernel='linear')": SVC(kernel='linear'),
          "SVM (kernel='sigmoid')": SVC(kernel='sigmoid')}

"""
          "StochasticGradientDescent": linear_model.SGDClassifier(),
          "Ridge":linear_model.RidgeClassifier(),
          "Perceptron":linear_model.Perceptron(random_state=0),
          "PassiveAggressive":linear_model.PassiveAggressiveClassifier(random_state=0),
"""

test_multi_model(asset, models, X, y, X_test, y_test)


#7.Get accuracy table with CSV format
# Get the accuracy table with CSV
data = [["Algorithm","False positive Rate", "False negative rate","True positive rate","True negative rate","Accuracy", "CV_Accuracy","Precision", "F1 score"]]

f = open("/content/drive/My Drive/NCTIR/{}/accuracy.txt".format(asset), "r")
lines = f.readlines()
for line in lines[3:-1]:
  line = line.strip()
  linedata = line[2:-2].split(" | ")
  data.append(linedata)
data = np.array(data)
r = pd.DataFrame(data=data[1:,1:],
                  index=data[1:,0],
                  columns=data[0,1:])

r.to_csv("/content/drive/My Drive/NCTIR/{}/accuracy.csv".format(asset))
r.head(10)

#8.Test one single model
model =  RandomForestClassifier(n_estimators=10,  random_state=0)
test_model(model, X, y, X_test, y_test)


#9.Get output function
def get_output(model, X, y, X_test, df):
  """Get the output only without test y"""
  model.fit(X, y)
  result = df.loc[:,["ID","TARGET"]]
  result["result"] = model.predict(X_test)
  result.to_excel("/content/drive/My Drive/NCTIR/{}/Result.xlsx".format(asset))
  
#10.Check the false positive/negative function
def get_fp(model, X, y, X_test, df):
  """Return false positive result as df"""
  model.fit(X, y)
  result = df.loc[:,["TARGET", "Feature_Type"]]
  result["result"] = model.predict(X_test)
  print("*"*50, "False Positive", "*"*50)
  fp = result.loc[(result['result'] ==1) & (result["TARGET"]==0)]
  if fp.empty:
    print("There is no false positive result!")
  else:  
    print("False positive results are: ")
    print(fp.to_string(), "\n")
  print("*"*50, "False Negative", "*"*50)
  fn = result.loc[(result['result'] ==0) & (result["TARGET"]==1)]
  if fn.empty:
    print("There is no false negative result!")
  else: 
    print("False negative results are: ")
    print(fn.to_string(), "\n")
  print("*"*50, "True Positive", "*"*50)
  tp = result.loc[(result['result'] ==1) & (result["TARGET"]==1)]
  print(tp.to_string())


get_fp(model, X, y, X_test, df_test)

#10. Feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score as acc
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import collections

def select_features(model, X, y, n=10):
  """Input the number of features you want to have"""
  candidate = []
  # Build step forward feature selection
  sfs1 = sfs(model,
            k_features=n,
            forward=True,
            floating=False,
            verbose=2,
            scoring='accuracy',
            cv=5)
  # Perform SFFS
  sfs1 = sfs1.fit(X, y)
  # The index list of the important features
  feat_cols = list(sfs1.k_feature_idx_)
  for idx in feat_cols:
    candidate.append(X.columns[idx])
  return candidate

model = RandomForestClassifier(n_estimators=5, random_state=0, max_depth=4)
selected_feats = select_features(model, X, y, n=10)
 
# Show the selected top 10 features
selected_feats
print()

clf = RandomForestClassifier(n_estimators=5, random_state=0, max_depth=4)
clf.fit(X.loc[:, selected_feats], y)

y_train_pred = clf.predict(X.loc[:, selected_feats])
print('Training accuracy on selected features: %.3f' % acc(y, y_train_pred))

y_test_pred = clf.predict(X_test.loc[:, selected_feats])
print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))

# Build full model on ALL features, for comparison
clf = RandomForestClassifier(n_estimators=5, random_state=0, max_depth=4)
clf.fit(X, y)

y_train_pred = clf.predict(X)
print('Training accuracy on all features: %.3f' % acc(y, y_train_pred))

y_test_pred = clf.predict(X_test)
print('Testing accuracy on all features: %.3f' % acc(y_test, y_test_pred))