# Project Overview
In this project, you will play detective, and put your machine learning skills to use by building an algorithm to identify Enron Employees who may have committed fraud based on the public Enron financial and email dataset.

Prepare for this project with: Intro to Machine Learning.

# Project Details

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, you will play detective, and put your new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. To assist you in your detective work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.

Resources Needed
You should have python and sklearn running on your computer, as well as the starter code (both python scripts and the Enron dataset) that you downloaded as part of the first mini-project in the Intro to Machine Learning course. You can get the starter code on git: git clone https://github.com/udacity/ud120-projects.git

The starter code can be found in the final_project directory of the codebase that you downloaded for use with the mini-projects. Some relevant files: 

poi_id.py : Starter code for the POI identifier, you will write your analysis here. You will also submit a version of this file for your evaluator to verify your algorithm and results. 

final_project_dataset.pkl : The dataset for the project, more details below. 

tester.py : When you turn in your analysis for evaluation by Udacity, you will submit the algorithm, dataset and list of features that you use (these are created automatically in poi_id.py). The evaluator will then use this code to test your result, to make sure we see performance that’s similar to what you report. You don’t need to do anything with this code, but we provide it for transparency and for your reference. 

emails_by_address : this directory contains many text files, each of which contains all the messages to or from a particular email address. It is for your reference, if you want to create more advanced features based on the details of the emails dataset. You do not need to process the e-mail corpus in order to complete the project.

Steps to Success
We will provide you with starter code that reads in the data, takes your features of choice, then puts them into a numpy array, which is the input form that most sklearn functions assume. Your job is to engineer the features, pick and tune an algorithm, and to test and evaluate your identifier. Several of the mini-projects were designed with this final project in mind, so be on the lookout for ways to use the work you’ve already done.

As preprocessing to this project, we've combined the Enron email and financial data into a dictionary, where each key-value pair in the dictionary corresponds to one person. The dictionary key is the person's name, and the value is another dictionary, which contains the names of all the features and their values for that person. The features in the data fall into three major types, namely financial features, email features and POI labels.

financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

POI label: [‘poi’] (boolean, represented as integer)

You are encouraged to make, transform or rescale new features from the starter features. If you do this, you should store the new feature to my_dataset, and if you use the new feature in the final algorithm, you should also add the feature name to my_feature_list, so your evaluator can access it during testing. For a concrete example of a new feature that you could add to the dataset, refer to the lesson on Feature Selection.

In addition, we advise that you keep notes as you work through the project. As part of your project submission, you will compose answers to a series of questions (also given on the next page) to understand your approach towards different aspects of the analysis. Your thought process is, in many ways, more important than your final project and we will by trying to probe your thought process in these questions.

# Results

1. Summarize for us the goal of this project and how machine learning is useful 
in trying to accomplish it. As part of your answer, give some background on the 
dataset and how it can be used to answer the project question. Were there any 
outliers in the data when you got it, and how did you handle those?  [relevant 
rubric items: “data exploration”, “outlier investigation”]

The goal of this project is to identify the people who comitted fraud in Enron,
based on financial and email data from Enron scandal.

20 out of 21 features all contained missing values the only feature that did 
not contain missing values was 'POI'. There are 146 people in the dataset and 
21 features availables, such as, 'salary', 'deferral_payments', 
'total_payments', 'loan_advances' or 'bonus' among others for each person. Only
18 out of 146 are POI (People of Interest), 35 according to our definition. And
128 non-POI.


The dataset contain numerous missing data (NaN values).

| Feature                    | Missing Values |
| -------------------------- |:--------------:|
| salary                     | 51             |
| to_messages                | 60             |
| deferral_payments          | 107            |
| total_payments             | 21             |
| loan_advances              | 142            |
| bonus                      | 64             |
| email_address              | 35             |
| restricted_stock_deferred  | 128            |
| total_stock_value          | 20             |
| shared_receipt_with_poi    | 60             |
| long_term_incentive        | 80             |
| exercised_stock_options    | 44             |
| from_messages              | 60             |
| other                      | 53             |
| from_poi_to_this_person    | 60             |
| from_this_person_to_poi    | 60             |
| poi                        | 0              |
| deferred_income            | 97             |
| expenses                   | 51             |
| restricted_stock           | 36             |
| director_fees              | 129            |


Through data visualization and the csv file I got from the dataset, we can 
easily appreciate the outlier, 'TOTAL', 'THE TRAVEL AGENCY IN THE PARK' and 
'LOCKHART EUGENE E'.

LOCKHART EUGENE E: Contains only NaN values.
TOTAL: It's an extreme outlier, you can see it in the scatter plot.
THE TRAVEL AGENCY IN THE PARK: It's not a name.

I will remove this records.

---

2. What features did you end up using in your POI identifier, and what 
selection process did you use to pick them? Did you have to do any scaling? Why 
or why not? As part of the assignment, you should attempt to engineer your own 
feature that does not come ready-made in the dataset -- explain what feature 
you tried to make, and the rationale behind it. (You do not necessarily have to 
use it in the final analysis, only engineer and test it.) In your feature 
selection step, if you used an algorithm like a decision tree, please also give 
the feature importances of the features that you use, and if you used an 
automated feature selection function like SelectKBest, please report the 
feature scores and reasons for your choice of parameter values.  [relevant 
rubric items: “create new features”, “intelligently select features”, “properly
scale features”]

I created two new features 'fraction_to_poi' and 'fraction_from_poi', which 
is the frequency an employee sent emails to POIs and the frequency and employee
received emails from POIs.
It seems important for me how often you have been in touch with a POIs in order 
to be involved in the fraud. However, these two new features, eventually will 
not be importants.

Using SelectKBest to get the best features, I compared the results (accuarcy, 
precision and recall) of all of them and eventually chose 7. You can see below 
the table and chart of the results. 

Normal Features:

| k   | Accuarcy | Precision  | Recall  |
| --- |:--------:|:----------:|:-------:|
| 2   | 0.85210  | 0.48304    | 0.30477 |
| 3   | 0.83949  | 0.45824    | 0.29800 |
| 4   | 0.84769  | 0.44512    | 0.27886 |
| 5   | 0.84675  | 0.45052    | 0.31850 |
| 6   | 0.85738  | 0.44924    | 0.35879 |
| 7   | 0.85261  | 0.43183    | 0.37583 |
| 8   | 0.84667  | 0.40897    | 0.37417 |
| 9   | 0.84535  | 0.37247    | 0.31758 |

With new Features:

| k   | Accuarcy | Precision  | Recall  |
| --- |:--------:|:----------:|:-------:|
| 2   | 0.84205  | 0.43570    | 0.28990 |
| 3   | 0.84128  | 0.44283    | 0.32108 |
| 4   | 0.83925  | 0.41376    | 0.30562 |
| 5   | 0.83225  | 0.40340    | 0.34635 |
| 6   | 0.84119  | 0.39630    | 0.37286 |
| 7   | 0.83571  | 0.37302    | 0.37644 |
| 8   | 0.83143  | 0.36088    | 0.37719 |
| 9   | 0.83023  | 0.32825    | 0.32117 |

![Comparation](../master/img/img1.png)

We can easily see that Precision drop when we use more than 5 features, 
meanwhile recall peack is between six and seven. Accuarcy keeps steady from
2 to 8 features. This is the reason why I chose only 7 features.
 
An alternative approach would be to select the features based on the scores 
provided. A cut-off point could be determined based on where the scores 
drop-off. For example, based on the scores provided, I see that scores fell 
significantly after the salary feature (4th highest).

Below we can see a table with the features with highest variance. 

| Feature                    | Score               |
| -------------------------- |:-------------------:|
| exercised_stock_options    | 24.815079733218194  |
| total_stock_value          | 24.182898678566879  |
| bonus                      | 20.792252047181535  |
| salary                     | 18.289684043404513  |
| deferred_income            | 11.458476579280369  |
| long_term_incentive        | 9.9221860131898225  |
| restricted_stock           | 9.2128106219771002  |
| total_payments             | 8.7727777300916756  |
| shared_receipt_with_poi    | 8.589420731682381   |
| loan_advances              | 7.1840556582887247  |
| expenses                   | 6.0941733106389453  |
| from_poi_to_this_person    | 5.2434497133749582  |
| other                      | 4.1874775069953749  |
| from_this_person_to_poi    | 2.3826121082276739  |
| director_fees              | 2.1263278020077054  |
| to_messages                | 1.6463411294420076  |
| deferral_payments          | 0.22461127473600989 |
| from_messages              | 0.16970094762175533 |
| restricted_stock_deferred  | 0.06549965290994214 |

I will use 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 
'deferred_income', 'long_term_incentive' and 'restricted_stock'.
After feature engineering and using SelectKBest, I scaled features using 
MinMaxScaler(). If we do not normalize the features, some machine learning 
algorithms might not work properly.

---
3. What algorithm did you end up using? What other one(s) did you try? How did 
model performance differ between algorithms?  [relevant rubric item: “pick an 
algorithm”]

I test 4 algorithm, SVM, Regression, KMeans and Naive Bayes. Below we can see 
the results with the best parameters for each case.

| Feature        | Accuarcy        | Precision      | Recall          |
| -------------- |:---------------:|:--------------:|:---------------:|
| Naive Bayes    | 0.854761904762  | 0.432977633478 | 0.373191558442  |
| K-means        | 0.337619047619  | 0.664069275329 | 0.324404761905  |
| Regression     | 0.859761904762  | 0.400333333333 | 0.190000721501  |
| SVM            | 0.866428571429  | 0.141666666667 | 0.0384523809524 |

SVM parameters:
    kernel = 'linear'
    C = 1
    gamma = 1
    
Regression parameters: 
    tol = 1
    C = 0.1

KMeans parameters:
    tol = 0.1
    n_clusters = 5

As final decision, we choose Naive Bayes algorythm, as it has the highest 
precision and recall. We can see below the accuarcy, precision and recall for
for each algorithm with the new features included. 

| Feature        | Accuarcy        | Precision      | Recall          |
| -------------- |:---------------:|:--------------:|:---------------:|
| Naive Bayes    | 0.842619047619  | 0.395617965368 | 0.37384992785   |
| K-means        | 0.369047619048  | 0.760431767809 | 0.37380952381   |
| Regression     | 0.859285714286  | 0.466736263736 | 0.244305916306  |
| SVM            | 0.86619047619   | 0.093833333333 | 0.0442738095238 |

SVM parameters:
    kernel = 'rbf'
    C = 0.1
    gamma = 1
    
Regression parameters: 
    tol = 1
    C = 0.1

KMeans parameters:
    tol = 1
    n_clusters = 5

As we expected, when we include the new features most of parameters (Accuarcy,
Precission and Recall) get worse.
                                                                
---
4. What does it mean to tune the parameters of an algorithm, and what can 
happen if you don’t do this well?  How did you tune the parameters of your 
particular algorithm? What parameters did you tune? (Some algorithms do not 
have parameters that you need to tune -- if this is the case for the one you 
picked, identify and briefly explain how you would have done it for the model 
that was not your final choice or a different model that does utilize parameter 
tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss 
parameter tuning”, “tune the algorithm”]

Tune the parameters of an algorithm refers to find the parameters which the
highest Accuarcy, Precission and Recall for an algorithm. This might be so 
effective but it might lead to overfit and get a bad results of the learning
process.

You can see in the question above the best parameters found for each of the 
algorithms tested. Naive Bayes, whcih is the choosen algorithm does not need
parameters.

I used GridSearchCV to get the best parameter for each algorithm.

SVM parameters:
    kernel = 'linear'
    C = 1
    gamma = 1
    
Regression parameters: 
    tol = 1
    C = 0.1

KMeans parameters:
    tol = 0.1
    n_clusters = 5


---
5. What is validation, and what’s a classic mistake you can make if you do it 
wrong? How did you validate your analysis?  [relevant rubric items: “discuss 
validation”, “validation strategy”]

Model validation is referred to as the process where a trained model is 
evaluated with a testing data set.

Model validation is carried out after model training. Together with model 
training, model validation aims to find an optimal model with the best 
performance.

A classic mistake is overfitting, it happend when the model performed well in
training but not in the test set. In order to avoid this overfitting, I have
created a function called evaluateClf which I calculate the mean of accuarcy, 
precision and recall of 100 different training data.


---
6. Give at least 2 evaluation metrics and your average performance for each of 
them.  Explain an interpretation of your metrics that says something 
human-understandable about your algorithm’s performance. [relevant rubric item: 
“usage of evaluation metrics”]

With the help of train_test_split I get the set of training data in a random
way. I use a test_size=0.3. We get a result similar to our model.

Results without new Features:

| Feature        | Accuarcy        | Precision      | Recall          |
| -------------- |:---------------:|:--------------:|:---------------:|
| Naive Bayes    | 0.854761904762  | 0.432977633478 | 0.373191558442  |

Results with new Features:

| Feature        | Accuarcy        | Precision      | Recall          |
| -------------- |:---------------:|:--------------:|:---------------:|
| Naive Bayes    | 0.835714285714  | 0.373023809524 | 0.376445165945  |

I have considered precision and recall the most importants parameters. 
Precision indicates the ratio of true positives to the records POIs. It means
that every 100 people there are 43 POIs. and only 37 are correctly classified
as POIs. Recall is the ratio of true positives to the records POIs.

# Corrections
There is still an issue when saving the code, you are actually saving a decision tree and not your algorithm.

As a side-note you could potentially improve the feature selection process by using an algorithmic approach to determine the number of features to be chosen (like RFE). I've left a comment regarding the matter, I hope you might find it interesting.

poi_id.py can be run to export the dataset, list of features and algorithm, so that the final algorithm can be checked easily using tester.py.
The issue raised by the previous reviewer has not been addressed, the final exported algorithm is a decision tree and does not reach 0.3 in precision and recall, one way to fix this and use your algorim is to comment out the following lines:
```javascript
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

prediction = clf.predict(features_test)
print accuracy_score(prediction, labels_test)
```
####Student response addresses the most important characteristics of the dataset and uses these characteristics to inform their analysis. Important characteristics include:
total number of data points
allocation across classes (POI/non-POI)
number of features used
are there features with many missing values? etc.
Pro Tip: There are several other options to deal with missing values like:
a. Replacing the values with means or medians.
b. Remove the features that have an exceeding number of missing values.
c. More complex approaches rely on analysing the distribution of missing values: https://en.wikipedia.org/wiki/Missing_data
http://scikit-learn.org/stable/modules/preprocessing.html


####Univariate or recursive feature selection is deployed, or features are selected by hand (different combinations of features are attempted, and the performance is documented for each one). Features that are selected are reported and the number of features selected is justified. For an algorithm that supports getting the feature importances (e.g. decision tree) or feature scores (e.g. SelectKBest), those are documented as well.
Pro Tip: Please note that you can leverage the power of recursive feature selection to automate the selection process and find a good indication of the number of relevant features, here is an example of how the code might look like:
```javascript
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
svc = SVC(kernel="linear")
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(labels, 50),
          scoring='precision')
rfecv.fit(features, labels)
print("Optimal number of features : %d" % rfecv.n_features_)
print rfecv.support_
features=features[:,rfecv.support_]
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
```


####At least two different algorithms are attempted and their performance is compared, with the best performing one used in the final analysis.
Pro Tip (Advanced): Xgboost, one of Kaggle’s top algorithms.
In the recent years one algorithm emerged as favourite in the machine learning community, it is actually one of the most used in Kaggle: Xgboost.
Here you can find an informative discussion on why that is the case: https://www.quora.com/Why-is-xgboost-given-so-much-less-attention-than-deep-learning-despite-its-ubiquity-in-winning-Kaggle-solutions
The algorithm is not available sci-kit learn, here is how you can start working with it:
http://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/


####Performance of the final algorithm selected is assessed by splitting the data into training and testing sets or through the use of cross validation, noting the specific type of validation performed.
Pro Tip: The dataset is small and skewed towards non-POI, we need a technique that accounts for that or the risk is that we would not be able to assess, in the validation phase, the real potential of our algorithm in terms of performance metrics. The chance of randomly splitting skewed and non representative validation sub-sets could be high, therefore the need to use stratification (preservation of the percentage of samples for each class) to achieve robustness in a dataset with the aforementioned limitations.
http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
