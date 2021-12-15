from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt  # 可视化模块

# TODO 1. --------data cleaning--------------
import numpy as np
import pandas as pd
from matplotlib.pyplot import clf
from sklearn import datasets, svm
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score

df = pd.read_csv("healthcare-dataset-stroke-data.csv")
# Determining that if missing patterns exist
df.apply(lambda x: sum(x.isnull()), axis=0)
bmi_mean = np.mean(df["bmi"])
target = df["bmi"][1]

# Dealing with the N/A value in the bmi column
for index in range(0, len(df["bmi"])):
    if str(df["bmi"][index])[0] == "n":
        df["bmi"][index] = format(bmi_mean, '.1f')
    else:
        df["bmi"][index] = format(df["bmi"][index], '.1f')

# Switching the float datatype of the column age into integer
for index in range(0, len(df["age"])):
    df["age"] = df["age"].astype(int)

# Switching the string datatype of the column gender into integer
for index in range(0, len(df["gender"])):
    if df["gender"][index] == "Male":
        df["gender"][index] = 1
    else:
        df["gender"][index] = 2

# Switching the string datatype of the column married into integer
for index in range(0, len(df["ever_married"])):
    if df["ever_married"][index] == "Yes":
        df["ever_married"][index] = 1
    else:
        df["ever_married"][index] = 0

# Switching the string datatype of the column married into integer
for index in range(0, len(df["work_type"])):
    if df["work_type"][index] == "Private":
        df["work_type"][index] = 1
    elif df["work_type"][index] == "Self-employed":
        df["work_type"][index] = 2
    else:
        df["work_type"][index] = 3

# Switching the string datatype of the column living into integer
for index in range(0, len(df["Residence_type"])):
    if df["Residence_type"][index] == "Urban":
        df["Residence_type"][index] = 1
    else:
        df["Residence_type"][index] = 2

# Switching the string datatype of the column smoking status into integer
for index in range(0, len(df["smoking_status"])):
    if df["smoking_status"][index] == "formerly smoked":
        df["smoking_status"][index] = 1
    elif df["smoking_status"][index] == "never smoked":
        df["smoking_status"][index] = 2
    else:
        df["smoking_status"][index] = 3

# Switching the integer datatype of the column stroke into string
for index in range(0, len(df["stroke"])):
    if df["stroke"][index] == 1:
        df["stroke"][index] = "Yes"
    else:
        df["stroke"][index] = "No"

# Knitting the dataframe into a csv format file
df.to_csv("preprocessed_data.csv")
print(df)

# TODO 2. --------feature selection------------
df = pd.read_csv("preprocessed_data.csv")
divide = np.random.rand(len(df)) < 0.8
train_data = df[divide]
test_data = df[~divide]
features = train_data.shape[1] - 1
# 我在下面split了, 这里可以不split了

# TODO 3. --------split data and run model-----
df_data = df.drop(df.columns[-1], axis=1)
df_label = df[df.columns[-1]]
X_train, X_test, Y_train, Y_test = train_test_split(df_data, df_label, test_size=0.2, random_state=0)

model = LogisticRegression(
    penalty="l2", random_state=None, solver="liblinear", max_iter=1000,
    multi_class='ovr', verbose=0,
)
# TODO 4. --------tune hyperparameters---------
parameters = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_search = GridSearchCV(model, parameters, n_jobs=-1, scoring='accuracy', cv=[5, 10, 20])
grid_search.fit(X_train, Y_train)
print(grid_search.grid_scores_, grid_search.best_params_, grid_search.best_score_)

prepro = grid_search.predict_proba(X_test)
acc = grid_search.score(X_test, Y_test)

# TODO 5. --detect and then prevent overfitting-
# 等1，2，3，4搞完，看模型的train的正确率和test的正确率，再决定是否有overfitting
(X, y) = datasets.load_digits(return_X_y=True)
# print(X[:2,:])

train_sizes, train_score, test_score = learning_curve(RandomForestClassifier(), X, y,
                                                      train_sizes=[0.1, 0.25, 0.5, 0.75, 1], cv=10, scoring='accuracy')
train_error = 1 - np.mean(train_score, axis=1)
test_error = 1 - np.mean(test_score, axis=1)
plt.plot(train_sizes, train_error, 'o-', color='r', label='training')
plt.plot(train_sizes, test_error, 'o-', color='g', label='testing')
plt.legend(loc='best')
plt.xlabel('training examples')
plt.ylabel('error')
plt.show()
