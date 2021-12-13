from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

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

# Switching the integer datatype of the column age into string
for index in range(0, len(df["stroke"])):
    if df["stroke"][index] == 1:
        df["stroke"][index] = "Yes"
    else:
        df["stroke"][index] = "No"

# Knitting the dataframe into a csv format file
df.to_csv("preprocessed_data.csv")
print(df)


# TODO 2. --------feature selection------------
data = pd.read_csv("preprocessed_data.csv")
divide = np.random.rand(len(data)) < 0.8
train_data = data[divide]
test_data = data[~divide]
features = train_data.shape[1] - 1

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

# TODO 5. --detect and then prevent overfitting-
