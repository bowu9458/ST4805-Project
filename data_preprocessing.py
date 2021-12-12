import numpy as np
import pandas as pd

df = pd.read_csv("stroke.csv")
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
# for index in range(0, len(df["stroke"])):
#     if df["stroke"][index] == 1:
#         df["stroke"][index] = "Yes"
#     else:
#         df["stroke"][index] = "No"

# Knitting the dataframe into a csv format file
# df.to_csv("preprocessed_data.csv")
print(df)
print(str(df["age"][1])[:-2])
