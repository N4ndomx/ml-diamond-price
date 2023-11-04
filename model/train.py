
from os import PathLike
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
import pandas as pd
import pathlib
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

def score_model(model, X_t, X_v, y_t, y_v):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

df = pd.read_csv(pathlib.Path('data/diamonds_data.csv'))


print(df.head())
y = df.price
X = df.drop('price', axis = 1)


x_train_full, x_valid_full, y_train, y_valid = train_test_split(X,y, test_size = 0.2)


# Get list of categorical variables
s = (x_train_full.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)

# Make copy to avoid changing original data
label_X_train = x_train_full.copy()
label_X_valid = x_valid_full.copy()

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(x_train_full[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(x_valid_full[object_cols])

# Select numerical columns
numerical_cols = [cname for cname in x_train_full.columns if x_train_full[cname].dtype in ['int64', 'float64']]

# Valores Nulos?
print( df.isnull().values.any())

# Define the models
model = RandomForestRegressor(n_estimators=100, random_state=0)

mae = score_model(model,label_X_train,label_X_valid, y_train, y_valid)
print("Model MAE: ", mae)

y_pred = model.predict(label_X_valid)
print ("Valid : \n",label_X_valid.head())

print ('Saving model...')

dump(model, pathlib.Path('model/model_ps.joblib'))