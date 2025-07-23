import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import joblib



df=pd.read_csv("airbnb_listings.csv")
X=df.drop(columns=["ListingID","PricePerNight"])
y=df["PricePerNight"]

preprocessor=ColumnTransformer(
    [("cat",OneHotEncoder(handle_unknown="ignore"),["City","RoomType"])],
    remainder="passthrough"
)

model=Pipeline([
    ("preprocessor",preprocessor),
    ("regressor",RandomForestRegressor(n_estimators=100,random_state=42))
]
)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)

#with mlflow.start_run():
model.fit(X_train,y_train)
joblib.dump(model,"airbnb.pkl")
print("model saved succesfully")
