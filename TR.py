import pymysql
import pandas as pd
import streamlit as st
import pandas as pd
import openpyxl
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
connection = pymysql .connect(host = 'localhost', user = 'root', password = 'mypass')
cursor = connection.cursor()

# Database creation
cr_db = "create database if not exists tourism2"
cursor.execute(cr_db)

#use databse
us_db = "use tourism2"
cursor.execute(us_db)

#region user Defined functions

def createtabel(tbname, columns):
    Query = f"create table if not exists {tbname} {columns}"
    cursor.execute(Query)

#endregion

#region tabel creation
createtabel('city', '(CityID varchar(50) primary key, CityName varchar(1000), CountryId varchar(50))')
createtabel('continent', '(ContenentId varchar(50) primary key, Contenent varchar(255))')
createtabel('country', '(CountryId varchar(50) primary key, Country varchar(255), RegionId varchar(50))')
createtabel('Item', '(AttractionId varchar(50) primary key, AttractionCityId varchar(50), AttractionTypeId varchar(50), Attraction varchar(255), AttractionAddress varchar(255))')
createtabel ('Mode', '(VisitModeId varchar(50) primary key, VisitMode varchar(50))')
createtabel('Region', '(Region varchar(255), RegionId varchar(50) primary key, ContentId varchar(50))')
createtabel('Transaction', '(TransactionId varchar(50) primary key, UserId varchar(50), VisitYear varchar(4), VisitMonth int, VisitMode varchar(2), AttractionId varchar(50), Rating int )')
createtabel('Type', '(AttractionTypeId varchar(50) primary key, AttractionType varchar(255))')
createtabel('User', '(UserId varchar(50) primary key, ContenentId varchar(50), RegionId varchar(50), CountryId varchar(50), CityId varchar(50))')
#endregion

def insert(path, tablename):
    df = pd.read_excel(path)
    df = df.dropna()
    df = df.replace("'", "", regex=True)
    df = df.where(pd.notnull(df), None)
    for col in df.columns:
        type = df[col].dtype
        if np.issubdtype(type, np.int64):
            df[col] = df[col].astype('Int32')
        elif np.issubdtype(type, np.float64):
            df[col] = df[col].astype(float)
        elif np.issubdtype(type, np.bool_):
            df[col] = df[col].astype(bool)
    columns = ", ".join(df.columns)
    s = ", ".join(['%s' for i in range (len(df.columns))])
    query = f"insert ignore into {tablename} ({columns}) values ({s})"
    value = [tuple(i) for i in df.values]
    cursor.executemany(query, value)
    connection.commit()

path_city = r"D:\Python Projects\Production\Tourism\City.xlsx"
path_continent = r"D:\Python Projects\Production\Tourism\Continent.xlsx"
path_country = r"D:\Python Projects\Production\Tourism\Country.xlsx"
path_item = r"D:\Python Projects\Production\Tourism\Item.xlsx"
path_mode = r"D:\Python Projects\Production\Tourism\Mode.xlsx"
path_region = r"D:\Python Projects\Production\Tourism\Region.xlsx"
path_transaction = r"D:\Python Projects\Production\Tourism\Transaction.xlsx"
path_type = r"D:\Python Projects\Production\Tourism\Type.xlsx"
path_user = r"D:\Python Projects\Production\Tourism\User.xlsx"

insert(path_city,'city')
insert(path_continent, 'continent')
insert(path_country, 'country')
insert(path_item,'item')
insert(path_mode, 'mode')
insert(path_region, 'region')
insert(path_transaction, 'transaction')
insert(path_type, 'type')
insert(path_user, 'user')

print("Data Transfered to database")


query_df = """select a.transactionid, a.userid,  k.countryid as 'User Country', l.regionID as 'User Region', m.cityID as 'User City',
a.visityear, a.visitmonth, a.visitmode, a.attractionid  ,  c.cityid, e.countryid, j.contenentid, a.rating     from transaction a
inner join item b on a. AttractionId = b.AttractionId
inner join city c on b.AttractionCityId = c.CityID
inner join type d on b.AttractionTypeId = d.AttractionTypeId
inner join country e on c.countryid = e.countryid
inner join user f on a.userid = f.userid
inner join mode g on a.visitmode = g.visitmodeid
inner join region h on e.regionid = h.regionid
inner join continent j on h.contentid = j.contenentid
inner join country k on f.countryid = k.countryid
inner join region l on k.regionid = l.regionid
inner join city m on CAST( f.cityid AS signed)  = m.CityID"""

cursor.execute(query_df)
columns = [col[0] for col in cursor.description]
rows = cursor.fetchall()

DF_REG = pd.DataFrame(rows,  columns=columns)
print(DF_REG.columns)
DF_REG.drop(['transactionid', 'userid'  , 'User Country' , 'visityear' , 'visitmonth' , 'User Region', 'visitmode','cityid','countryid', 'contenentid'], axis=1 , inplace = True)
DF_REG = DF_REG.astype(float)
print(DF_REG.dtypes)


x = DF_REG.drop(['rating'], axis=1 )
y = DF_REG['rating']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

#model = DecisionTreeRegressor( max_depth= 5, min_samples_leaf= 4, min_samples_split= 2, splitter= 'best')
#model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
#model = LinearRegression()
model = RandomForestRegressor(bootstrap = True, max_depth = 10, min_samples_leaf = 1, min_samples_split = 10, n_estimators = 200)

"""param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],    # Minimum number of samples required to be at a leaf node
    'bootstrap': [True, False]        # Whether bootstrap samples are used when building trees
}

# Step 5: Set up GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Step 6: Fit the model with GridSearchCV
grid_search.fit(x_train, y_train)

# Step 7: Get the best parameters and evaluate the model
print("Best Hyperparameters:", grid_search.best_params_)"""

model.fit(x_train,y_train)
train_prediction = model.predict(x_train)
test_prediction = model.predict(x_test)
DF_REG['Prediction'] = model.predict(x)
DF_REG.to_csv('final3.csv')
r2_scores = r2_score(y_test, test_prediction)
r2_scores_train = r2_score(y_train, train_prediction)

print(r2_scores, r2_scores_train)



query_eda = """select a.transactionid, a.userid,  k.country as 'User Country', l.region as 'User Region', m.cityname as 'User City',
a.visityear, a.visitmonth, g.visitmode, b.attraction, b.attractionaddress, c.cityname, e.country, j.contenent, a.rating     from transaction a
inner join item b on a. AttractionId = b.AttractionId
inner join city c on b.AttractionCityId = c.CityID
inner join type d on b.AttractionTypeId = d.AttractionTypeId
inner join country e on c.countryid = e.countryid
inner join user f on a.userid = f.userid
inner join mode g on a.visitmode = g.visitmodeid
inner join region h on e.regionid = h.regionid
inner join continent j on h.contentid = j.contenentid
inner join country k on f.countryid = k.countryid
inner join region l on k.regionid = l.regionid
inner join city m on CAST( f.cityid AS signed)  = m.CityID"""
"""cursor.execute(query_eda)
columns = [col[0] for col in cursor.description]
rows = cursor.fetchall()

DF_EDA = pd.DataFrame(rows,  columns=columns)"""
