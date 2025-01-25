from mysql import connector
import pandas as pd
import streamlit as st
import pandas as pd
import openpyxl
connection = connector.connect(host = 'localhost', user = 'root', password = '1234')
cursor = connection.cursor()

# Database creation
cr_db = "create database if not exists tourism"
cursor.execute(cr_db)

#use databse
us_db = "use tourism"
cursor.execute(us_db)

#region user Defined functions

def createtabel(tbname, columns):
    Query = f"create table if not exists {tbname} {columns}"
    cursor.execute(Query)

#endregion
path_city = r"D:\Python Projects\Production\Tourism\City.xlsx"
df2 = pd.read_excel(path_city)
print(df2)

#region tabel creation
createtabel('city', '(CityID varchar(50) primary key, CityName varchar(255), CountryId varchar(50))')
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
    columns = ", ".join(df.columns)
    s = ", ".join(['%s' for i in range (len(df.columns))])
    query = f"insert into {tablename} ({columns}) values ({s})"
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
print("city info added")
insert(path_continent, 'continent')
insert(path_item,'item')
insert(path_mode, 'mode')
insert(path_region, 'region')
insert(path_transaction, 'transaction')
insert(path_type, 'type')
insert(path_user, 'user')

print("Data Transfered to database")














