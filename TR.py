import pymysql
import pandas as pd
import streamlit as st
import pandas as pd
import openpyxl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans

connection = pymysql .connect(host = 'localhost', user = 'root', password = 'mypass')
cursor = connection.cursor()

tab1, tab2 ,tab3 = st.tabs(["Home Page", "Recommendation System" , "Insights"])
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


query_df = """select  l.region as 'User Region', m.cityname as 'User City',
 g.visitmode, b.attraction,  round(avg(a.rating)) as 'rating'      from transaction a
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
inner join city m on CAST( f.cityid AS signed)  = m.CityID
group by   l.region, m.cityname, g.visitmode, b.attraction"""

cursor.execute(query_df)
columns = [col[0] for col in cursor.description]
rows = cursor.fetchall()

DF_REG = pd.DataFrame(rows,  columns=columns)

DF_REG.drop(DF_REG[DF_REG['User Region'] == '-'].index, inplace=True)

DF_REG_IN = DF_REG.copy()
DF_REG = DF_REG.rename(columns={'User Region' : 'User_Region_Name', 'User City' : 'User_City_Name', 'visitmode': 'visitmode_Name', 'attraction': 'attraction_Name' })



DF_REG = DF_REG[(DF_REG!= '-').all(axis=1)]


lbl = LabelEncoder()
DF_REG['User_Region_Name'] = lbl.fit_transform(DF_REG['User_Region_Name'])
DF_REG['User_City_Name'] = lbl.fit_transform(DF_REG['User_City_Name'])
DF_REG['attraction_Name'] = lbl.fit_transform(DF_REG['attraction_Name'])
DF_REG['visitmode_Name'] = lbl.fit_transform(DF_REG['visitmode_Name'])
#DF_REG = pd.get_dummies(DF_REG, columns=['visitmode'] )

DF_REG = DF_REG.astype('Int32')
#DF_CL = DF_REG.copy()

DF_INPUT = pd.concat([DF_REG_IN, DF_REG], axis =1)


x = DF_REG.drop(['rating'], axis=1 )
y = DF_REG['rating']

xc = DF_REG.drop(['visitmode_Name'], axis=1)
yc = DF_REG['visitmode_Name']


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
xc_train, xc_test, yc_train, yc_test = train_test_split(xc, yc, test_size=0.2)

model = RandomForestRegressor(bootstrap= True, max_depth = 10, max_features = 'sqrt', min_samples_leaf = 2, min_samples_split = 10, n_estimators = 500)
model_cl = RandomForestClassifier(max_depth=10, n_estimators= 500)

#gd_sch = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

#model = DecisionTreeRegressor( max_depth= 5, min_samples_leaf= 4, min_samples_split= 2, splitter= 'best')
#model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
#model = LinearRegression()
#model = RandomForestRegressor(bootstrap = True, max_depth = 10, min_samples_leaf = 2, min_samples_split = 10, n_estimators = 200)

#model1 = [RandomForestRegressor(n_estimators=1000, max_depth=10), AdaBoostRegressor(n_estimators=200000, learning_rate=0.001 ), GradientBoostingRegressor(n_estimators= 2000, learning_rate= 0.001, max_depth=50 )]

model.fit(x_train,y_train)
train_prediction = np.round(model.predict(x_train))
test_prediction = np.round(model.predict(x_test))
mse = mean_squared_error(y_test, test_prediction)
mse_train = mean_squared_error(y_train, train_prediction)
print(f" mse_test : {mse}, mse_train: {mse_train}")

model_cl.fit(xc_train, yc_train)
traincl_prd = model_cl.predict(xc_train)
testcl_prd = model_cl.predict(xc_test)
ac_score_test =accuracy_score(yc_test, testcl_prd)
ac_score_train = accuracy_score(yc_train, traincl_prd)
print(f"f1 score: test : {ac_score_test}, train: {ac_score_train}")
print(traincl_prd, y_train)


with tab1:
    st.header("Tourism Experience Analytics: Classification, Prediction, and Recommendation System")
    st.markdown("""<h3 style='color: #0000FF;'>User Rating Prediction</h3>""", unsafe_allow_html=True)
    st.write(f"Model : RandomForestRegressor")
    st.write(f"Mean Squared Error-Test Data: {mse}")
    st.write(f"Mean Squared Error-Train Data: {mse_train}")
    NEW_DF = pd.DataFrame()
    REG_UN = np.array(list(set([i for i in DF_REG_IN['User Region']])))
    REG_IN = REG_UN.tolist()
    REG_NAME = st.selectbox("Select Region", REG_IN)
    User_Region_Name = DF_INPUT.loc[DF_INPUT['User Region'] == REG_NAME, 'User_Region_Name'].iloc[0]
  

    CITY_UN = np.array(list(set([i for i in DF_REG_IN['User City']])))
    CITY_IN = CITY_UN.tolist()
    CITY_NAME = st.selectbox("Select City", CITY_IN)
    User_City_Name = DF_INPUT.loc[DF_INPUT['User City'] == CITY_NAME, 'User_City_Name'].iloc[0]

    ATTRACTION_UN = np.array(list(set([i for i in DF_REG_IN['attraction']])))
    ATTRACTION_IN = ATTRACTION_UN.tolist()
    ATTRACTION_NAME = st.selectbox("Select Attraction", ATTRACTION_IN)
    attraction_Name = DF_INPUT.loc[DF_INPUT['attraction'] == ATTRACTION_NAME, 'attraction_Name'].iloc[0]


    VISITMODE_UN = np.array(list(set([i for i in DF_REG_IN['visitmode']])))
    VISITMODE_IN = VISITMODE_UN.tolist()
    VISITMODE_NAME = st.selectbox("Select Visit Mode", VISITMODE_IN)
    visitmode_Name = DF_INPUT.loc[DF_INPUT['visitmode'] == VISITMODE_NAME, 'visitmode_Name'].iloc[0]

    NEW_DF['User_Region_Name'] = [User_Region_Name] 
    NEW_DF['User_City_Name'] = [User_City_Name]
    NEW_DF['visitmode_Name'] = [visitmode_Name]
    NEW_DF['attraction_Name'] = [attraction_Name]

    if st.button("Prediction"):
        Rating_new = np.round(model.predict(NEW_DF))
        st.write(f"Predicted Rating: {Rating_new}")

    st.markdown("""<h3 style='color: #0000FF;'>Visit Mode Prediction</h3>""", unsafe_allow_html=True)
    st.write(f"Model : DecisionTreeClassifier")
    st.write(f"Accuracy score-Test Data: {ac_score_test}")
    st.write(f"Accuracy Score-Train Data: {ac_score_train}")
    NEWCL_DF = pd.DataFrame()
    REGCL_UN = np.array(list(set([i for i in DF_REG_IN['User Region']])))
    REGCL_IN = REGCL_UN.tolist()
    REGCL_NAME = st.selectbox("Region ", REGCL_IN)
    User_Region_NameCL = DF_INPUT.loc[DF_INPUT['User Region'] == REGCL_NAME, 'User_Region_Name'].iloc[0]
  

    CITYCL_UN = np.array(list(set([i for i in DF_REG_IN['User City']])))
    CITYCL_IN = CITYCL_UN.tolist()
    CITYCL_NAME = st.selectbox("City", CITYCL_IN)
    User_City_NameCL = DF_INPUT.loc[DF_INPUT['User City'] == CITYCL_NAME, 'User_City_Name'].iloc[0]

    ATTRACTIONCL_UN = np.array(list(set([i for i in DF_REG_IN['attraction']])))
    ATTRACTIONCL_IN = ATTRACTIONCL_UN.tolist()
    ATTRACTIONCL_NAME = st.selectbox("Attraction", ATTRACTIONCL_IN)
    attraction_NameCL = DF_INPUT.loc[DF_INPUT['attraction'] == ATTRACTIONCL_NAME, 'attraction_Name'].iloc[0]


    rating = st.number_input("Enter Rating")
    

    NEWCL_DF['User_Region_Name'] = [User_Region_NameCL] 
    NEWCL_DF['User_City_Name'] = [User_City_NameCL]
    NEWCL_DF['attraction_Name'] = [attraction_NameCL]
    NEWCL_DF['rating'] = [rating]
    NEW_DF.astype('int32')


    if st.button("Visit Mode Prediction"):
        visitmode_new = np.int32(model_cl.predict(NEWCL_DF))
        visitmode_Name_CL = DF_INPUT.loc[DF_INPUT['visitmode_Name'] == visitmode_new[0], 'visitmode'].iloc[0]
        st.write(f"Predicted Visit Mode: {visitmode_Name_CL}")

with tab2:
    qr_re = """select  l.region as 'User Region', m.cityname as 'User City',
    g.visitmode, b.attraction, e.country as 'Attraction Country',  round(avg(a.rating)) as 'rating'      from transaction a
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
    inner join city m on CAST( f.cityid AS signed)  = m.CityID
    group by   l.region, m.cityname, g.visitmode, b.attraction, e.country"""

    cursor.execute(qr_re)
    col1 = [col[0] for col in cursor.description]
    rows = cursor.fetchall()
    DF_REC = pd.DataFrame(rows, columns = col1)
    DFN_REC = DF_REC.copy()
    DF_REC = DF_REC.rename(columns= {'User Region' : 'User Region_name', 'User City' : 'User City_Name', 'visitmode' : 'visitmode_Name', 'attraction': 'attraction_name', 'Attraction Country': 'Attraction Country_Name', 'rating': 'rating' })
    DF_REC['User Region_name'] = lbl.fit_transform(DF_REC['User Region_name'])
    DF_REC['User City_Name'] = lbl.fit_transform(DF_REC['User City_Name'])
    DF_REC['visitmode_Name'] = lbl.fit_transform( DF_REC['visitmode_Name'])
    DF_REC['Attraction Country_Name'] = lbl.fit_transform( DF_REC['Attraction Country_Name'])

    DFREC_INPUT = pd.concat([DFN_REC, DF_REC], axis =1)
    print(DFREC_INPUT.columns)

    x_rec = DF_REC.drop(['attraction_name'], axis=1)
    y_rec = DF_REC['attraction_name']

    kmeans = KMeans(n_clusters=3)

    kmeans.fit(x_rec, y_rec)
    test_prediction = kmeans.predict(x_rec)
    DF_REC['Prediction'] = kmeans.predict(x_rec)
    print(DF_REC)
    NEW_DF_REC = pd.DataFrame()

    REGR_UN = np.array(list(set([i for i in DFREC_INPUT['User Region']])))
    REGR_IN = REGR_UN.tolist()
    REGFR_NAME = st.selectbox("User_Region", REGR_IN)
    User_Region_Name_REC = DFREC_INPUT.loc[DFREC_INPUT['User Region'] == REGFR_NAME, 'User Region_name'].iloc[0]
  

    CITYRE_UN = np.array(list(set([i for i in DFREC_INPUT['User City']])))
    CITYRE_IN = CITYRE_UN.tolist()
    CITYRE_NAME = st.selectbox("User_City", CITYRE_IN)
    User_City_Name_REC = DFREC_INPUT.loc[DFREC_INPUT['User City'] == CITYRE_NAME, 'User City_Name'].iloc[0]

    VSTRE_UN = np.array(list(set([i for i in DFREC_INPUT['visitmode']])))
    VSTRE_IN = VSTRE_UN.tolist()
    VSTRE_NAME = st.selectbox("Visit Mode", VSTRE_IN)
    visitmode_REC = DFREC_INPUT.loc[DFREC_INPUT['visitmode'] == VSTRE_NAME, 'visitmode_Name'].iloc[0]

    ATTCNRE_UN = np.array(list(set([i for i in DFREC_INPUT['Attraction Country']])))
    ATTCNRE_IN = ATTCNRE_UN.tolist()
    ATTCNRE_NAME = st.selectbox("Attraction Country", ATTCNRE_IN)
    attraction_Name_REC = DFREC_INPUT.loc[DFREC_INPUT['Attraction Country'] == ATTCNRE_NAME, 'Attraction Country_Name'].iloc[0]


    rating_rec = st.number_input("Rating")
    

    NEW_DF_REC['User Region_name'] = [User_Region_Name_REC] 
    NEW_DF_REC['User City_Name'] = [User_City_Name_REC]
    NEW_DF_REC['visitmode_Name'] = [visitmode_REC]
    NEW_DF_REC['Attraction Country_Name'] = [attraction_Name_REC]
    NEW_DF_REC['rating'] = [rating_rec]
    NEW_DF_REC.astype('int32')

    if st.button("Recomendation"):
        prd = np.int32(kmeans.predict(NEW_DF_REC))
        pt = [i for i in prd]
        DF_REC_FINAL = DF_REC[DF_REC['Prediction'] == pt[0]]
        st.subheader("The Recommended Attraction Sites")
        st.write(DF_REC_FINAL['attraction_name'].unique())


with tab3:
    options = st.selectbox("Select a Query",("None",
    "Top 5 Country With More Tourist",
    "Top 5 Attraction Sites", 
    "Tourist Distribution",
    "Visit Mode"))
    if st.button("Result", type="primary"):
        if options == "Top 5 Country With More Tourist":
            qr= """select  k.country as 'User Country', count(a.userid)  as 'User Count'   from transaction a
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
            inner join city m on CAST( f.cityid AS signed)  = m.CityID
            group by k.country
            order by 2 desc
            limit 5;"""
            cursor.execute(qr)
            col = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            df_qr = pd.DataFrame(rows, columns=col)
            st.write(df_qr)
            st.bar_chart(df_qr.set_index('User Country')['User Count'])
        elif options == "Top 5 Attraction Sites":
            qr = """select   b.attraction , count(a.userid)  as 'User Count'   from transaction a
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
            inner join city m on CAST( f.cityid AS signed)  = m.CityID
            group by b.attraction
            order by 2 desc
            limit 5;"""
            cursor.execute(qr)
            col = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            df_qr = pd.DataFrame(rows, columns=col)
            st.write(df_qr)
            st.bar_chart(df_qr.set_index('User Country')['User Count'])

            


    




 
