import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)



### İş Problemi

'''

Her bir eve ait özelliklerin ve ev fiyatlarının bulunduğu veriseti kullanılarak,
farklı tipteki evlerin fiyatlarına ilişkin bir makine öğrenmesi projesi
gerçekleştirilmek istenmektedir.

'''

### Veri seti & Değişkenler

'''

Ames, Lowa’daki konut evlerinden oluşan bu veri seti içerisinde 79 açıklayıcı değişken bulunduruyor. Kaggle üzerinde bir yarışması
da bulunan projenin veri seti ve yarışma sayfasına aşağıdaki linkten ulaşabilirsiniz. Veri seti bir kaggle yarışmasına ait
olduğundan dolayı train ve test olmak üzere iki farklı csv dosyası vardır. Test veri setinde ev fiyatları boş bırakılmış olup, bu
değerleri sizin tahmin etmeniz beklenmektedir.


train.csv

Toplam Gözlem               Sayısal Değişken            Kategorik Değişken
    1460                          38                            43

'''


################################
# Görev 1: Keşifçi Veri Analizi
################################

### Adım 1: Train ve Test veri setlerini okutup birleştiriniz. Birleştirdiğiniz veri üzerinden ilerleyiniz.


train = pd.read_csv("Miuul_Course_1/Machine-Learning/Datasets/train.csv")
test = pd.read_csv("Miuul_Course_1/Machine-Learning/Datasets/test.csv")

df = pd.concat([train, test])

df.reset_index(inplace=True, drop=True)

df.info()


### Adım 2: Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# df[cat_but_car].value_counts()
# cat_cols.append(cat_but_car[0])

### Adım 3: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)

# df[num_cols].info()
# df[cat_cols].info()


df[cat_cols] = df[cat_cols].astype("object")

### Adım 4: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.

#### Katergorik değişkenlerin değerlendirilmesi

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

#### Numeric değişkenlerin değerlendirilmesi

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col)

### Adım 5: Kategorik değişkenler ile hedef değişken incelemesini yapınız.


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}).sort_values("TARGET_MEAN", ascending=False), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "SalePrice", col)


### Adım 6: Aykırı gözlem var mı inceleyiniz.

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, 0.05, 0.95)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(check_outlier(df, col))


### Adım 7: Eksik gözlem var mı inceleyiniz.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns, missing_df

missing_values_table(df)



################################
# Görev 2: Feature Engineering
################################

### Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

#### Eksik gözlemler

na_columns, miss_df = missing_values_table(df, True)
'''
(df["FireplaceQu"].fillna("0") + "_" + df["Fireplaces"].astype("string")).value_counts()
'''

##### Bazı değişkenlerdeki boş değerler evin o özelliğe sahip olmadığını ifade etmektedir

no_cols = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu",
           "GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]

for col in no_cols:
    df[col].fillna("No", inplace=True)

def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data


df = quick_missing_imp(df)


#### Aykırı gözlemler

for col in num_cols:
    if col != "SalePrice":
        print(check_outlier(df, col))

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df, col)

for col in num_cols:
    if col != "SalePrice":
        print(check_outlier(df, col))


### Adım 2: Rare Encoder uygulayınız.

#### Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.


##### Analiz

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "SalePrice", cat_cols)


#### Encoder

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


temp_df = rare_encoder(df, 0.02)

rare_analyser(temp_df, "SalePrice", cat_cols)



### Adım 3: Yeni değişkenler oluşturunuz.


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(temp_df)

final_df = temp_df.copy()


# Bina yaşını hesaplanması

final_df["HouseAge"] = final_df["YrSold"] - final_df["YearBuilt"]
final_df.loc[final_df["HouseAge"] < 0, "HouseAge"] = 0


# Bina toplam alanların hesabı

final_df["TotalArea"] = final_df["TotalBsmtSF"] + final_df["1stFlrSF"] + final_df["2ndFlrSF"]

final_df["TotalSqFeet"] = final_df["GrLivArea"] + final_df["TotalBsmtSF"]

final_df["TotalPorch"] = final_df["WoodDeckSF"] + final_df["OpenPorchSF"] + final_df["EnclosedPorch"] + final_df["3SsnPorch"] + final_df["ScreenPorch"]


# Metre kare başına fiyat

final_df["Price/Area"] = final_df["SalePrice"] / final_df["TotalArea"]


# Satışın mevsimi

final_df["SeasonSold"] = (final_df["MoSold"].astype("int")).astype("string") + "." + final_df["YrSold"].astype("string")


# Malzemelerin Kalite ve durumu üzerinden değerlendirme

final_df["Quality"] = final_df["OverallQual"] + final_df["OverallCond"]


# Mahallenin Price ortalaması üzerinden değerlendirme

neigh_dict = final_df.groupby("Neighborhood").agg({"SalePrice": "mean"}).to_dict()["SalePrice"]

for key, value in neigh_dict.items():
    final_df.loc[final_df["Neighborhood"] == key, "NeighborhoodValue"] = value

final_df["NeighborhoodValue"].value_counts()





# Havuz varmı yok mu

final_df.loc[final_df["PoolQC"] != "No", "PoolQC"] = "Yes"


# Garaj var mı yok mu

final_df.loc[final_df["GarageArea"] == 0, "HasGarage"] = "No"
final_df.loc[final_df["HasGarage"] != "No", "HasGarage"] = "Yes"

final_df["HasGarage"].value_counts()



# Şömine varmı yokmu

final_df.loc[final_df["Fireplaces"] == 0, "HasFireplace"] = "No"
final_df.loc[final_df["HasFireplace"] != "No", "HasFireplace"] = "Yes"

final_df["HasFireplace"].value_counts()



# Ev yenilenmişmi yenilememişmi

final_df["YearRemodAdd"].astype("int")
final_df["YearBuilt"].astype("int")

final_df.loc[final_df["YearBuilt"] == final_df["YearRemodAdd"], "Remodeled"] = "No"
final_df.loc[final_df["YearBuilt"] != final_df["YearRemodAdd"], "Remodeled"] = "Yes"


final_df["Remodeled"].value_counts()


# Evdeki toplam banyo sayısı


bath_cols = [col for col in final_df.columns if col.__contains__("Bath")]
final_df["TotalBath"] = 0

for i in bath_cols:
    final_df["TotalBath"] += final_df[i]

final_df["TotalBath"].value_counts()



# Droplanacak değişkenler

drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope","Heating", "PoolQC", "MiscFeature", "Neighborhood", "YearBuilt", "YearRemodAdd", "GarageYrBlt", "3SsnPorch", "PoolArea"]


final_df.drop(drop_list, axis=1, inplace=True)


### Adım 4: Encoding işlemlerini gerçekleştiriniz.

#### Label Encoding

binary_cols = [col for col in final_df.columns if final_df[col].dtypes == "O" and len(final_df[col].unique()) == 2]

cat_cols, num_cols, cat_but_car = grab_col_names(final_df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for i in binary_cols:
    label_encoder(final_df, i)

cat_cols.append('NeighborhoodValue')
num_cols.remove('NeighborhoodValue')

#### One - Hot Encoder

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

final_df = one_hot_encoder(final_df, cat_cols, drop_first=True)


################################
# Görev 3: Model Kurma
################################

### Adım 1: Train ve Test verisini ayırınız. (SalePrice değişkeni boş olan değerler test verisidir.)

train_df = final_df.loc[~final_df["SalePrice"].isnull()]
test_df = final_df.loc[final_df["SalePrice"].isnull()].reset_index(drop=True)


### Adım 2: Train verisi ile model kurup, model başarısını değerlendiriniz.

import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

y = train_df["SalePrice"] # np.log1p(train_df['SalePrice'])
X = train_df.astype("float64").drop(["Id", "SalePrice"], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)



models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]


for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


# RMSE: 32111.9934 (LR)
# RMSE: 31298.8475 (Ridge)
# RMSE: 20092.0672 (Lasso)
# RMSE: 17743.9985 (ElasticNet)
# RMSE: 44060.3042 (KNN)
# RMSE: 19709.1482 (CART)
# RMSE: 11027.8938 (RF)
# RMSE: 81132.6532 (SVR)
# RMSE: 9873.947 (GBM)
# RMSE: 10668.8411 (XGBoost)
# RMSE: 11844.0483 (LightGBM)
# RMSE: 8725.8363 (CatBoost) *************


### Adım 3: Hiperparemetre optimizasyonu gerçekleştiriniz.


model = CatBoostRegressor(verbose=False)

params = {'learning_rate': [0.01, 0.05, 0.1],
          'depth': [4, 6, 8],
          'l2_leaf_reg': [1, 3, 5]}

cat_cv = GridSearchCV(model, params, cv=5, scoring='neg_root_mean_squared_error')
cat_cv.fit(X_train, y_train)


print("En iyi parametreler: ", cat_cv.best_params_)
# En iyi parametreler:  {'depth': 4, 'l2_leaf_reg': 1, 'learning_rate': 0.05}
print("En iyi RMSE değeri: ", -cat_cv.best_score_)
# En iyi RMSE değeri:  8167.146088572644


#### Final Model

final_model = CatBoostRegressor(verbose=False,
                                learning_rate=cat_cv.best_params_['learning_rate'],
                                depth=cat_cv.best_params_['depth'],
                                l2_leaf_reg=cat_cv.best_params_['l2_leaf_reg'])

final_model.fit(X_train, y_train)



### Adım 4: Değişken önem düzeyini inceleyeniz.

def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")


plot_importance(final_model, X)


##### Test Veri setinin tahmin edilmesi

valid_df = test_df.astype("float64").drop(["Id", "SalePrice"], axis=1)

test["PredPrice"] = final_model.predict(valid_df)


