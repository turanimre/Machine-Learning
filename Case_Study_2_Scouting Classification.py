import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder



pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


### İş Problemi

'''

Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf
(average, highlighted) oyuncu olduğunu tahminleme.

'''

### Veri seti & Değişkenler

'''

Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların değerlendirdikleri futbolcuların, maç
içerisinde puanlanan özellikleri ve puanlarını içeren bilgilerden oluşmaktadır.


scoutium_attributes.csv

task_response_id Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
match_id:  İlgili maçın id'si
evaluator_id:  Değerlendiricinin(scout'un) id'si
player_id:  İlgili oyuncunun id'si
position_id:  İlgili oyuncunun o maçta oynadığı pozisyonun id’si
              1: Kaleci
              2: Stoper
              3: Sağ bek
              4: Sol bek
              5: Defansif orta saha
              6: Merkez orta saha
              7: Sağ kanat
              8: Sol kanat
              9: Ofansif orta saha
              10: Forvet
analysis_id:  Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme
attribute_id:  Oyuncuların değerlendirildiği her bir özelliğin id'si
attribute_value:  Bir scoutun bir oyuncunun bir özelliğine verdiği değer(puan)


scoutium_potential_labels.csv

task_response_id:  Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
match_id:  İlgili maçın id'si
evaluator_id:  Değerlendiricinin(scout'un) id'si
player_id:  İlgili oyuncunun id'si
potential_label:  Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket. (hedef değişken)

'''


### Adım 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.

att_df = pd.read_csv("Miuul_Course_1/Machine-Learning/Datasets/scoutium_attributes.csv", sep=";")
pot_df = pd.read_csv("Miuul_Course_1/Machine-Learning/Datasets/scoutium_potential_labels.csv", sep=";")



### Adım 2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.
# ("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)


df = att_df.merge(pot_df, on=["task_response_id", 'match_id', 'evaluator_id', "player_id"])


### Adım 3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.

df = df[df["position_id"] != 1]


### Adım 4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.( below_average sınıfı tüm verisetinin %1'ini oluşturur)

len(df[df["potential_label"] == "below_average"]) * 100 / len(df)

df = df[df["potential_label"] != "below_average"]


### Adım 5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz. Bu pivot table'da her satırda bir oyuncu olacak şekilde manipülasyon yapınız.

#### Adım 1: İndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan “attribute_value” olacak şekilde pivot table’ı oluşturunuz.

df_pivot = df.pivot_table(index=["player_id", "position_id", "potential_label"], columns="attribute_id", values="attribute_value")


#### Adım 2: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini stringe çeviriniz.

df_pivot = df_pivot.reset_index()

df_pivot.columns = df_pivot.columns.astype("string")


### Adım 6: Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz.

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

label_encoder(df_pivot, "potential_label")



### Adım 7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.

num_cols = [col for col in df_pivot.columns if "4" in col]


### Adım 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız.

ss = StandardScaler()

for i in num_cols:
    df_pivot[[i]] = ss.fit_transform(df_pivot[[i]])


### Adım 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate


import warnings
warnings.simplefilter(action="ignore")

y = df_pivot[["potential_label"]]
X = df_pivot.drop("potential_label", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

### Model Seçimi


models = {'K-Nearest Neighbors': KNeighborsClassifier(),
           'Support Vector Machines': SVC(),
           'Decision Trees': DecisionTreeClassifier(),
           'Random Forest': RandomForestClassifier(),
           'Gradient Boosting': GradientBoostingClassifier(),
           'XGBoost': XGBClassifier(),
           'LightGBM': LGBMClassifier(),
           'CatBoost': CatBoostClassifier()}


def evaluate_classification_models(X, y, models, cv=5):
    """
    X: veri seti
    y: hedef değişken
    models: kullanılacak modellerin sözlük formatında belirtilmesi
    cv: cross-validation sayısı (varsayılan 5)
    """
    # Metriklerin tanımlanması
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1': make_scorer(f1_score),
               'roc_auc': make_scorer(roc_auc_score)}

    # Her model için cross-validation yapma
    for model_name, model in models.items():
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)

        # Cross-validation sonuçlarının yazdırılması
        print(f"Results for {model_name}:")
        print(f"Accuracy: {cv_results['test_accuracy'].mean():.4f} (+/- {cv_results['test_accuracy'].std():.4f})")
        print(f"Precision: {cv_results['test_precision'].mean():.4f} (+/- {cv_results['test_precision'].std():.4f})")
        print(f"Recall: {cv_results['test_recall'].mean():.4f} (+/- {cv_results['test_recall'].std():.4f})")
        print(f"F1: {cv_results['test_f1'].mean():.4f} (+/- {cv_results['test_f1'].std():.4f})")
        print(f"ROC AUC: {cv_results['test_roc_auc'].mean():.4f} (+/- {cv_results['test_roc_auc'].std():.4f})\n")



evaluate_classification_models(X, y, models)

'''

Results for Random Forest:
Accuracy: 0.8745 (+/- 0.0274)
Precision: 0.8743 (+/- 0.1123)
Recall: 0.4939 (+/- 0.2167)
F1: 0.5899 (+/- 0.1669)
ROC AUC: 0.7330 (+/- 0.0952)

Results for CatBoost:
Accuracy: 0.8745 (+/- 0.0215)
Precision: 0.8985 (+/- 0.1289)
Recall: 0.4773 (+/- 0.1639)
F1: 0.5959 (+/- 0.1145)
ROC AUC: 0.7270 (+/- 0.0690)

'''


#### Model Tuning

### RandomForestClassifier GridSearchCV

param_grid = {'n_estimators': [50, 100, 150],
              'max_features': ['sqrt', 'log2'],
              'max_depth': [None, 10, 20],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'bootstrap': [True, False]}

model = RandomForestClassifier(random_state=42)
grid_search_RF = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='accuracy')

grid_search_RF.fit(X_train, y_train)

print("Best Parameters: ", grid_search_RF.best_params_)
# Best Parameters:  {'bootstrap': True, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}
print("Best score: ", grid_search_RF.best_score_)
# Best score:  0.9120507399577166 ******************


### CatBoostClassifier GridSearchCV

param_grid = {'iterations': [100, 200],
              'learning_rate': [0.01, 0.05, 0.1],
              'depth': [3, 5, 7],
              'l2_leaf_reg': [1, 3, 5, 7, 9]}

model = CatBoostClassifier(random_seed=42)
grid_search_cat = GridSearchCV(model, param_grid=param_grid, n_jobs=-1,
                           cv=5, scoring='accuracy')

grid_search_cat.fit(X_train, y_train)
print("Best Parameters: ", grid_search_cat.best_params_)
# Best Parameters:  {'depth': 3, 'iterations': 100, 'l2_leaf_reg': 1, 'learning_rate': 0.05}
print("Best score: ", grid_search_cat.best_score_)
# Best score:  0.907505285412262



### Adım 10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.


final_model = RandomForestClassifier(bootstrap=True,
                                     max_depth=None,
                                     max_features='sqrt',
                                     min_samples_leaf=2,
                                     min_samples_split=2,
                                     n_estimators=100)
final_model.fit(X_train, y_train)


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