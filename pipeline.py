import dill
import pandas as pd
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def brand(df):
    df_1 = df.copy()
    summa = len(df_1["device_brand"])
    device_brand = df_1.groupby(["device_brand"]).agg({"device_brand": "count"})
    device_brand["device_brand"] = device_brand["device_brand"].apply(lambda x: x / summa * 100)
    top_brands = dict(zip(device_brand.index, device_brand["device_brand"]))
    top_brands = {key: val for key, val in top_brands.items() if val > 1.0}
    # оставляем признак если частота встречаемости более 1 процента, иначе заменяем на "other"

    def top_brand(x):
        if x in top_brands:
            return x
        else:
            x = "other"
            return x

    # Добавляем фичу "top_brand"
    df_1.loc[:, 'top_brand'] = df_1['device_brand'].apply(top_brand)
    df_1 = df_1.drop(columns=["device_brand"], axis=1)
    return df_1


def date(df):
    df_2 = df.copy()
    df_2["visit_date"] = pd.to_datetime(df_2["visit_date"])       #приводим данные к типу дата, создаем новые признаки месяц и день недели
    df_2["visit_month"] = df_2["visit_date"].dt.month.astype(int)
    df_2["visit_day"] = df_2["visit_date"].dt.dayofweek.astype(int)
    df_2 = df_2.drop(columns=["visit_date"], axis=1)
    return df_2


def source(df):
    df_3 = df.copy()
    summa = len(df_3["utm_source"])
    utm_source = df_3.groupby(["utm_source"]).agg({"utm_source": "count"})
    utm_source["utm_source"] = utm_source["utm_source"].apply(lambda x: x / summa * 100)
    top_sources = dict(zip(utm_source.index, utm_source["utm_source"]))
    top_sources = {key: val for key, val in top_sources.items() if val > 1.0}

    # оставляем признак если частота встречаемости более 1 процента, иначе заменяем на "other"

    def top_source(x):
        if x in top_sources:
            return x
        else:
            x = "other"
            return x

    # Добавляем фичу "top_brand"
    df_3.loc[:, 'top_source'] = df_3['utm_source'].apply(top_source)
    df_3 = df_3.drop(columns=["utm_source"], axis=1)
    return df_3


def medium(df):
    df_4 = df.copy()
    summa = len(df_4["utm_medium"])
    utm_medium = df_4.groupby(["utm_medium"]).agg({"utm_medium": "count"})
    utm_medium["utm_medium"] = utm_medium["utm_medium"].apply(lambda x: x / summa * 100)
    top_medium = dict(zip(utm_medium.index, utm_medium["utm_medium"]))
    top_medium = {key: val for key, val in top_medium.items() if val > 1.0}

    # оставляем признак если частота встречаемости более 1 процента, иначе заменяем на "other"

    def top_med(x):
        if x in top_medium:
            return x
        else:
            x = "other"
            return x
    # создаем дополнительный признак органический траффик
    org = ['organic', 'referral', '(none)']

    def organic(x):
        if x in org:
            x = 0
            return x
        else:
            x = 1
            return x

    df_4.loc[:, 'is_organic_traffic'] = df_4['utm_medium'].apply(organic)
    # Добавляем фичу "top_brand"
    df_4.loc[:, 'top_medium'] = df_4['utm_medium'].apply(top_med)
    df_4 = df_4.drop(columns=["utm_medium"], axis=1)
    return df_4


def adcontent(df):
    df_5 = df.copy()
    summa = len(df_5["utm_adcontent"])
    utm_adcontent = df_5.groupby(["utm_adcontent"]).agg({"utm_adcontent": "count"})
    utm_adcontent["utm_adcontent"] = utm_adcontent["utm_adcontent"].apply(lambda x: x / summa * 100)
    top_adcontent = dict(zip(utm_adcontent.index, utm_adcontent["utm_adcontent"]))
    top_adcontent = {key: val for key, val in top_adcontent.items() if val > 1.0}

    # оставляем признак если частота встречаемости более 1 процента, иначе заменяем на "other"

    def top_ad(x):
        if x in top_adcontent:
            return x
        else:
            x = "other"
            return x

    # Добавляем фичу "top_brand"
    df_5.loc[:, 'top_adcontent'] = df_5['utm_adcontent'].apply(top_ad)
    df_5 = df_5.drop(columns=["utm_adcontent"], axis=1)
    return df_5


def geo(df):   #переводим признак страна в числовой широта и долгота, а также создаем дополнительные бинарные признаки
    df_6 = df.copy() #Москва, Санкт-Петербург, Россия
    coord = pd.read_csv("data/world_coordinates.csv")
    countries_lat = dict(zip(coord.Country, coord.latitude))
    countries_long = dict(zip(coord.Country, coord.longitude))

    def country_lat(x):
        if x in countries_lat:
            lat = countries_lat[x]
            return lat
        else:
            lat = 0.0
            return lat

    def country_long(x):
        if x in countries_long:
            long = countries_long[x]
            return long
        else:
            long = 0.0
            return long

    df_6.loc[:, 'lat'] = df_6['geo_country'].apply(country_lat)
    df_6.loc[:, 'long'] = df_6['geo_country'].apply(country_long)
    df_6["long"] = df_6["long"].astype("float")
    df_6["lat"] = df_6["lat"].astype("float")

    def is_russia(x):
        if x == "Russia":
            x = 1
            return x
        else:
            x = 0
            return x

    def is_sp(x):
        if x == "Saint Petersburg":
            x = 1
            return x
        else:
            x = 0
            return x

    def is_moscow(x):
        if x == "Moscow":
            x = 1
            return x
        else:
            x = 0
            return x

    df_6.loc[:, 'is_russia'] = df_6['geo_country'].apply(is_russia)
    df_6.loc[:, 'is_sp'] = df_6['geo_city'].apply(is_sp)
    df_6.loc[:, 'is_moscow'] = df_6['geo_city'].apply(is_moscow)
    df_6 = df_6.drop(columns=["geo_country", "geo_city"], axis=1)
    print(df_6.info())
    return df_6


def short_browser(df): #обрабатываем категориальный признак браузер
    df_8 = df.copy()

    def browser(x):
        x = x.split(" ")
        return x[0]

    df_8['device_browser'] = df_8['device_browser'].apply(browser)
    return df_8


def hours(df): #создаем дополнительный признак час визита
    df_9 = df.copy()

    def hour(x):
        x = x.split(":")
        return x[0]

    df_9['visit_time'] = df_9["visit_time"].apply(hour)
    return df_9


def campaign(df):
    df_10 = df.copy()
    summa = len(df_10["utm_campaign"])
    utm_campaign = df_10.groupby(["utm_campaign"]).agg({"utm_campaign": "count"})
    utm_campaign["utm_campaign"] = utm_campaign["utm_campaign"].apply(lambda x: x / summa * 100)
    top_campaign = dict(zip(utm_campaign.index, utm_campaign["utm_campaign"]))
    top_campaign = {key: val for key, val in top_campaign.items() if val > 1.0}

    # оставляем признак если частота встречаемости более 1 процента, иначе заменяем на "other"

    def top_camp(x):
        if x in top_campaign:
            return x
        else:
            x = "other"
            return x

    # Добавляем фичу "top_brand"
    df_10['utm_campaign'] = df_10['utm_campaign'].apply(top_camp)
    return df_10


def final(df): #удаляем ненужные столбцы
    df = df.drop(columns=["client_id", "session_id", "utm_keyword",
                          "device_model", "device_screen_resolution"], axis=1)
    return df


def main():
    df = pd.read_csv('data/ga_sessions.csv', low_memory=False)
    df1 = pd.read_csv("data/ga_hits.csv", usecols=["session_id", "event_action"], low_memory=False)

    def action(x):               #создаем целевую переменную
        actions = ['sub_car_claim_click', 'sub_car_claim_submit_click',
                   'sub_open_dialog_click', 'sub_custom_question_submit_click',
                   'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success',
                   'sub_car_request_submit_click']
        if x in actions:
            x = 1
            return x
        else:
            x = 0
            return x

    df1.loc[:, "goal_action"] = df1["event_action"].apply(action)
    df1 = df1.groupby("session_id")["goal_action"].max()
    df1 = pd.DataFrame(df1)
    df = df.merge(df1, how="inner", on="session_id")  # объединяем датафреймы
    df = df.drop_duplicates(keep="first")
    x = df.drop(columns=["goal_action"], axis=1)
    y = df["goal_action"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=42)

    initial_pipe = Pipeline(steps=[
        ("final", FunctionTransformer(final)),
        ("brand", FunctionTransformer(brand)),
        ("source", FunctionTransformer(source)),
        ("medium", FunctionTransformer(medium)),
        ("adcontent", FunctionTransformer(adcontent)),
        ("date", FunctionTransformer(date)),
        ("geo", FunctionTransformer(geo)),
        ("short_browser", FunctionTransformer(short_browser)),
        ("hours", FunctionTransformer(hours)),
        ("campaign", FunctionTransformer(campaign))

    ])

    pipe_num = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    pipe_cat = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', pipe_num, make_column_selector(dtype_include="number")),
        ("categorical", pipe_cat, make_column_selector(dtype_include="object"))
    ])

    models = (
        LogisticRegression(random_state=42, C=1.0, n_jobs=-1, class_weight='balanced'),
        RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1,
                               n_estimators=280, min_samples_split=10, min_samples_leaf=5, oob_score=True),
        MLPClassifier(random_state=42, max_iter=200, hidden_layer_sizes=(100, 20), activation="tanh")
    )

    best_score = .0
    best_pipe = None

    for model in models:
        pipe = Pipeline(steps=[
            ('initial', initial_pipe),
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipe.fit(x_train, y_train)
        prediction_test = pipe.predict_proba(x_test)[:, 1]
        roc_auc_test = roc_auc_score(y_test, prediction_test)
        print(f"model: {model},  roc_auc_score: {roc_auc_test:.4f}")

        if roc_auc_test > best_score:
            best_score = roc_auc_test
            best_pipe = pipe
    best_pipe.fit(x, y)
    print(f'best model: {type(best_pipe.named_steps["model"]).__name__}, roc_auc_test: {best_score:.4f}')
    with open("model.pkl", "wb") as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': "Sber_auto_podpiska",
                'author': 'Yana Kuptsova',
                'version': 1,
                'type': type(best_pipe.named_steps["model"]).__name__,
                'roc_auc_test': best_score}
        }, file, recurse=True)


if __name__ == '__main__':
    main()

