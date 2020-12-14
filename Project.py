"""Author: Dvir Sadon"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as skl


def load_data():
    energy_df = pd.read_csv("energy_dataset.csv")
    weather_df = pd.read_csv("weather_features.csv")
    return energy_df, weather_df


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""DATA CLEANING""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def clean_data(energy_df, weather_df):
    ce = clean_energy(energy_df)
    cw = clean_weather(weather_df)
    final_df = merge_datasets(ce, cw)
    return final_df


def clean_energy(energy_df):
    # Dropping al the columns I will not be using (Some have no values and some are forecasts that will hurt the model).
    energy_df = energy_df.drop(['generation fossil coal-derived gas', 'generation fossil oil shale',
                                'generation fossil peat', 'generation geothermal',
                                'generation hydro pumped storage aggregated', 'generation marine',
                                'generation wind offshore', 'forecast wind offshore eday ahead',
                                'total load forecast', 'forecast solar day ahead',
                                'forecast wind onshore day ahead'], axis=1)

    energy_df['time'] = pd.to_datetime(energy_df['time'], utc=True, infer_datetime_format=True)
    energy_df = energy_df.set_index('time')

    # print(df_energy.isnull().sum(axis=0))
    # Because "Total load actual" has 36 rows where it is NaN and deleting
    # them is out of the question, I will try to predict price instead ("price actual" has none)

    energy_df.interpolate(method='linear', limit_direction='forward', inplace=True, axis=0)  # Just interpolating Na
    # values that are left linearly for now
    return energy_df


def clean_weather(weather_df):
    weather_df = df_convert_dtypes(weather_df, np.int64, np.float64)  # Convert all values to float64

    # Change the time type to datetime
    weather_df['time'] = pd.to_datetime(weather_df['dt_iso'], utc=True, infer_datetime_format=True)
    weather_df = weather_df.drop(['dt_iso'], axis=1)
    weather_df = weather_df.set_index('time')

    # Drop duplicate rows for each city
    weather_df_2 = weather_df.reset_index().drop_duplicates(subset=['time', 'city_name'],
                                                            keep='last').set_index('time')

    weather_df = weather_df.reset_index().drop_duplicates(subset=['time', 'city_name'],
                                                          keep='first').set_index('time')

    # Dropping columns with quality descriptions of the weather
    weather_df = weather_df.drop(['weather_main', 'weather_id',
                                  'weather_description', 'weather_icon'], axis=1)

    # Removing (making into null) the values in pressure and wind speed columns that are not realistic
    weather_df.loc[weather_df.pressure > 1051, 'pressure'] = np.nan
    weather_df.loc[weather_df.pressure < 931, 'pressure'] = np.nan
    weather_df.loc[weather_df.wind_speed > 50, 'wind_speed'] = np.nan

    # Linearly interpolate the null values we deleted
    weather_df.interpolate(method='linear', limit_direction='forward', inplace=True, axis=0)
    return weather_df


def merge_datasets(energy_df, weather_df):
    df_1, df_2, df_3, df_4, df_5 = [x for _, x in weather_df.groupby('city_name')]
    dfs = [df_1, df_2, df_3, df_4, df_5]

    df_final = energy_df

    for df in dfs:
        city = df['city_name'].unique()
        city_str = str(city).replace("'", "").replace('[', '').replace(']', '').replace(' ', '')
        df = df.add_suffix('_{}'.format(city_str))
        df_final = df_final.merge(df, on=['time'], how='outer')
        df_final = df_final.drop('city_name_{}'.format(city_str), axis=1)

    print(df_final.columns)
    return df_final


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""APPLY MODELS"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def get_divided(df):
    X_train = df.drop('price actual', axis=1).as_matrix()
    y_train = df['price actual'].as_matrix()

    X_train, X_test, y_train, y_test = skl.train_test_split(X_train, y_train, test_size=0.2)
    return X_train, X_test, y_train, y_test


def naive_model():
    pass


def define_network():
    features = 71
    hidden_layer_nodes = 10
    x = tf.placeholder(tf.float32, [None, features])
    y_ = tf.placeholder(tf.float32, [None, 1])


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""UTILITY""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def plot_series(df=None, column=None, series=pd.Series([]),
                label=None, ylabel=None, title=None, start=0, end=None):
    sns.set()
    fig, ax = plt.subplots(figsize=(30, 12))
    ax.set_xlabel('Time', fontsize=16)
    if column:
        ax.plot(df[column][start:end], label=label)
        ax.set_ylabel(ylabel, fontsize=16)
    if series.any():
        ax.plot(series, label=label)
        ax.set_ylabel(ylabel, fontsize=16)
    if label:
        ax.legend(fontsize=16)
    if title:
        ax.set_title(title, fontsize=24)
    ax.grid(True)
    return ax


def df_convert_dtypes(df, convert_from, convert_to):
    cols = df.select_dtypes(include=[convert_from]).columns
    for col in cols:
        df[col] = df[col].values.astype(convert_to)
    return df


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""MAIN"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def main():
    edf, wdf = load_data()
    fdf = clean_data(edf, wdf)
    # print(wdf.head())

    ax = plot_series(df=edf, column='total load actual', ylabel='Total Load (MWh)',
                     title='Actual Total Load (First 2 weeks - Original)', end=24 * 7 * 2)
    plt.show()

    # print(edf[edf.isnull().any(axis=1)].tail())

    pd.set_option('max_columns', 20)
    print(fdf.describe().round(2))

    print(fdf.shape)


main()
