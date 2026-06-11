import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from sklearn.preprocessing import StandardScaler

def load_data(file_path: Path, pressure_level: int) -> xr.Dataset:
    """"
    Загружает данные из NetCDF файла и фильтрует по заданному уровню давления.
    args:
        file_path: Путь к NetCDF файлу
        pressure_level: Уровень давления для фильтрации данных
    returns:
        nc_data: данные отфильтрованные по уровню давлению
    """

    ds = xr.open_dataset(file_path, engine="netcdf4")

    if "time_bnds" in ds.data_vars:
        ds = ds.drop_vars("time_bnds")

    ds = ds.isel(lev=pressure_level)

    return ds

def dataset_to_dataframe(ds: xr.Dataset, features: list) -> pd.DataFrame:
    """
    Преобразует xarray Dataset в pandas DataFrame с MultiIndex (time, lon, lat) и указанными признаками.
    
    params:
        ds: Dataset с координатами time, lon, lat и переменными признаков
        features: Список имен переменных признаков для включения в DataFrame
    return:
        df: DataFrame с MultiIndex (time, lon, lat) и столбцами признаков
    """
    df = ds[features].to_dataframe(dim_order=('time', 'lon', 'lat'))
    return df

def normalize_dataframe(ds: xr.Dataset, features: list) -> pd.DataFrame:
    """
    Нормализует указанные признаки в DataFrame
    
    params:
        df: DataFrame с MultiIndex (time, lon, lat) и столбцами признаков
        features: Список имен переменных признаков для нормализации
    return:
        df_scaled: DataFrame с нормализованными признаками
        scaler_params: Словарь с параметрами нормализации (mean, scale) для каждого признака
    """

    mean= ds.mean(dim='time')
    std = ds.std(dim='time')

    norm_ds = (ds-mean)/std  
    norm_df = norm_ds[features].to_dataframe(dim_order=('time', 'lon', 'lat')).reset_index()
    norm_df=norm_df.iloc[:,-3:-1] 

    return norm_df


