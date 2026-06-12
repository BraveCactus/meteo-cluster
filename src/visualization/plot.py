import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Tuple, Optional
from scipy import stats

def plot_cluster_map(ds: xr.Dataset, 
                     time_idx: int, 
                     output_path: Optional[Path] = None,
                     region: Optional[Tuple[float, float, float, float]] = None) -> plt.Figure:
    """
    Рисует карту кластеров для заданного временного индекса.

    params:
        ds: xarray Dataset с координатами 'lon', 'lat' и переменной 'cluster'
        time_idx: Временной индекс для отображения
        output_path: Необязательный путь для сохранения фигуры
        region: Необязательный кортеж (min_lon, max_lon, min_lat, max_lat) для выделения рассматриваемого региона
    returns:
        fig: карта кластеров для заданного временного индекса
    """

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    data = ds["cluster"].isel(time=time_idx)
    values = data.to_numpy()

    unique_labels = np.unique(values)
    n_clusters = int(unique_labels.size)

    if n_clusters <= 1:
        raise ValueError("At least two clusters are required for plotting")
    
    cmap = plt.get_cmap("tab20", n_clusters)
    levels = np.arange(n_clusters + 1)

    if region:
        ax.set_extent(region, crs=ccrs.PlateCarree())

    im = data.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        levels=levels,
        add_colorbar=False,
    )

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5)

    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(n_clusters))
    cbar.set_label("Cluster")

    time_str = str(ds.time.isel(time=time_idx).values)[:10]
    ax.set_title(f"Cluster Distribution - {time_str}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)    
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig

def plot_cluster_map_period(ds: xr.Dataset, 
                            time_range: Tuple[int, int], 
                            output_path: Optional[Path] = None,
                            region: Optional[Tuple[float, float, float, float]] = None) -> plt.Figure:
    """
    Рисует карту доминирующего кластера за указанный период.
    
    params:
        ds: xarray Dataset с переменной 'cluster'
        time_range: Кортеж (start_idx, end_idx) - интервал времени
        output_path: Необязательный путь для сохранения карты
        region: Необязательный кортеж (min_lon, max_lon, min_lat, max_lat)
    returns:
        fig: карта доминирующего кластера за указанный период
    """    
    
    start_idx, end_idx = time_range    
   
    data = ds["cluster"].isel(time=slice(start_idx, end_idx))    
    
    mode_result = xr.apply_ufunc(
        stats.mode,
        data,
        input_core_dims=[['time']],
        output_core_dims=[[], []],
        vectorize=True
    )
    
    mode = mode_result[0]
    
    unique_clusters = np.unique(mode.values)
    n_clusters = len(unique_clusters)
    
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    if region:
        ax.set_extent(region, crs=ccrs.PlateCarree())    
    
    im = mode.plot(ax=ax, transform=ccrs.PlateCarree(), 
                   cmap="tab10", levels=np.arange(n_clusters + 1), add_colorbar=False)
    
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5)    
    
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(n_clusters))
    cbar.set_label("Dominant Cluster")    
    
    start_date = str(ds.time.isel(time=start_idx).values)[:10]
    end_date = str(ds.time.isel(time=end_idx - 1).values)[:10]
    ax.set_title(f"Dominant Cluster - {start_date} to {end_date}")
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)    
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig

def plot_seasonal_similarity(ds: xr.Dataset, 
                             period1: Tuple[int, int], 
                             period2: Tuple[int, int],
                             output_path: Optional[Path] = None,
                             region: Optional[Tuple[float, float, float, float]] = None,
                             cmap: str = "viridis") -> plt.Figure:
    """
    Сравнивает два временных периода и рисует карту сходства кластеров.
    
    params:
        ds: xarray Dataset с переменной 'cluster'
        period1: Кортеж (start_idx, end_idx) для первого периода
        period2: Кортеж (start_idx, end_idx) для второго периода
        output_path: Необязательный путь для сохранения карты
        region: Необязательный кортеж (min_lon, max_lon, min_lat, max_lat)
        cmap: Цветовая карта
    returns:
        fig: карта сходства кластеров между двумя периодами
    """
    start1, end1 = period1
    start2, end2 = period2    
    
    data1 = ds["cluster"].isel(time=slice(start1, end1))
    data2 = ds["cluster"].isel(time=slice(start2, end2))
    
    n1 = end1 - start1
    n2 = end2 - start2    
    
    data1_aligned = data1.assign_coords(time=range(n1))
    data2_aligned = data2.assign_coords(time=range(n2))    
    
    data1_aligned = data1_aligned.rename('r1')
    data2_aligned = data2_aligned.rename('r2')    
    
    min_len = min(n1, n2)
    r_da = (data1_aligned.isel(time=slice(0, min_len)) == 
            data2_aligned.isel(time=slice(0, min_len))).mean(dim='time')
    
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    if region:
        ax.set_extent(region, crs=ccrs.PlateCarree())
    
    im = r_da.plot(ax=ax, transform=ccrs.PlateCarree(), 
                   cmap=cmap, vmin=0, vmax=1, add_colorbar=False)
    
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5)
    
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 0.25, 0.5, 0.75, 1])
    cbar.set_label("Similarity (fraction of days with same cluster)")
    
    start_date1 = str(ds.time.isel(time=start1).values)[:10]
    end_date1 = str(ds.time.isel(time=end1 - 1).values)[:10]
    start_date2 = str(ds.time.isel(time=start2).values)[:10]
    end_date2 = str(ds.time.isel(time=end2 - 1).values)[:10]
    ax.set_title(f"Cluster Similarity: {start_date1}-{end_date1} vs {start_date2}-{end_date2}")
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig

def plot_parameter_change(ds: xr.Dataset,
                            period1: Tuple[int, int],
                            period2: Tuple[int, int],
                            variable: str = "T",
                            output_path: Optional[Path] = None,
                            region: Optional[Tuple[float, float, float, float]] = None,
                            cmap: str = "RdBu_r") -> plt.Figure:
    """
    Сравнивает параметр между двумя периодами.
    
    params:
        ds: xarray Dataset с переменными H, T, U, V
        period1: Кортеж (start_idx, end_idx) для первого периода
        period2: Кортеж (start_idx, end_idx) для второго периода
        variable: Имя параметра ('T', 'H', 'U', 'V')
        output_path: Необязательный путь для сохранения карты
        region: Необязательный кортеж (min_lon, max_lon, min_lat, max_lat)
        cmap: Цветовая карта
    returns:
        fig: карта изменения параметра между двумя периодами
    """
    start1, end1 = period1
    start2, end2 = period2    
    
    data1 = ds[variable].isel(time=slice(start1, end1))
    data2 = ds[variable].isel(time=slice(start2, end2))    
   
    n1 = end1 - start1
    n2 = end2 - start2
        
    data1_aligned = data1.assign_coords(time=range(n1))
    data2_aligned = data2.assign_coords(time=range(n2))    
    
    min_len = min(n1, n2)    
    
    diff = (data1_aligned.isel(time=slice(0, min_len)) - 
            data2_aligned.isel(time=slice(0, min_len))).mean(dim='time')    
    
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    if region:
        ax.set_extent(region, crs=ccrs.PlateCarree())    
   
    vmax = max(abs(diff.min().item()), abs(diff.max().item()))
    
    im = diff.plot(ax=ax, transform=ccrs.PlateCarree(), 
                   cmap=cmap, vmin=-vmax, vmax=vmax, add_colorbar=False)
    
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5)
    
    cbar = plt.colorbar(im, ax=ax)    
    
    var_names = {'T': 'Temperature', 'H': 'Geopotential', 'U': 'U-wind', 'V': 'V-wind'}
    var_name = var_names.get(variable, variable)
    units = {'T': 'K', 'H': 'm²/s²', 'U': 'm/s', 'V': 'm/s'}
    unit = units.get(variable, '')
    
    cbar.set_label(f"{var_name} Difference ({unit})")    
    
    start_date1 = str(ds.time.isel(time=start1).values)[:10]
    end_date1 = str(ds.time.isel(time=end1 - 1).values)[:10]
    start_date2 = str(ds.time.isel(time=start2).values)[:10]
    end_date2 = str(ds.time.isel(time=end2 - 1).values)[:10]
    ax.set_title(f"{var_name} Change: {start_date1}-{end_date1} vs {start_date2}-{end_date2}")
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig