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
        fig: matplotlib Figure
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

    if output_path and not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path:
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
        fig: matplotlib Figure
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
    
    if output_path and not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig