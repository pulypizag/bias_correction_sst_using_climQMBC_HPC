##### ZONA MAS PEQUENA, NO GLOBAL #####
# Importar librerias
import sys
import os
from climQMBC.methods import QM
import xarray as xr
import numpy as np
import logging
import time as time_seconds
from dask import delayed
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import datetime
import pandas as pd

### CONFIGURAR LOGGING ###
logging.basicConfig(
    filename="procesamiento_puntos1.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

### CONFIGURAR DASK ###
# Configurar el cluster Dask para SLURM
cluster = SLURMCluster(
    queue="compute-fit",        # Nombre de la particion
    cores=128,                     # Nucleos por trabajo
    memory="512GB",                # Memoria por trabajo
    walltime="2-00:00:00",       # Tiempo maximo de ejecucion
    
)

cluster.scale(jobs=2)      # Escalar a 2 nodos
#Crear cliente Dask para distribuir tareas
client = Client(cluster)
print("Dashboard Dask:", client.dashboard_link)
logging.info("Cluster Dask configurado.")

### CARGAR DATA OBSERVADA ###
t0 = time_seconds.perf_counter()
logging.info("Cargando datos observados...")
ds = xr.open_dataset('/nfs_home/ppizarro/sst.nc')
sst = ds.sst[94 * 12:170 * 12]
sst = sst.isel(lat=slice(None, None, -1))
sst = sst.isel(time=slice(0, 804))
sst = sst.fillna(0)
lat = sst.lat.values
lon = sst.lon.values
sst_100 = np.tile(sst, (100, 1, 1))
t1 = time_seconds.perf_counter()
logging.info(f"Datos observados cargados en {t1 - t0:.2f} segundos.")

### RESTRICCION A REGION ###
lat_lo = -50
lat_hi = -18
lon_lo = 250
lon_hi = 290
model_lon_indices = (sst.lon >= lon_lo) & (sst.lon <= lon_hi)
model_lat_indices = (sst.lat >= lat_lo) & (sst.lat <= lat_hi)
sst_region = sst[:, model_lat_indices, :][:, :, model_lon_indices]
sst_100_region = np.tile(sst_region, (100, 1, 1))
lat = sst_region.lat.values
lon = sst_region.lon.values

### CARGAR DATOS MODELO LENS2 ###
def cargar_datos(file_path):
    ds = xr.open_dataset(file_path)
    realizacion = ds.SST.isel(time=slice(1175, 1979))
    realizacion = realizacion.where(realizacion != 0, drop=False) - 273.15
    return realizacion

logging.info("Cargando datos del modelo...")
input_dir_h = '/nfs_home/ppizarro/Data_sst_HPC/Historic_Regridded'
nc_files_h = [os.path.join(input_dir_h, f) for f in os.listdir(input_dir_h) if f.endswith('.nc')]
realizaciones_his = {i: cargar_datos(file)[:, model_lat_indices, :][:, :, model_lon_indices] for i, file in enumerate(nc_files_h)}
logging.info("Datos del modelo cargados.")

### PROCESAR PUNTOS CON DASK ###
N_realizaciones = 100
Lobs = 804

def procesar_punto(jk):
    j, k = jk
    t_start = time_seconds.perf_counter()
    # Validar datos historicos
    try:
        sst_his = np.concatenate([realizaciones_his[i][:, j, k] for i in range(N_realizaciones)])
        if len(sst_his) == 0 or np.isnan(sst_his).all():
            raise ValueError("sst_his esta vacio o lleno de NaNs")
        obs_data = sst_100_region[:, j, k]
        if len(obs_data) == 0 or np.isnan(obs_data).all():
            raise ValueError("sst_100_region esta vacio o lleno de NaNs")
    except Exception as e:
        logging.warning(f"Error en punto ({j}, {k}): {e}")
        return j, k, None

    # Aplicar correccion de sesgo
    try:
        resultado_QM = QM(obs=obs_data, mod=sst_his, allow_negatives=0, frq="M")
    except ValueError as e:
        logging.warning(f"Error en QM para punto ({j}, {k}): {e}")
        return j, k, None

    t_end = time_seconds.perf_counter()
    logging.info(f"Punto ({j}, {k}) procesado en {t_end - t_start:.2f} segundos.")
    return j, k, resultado_QM

#crear grilla puntos para generar tareas en cada nodo?
grilla_puntos = [(j, k) for j in range(len(lat)) for k in range(len(lon))]
#Crear tareas Dask para cada punto
tareas = [delayed(procesar_punto)(punto) for punto in grilla_puntos]
#computar y recolectar resultados
resultados = client.gather(client.compute(tareas))
logging.info("Procesamiento completado.")



### INICIALIZAR MATRIZ DE RESULTADOS ###
sst_his_corrected = np.zeros((Lobs * N_realizaciones, len(lat), len(lon)))

# Llenar la matriz con los resultados de QM
for res in resultados:
    if res[2] is None:
        continue
    j, k, his_corr = res
    sst_his_corrected[:, j, k] = his_corr

### CONFIGURAR FECHAS REALES ###
# Generar fechas reales desde enero de 1948
time_original = pd.date_range(start="1948-01-01", periods=Lobs, freq="M")
time_replica = np.tile(time_original, N_realizaciones)

### GUARDAR RESULTADOS MEJORADOS ###
# Crear DataArray para datos corregidos
sst_his_corrected_da = xr.DataArray(
    sst_his_corrected,
    dims=["time", "lat", "lon"],
    coords={"time": time_replica, "lat": lat, "lon": lon},
    attrs={
        "description": "Temperatura superficial del mar histórica corregida (QM)",
        "units": "°C",
        "method": "Quantile Mapping",
        "source": "Datos observados concatenados con modelo LENS2",
    },
)

# Crear DataArray para datos observados
sst_obs_da = xr.DataArray(
    sst_100_region,
    dims=["time", "lat", "lon"],
    coords={"time": time_replica, "lat": lat, "lon": lon},
    attrs={
        "description": "Temperatura superficial del mar observada (100 replicaciones)",
        "units": "°C",
    },
)

# Crear Dataset
ds = xr.Dataset(
    {
        "sst_his_corrected": sst_his_corrected_da,
        "sst_obs": sst_obs_da,
    },
    attrs={
        "title": "Resultados de correccion de sesgo QM para SST",
        "institution": "Universidad Adolfo Ibanez Stgo Chile",
        "created_on": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    },
)


# Guardar el Dataset en un archivo NetCDF
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"sst_corrected_QM_{timestamp}.nc"
ds.to_netcdf(output_filename, format="NETCDF4", encoding={"sst_his_corrected": {"zlib": True}})
logging.info(f"Resultados guardados en {output_filename}")

t1 = time_seconds.perf_counter()

logging.info("Se corrigio el sesgo correctamente en todos los puntos y se guardo el resultado.")
logging.info(f"Tiempo total: {t1 - t0:.2f} segundos.")