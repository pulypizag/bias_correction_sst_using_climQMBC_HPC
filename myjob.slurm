#!/bin/sh

## save this file as myjob.slurm
## nodes request and configuration

#SBATCH --partition=compute-fit          # Nombre de la particion a usar
#SBATCH --job-name= Divide_Zone_Bias     # Nombre del trabajo
#SBATCH --ntasks-per-node=1              # 1 proceso x nodo
#SBATCH --nodes=2                        # 2 nodos a utilizar
#SBATCH --cpus-per-task=128              # 128 hilos x proceso
#SBATCH --mem=512G                       # Mmeoria x nodo
#SBATCH --time=2-00:00:00                # Tiempo maximo de ejecucion (2 dias)


#SBATCH --output=/nfs_home/ppizarro/carpeta_de_outputs/%j.out              # output  userName_jobName_jobId.out
#SBATCH --error=/nfs_home/ppizarro/carpeta_de_outputs/%j.err


echo "Activating python module"
ml rhpython/3.8

echo "Currently using this python $(which python)"

echo "Creating python environment"
python -m venv my_env
source ./my_env/bin/activate

echo "Installing python packages"
python -m pip install --upgrade pip
python -m pip install numpy
python -m pip install scipy
python -m pip install matplotlib
python -m pip install pandas
python -m pip install seaborn
python -m pip install xarray
python -m pip install netCDF4
python -m pip install dask
python -m pip install dask-jobqueue
echo "Installed python packages"

#cargar modulos necesarios
module load dask-jobqueue

# Ejecutar tu script de Python
echo "STARTING SCRIPT"
srun python ./DivideZonesBiasMejorGuardado.py
