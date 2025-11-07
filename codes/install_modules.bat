@REM @echo off
@REM REM ============================================
@REM REM Batch script to create and configure a conda environment on WINDOWS
@REM REM ============================================

@REM REM Default environment name
@REM set default_env_name=merna

@REM REM If the environment is active, deactivate it
@REM IF NOT "%CONDA_DEFAULT_ENV%"=="" (
@REM     echo Deactivate environment: %CONDA_DEFAULT_ENV%
@REM     CALL conda deactivate
@REM )

@REM REM Prompt user for environment name
@REM set /p env_name=Enter the name of the environment to create (press Enter to use '%default_env_name%'):

@REM REM If user input is empty, use default
@REM if "%env_name%"=="" (
@REM     set env_name=%default_env_name%
@REM )

@REM REM Ensure conda is available
@REM CALL conda --version

@REM REM Remove existing env if exists
@REM CALL conda env list | findstr /R /C:"%env_name%" >nul
@REM IF %ERRORLEVEL%==0 (
@REM     echo Environment '%env_name%' already exists.
@REM     echo Removing existing environment...
@REM     CALL conda remove --name %env_name% --all -y
@REM )

@REM REM CALL conda create -n %env_name% python=3.10 -c conda-forge -y

@REM CALL conda install -n base -c conda-forge mamba -y
@REM CALL mamba create -n %env_name% python=3.10 -c conda-forge -y
@REM CALL mamba install numpy scipy matplotlib shapely gdal pyproj ipympl ipywidgets -c conda-forge -y
@REM CALL pip install harmonica 

@REM REM Activate the environment
@REM CALL conda activate %env_name%

@REM REM Install packages
@REM REM CALL conda config --env --set channel_priority strict
@REM REM CALL conda install numpy scipy matplotlib shapely gdal pyproj -c conda-forge -y --repodata-fn=repodata.json
@REM REM CALL conda install ipympl ipywidgets -c conda-forge -y
@REM REM CALL pip install pdfkit
@REM REM CALL conda install -c conda-forge wkhtmltopdf -y

@REM echo.
@REM echo ============================================
@REM echo Done! The environment '%env_name%' is ready.
@REM echo ============================================


@echo off
set default_env_name=merna
IF NOT "%CONDA_DEFAULT_ENV%"=="" (
    echo Deactivating current environment: %CONDA_DEFAULT_ENV%
    CALL conda deactivate
)

set /p env_name=Enter the name of the environment to create (press Enter to use '%default_env_name%'):
if "%env_name%"=="" set env_name=%default_env_name%

REM 1) Installa il solver veloce
CALL conda install -n base -c conda-forge conda-libmamba-solver -y
CALL conda config --set solver libmamba
CALL conda config --set channel_priority strict
CALL conda config --add channels conda-forge
CALL conda config --remove channels defaults  2>nul

REM 2) Crea l’ambiente con pin “furbi” (riduce i conflitti)
CALL conda create -n merna -y ^
  python=3.10 numpy>=1.26,<2 scipy matplotlib ^
  "gdal=3.9.*" "proj=9.4.*" "geos=3.12.*" shapely pyproj ipympl ipywidgets harmonica

CALL conda activate merna
