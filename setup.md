# Environment Setups For ML-FAC and PRIME

## ML-FAC Environment Setup

### Create Conda Environment

```bash
conda create --name ML_FAC python=3.10 -y
conda activate ML_FAC
conda install pyarrow nomkl openblas -c conda-forge -y
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.9 -y
```

### Configure Pathing

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

echo 'export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

echo 'export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo 'unset OLD_LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

conda deactivate
conda activate ML_FAC
```

### Install Everything From Scratch

```bash
pip install numpy scipy pandas matplotlib scikit-learn netCDF4 tqdm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### In Case Environment Goes Kaput

```bash
conda deactivate
conda remove --name ML_FAC --all -y
```

## PRIME Environment Setup

### Create PRIME Conda Environment

```bash
conda create --name PRIME_ENV python=3.10 -y
conda activate PRIME_ENV
```

### Download Optimizations

```bash
# For AMD Computers
conda install nomkl pyarrow openblas -c conda-forge -y

# For Intel Computers
conda install numpy scipy pandas scikit-learn -c conda-forge -y
```

### Install Prime

```bash
pip install primesw cdasws
```

### In Case Environment Dies

```bash
conda deactivate
conda remove --name PRIME_ENV --all -y
```
