```bash
conda create --name py38 python=3.8
```

```bash
conda activate py38
```

```bash
conda install cpuonly -c pytorch -c conda-forge --file requirements.txt
```

## 4. It is required to download the [dataset](https://drive.google.com/u/3/uc?id=1DaFBO4K9Z7fE9k5C_LrdyWJSs76OsvYl&export=download) into the `Data` folder.

```bash
python ./main.py
```