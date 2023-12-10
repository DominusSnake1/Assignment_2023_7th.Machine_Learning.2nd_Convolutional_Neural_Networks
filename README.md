```bash
conda create --name py38 python=3.8
```

```bash
conda activate py38
```

```bash
conda install cpuonly -c pytorch --file requirements.txt
```

```bash
python ./main.py
```