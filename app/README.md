# Project: Content Based - Movie Recommender System!

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n movie python=3.7.10 -y
```

```bash
conda activate movie
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

### STEP 02- run!
Linux:
```bash
streamlit run app.py & (cd backend && flask run --port=5000)
```

Windows:
```bash
concurrently "streamlit run app.py" "(cd backend && flask run --port=5000)"
```