# DataWhispers Recommender System for Data Exploration DHBW course

## Our tech stack

- Git: Version control 
- Streamlit: UI & Cloud Hosting
- Scikit, Torch and Sentence Transformer: For the models
- EDA and Preprocessing: Pandas

## View of cloud-hosted web-app

It will load longer since the models will be trained from scratch.
Estimated time: 4mins

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://datawhispers-movie-recommender.streamlit.app/)

## Setup from scratch

1. clone the repository

    ``` bash
    git clone https://github.com/GermanPaul12/DataWhispers-Recommender-System-DHBW.git
    ```

2. Change directory to the project

    ``` bash
    cd DataWhispers-Recommender-System-DHBW/
    ```

3. Create a virtual environment:

    Linux or mac OS

    ``` bash
    python3 -m venv venv/
    ```

    Windows

    ``` bash
    python -m venv venv/
    ```

4. Activate the virtual environment

    Linux or mac OS

    ``` bash
    source venv/bin/activate
    ```

    Windows

    ``` bash
    .\venv\Scripts\Activate.ps1
    ```

5. Install the dependencies

    ``` bash
    pip install -r requirements.txt
    ```

6. Start the application

    ``` bash
    streamlit run app/app.py 
    ```

7. View the web-app at http://localhost:8501/