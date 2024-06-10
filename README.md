# DataWhispers Recommender System for Data Exploration DHBW course

## Our tech stack

- Git: to track the versions
- Streamlit: UI
- Flask: Backend
- Bert and Scikit: For the models
- EDA and Preprocessing: Pandas

## View of cloud-hosted web-app

It will load longer since the models will be trained from scratch.
Estimated time: 4mins



## Setup from scratch

1. clone the repository

    ``` bash
    git clone https://github.com/GermanPaul12/DataWhispers-Recommender-System-DHBW.git
    ```

2. Download the large files here:

   [Large files link](https://stadsinitiative-my.sharepoint.com/:f:/g/personal/german_paul_stads_de/EohrgaWKqj1MiAwoA4b-QaUBQlr8Qta-gO0P4GAMqWa_zQ?e=edlDrd)
3. Add the large files to the `data` directory
4. Create a virtual environment:

    Linux or mac OS

    ``` bash
    python3 -m venv venv/
    ```

    Windows

    ``` bash
    python -m venv venv/
    ```

5. Activate the virtual environment

    Linux or mac OS

    ``` bash
    source venv/bin/activate
    ```

    Windows

    ``` bash
    .\venv\Scripts\Activate.ps1
    ```

6. Install the dependencies

    ``` bash
    pip install -r requirements.txt
    ```

7. Change directory to the app

    ``` bash
    cd app/
    ```

8. Start the application

    ``` bash
    streamlit run app/app.py 
    ```

9. View the web-app at http://localhost:8501/
