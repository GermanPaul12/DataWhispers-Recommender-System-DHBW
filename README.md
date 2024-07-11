# DataWhispers Recommender System for Data Exploration DHBW course

## Github Link

![Click here for the Github Repo](https://img.shields.io/badge/click%20on%20the%20badge%20to%20the%20right%20for%20the%20github%20repository-FFFFFF?style=for-the-badge) [![Click here for the GitHub Repository](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/GermanPaul12/DataWhispers-Recommender-System-DHBW)

## Our Goal

Our goal is to develop a content-based recommender system that makes recommendations for other similar films based on one or more films watched. In addition, the system is to be presented in a user-friendly interface.

## Team Members

:bust_in_silhouette: German Paul

:bust_in_silhouette: Michael Greif

:bust_in_silhouette: Nico Dobberkau

## Our tech stack

![Version Control](https://img.shields.io/badge/version%20control-000000?style=for-the-badge) ![Git](https://img.shields.io/badge/GIT-E44C30?style=for-the-badge&logo=git&logoColor=white)

![User Interface](https://img.shields.io/badge/user%20interface-000000?style=for-the-badge) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
 
![Model Creation](https://img.shields.io/badge/model%20creation-000000?style=for-the-badge) ![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![Sentence Transformer](https://img.shields.io/badge/sentence%20transformer-000000?style=for-the-badge)

![Explanatory Data Analysis and Data Preprocessing](https://img.shields.io/badge/Explanatory%20Data%20Analysis%20and%20Data%20Preprocessing-000000?style=for-the-badge) ![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white) ![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)  

![Data Visualization](https://img.shields.io/badge/data%20visualization-000000?style=for-the-badge) ![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

![Datasets](https://img.shields.io/badge/dataset-000000?style=for-the-badge) ![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)

## Our structure

```
📦root directory
 ┣ 📂app
 ┃ ┣ 📜Streamlit App
 ┣ 📂data
 ┃ ┣ 📜Logos and movie image placeholder
 ┃ ┣ 📜Pretrained model files
 ┃ ┣ 📜Report
 ┃ ┣ 📜Presentation
 ┃ ┣ 📜Our datasets and preprocessed datasets
 ┣ 📂EDA
 ┃ ┣ 📜Explanatory Data Analysis Notebook
 ┣ 📂model
 ┃ ┣ 📜Model creation files
 ┃ ┣ 📜Creation of preprocessed datasets scripts
 ┣ 📜README
 ┣ 📜dependencies
 ┗ 📜files to be ignored by github
```

## Download and view our report on this project here

![Click here for the report](https://img.shields.io/badge/click%20on%20the%20badge%20to%20the%20right%20for%20the%20report-FFFFFF?style=for-the-badge) [![Click here to download the PDF](https://img.shields.io/badge/Download%20AS%20pdf-EF3939?style=for-the-badge&logo=adobeacrobatreader&logoColor=white&color=black&labelColor=ec1c24)](https://raw.githubusercontent.com/GermanPaul12/DataWhispers-Recommender-System-DHBW/main/data/report/project_report_4.pdf)

## Download and view our PowerPoint for the project here

![Click here for the presentation](https://img.shields.io/badge/click%20on%20the%20badge%20to%20the%20right%20for%20the%20presentation-FFFFFF?style=for-the-badge) [![Click here to download the PDF](https://img.shields.io/badge/Download%20AS%20pdf-EF3939?style=for-the-badge&logo=adobeacrobatreader&logoColor=white&color=black&labelColor=ec1c24)](https://raw.githubusercontent.com/GermanPaul12/DataWhispers-Recommender-System-DHBW/main/data/presentation/project_presentation_4.pdf)

## Prerequisites for the local setup

:information_source: Should work with all python 3 versions >= ![Python 3.9](https://img.shields.io/badge/python-3.9-white.svg), but if you don't have python 3 installed on your machine we recommend to install the same version as we have 

:snake: Click here to get to the download page of our python version: [![Python 3.11.2](https://img.shields.io/badge/python-3.11.2-blue.svg)](https://www.python.org/downloads/release/python-3112/)

:computer: Also you need a RAM space capacity of at least 8GB, since models will be saved in memory ;-)

## Setup project locally on your machine from scratch

1. clone the repository

   ```bash
   git clone https://github.com/GermanPaul12/DataWhispers-Recommender-System-DHBW.git
   ```
2. Change directory to the project

   ```bash
   cd DataWhispers-Recommender-System-DHBW/
   ```
3. Create a virtual environment:

   Linux or mac OS

   ```bash
   python3 -m venv venv/
   ```
   Windows

   ```bash
   python -m venv venv/
   ```
4. Activate the virtual environment

   Linux or mac OS

   ```bash
   source venv/bin/activate
   ```
   Windows

   ```bash
   .\venv\Scripts\Activate
   ```
5. Install the dependencies

   ```bash
   pip install -r requirements.txt
   ```
6. Start the application

   ```bash
   streamlit run app/app.py 
   ```
7. View the web-app at http://localhost:8501/
