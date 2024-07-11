# DataWhispers Recommender System for Data Exploration DHBW course

## Github Link

[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/GermanPaul12/DataWhispers-Recommender-System-DHBW)

## Our Goal

Our goal is to develop a content-based recommender system that makes recommendations for other similar films based on one or more films watched. In addition, the system is to be presented in a user-friendly interface.

## Team Members

- German Paul
- Michael Greif
- Nico Dobberkau

## Our tech stack

- Git: Version control
- Streamlit: UI
- Scikit-Learn, Torch and Sentence Transformer: Model creation
- EDA and Preprocessing: Pandas, Numpy, String, Missingno

## Our structure

1. App Directory
   - Streamlit App
2. Data Directory
   - All the data we need for the project
   - Logos and images
   - Pretrained model files
   - Our paper
   - Our presentation
   - Our datasets and preprocessed datasets
3. EDA directory
   - Notebook for the Explanatory Data Analysis
4. Model directory
   - Model creation files
   - Creation of preprocessed datasets scripts
5. Root directory
   - All the directories
   - Our documentation, dependencies and files that should be ignored by git

## Download and view our report on this project here

[![Download as PDF Button](https://img.shields.io/badge/Download%20AS%20pdf-EF3939?style=for-the-badge&logo=adobeacrobatreader&logoColor=white&color=black&labelColor=ec1c24)](https://github.com/GermanPaul12/DataWhispers-Recommender-System-DHBW/data/report/project_report_4.pdf?raw=true)

## Download and view our PowerPoint for the project here

[![Download PP as PDF Button](https://img.shields.io/badge/Download%20AS%20pdf-EF3939?style=for-the-badge&logo=adobeacrobatreader&logoColor=white&color=black&labelColor=ec1c24)](https://github.com/GermanPaul12/DataWhispers-Recommender-System-DHBW/data/presentation/project_presentation_4.pdf?raw=true)

## Prerequisites for the local setup

Probably should work with most python3 versions >= 3.9, but if you don't have python3 installed on your machine we recommend to install the same version:

[![Python 3.11.2](https://img.shields.io/badge/python-3.11.2-blue.svg)](https://www.python.org/downloads/release/python-3112/)

Also you need a RAM space capacity of at least 8GB ;-)

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
