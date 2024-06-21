# DataWhispers Recommender System for Data Exploration DHBW course

## Our tech stack

- Git: Version control
- Streamlit: UI & Cloud Hosting
- Scikit, Torch and Sentence Transformer: For the models
- EDA and Preprocessing: Pandas

## View our paper on this project

[![Download as PDF Button](https://img.shields.io/badge/Download%20AS%20pdf-EF3939?style=for-the-badge&logo=adobeacrobatreader&logoColor=white&color=black&labelColor=ec1c24)](https://github.com/GermanPaul12/DataWhispers-Recommender-System-DHBW/data/paper/Ausarbeitung.pdf?raw=true)

## Prerequisites

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
