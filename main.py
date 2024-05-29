import platform
import subprocess
if platform.system == "Windows":
    subprocess.run("python -m venv venv/")
    subprocess.run(r".\venv\Scripts\Activate.ps1")
    subprocess.run("pip install -r requirements.txt")
    subprocess.run("cd app/")
    subprocess.run("streamlit run app.py")
    subprocess.run("cd backend && flask run --port=5000")
else:
    subprocess.run("python3 -m venv venv/")
    subprocess.run("source venv/bin/activate")    
    subprocess.run("pip install -r requirements.txt")
    subprocess.run("cd app/")
    subprocess.run("streamlit run app.py & (cd backend && flask run --port=5000)")