# run_both.py
import os
import subprocess
import uvicorn

def start_chainlit():
    subprocess.Popen(["chainlit", "run", "app.py"])

if __name__ == "__main__":
    # Set environment variable
    os.environ["ENCRYPTION_KEY"] = "your_secret_key"
    start_chainlit()
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
