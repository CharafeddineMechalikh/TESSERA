@echo off
setlocal
cd /d "%~dp0"
echo [launch] Working directory: %CD%

if not "%1"=="fast" (
  echo [launch] Installing pinned dependencies...
  python -m pip install -r requirements.txt
)

echo [launch] Generation runs through the distributed decoder, not Ollama
echo [launch] Serving on http://127.0.0.1:8000
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
