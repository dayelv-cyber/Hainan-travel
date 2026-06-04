@echo off
cd /d "%~dp0"
D:\aca\python.exe -m streamlit run app.py --server.port 8502
pause
