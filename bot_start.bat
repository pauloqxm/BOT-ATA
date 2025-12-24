@echo off
title Transcricao Bot
cd /d C:\Users\paulo.ferreira\bot_decifraVOZ

call .venv\Scripts\activate.bat

echo Iniciando Streamlit...
streamlit run str_chip.py --server.port 8501


pause
