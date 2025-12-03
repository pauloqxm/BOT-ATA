@echo off
title Configuracao Transcricao Bot
cls

echo =====================================
echo   CONFIGURACAO DO AMBIENTE
echo =====================================
echo.

cd /d C:\transcricao_bot

echo Verificando Python 3.13...
if not exist "C:\Users\dayana.magalhaes\AppData\Local\Programs\Python\Python313\python.exe" (
    echo ERRO: Python 3.13 nao encontrado.
    echo.
    echo Instale o Python 3.13 em:
    echo C:\Users\dayana.magalhaes\AppData\Local\Programs\Python\Python313\
    echo.
    pause
    exit /b
)

echo Python 3.13 encontrado.
echo.

REM Remove ambiente virtual antigo se existir
if exist ".venv" (
    echo Removendo ambiente virtual antigo...
    rmdir /s /q .venv
    echo Ambiente virtual removido.
    echo.
)

REM Cria novo ambiente virtual
echo Criando novo ambiente virtual...
"C:\Users\dayana.magalhaes\AppData\Local\Programs\Python\Python313\python.exe" -m venv .venv

if %ERRORLEVEL% NEQ 0 (
    echo ERRO ao criar ambiente virtual.
    pause
    exit /b
)

echo Ambiente virtual criado com sucesso.
echo.

REM Ativa o venv
call .venv\Scripts\activate.bat

if %ERRORLEVEL% NEQ 0 (
    echo ERRO ao ativar o ambiente virtual.
    pause
    exit /b
)

echo Ambiente virtual ativo.
echo.

REM Atualiza pip
echo Atualizando pip...
python -m pip install --upgrade pip

echo.
echo Pip atualizado.
echo.

REM Instala dependencias
if exist "requirements.txt" (
    echo Instalando dependencias do requirements.txt...
    echo.
    python -m pip install -r requirements.txt
    
    if %ERRORLEVEL% NEQ 0 (
        echo AVISO: Alguns pacotes podem ter falhado.
        echo Verifique os logs acima.
    )
    
    echo.
    echo Dependencias instaladas!
) else (
    echo AVISO: requirements.txt nao encontrado.
    echo.
    echo Criando arquivo requirements.txt basico...
    (
        echo streamlit
        echo openai-whisper
        echo pydub
        echo numpy
        echo torch
        echo ffmpeg-python
    ) > requirements.txt
    echo Arquivo requirements.txt criado.
    echo.
    echo Instalando dependencias basicas...
    python -m pip install -r requirements.txt
)

echo.
echo =====================================
echo   CONFIGURACAO CONCLUIDA!
echo =====================================
echo.
echo Execute RUN.bat para iniciar a aplicacao.
echo.
pause
