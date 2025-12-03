@echo off
title Transcricao Bot
cd /d C:\transcricao_bot

echo ================================
echo    TRANSCRICAO BOT
echo ================================
echo.

:menu
echo [1] Sem proxy
echo [2] Com proxy (credenciais salvas)
echo [3] Com proxy (digitar credenciais)
echo.
set /p opcao="Escolha [1-3]: "

if "%opcao%"=="1" goto sem_proxy
if "%opcao%"=="2" goto proxy_salvo
if "%opcao%"=="3" goto proxy_digitado
goto menu

:sem_proxy
echo.
echo Executando sem proxy...
set HTTP_PROXY=
set HTTPS_PROXY=
goto iniciar

:proxy_salvo
echo.
echo Usando proxy com credenciais salvas...
set HTTP_PROXY=http://dayana.magalhaes:Daniel.2021@172.31.136.14:128
set HTTPS_PROXY=http://dayana.magalhaes:Daniel.2021@172.31.136.14:128
goto iniciar

:proxy_digitado
echo.
set /p usuario="Usuario: "
set /p senha="Senha: "
set HTTP_PROXY=http://%usuario%:%senha%@172.31.136.14:128
set HTTPS_PROXY=http://%usuario%:%senha%@172.31.136.14:128
echo Proxy configurado.
goto iniciar

:iniciar
call .venv\Scripts\activate.bat
echo.
echo Iniciando aplicacao...
streamlit run str_chip.py
echo.
pause
