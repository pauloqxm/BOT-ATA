@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Corrigir Instalacao - Transcricao Bot
cd /d C:\Users\paulo.ferreira\bot_decifraVOZ

REM ==========================================
REM PYTHON FIXO (SEU PYTHON 3.13)
REM ==========================================
set "PYTHON_EXE=C:\Users\paulo.ferreira\AppData\Local\Programs\Python\Python313\python.exe"
set "LOG=instalacao_log.txt"

echo. > "%LOG%"
echo INICIO %date% %time% >> "%LOG%"

if not exist "%PYTHON_EXE%" (
    echo ERRO. Python nao encontrado em:
    echo %PYTHON_EXE%
    echo ERRO. Python nao encontrado em %PYTHON_EXE% >> "%LOG%"
    pause
    exit /b 1
)

echo ================================
echo   CORRIGIR INSTALACAO
echo ================================
echo Pasta: %cd%
echo Python: %PYTHON_EXE%
"%PYTHON_EXE%" --version
echo Log: %LOG%
echo.

set "PROXY_HOST=172.31.136.14"
set "PROXY_PORT=128"
set "PIP_PROXY="

:menu
echo [1] Instalar sem proxy
echo [2] Instalar com proxy (digitar credenciais)
echo [3] Recriar .venv do zero
echo [0] Sair
echo.
set /p opcao="Escolha: "

if "%opcao%"=="1" goto sem_proxy
if "%opcao%"=="2" goto proxy_digitado
if "%opcao%"=="3" goto recriar_venv
if "%opcao%"=="0" goto fim
echo Opcao invalida.
echo.
goto menu

:sem_proxy
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "PIP_PROXY="
echo Proxy: desativado >> "%LOG%"
goto fluxo

:proxy_digitado
echo.
set /p usuario="Usuario: "
set /p senha="Senha: "

REM URL-encode seguro via PowerShell (nao depende de Python)
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "[uri]::EscapeDataString('%usuario%')"`) do set "USER_ENC=%%i"
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "[uri]::EscapeDataString('%senha%')"`) do set "PASS_ENC=%%i"

set "HTTP_PROXY=http://%USER_ENC%:%PASS_ENC%@%PROXY_HOST%:%PROXY_PORT%"
set "HTTPS_PROXY=http://%USER_ENC%:%PASS_ENC%@%PROXY_HOST%:%PROXY_PORT%"
set "PIP_PROXY=http://%USER_ENC%:%PASS_ENC%@%PROXY_HOST%:%PROXY_PORT%"

echo Proxy configurado.
echo Proxy configurado. >> "%LOG%"
goto fluxo

:recriar_venv
echo.
set /p confirma="Vai apagar .venv. Digite SIM: "
if /I not "%confirma%"=="SIM" (
    echo Cancelado.
    echo Cancelado recriacao venv >> "%LOG%"
    echo.
    goto menu
)
if exist ".venv" rmdir /s /q ".venv"
echo Recriando .venv >> "%LOG%"
goto fluxo


:fluxo
call :criar_ou_validar_venv
if errorlevel 1 goto fatal

call :testar_conexao_pypi
if errorlevel 1 goto fatal

call :instalar_dependencias
if errorlevel 1 goto fatal

call :testar_importacoes
if errorlevel 1 goto fatal

call :gerar_requirements
if errorlevel 1 goto fatal

echo.
echo ================================
echo   CONCLUIDO COM SUCESSO
echo ================================
echo.
echo Para rodar:
echo .venv\Scripts\streamlit.exe run str_chip.py
echo.
echo Log: %LOG%
echo.
pause
goto fim


:criar_ou_validar_venv
echo.
echo ================================
echo   ETAPA 1  Criar/Validar VENV
echo ================================
echo.
echo Verificando venv... >> "%LOG%"

if not exist ".venv\Scripts\python.exe" (
    echo Criando .venv...
    echo Criando .venv... >> "%LOG%"
    "%PYTHON_EXE%" -m venv .venv >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo ERRO ao criar o .venv. Veja o log.
        echo ERRO ao criar o .venv >> "%LOG%"
        pause
        exit /b 1
    )
)

set "VENV_PY=%cd%\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo ERRO: python.exe do venv nao encontrado.
    echo ERRO: python.exe do venv nao encontrado >> "%LOG%"
    pause
    exit /b 1
)

echo OK. Venv pronto.
echo OK. Venv pronto. >> "%LOG%"
exit /b 0


:testar_conexao_pypi
echo.
echo ================================
echo   ETAPA 2  Testar conexao PyPI
echo ================================
echo.
echo Testando acesso ao PyPI (real)... >> "%LOG%"

set "PIP_CMD=%VENV_PY% -m pip"
if not "%PIP_PROXY%"=="" (
    set "PIP_CMD=%VENV_PY% -m pip --proxy %PIP_PROXY%"
)

REM Teste REAL: baixa um pacote pequeno (wheel) do index
%PIP_CMD% download --no-deps --dest ._pip_test wheel -q >> "%LOG%" 2>&1

if errorlevel 1 (
    echo ERRO. Ainda sem acesso ao PyPI via proxy.
    echo Procure por "407" no log.
    echo.
    echo ERRO. Ainda sem acesso ao PyPI >> "%LOG%"
    pause
    exit /b 1
)

if exist "._pip_test" rmdir /s /q "._pip_test"

echo OK. PyPI acessivel de verdade.
echo OK. PyPI acessivel de verdade. >> "%LOG%"
exit /b 0


:instalar_dependencias
echo.
echo ================================
echo   ETAPA 3  Instalar pacotes
echo ================================
echo.
echo Instalando pacotes... >> "%LOG%"

set "PIP_CMD=%VENV_PY% -m pip"
if not "%PIP_PROXY%"=="" (
    set "PIP_CMD=%VENV_PY% -m pip --proxy %PIP_PROXY%"
)

REM Atualiza pip
%PIP_CMD% install --upgrade pip >> "%LOG%" 2>&1

REM Base (leve)
%PIP_CMD% install wheel setuptools==70.0.0 >> "%LOG%" 2>&1
if errorlevel 1 exit /b 1

REM ==========================================
REM Pesados: so aceita WHEEL (sem compilar)
REM Isso evita erro de compiler no Windows
REM ==========================================
echo Instalando pesados (somente wheel)...
echo Instalando pesados (somente wheel)... >> "%LOG%"

%PIP_CMD% install --only-binary=:all: numpy >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ERRO: Nao achei wheel do numpy para seu Python 3.13.
    echo Troque para Python 3.11 ou instale Build Tools.
    pause
    exit /b 1
)

%PIP_CMD% install --only-binary=:all: scipy >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ERRO: Nao achei wheel do scipy para seu Python 3.13.
    echo Troque para Python 3.11 ou instale Build Tools.
    pause
    exit /b 1
)

%PIP_CMD% install --only-binary=:all: pandas >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ERRO: Nao achei wheel do pandas para seu Python 3.13.
    echo Troque para Python 3.11 ou instale Build Tools.
    pause
    exit /b 1
)

REM Numba/llvmlite podem nao ter wheel pra 3.13 dependendo da versao
%PIP_CMD% install --only-binary=:all: llvmlite >> "%LOG%" 2>&1
if errorlevel 1 (
    echo AVISO: llvmlite sem wheel pra 3.13 no seu ambiente. Vou seguir sem numba.
    echo AVISO: llvmlite sem wheel >> "%LOG%"
)

%PIP_CMD% install --only-binary=:all: numba >> "%LOG%" 2>&1
if errorlevel 1 (
    echo AVISO: numba sem wheel pra 3.13 no seu ambiente. Vou seguir sem numba.
    echo AVISO: numba sem wheel >> "%LOG%"
)

REM Outros pacotes
%PIP_CMD% install psutil python-dotenv tqdm regex pyyaml protobuf soundfile matplotlib altair librosa >> "%LOG%" 2>&1
if errorlevel 1 exit /b 1

REM Torch no Windows + Python 3.13 pode ser chato. Se falhar, a saida do log vai dizer.
%PIP_CMD% install torch torchaudio >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ERRO: torch/torchaudio falhou no Python 3.13.
    echo Solucao que sempre funciona: usar Python 3.11.
    pause
    exit /b 1
)

%PIP_CMD% install openai-whisper streamlit >> "%LOG%" 2>&1
if errorlevel 1 exit /b 1

exit /b 0


:testar_importacoes
echo.
echo ================================
echo   ETAPA 4  Testar imports
echo ================================
echo.
"%VENV_PY%" -c "import numpy; import pandas; import psutil; import streamlit; print('OK basicos')" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ERRO em imports basicos. Veja o log.
    pause
    exit /b 1
)

"%VENV_PY%" -c "import torch; import torchaudio; print('OK torch', torch.__version__)" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ERRO no torch/torchaudio. Veja o log.
    pause
    exit /b 1
)

"%VENV_PY%" -c "import whisper; print('OK whisper')" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ERRO no whisper. Veja o log.
    pause
    exit /b 1
)

exit /b 0


:gerar_requirements
echo.
echo ================================
echo   ETAPA 5  Gerar requirements
echo ================================
echo.
"%VENV_PY%" -m pip freeze > requirements.txt
if errorlevel 1 (
    echo ERRO ao gerar requirements.txt
    pause
    exit /b 1
)
exit /b 0


:fatal
echo.
echo ================================
echo   FALHOU
echo ================================
echo Abre o "%LOG%" e me mande as linhas do erro (as ultimas 40).
pause
goto fim


:fim
echo FIM %date% %time% >> "%LOG%"
endlocal
exit /b 0
