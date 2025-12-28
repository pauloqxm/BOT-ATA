# ============================================================
# PROXY FRONTEND (TEM QUE SER O PRIMEIRO BLOCO DO ARQUIVO)
# ============================================================
import os
import socket
import streamlit as st
from urllib.parse import quote

# Configuração inicial
st.set_page_config(
    page_title="DECIFRAVOZ| Configuração de Rede",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

PROXY_HOST = "172.31.136.14"
PROXY_PORT = "128"


def _clear_proxy_env():
    for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        os.environ.pop(k, None)


def _set_proxy_env(user, password, host, port, url_encode):
    if url_encode:
        user = quote(user, safe="")
        password = quote(password, safe="")
    proxy_url = f"http://{user}:{password}@{host}:{port}"
    os.environ.update({
        "HTTP_PROXY": proxy_url,
        "HTTPS_PROXY": proxy_url,
        "http_proxy": proxy_url,
        "https_proxy": proxy_url,
    })


def _test_proxy_connection():
    """Tenta abrir uma conexão socket para validar o host/porta"""
    try:
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((PROXY_HOST, int(PROXY_PORT)))
        return True
    except Exception:
        return False


def _proxy_selector_ui_gate():
    if st.session_state.get("proxy_configured"):
        return

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

    #MainMenu, footer, header {visibility: hidden;}
    .main { background-color: #f8fafc; }

    .proxy-container { max-width: 850px; margin: 40px auto; }
    .main-card {
        border-radius: 24px;
        overflow: hidden;
        box-shadow: 0 20px 50px rgba(0,0,0,0.1);
        background: white;
    }
    .hero-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 7px;
        margin: -9rem auto;
        text-align: center;
        margin-bottom: 0;
    }
    .config-body { padding: 40px; margin-top: 0; }

    .badge {
        background: rgba(255,255,255,0.2);
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .stTextInput > div > div > input {
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
        padding: 12px !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 1rem 2rem !important;
        transition: all 0.3s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.3);
    }
    .info-box {
        background-color: #f1f5f9;
        border-left: 4px solid #4f46e5;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 25px;
        font-size: 0.9rem;
        color: #475569;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="proxy-container">', unsafe_allow_html=True)
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="hero-header">
        <span class="badge">VERSÃO v2.0</span>
        <h1 style="margin: 15px 0 5px 0; font-weight:800; font-size: 2.5rem;">DECIFRAVOZ</h1>
        <p style="opacity: 0.9; font-size: 1.1rem;">Sistema de trascrição de áudio.</p>
        <div style="margin-top: 20px; display: flex; justify-content: center; gap: 20px;">
            <span>📍 <b>Host:</b> {PROXY_HOST}</span>
            <span>🔌 <b>Porta:</b> {PROXY_PORT}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="config-body">', unsafe_allow_html=True)

    col_mode, col_test = st.columns([3, 1])
    with col_mode:
        modo = st.segmented_control(
            "Selecione o modo de acesso",
            options=["Sem Proxy", "Proxy Autenticado"],
            default="Sem Proxy"
        )

    with col_test:
        st.write("")
        if st.button("🛜 Testar Rede", use_container_width=True):
            with st.spinner("Checando..."):
                if _test_proxy_connection():
                    st.toast("Conexão com o servidor OK!", icon="✅")
                else:
                    st.toast("Servidor inacessível.", icon="❌")

    st.markdown("<br>", unsafe_allow_html=True)

    if modo == "Sem Proxy":
        st.markdown("""
        <div class="info-box">
            <b>Conexão Direta Habilitada:</b> O sistema tentará acessar os servidores do Whisper
            utilizando a rota padrão da rede local, sem túneis de autenticação.
        </div>
        """, unsafe_allow_html=True)
        _clear_proxy_env()
        user = ""
        password = ""
        encode = False
    else:
        st.markdown("##### Credenciais Corporativas")
        c1, c2 = st.columns(2)
        with c1:
            user = st.text_input("Usuário AD", placeholder="ex: joao.silva")
        with c2:
            password = st.text_input("Senha de Rede", type="password", placeholder="••••••••")

        st.info("💡 Dica: Se sua senha possui caracteres como '@' ou '!', ative o encoding abaixo.")
        encode = st.toggle("Habilitar URL Encoding para segurança")

    st.divider()

    if st.button("🚀 Inicializar Sistema DECIFRAVOZ", type="primary", use_container_width=True):
        if modo == "Proxy Autenticado":
            if not user or not password:
                st.error("Por favor, preencha as credenciais de acesso.")
                st.stop()
            _set_proxy_env(user, password, PROXY_HOST, PROXY_PORT, encode)

        st.session_state.proxy_configured = True
        with st.status("Autenticando e carregando modelos...", expanded=True) as status:
            st.write("Configurando variáveis de ambiente...")
            st.write("Validando gateway...")
            status.update(label="Acesso Autorizado!", state="complete", expanded=False)

        st.balloons()
        st.rerun()

    st.markdown('</div></div></div>', unsafe_allow_html=True)
    st.stop()


# Gatekeeper
_proxy_selector_ui_gate()

# ============================================================
# A PARTIR DAQUI PODE CARREGAR O RESTO (IMPORTS PESADOS)
# ============================================================
import time
import warnings
import tempfile
import json
from pathlib import Path
from datetime import datetime
import re
import subprocess
import torch
import psutil
import platform
import librosa
import pandas as pd

# Whisper oficial
import whisper

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
warnings.filterwarnings("ignore", message=".*huggingface_hub.*")

# Ajuste de threads para não brigar com o Streamlit
num_threads = os.cpu_count() or 4
try:
    torch.set_num_threads(num_threads)
except RuntimeError:
    pass
os.environ["OMP_NUM_THREADS"] = str(num_threads)

# Anchor topo
st.markdown('<a id="top"></a>', unsafe_allow_html=True)

# ============================================================
# CSS PRINCIPAL (SEM CSS SEPARADO) + BARRA BRANCA DAS ABAS
# ============================================================
st.markdown("""
<style>
    /* Tema principal */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    .main-container {
        background: white;
        border-radius: 20px;
        padding: 0rem;
        margin: -8rem auto;
        box-shadow: 0 20px 60px rgba(0,0,0,0.1);
        max-width: 95%;
    }

    /* Botões modernos */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        width: 100%;
    }

    .stButton > button:hover:not(:disabled) {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    .stButton > button:disabled {
        background: #cccccc;
        box-shadow: none;
    }

    /* Uploader estilizado */
    .uploadedFile {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.2);
    }

    /* Métricas */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    .metric-value {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #333 !important;
        margin: 0.5rem 0;
    }

    .metric-label {
        color: #666 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600 !important;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }

    /* Cards */
    .custom-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
        margin: 1rem 0;
        border: 1px solid #f0f0f0;
    }

    .success-card {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
    }

    .warning-card {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 5px solid #ffc107;
    }

    .error-card {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
    }

    .info-card {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-left: 5px solid #17a2b8;
    }

    /* Tabs: barra branca abaixo dos nomes */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        background: #ffffff;
        border-radius: 14px;
        margin: 2rem auto;
        padding: 10px 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.06);
        margin-bottom: 1.25rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        padding: 0.65rem 1.1rem;
        font-weight: 700;
        color: #555;
        transition: all 0.25s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 6px 18px rgba(102, 126, 234, 0.25);
    }

    /* Timestamps */
    .timestamp-item {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }

    .timestamp-item:hover {
        transform: translateX(5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    /* Texto prévia */
    .text-preview {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        font-family: 'Courier New', monospace;
        line-height: 1.6;
        max-height: 300px;
        overflow-y: auto;
    }

    /* Status pills */
    .pill {
        padding: 0.45rem 0.9rem;
        border-radius: 999px;
        font-weight: 800;
        font-size: 0.9rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 72px;
        color: white;
        letter-spacing: 0.5px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }
    .pill-processing {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .pill-ok {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    }
    .pill-warn {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
    }

    .main-content { margin-bottom: 80px; }

    /* Botão voltar topo */
    .top-btn-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
    }
    .top-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        font-size: 24px;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        text-decoration: none;
    }
    .top-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# =============================
# Cabeçalho com imagem personalizada
# =============================
st.markdown("""
<div style="width: 100%; display: flex; justify-content: center; margin-bottom: 2rem;">
    <img src="https://i.ibb.co/6hdSJFc/Gemini-Generated-Image-ueiwonueiwonueiw.png"
         style="width: 100%; max-width: 1250px; margin: -4rem auto; border-radius: 14px; box-shadow: 0 6px 20px rgba(0,0,0,0.15);">
</div>
""", unsafe_allow_html=True)

# Container principal
st.markdown('<div class="main-container main-content">', unsafe_allow_html=True)

# =============================
# Arquivo de correções e histórico
# =============================
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path(".")
CORRECOES_FILE = BASE_DIR / "correcoes_custom.json"
HISTORICO_FILE = BASE_DIR / "historico_transcricoes.json"


def carregar_correcoes_custom():
    if CORRECOES_FILE.exists():
        try:
            with open(CORRECOES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
    return {}


def salvar_correcoes_custom(data: dict):
    try:
        with open(CORRECOES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Erro ao salvar correções. {e}")


def carregar_historico():
    if HISTORICO_FILE.exists():
        try:
            with open(HISTORICO_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            return []
    return []


def salvar_historico(lista: list):
    try:
        with open(HISTORICO_FILE, "w", encoding="utf-8") as f:
            json.dump(lista, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Erro ao salvar histórico. {e}")


# =============================
# Estado da aplicação
# =============================
if "correcoes_custom" not in st.session_state:
    st.session_state["correcoes_custom"] = carregar_correcoes_custom()

if "texto_transcrito" not in st.session_state:
    st.session_state["texto_transcrito"] = ""

if "texto_paragrafado" not in st.session_state:
    st.session_state["texto_paragrafado"] = ""

if "texto_pos_processado" not in st.session_state:
    st.session_state["texto_pos_processado"] = ""

if "historico_transcricoes" not in st.session_state:
    st.session_state["historico_transcricoes"] = carregar_historico()

# flags para evitar erro no session_state do text_area
if "pp_apply_pending" not in st.session_state:
    st.session_state["pp_apply_pending"] = False
if "pp_apply_text" not in st.session_state:
    st.session_state["pp_apply_text"] = ""


# =============================
# Utilitários gerais
# =============================
BASE_PROMPT = (
    "Transcrição em português brasileiro formal, com pontuação correta, "
    "acentuação adequada e frases completas. Use nomes próprios, siglas e "
    "termos técnicos conforme aparecem no áudio. Evite inventar trechos."
)


def get_correcoes_dicionario():
    correcoes_base = {
        "pq": "porque",
        "tb": "também",
        "vc": "você",
        "d": "de",
        "q": "que",
        "ta": "está",
        "tava": "estava",
        "pra": "para",
        "ne": "não é",
        "naum": "não",
        "entao": "então",
        "tbm": "também",
        "obg": "obrigado",
        "vlw": "valeu",
        "blz": "beleza",
        "cm": "com",
        "td": "tudo",
        "qd": "quando",
        "qq": "qualquer",
    }
    raw_custom = st.session_state.get("correcoes_custom", {})
    correcoes_custom = {}
    for k, v in raw_custom.items():
        key_clean = str(k).strip()
        val_clean = str(v).strip()
        if key_clean:
            correcoes_custom[key_clean] = val_clean

    correcoes = {}
    correcoes.update(correcoes_base)
    correcoes.update(correcoes_custom)
    return correcoes


def pos_processar_texto(texto: str) -> str:
    if not texto:
        return ""
    correcoes = get_correcoes_dicionario()
    texto = re.sub(r"\s+", " ", texto)
    for errado, correto in correcoes.items():
        padrao = r"\b{}\b".format(re.escape(errado))
        texto = re.sub(padrao, correto, texto, flags=re.IGNORECASE)
    texto = re.sub(r"\s+([.,!?])", r"\1", texto)
    return texto.strip()


def organizar_paragrafos(texto: str, max_caracteres=500) -> str:
    if not texto:
        return ""
    frases = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+", texto)
    paragrafos = []
    paragrafo_atual = ""
    for frase in frases:
        if not frase.strip():
            continue
        if len(paragrafo_atual) + len(frase) > max_caracteres and paragrafo_atual:
            paragrafos.append(paragrafo_atual.strip())
            paragrafo_atual = ""
        paragrafo_atual += frase + " "
    if paragrafo_atual:
        paragrafos.append(paragrafo_atual.strip())
    return "\n\n".join(paragrafos)


def capitalizar_frases(texto: str) -> str:
    if not texto:
        return ""
    frases = re.split(r"(?<=[.!?])\s+", texto)
    frases_capitalizadas = []
    for frase in frases:
        if frase:
            frase = frase.strip()
            if frase:
                frase = frase[0].upper() + frase[1:]
                frases_capitalizadas.append(frase)
    return " ".join(frases_capitalizadas)


def corrigir_pontuacao(texto: str) -> str:
    if not texto:
        return ""
    texto = re.sub(r"\s+([.,!?:;])", r"\1", texto)
    texto = re.sub(r"([.,!?:;])(?!\s|$)", r"\1 ", texto)
    texto = re.sub(r"([.,!?:;]){2,}", r"\1", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def formatar_ata(texto: str) -> str:
    if not texto:
        return ""
    if not texto.startswith("ATA DA REUNIÃO"):
        data_atual = datetime.now().strftime("%d/%m/%Y")
        texto = f"ATA DA REUNIÃO\nData: {data_atual}\n\n{texto}"
    if "Encerramento" not in texto and "FIM DA ATA" not in texto:
        texto += "\n\n---\nFIM DA ATA\n"
    return texto


def dividir_em_chunks(audio, sr, chunk_seg=120):
    partes = []
    tam = int(chunk_seg * sr)
    total = len(audio)
    for i in range(0, total, tam):
        parte = audio[i: i + tam]
        t_ini = i / sr
        t_fim = (i + len(parte)) / sr
        partes.append((parte, t_ini, t_fim))
    return partes


def formatar_tempo(segundos: float) -> str:
    minutos = int(segundos // 60)
    seg = int(segundos % 60)
    return f"{minutos:02d}:{seg:02d}"


def formatar_timestamps(timestamps, max_chars=400):
    linhas = []
    for ts in timestamps:
        texto = ts["text"]
        if len(texto) > max_chars:
            texto = texto[:max_chars] + "..."
        inicio = formatar_tempo(ts["start"])
        fim = formatar_tempo(ts["end"])
        linhas.append(f"<div class='timestamp-item'><b>[{inicio} - {fim}]</b> {texto}</div>")
    return "\n".join(linhas)


# =============================
# Detecção de NPU / GPU / Placa de vídeo (Windows)
# =============================
def detectar_npu(cpu_name: str):
    if not cpu_name:
        return False, "Não identificado"

    cpu_lower = cpu_name.lower()
    tem_npu = False
    descricao = "Não identificado"

    if "core ultra" in cpu_lower or "ultra 5" in cpu_lower or "ultra 7" in cpu_lower or "ultra 9" in cpu_lower:
        tem_npu = True
        descricao = "Intel NPU (linha Core Ultra)"
    elif "snapdragon" in cpu_lower or "qualcomm" in cpu_lower:
        tem_npu = True
        descricao = "NPU integrada (SoC Qualcomm)"

    return tem_npu, descricao


def detectar_gpu_e_placa_video():
    gpu_cuda = None
    placas_video = []

    if torch.cuda.is_available():
        try:
            gpu_cuda = torch.cuda.get_device_name(0)
        except Exception:
            gpu_cuda = "GPU CUDA detectada"

    try:
        if platform.system() == "Windows":
            creationflags = 0
            if hasattr(subprocess, "CREATE_NO_WINDOW"):
                creationflags = subprocess.CREATE_NO_WINDOW

            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "Name"],
                capture_output=True,
                text=True,
                creationflags=creationflags
            )
            linhas = [l.strip() for l in result.stdout.splitlines() if l.strip() and "Name" not in l]
            if linhas:
                placas_video.extend(linhas)

            if not placas_video:
                result_ps = subprocess.run(
                    ["powershell", "-Command",
                     "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                    capture_output=True,
                    text=True,
                    creationflags=creationflags
                )
                ps_lines = [l.strip() for l in result_ps.stdout.splitlines() if l.strip()]
                if ps_lines:
                    placas_video.extend(ps_lines)

    except Exception:
        pass

    if not placas_video:
        placas_video = ["Nenhuma placa identificada"]

    return gpu_cuda, placas_video


# =============================
# ACELERAÇÃO AUTOMÁTICA UNIVERSAL
# =============================
def detectar_acelerador():
    """
    Detecta automaticamente o melhor acelerador disponível:
    - CUDA (NVIDIA)
    - OpenVINO (Intel CPU / Intel GPU / NPU), se estiver instalado
    - CPU (fallback)
    """
    if torch.cuda.is_available():
        try:
            nome_gpu = torch.cuda.get_device_name(0)
            return {"engine": "cuda", "device": "cuda", "name": nome_gpu, "fp16": True}
        except Exception:
            pass

    try:
        import openvino  # noqa: F401
        from openvino.runtime import Core

        core = Core()
        dispositivos = core.available_devices
        prioridade = ["GPU", "NPU", "CPU"]

        for preferido in prioridade:
            for disp in dispositivos:
                if preferido in disp:
                    return {"engine": "openvino", "device": disp, "name": disp, "fp16": False}
    except Exception:
        pass

    return {"engine": "cpu", "device": "cpu", "name": "Processamento via CPU", "fp16": False}


@st.cache_resource(show_spinner=True)
def carregar_whisper_inteligente(modelo_nome, acelerador):
    engine = acelerador["engine"]
    device = acelerador["device"]

    st.info(f"Acelerador selecionado: **{acelerador['name']}** ({engine})")

    if engine in ("cuda", "cpu"):
        return whisper.load_model(modelo_nome, device=engine)

    if engine == "openvino":
        try:
            from openvino_whisper import load_model as load_ov
            return load_ov(modelo_nome, device=device)
        except Exception:
            st.warning("OpenVINO não está totalmente disponível. Voltando para CPU.")
            return whisper.load_model(modelo_nome, device="cpu")

    return whisper.load_model(modelo_nome, device="cpu")


def transcrever_com_whisper(audio, sr, modelo_nome: str, chunk_seg: int):
    acel = detectar_acelerador()
    engine = acel.get("engine", "cpu")
    fp16 = acel.get("fp16", False)
    device_name = acel.get("name", "cpu")

    st.markdown(f"""
    <div class="info-card">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="font-size: 2rem;">⚙️</div>
            <div>
                <h4 style="margin: 0;">Configuração do Sistema</h4>
                <p style="margin: 0;">Acelerador detectado: {device_name} ({engine})</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    duracao_min = len(audio) / sr / 60
    engine_label = str(engine).upper()

    st.markdown(f"""
    <div class="custom-card">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h3 style="margin: 0;">🎯 Modelo Selecionado</h3>
                <p style="margin: 0; color: #666;">{modelo_nome.upper()} em {engine_label}</p>
            </div>
            <span class="pill pill-processing">PRONTO</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner(f"Carregando modelo Whisper {modelo_nome}"):
        model = carregar_whisper_inteligente(modelo_nome, acel)

    partes = dividir_em_chunks(audio, sr, chunk_seg)
    total_partes = len(partes)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Partes para processar</div>
            <div class="metric-value">{total_partes}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Duração total</div>
            <div class="metric-value">{formatar_tempo(duracao_min * 60)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Tamanho do chunk</div>
            <div class="metric-value">{chunk_seg}s</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 Progresso da Transcrição")
    progress_bar = st.progress(0)
    progress_col1, progress_col2 = st.columns([4, 1])
    with progress_col2:
        percent_text = st.empty()

    texto_final = ""
    timestamps = []
    tempos_partes = []
    inicio_geral = time.time()

    for idx, (parte, t_ini, t_fim) in enumerate(partes, start=1):
        janela_min = t_ini / 60
        janela_max = t_fim / 60

        card = st.empty()

        # Render inicial: processando
        card.markdown(f"""
        <div class="custom-card">
            <div style="display:flex; align-items:center; justify-content:space-between; gap: 1rem;">
                <div style="flex:1;">
                    <h3 style="margin:0;">Parte {idx}/{total_partes}</h3>
                    <p style="margin:0.35rem 0 0 0; color:#666;">Janela: {janela_min:.1f}min - {janela_max:.1f}min</p>
                </div>
                <span class="pill pill-processing">PROCESSANDO</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        inicio_parte = time.time()

        if parte is None or len(parte) == 0 or float(abs(parte).max()) < 1e-6:
            tempos_partes.append(0.0)
            card.markdown(f"""
            <div class="custom-card">
                <div style="display:flex; align-items:center; justify-content:space-between; gap: 1rem;">
                    <div style="flex:1;">
                        <h3 style="margin:0;">Parte {idx}/{total_partes}</h3>
                        <p style="margin:0.35rem 0 0 0; color:#666;">Janela: {janela_min:.1f}min - {janela_max:.1f}min</p>
                        <p style="margin:0.65rem 0 0 0; color:#856404;"><b>Sem áudio detectado</b></p>
                    </div>
                    <span class="pill pill-warn">SEM ÁUDIO</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            progresso = idx / total_partes
            progress_bar.progress(progresso)
            percent_text.markdown(f"**{progresso*100:.0f}%**")
            continue

        kwargs = {"language": "pt", "task": "transcribe", "initial_prompt": BASE_PROMPT}
        kwargs["fp16"] = bool(engine == "cuda" and torch.cuda.is_available() and fp16)

        result = model.transcribe(parte, **kwargs)

        tempo_parte = time.time() - inicio_parte
        tempos_partes.append(float(tempo_parte))

        segs = result.get("segments", []) or []
        preview = ""
        if segs:
            for seg in segs:
                ttxt = seg["text"]
                start = float(seg["start"]) + t_ini
                end = float(seg["end"]) + t_ini
                timestamps.append({"start": start, "end": end, "text": ttxt})
                texto_final += ttxt + " "
            preview = (segs[0]["text"] or "").strip()
        else:
            preview = ""

        preview_short = preview[:140] + ("..." if len(preview) > 140 else "")
        tempo_label = f"{tempo_parte:.1f}s"

        if segs:
            card.markdown(f"""
            <div class="custom-card">
                <div style="display:flex; align-items:center; justify-content:space-between; gap: 1rem;">
                    <div style="flex:1;">
                        <h3 style="margin:0;">Parte {idx}/{total_partes}</h3>
                        <p style="margin:0.35rem 0 0 0; color:#666;">Janela: {janela_min:.1f}min - {janela_max:.1f}min</p>
                        <p style="margin:0.65rem 0 0 0; color:#155724;"><b>Prévia</b> {preview_short}</p>
                        <p style="margin:0.25rem 0 0 0; color:#0c5460;"><b>Tempo</b> {tempo_label}</p>
                    </div>
                    <span class="pill pill-ok">OK</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            card.markdown(f"""
            <div class="custom-card">
                <div style="display:flex; align-items:center; justify-content:space-between; gap: 1rem;">
                    <div style="flex:1;">
                        <h3 style="margin:0;">Parte {idx}/{total_partes}</h3>
                        <p style="margin:0.35rem 0 0 0; color:#666;">Janela: {janela_min:.1f}min - {janela_max:.1f}min</p>
                        <p style="margin:0.65rem 0 0 0; color:#856404;"><b>Sem áudio detectado</b></p>
                        <p style="margin:0.25rem 0 0 0; color:#856404;"><b>Tempo</b> {tempo_label}</p>
                    </div>
                    <span class="pill pill-warn">SEM ÁUDIO</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        progresso = idx / total_partes
        progress_bar.progress(progresso)
        percent_text.markdown(f"**{progresso*100:.0f}%**")

    tempo_total = time.time() - inicio_geral
    return texto_final, timestamps, tempo_total, duracao_min, total_partes, tempos_partes


# =============================
# Sidebar – configurações
# =============================
with st.sidebar:
    st.markdown("""
    <div style="padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; border-radius: 15px; margin-bottom: 2rem;">
        <h3 style="margin: 0;">⚙️ Configurações</h3>
        <p style="margin: 0; opacity: 0.9;">Ajuste os parâmetros de processamento</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🌐 Proxy")
    proxy_atual = os.environ.get("HTTP_PROXY", "") or os.environ.get("http_proxy", "")
    if proxy_atual:
        st.success("Proxy ativo no ambiente.", icon="✅")
    else:
        st.info("Sem proxy no ambiente.", icon="ℹ️")

    if st.button("🔁 Trocar proxy", use_container_width=True):
        st.session_state.proxy_configured = False
        st.rerun()

    st.markdown("---")

    st.markdown("### 🎯 Modelo Whisper")
    modelos = {
        "🧠 tiny – velocidade máxima": "tiny",
        "⚡ base – equilíbrio ideal": "base",
        "🎯 small – precisão superior": "small",
        "🏆 medium – qualidade premium": "medium",
        "👑 large-v3 – excelência máxima": "large-v3",
    }
    modelo_label = st.selectbox("Selecione o modelo:", list(modelos.keys()), index=1)
    modelo_whisper = modelos[modelo_label]

    st.markdown("---")

    st.markdown("### 📊 Tamanho das Partes")
    chunk_segundos = st.slider(
        "Duração (segundos):",
        min_value=30,
        max_value=300,
        value=120,
        step=30,
        help="Partes menores = mais preciso\nPartes maiores = mais rápido"
    )

    st.markdown("---")
    st.markdown("### 💻 Sistema")

    try:
        cpu_info = platform.processor()
        if not cpu_info:
            cpu_info = "Processador não identificado"
    except Exception:
        cpu_info = "Processador não identificado"

    ram_total = psutil.virtual_memory().total / (1024**3)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Threads", num_threads)
        st.metric("RAM Total", f"{ram_total:.1f} GB")
    with c2:
        st.metric("PyTorch", str(torch.__version__)[:6])
        st.metric("Sistema", platform.system())

    with st.expander("📋 Detalhes do Sistema"):
        st.write(f"**Processador:** {cpu_info}")
        st.write(f"**Arquitetura:** {platform.machine()}")
        st.write(f"**Python:** {platform.python_version()}")
        st.write(f"**Whisper:** {whisper.__version__ if hasattr(whisper, '__version__') else 'N/A'}")

        mem = psutil.virtual_memory()
        st.write(f"**RAM Usada:** {mem.percent}%")
        st.write(f"**RAM Disponível:** {mem.available / (1024**3):.1f} GB")

        tem_npu, desc_npu = detectar_npu(cpu_info)
        st.write(f"**NPU:** {desc_npu if tem_npu else 'não detectada'}")

        gpu_cuda, placas_video = detectar_gpu_e_placa_video()
        if gpu_cuda:
            st.write(f"**GPU (CUDA):** {gpu_cuda}")
            try:
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                st.write(f"**VRAM Total:** {vram_total:.1f} GB")
            except Exception:
                pass
        else:
            st.write("**GPU (CUDA):** não detectada")

        st.markdown("**Placa(s) de vídeo detectada(s):**")
        for nome in placas_video:
            st.write(f"• {nome}")


# =============================
# Abas principais
# =============================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎧 TRANSCREVER ÁUDIO",
    "📚 BIBLIOTECA DE CORREÇÕES",
    "📝 PÓS-PROCESSAMENTO",
    "📊 HISTÓRICO"
])

# =============================
# Aba 1 – Transcrição
# =============================
with tab1:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>🎤 Envie seu Áudio</h2>
        <p style="color: #666;">Suporta MP3, WAV, M4A, OGG, FLAC, AAC, WMA</p>
    </div>
    """, unsafe_allow_html=True)

    audio_file = st.file_uploader(
        "Faça o upload do áudio",
        type=["mp3", "wav", "m4a", "ogg", "flac", "aac", "wma"],
        label_visibility="visible",
        key="audio_uploader_tab1"
    )

    if audio_file is not None:
        tamanho_mb = audio_file.size / 1024 / 1024
        st.markdown(f"""
        <div class="uploadedFile">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div style="flex: 1;">
                    <h4 style="margin: 0;">✅ {audio_file.name}</h4>
                    <p style="margin: 0; opacity: 0.9;">Arquivo pronto para transcrição</p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📁</div>
                    <h3 style="margin: 0;">{tamanho_mb:.1f} MB</h3>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        transcribe_clicked = st.button(
            "🚀 INICIAR TRANSCRIÇÃO",
            disabled=(audio_file is None),
            use_container_width=True,
            type="primary",
            key="transcribe_button_tab1"
        )

    if transcribe_clicked:
        if audio_file is None:
            st.error("⚠️ Por favor, envie um arquivo de áudio primeiro.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=audio_file.name) as tmp:
                tmp.write(audio_file.read())
                caminho_audio = tmp.name

            try:
                with st.spinner("Preparando áudio para processamento"):
                    audio, sr_original = librosa.load(caminho_audio, sr=None, mono=True)

                    max_abs = max(1e-8, float(abs(audio).max()))
                    audio = audio / max_abs * 0.9

                    if sr_original != 16000:
                        audio = librosa.resample(audio, orig_sr=sr_original, target_sr=16000)
                        sr = 16000
                    else:
                        sr = sr_original

                    duracao_min_pre = len(audio) / sr / 60
                    partes_preview = dividir_em_chunks(audio, sr, chunk_segundos)
                    total_partes_preview = len(partes_preview)

                st.markdown("### 📊 Visão Geral do Arquivo")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Duração Total</div>
                        <div class="metric-value">{formatar_tempo(duracao_min_pre * 60)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Tamanho</div>
                        <div class="metric-value">{tamanho_mb:.1f} MB</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_c:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Partes</div>
                        <div class="metric-value">{total_partes_preview}</div>
                    </div>
                    """, unsafe_allow_html=True)

                (
                    texto_bruto,
                    ts,
                    tempo_proc,
                    duracao_min,
                    total_partes,
                    tempos_partes,
                ) = transcrever_com_whisper(audio, sr, modelo_whisper, chunk_segundos)

                texto_pos = pos_processar_texto(texto_bruto)
                texto_corrigido = corrigir_pontuacao(capitalizar_frases(texto_pos))
                texto_paragrafado = organizar_paragrafos(texto_corrigido)

                st.session_state["texto_transcrito"] = (texto_bruto or "").strip()
                st.session_state["texto_paragrafado"] = texto_paragrafado
                st.session_state["texto_pos_processado"] = texto_paragrafado

                if not texto_corrigido.strip():
                    st.error("Nenhum texto final gerado. Verifique se o áudio tem fala clara.")
                else:
                    hist = st.session_state.get("historico_transcricoes", [])
                    item = {
                        "timestamp": datetime.now().isoformat(),
                        "arquivo": audio_file.name,
                        "modelo": modelo_whisper,
                        "duracao_min": float(duracao_min),
                        "tempo_proc": float(tempo_proc),
                        "palavras": len(texto_corrigido.split()),
                        "preview": texto_paragrafado[:1000] + ("..." if len(texto_paragrafado) > 1000 else "")
                    }
                    hist.insert(0, item)
                    st.session_state["historico_transcricoes"] = hist[:20]
                    salvar_historico(st.session_state["historico_transcricoes"])

                    st.markdown("""
                    <div class="success-card" style="padding: 2rem;">
                        <div style="text-align: center;">
                            <h2 style="margin: 0; color: #155724;">Transcrição Concluída</h2>
                            <p style="margin: 0; color: #0c5460;">Processamento finalizado com sucesso</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("### 📈 Estatísticas de Processamento")

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Duração Áudio</div>
                            <div class="metric-value">{formatar_tempo(duracao_min * 60)}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Tempo Process.</div>
                            <div class="metric-value">{formatar_tempo(tempo_proc)}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c3:
                        velocidade_x = (duracao_min * 60) / tempo_proc if tempo_proc > 0 else 0
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Velocidade</div>
                            <div class="metric-value">{velocidade_x:.1f}x</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c4:
                        palavras = len(texto_corrigido.split())
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Palavras</div>
                            <div class="metric-value">{palavras}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    if tempos_partes:
                        st.markdown("### 📊 Desempenho por Parte")
                        df_tempos = pd.DataFrame({
                            "Parte": list(range(1, total_partes + 1)),
                            "Tempo (s)": tempos_partes,
                        })
                        st.bar_chart(df_tempos.set_index("Parte"))

                    st.markdown("### 🧾 Prévia da Transcrição (com parágrafos)")
                    preview_texto = texto_paragrafado[:800] + "..." if len(texto_paragrafado) > 800 else texto_paragrafado
                    st.markdown(f"""
                    <div class="text-preview">
                        {preview_texto.replace("\\n\\n", "<br><br>")}
                        <br><br><small><i>Total: {len(texto_corrigido)} caracteres, {len(texto_corrigido.split())} palavras</i></small>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("### ⏱️ Timestamps Detalhados")
                    if ts:
                        timestamps_html = formatar_timestamps(ts)
                        st.markdown(f"""
                        <div style="max-height: 400px; overflow-y: auto; padding: 1rem;">
                            {timestamps_html}
                        </div>
                        """, unsafe_allow_html=True)
                        texto_ts = "\n".join([
                            f"[{formatar_tempo(t['start'])} - {formatar_tempo(t['end'])}] {t['text'][:400]}"
                            for t in ts
                        ])
                    else:
                        st.info("Nenhum timestamp disponível")
                        texto_ts = ""

                    st.markdown("### 📥 Download dos Resultados")
                    nome_base = os.path.splitext(audio_file.name)[0]
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        st.download_button(
                            "📄 Baixar Transcrição com Parágrafos",
                            data=texto_paragrafado,
                            file_name=f"transcricao_paragrafada_{nome_base}_{timestamp_str}.txt",
                            mime="text/plain",
                            use_container_width=True,
                            key="download_paragrafada"
                        )
                    with dl_col2:
                        if ts:
                            st.download_button(
                                "⏱️ Baixar Timestamps",
                                data=texto_ts,
                                file_name=f"timestamps_{nome_base}_{timestamp_str}.txt",
                                mime="text/plain",
                                use_container_width=True,
                                key="download_timestamps_tab1"
                            )

            finally:
                try:
                    os.unlink(caminho_audio)
                except Exception:
                    pass


# =============================
# Aba 2 – Biblioteca de correções (EM DUAS COLUNAS) + SEM "LIMPAR TUDO"
# =============================
with tab2:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>📚 Biblioteca de Correções</h2>
        <p style="color: #666;">Gerencie as substituições automáticas aplicadas nas transcrições</p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1])

    # Coluna esquerda: Correções Ativas
    with col_left:
        st.markdown("### 📋 Correções Ativas")
        dicionario_atual = get_correcoes_dicionario()

        if dicionario_atual:
            df_correcoes = pd.DataFrame([{"Original": k, "Substituir por": v} for k, v in dicionario_atual.items()])

            st.dataframe(
                df_correcoes,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Original": st.column_config.TextColumn("Palavra Original", help="Termo que será substituído"),
                    "Substituir por": st.column_config.TextColumn("Substituição", help="Termo que substituirá o original")
                }
            )

            st.markdown(f"""
            <div class="info-card">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <h4 style="margin: 0;">Resumo</h4>
                        <p style="margin: 0;">{len(dicionario_atual)} correções ativas</p>
                    </div>
                    <span class="pill pill-ok">ATIVO</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-card">
                <div style="text-align: center; padding: 2rem;">
                    <div style="font-size: 3rem;">📝</div>
                    <h4>Nenhuma correção cadastrada</h4>
                    <p>Adicione sua primeira correção ao lado</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Coluna direita: Adicionar Novas Correções (SEM botão Limpar Tudo)
    with col_right:
        st.markdown("### ➕ Novas Correções")

        with st.form("form_add_correcoes"):
            st.markdown("""
            <div class="custom-card">
                <h4 style="margin-top:0;">Múltiplas Palavras</h4>
                <p style="color: #666; font-size: 0.9rem;">Campos vazios serão ignorados</p>
            """, unsafe_allow_html=True)

            correcoes_inputs = []
            for i in range(8):
                c_orig, c_sub = st.columns([1, 1])
                with c_orig:
                    original = st.text_input(
                        f"Original {i+1}",
                        placeholder="Ex: vc, tb, d+, etc.",
                        key=f"original_input_{i}"
                    )
                with c_sub:
                    substituir = st.text_input(
                        f"Substituir por {i+1}",
                        placeholder="Ex: você, também, muito, etc.",
                        key=f"substituir_input_{i}"
                    )
                correcoes_inputs.append((original, substituir))

            submit_col1, submit_col2 = st.columns([2, 1])
            with submit_col1:
                submitted = st.form_submit_button(
                    "➕ Adicionar Todas",
                    use_container_width=True,
                    type="primary"
                )
            with submit_col2:
                add_selected = st.form_submit_button(
                    "📝 Incluir",
                    use_container_width=True
                )

            st.markdown("</div>", unsafe_allow_html=True)

            if submitted or add_selected:
                correcoes_adicionadas = []
                for o, s in correcoes_inputs:
                    if o.strip() and s.strip():
                        chave = o.strip()
                        valor = s.strip()
                        st.session_state["correcoes_custom"][chave] = valor
                        correcoes_adicionadas.append((chave, valor))

                if correcoes_adicionadas:
                    salvar_correcoes_custom(st.session_state["correcoes_custom"])
                    st.success(f"{len(correcoes_adicionadas)} correções adicionadas")
                    st.rerun()
                else:
                    st.warning("Nenhuma correção válida para adicionar. Preencha pelo menos um par.")


# =============================
# Aba 3 – Pós-processamento (SEM ERRO: usa callback)
# =============================
def _pp_aplicar_correcoes():
    base = st.session_state.get("texto_pos_processado_area", "")
    texto_corr = pos_processar_texto(base)
    texto_corr = corrigir_pontuacao(capitalizar_frases(texto_corr))
    texto_corr = organizar_paragrafos(texto_corr)

    # Atualiza o estado do widget no callback (permitido)
    st.session_state["texto_pos_processado_area"] = texto_corr
    st.session_state["texto_pos_processado"] = texto_corr
    st.session_state["pp_msg_ok"] = "Correções aplicadas."

def _pp_resetar():
    base = (
        st.session_state.get("texto_paragrafado", "").strip()
        or st.session_state.get("texto_transcrito", "").strip()
    )
    st.session_state["texto_pos_processado_area"] = base
    st.session_state["texto_pos_processado"] = base
    st.session_state["pp_msg_ok"] = "Texto restaurado."

def _pp_limpar():
    st.session_state["texto_pos_processado_area"] = ""
    st.session_state["texto_pos_processado"] = ""
    st.session_state["pp_msg_ok"] = "Texto limpo."

with tab3:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>📝 Pós-processamento do Texto</h2>
        <p style="color: #666;">À esquerda o texto bruto. À direita o texto corrigido.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.get("texto_transcrito", "").strip():
        st.info("Ainda não há transcrição disponível. Faça uma transcrição na aba de transcrição.")
    else:
        # Inicializa a área corrigida só se ainda não existir
        if "texto_pos_processado_area" not in st.session_state:
            st.session_state["texto_pos_processado_area"] = (
                st.session_state.get("texto_paragrafado", "").strip()
                or st.session_state.get("texto_transcrito", "").strip()
            )

        col_bruto, col_corr = st.columns(2)

        with col_bruto:
            st.markdown("#### 🎧 Texto bruto (saída direta do modelo)")
            st.text_area(
                "Texto bruto",
                value=st.session_state.get("texto_transcrito", ""),
                height=420,
                key="texto_bruto_view",
                disabled=True
            )

        with col_corr:
            st.markdown("#### ✨ Texto corrigido e revisado")
            st.text_area(
                "Texto corrigido",
                height=420,
                key="texto_pos_processado_area"
            )

        bcol1, bcol2, bcol3 = st.columns([1.2, 1, 1])

        with bcol1:
            st.button(
                "⚙️ Aplicar correções",
                use_container_width=True,
                type="primary",
                on_click=_pp_aplicar_correcoes
            )
        with bcol2:
            st.button(
                "↩️ Voltar pro texto da transcrição",
                use_container_width=True,
                on_click=_pp_resetar
            )
        with bcol3:
            st.button(
                "🧹 Limpar",
                use_container_width=True,
                on_click=_pp_limpar
            )

        if st.session_state.get("pp_msg_ok"):
            st.success(st.session_state["pp_msg_ok"])
            # opcional: apaga a msg depois de mostrar uma vez
            # del st.session_state["pp_msg_ok"]

        st.markdown("### 📥 Download do Texto Corrigido")
        st.download_button(
            "📄 Baixar texto corrigido",
            data=st.session_state.get("texto_pos_processado_area", ""),
            file_name=f"texto_corrigido_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
            key="download_pos_processado"
        )


# =============================
# Aba 4 – Histórico
# =============================
with tab4:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>📊 Histórico de Transcrições</h2>
        <p style="color: #666;">Veja as últimas transcrições realizadas e recarregue para editar</p>
    </div>
    """, unsafe_allow_html=True)

    historico = st.session_state.get("historico_transcricoes", [])

    if not historico:
        st.info("Ainda não há itens no histórico. Faça uma transcrição para começar.")
    else:
        df_hist = pd.DataFrame(historico)
        df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"], errors="coerce")
        df_hist["Quando"] = df_hist["timestamp"].dt.strftime("%d/%m/%Y %H:%M")

        st.markdown("### 📋 Lista de transcrições")
        st.dataframe(
            df_hist[["Quando", "arquivo", "modelo", "palavras", "duracao_min", "tempo_proc"]],
            use_container_width=True,
            hide_index=True
        )

        opcoes = [
            f"{i+1} • {item['arquivo']} • {pd.to_datetime(item['timestamp']).strftime('%d/%m/%Y %H:%M')}"
            for i, item in enumerate(historico)
        ]
        escolha = st.selectbox("Selecione uma transcrição para carregar", opcoes)

        idx_escolhido = opcoes.index(escolha)
        item_sel = historico[idx_escolhido]

        st.markdown("### 🔍 Prévia")
        st.markdown(f"""
        <div class="text-preview">
            {item_sel['preview'].replace("\\n\\n", "<br><br>")}
        </div>
        """, unsafe_allow_html=True)

        if st.button("📥 Carregar no pós-processamento", use_container_width=True):
            st.session_state["texto_transcrito"] = item_sel["preview"]
            st.session_state["texto_paragrafado"] = item_sel["preview"]
            st.session_state["texto_pos_processado"] = item_sel["preview"]
            st.success("Texto carregado. Vá na aba de pós-processamento para editar.")


# Fecha container principal
st.markdown("</div>", unsafe_allow_html=True)

# Botão topo
st.markdown("""
<div class="top-btn-container">
    <a href="#top" class="top-btn">↑</a>
</div>
""", unsafe_allow_html=True)

# Rodapé
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1.5rem;">
    <p style="font-size: 1.1rem; font-weight: 700;">Transcrição Inteligente v4.3</p>
    <p style="color: #999; font-size: 0.9rem;">
        Whisper OpenAI • Processamento em tempo real • Correções automáticas • Interface moderna
    </p>
    <p style="color: #aaa; font-size: 0.8rem; margin-top: 1rem;">
        © 2024 • Para uso profissional • Desenvolvido com Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
