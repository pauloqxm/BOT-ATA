import os
import time
import warnings
import tempfile
import json
from pathlib import Path
from datetime import datetime
import re
import subprocess  # NOVO: para detectar placa de vídeo via sistema

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
warnings.filterwarnings("ignore", message=".*huggingface_hub.*")

import torch
import psutil
import platform

# Ajuste de threads para não brigar com o Streamlit
num_threads = os.cpu_count() or 4
try:
    torch.set_num_threads(num_threads)
except RuntimeError:
    pass
os.environ["OMP_NUM_THREADS"] = str(num_threads)

import librosa
import streamlit as st
import pandas as pd

# Whisper oficial
import whisper

# =============================
# Configuração Streamlit com tema moderno
# =============================
st.set_page_config(
    page_title="Transcrição ATA – Whisper oficial",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Anchor para o botão "voltar ao topo"
st.markdown('<a id="top"></a>', unsafe_allow_html=True)

# CSS personalizado para interface moderna
st.markdown("""
<style>
    /* Tema principal */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem auto;
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
    
    /* Botões secundários */
    .secondary-btn {
        background: linear-gradient(135deg, #6c757d 0%, #495057 100%) !important;
    }
    
    .success-btn {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
    }
    
    .warning-btn {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%) !important;
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
    
    /* Métricas estilizadas */
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
    
    /* Progress bar moderna */
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
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
    }
    
    /* Tabs estilizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: transparent;
        border-bottom: 2px solid #f0f0f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px 10px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: #666;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
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
    
    /* Editor de texto */
    .text-editor {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
        font-family: 'Arial', sans-serif;
        line-height: 1.8;
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    .text-editor:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
    }
    
    /* Parágrafos */
    .paragraph {
        margin-bottom: 1.5rem;
            padding: 1rem;
        border-left: 4px solid #28a745;
        background: linear-gradient(135deg, #f8fff9 0%, #f0fdf4 100%);
        border-radius: 8px;
    }
    
    /* Status indicators */
    .status-processing {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    
    .status-success {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    /* Header */
    .page-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .page-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .page-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Form inputs */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
    }
    
    /* Download buttons */
    .download-btn {
        background: linear-gradient(135deg, #20c997 0%, #12b886 100%) !important;
        margin: 0.5rem 0;
    }
    
    .download-btn:hover {
        background: linear-gradient(135deg, #12b886 0%, #0ca678 100%) !important;
    }
    
    /* Toggle buttons */
    .stCheckbox > div {
        padding: 0.5rem;
        border-radius: 10px;
        background: #f8f9fa;
    }
    
    /* Slider estilizado */
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Botão voltar ao início */
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
    
    .main-content {
        margin-bottom: 80px;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# Cabeçalho moderno
# =============================
st.markdown("""
<div class="page-header">
    <h1>🎙️ Transcrição Inteligente</h1>
    <p>Whisper OpenAI • Processamento Inteligente • Correções Automáticas</p>
</div>
""", unsafe_allow_html=True)

# Container principal
st.markdown('<div class="main-container main-content">', unsafe_allow_html=True)

# =============================
# Arquivo de correções personalizadas
# =============================
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path(".")
CORRECOES_FILE = BASE_DIR / "correcoes_custom.json"

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
    frases = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+', texto)
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
    frases = re.split(r'(?<=[.!?])\s+', texto)
    frases_capitalizadas = []
    for frase in frases:
        if frase:
            frase = frase.strip()
            if frase:
                frase = frase[0].upper() + frase[1:]
                frases_capitalizadas.append(frase)
    return ' '.join(frases_capitalizadas)

def corrigir_pontuacao(texto: str) -> str:
    if not texto:
        return ""
    texto = re.sub(r'\s+([.,!?:;])', r'\1', texto)
    texto = re.sub(r'([.,!?:;])(?!\s|$)', r'\1 ', texto)
    texto = re.sub(r'([.,!?:;]){2,}', r'\1', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

def formatar_ata(texto: str) -> str:
    # Mantida para uso futuro, mas não exposta na UI de pós-processamento
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
        texto = ts['text']
        if len(texto) > max_chars:
            texto = texto[:max_chars] + "..."
        inicio = formatar_tempo(ts['start'])
        fim = formatar_tempo(ts['end'])
        linhas.append(f"<div class='timestamp-item'><b>[{inicio} - {fim}]</b> {texto}</div>")
    return "\n".join(linhas)

# =============================
# Detecção de NPU / GPU / Placa de vídeo
# =============================
def detectar_npu(cpu_name: str):
    """
    Detecção simples de NPU baseada no nome do processador.
    Não é perfeita, mas já indica se é uma linha com NPU dedicada.
    """
    if not cpu_name:
        return False, "Não identificado"

    cpu_lower = cpu_name.lower()
    tem_npu = False
    descricao = "Não identificado"

    # Intel Core Ultra (geralmente vem com NPU)
    if "core ultra" in cpu_lower or "ultra 5" in cpu_lower or "ultra 7" in cpu_lower or "ultra 9" in cpu_lower:
        tem_npu = True
        descricao = "Intel NPU (linha Core Ultra)"

    # Qualcomm / ARM com NPU integrada
    elif "snapdragon" in cpu_lower or "qualcomm" in cpu_lower:
        tem_npu = True
        descricao = "NPU integrada (SoC Qualcomm)"

    return tem_npu, descricao

def detectar_gpu_e_placa_video():
    """
    Detecta GPU CUDA e placas de vídeo usando:
    1. PyTorch CUDA
    2. WMIC (Windows)
    3. PowerShell (fallback para Windows)
    4. lspci (Linux)
    """

    gpu_cuda = None
    placas_video = []

    # ---- 1. Tenta detectar GPU CUDA via PyTorch ----
    if torch.cuda.is_available():
        try:
            gpu_cuda = torch.cuda.get_device_name(0)
        except Exception:
            gpu_cuda = "GPU CUDA detectada"

    # ---- 2. Detecção via Windows (WMIC) ----
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
            linhas = [
                l.strip()
                for l in result.stdout.splitlines()
                if l.strip() and "Name" not in l
            ]
            if linhas:
                placas_video.extend(linhas)

            # ---- 3. Fallback via PowerShell ----
            if not placas_video:
                result_ps = subprocess.run(
                    ["powershell", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                    capture_output=True,
                    text=True,
                    creationflags=creationflags
                )
                ps_lines = [
                    l.strip()
                    for l in result_ps.stdout.splitlines()
                    if l.strip()
                ]
                if ps_lines:
                    placas_video.extend(ps_lines)

    except Exception:
        pass

    # Garantir mensagem mais amigável
    if not placas_video:
        placas_video = ["Nenhuma placa identificada"]

    return gpu_cuda, placas_video

# =============================
# Whisper oficial
# =============================
@st.cache_resource(show_spinner=True)
def carregar_modelo_whisper(nome_modelo: str, device: str):
    return whisper.load_model(nome_modelo, device=device)

def transcrever_com_whisper(audio, sr, modelo_nome: str, chunk_seg: int):
    if torch.cuda.is_available():
        device = "cuda"
        fp16 = True
        device_msg = f"🎮 GPU NVIDIA detectada: {torch.cuda.get_device_name(0)}"
    else:
        device = "cpu"
        fp16 = False
        device_msg = "💻 Usando CPU - Processamento pode ser mais lento"

    st.markdown(f"""
    <div class="info-card">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="font-size: 2rem;">⚙️</div>
            <div>
                <h4 style="margin: 0;">Configuração do Sistema</h4>
                <p style="margin: 0;">{device_msg}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    duracao_min = len(audio) / sr / 60
    modelo_efetivo = modelo_nome
    
    st.markdown(f"""
    <div class="custom-card">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h3 style="margin: 0;">🎯 Modelo Selecionado</h3>
                <p style="margin: 0; color: #666;">{modelo_efetivo.upper()} em {device.upper()}</p>
            </div>
            <div class="status-processing">
                PRONTO PARA PROCESSAR
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner(f"🔧 Carregando modelo Whisper {modelo_efetivo}..."):
        model = carregar_modelo_whisper(modelo_efetivo, device)

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
        
        st.markdown(f"""
        <div class="custom-card">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h4 style="margin: 0;">📝 Parte {idx}/{total_partes}</h4>
                    <p style="margin: 0; color: #666;">
                        Janela: {janela_min:.1f}min - {janela_max:.1f}min
                    </p>
                </div>
                <div class="status-processing">
                    PROCESSANDO...
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        inicio_parte = time.time()
        result = model.transcribe(
            parte,
            language="pt",
            task="transcribe",
            temperature=[0.0, 0.2],
            best_of=5,
            initial_prompt=BASE_PROMPT,
            fp16=fp16,
        )
        tempo_parte = time.time() - inicio_parte
        tempos_partes.append(tempo_parte)

        segs = result.get("segments", [])
        if segs:
            for seg in segs:
                texto = seg["text"]
                start = float(seg["start"]) + t_ini
                end = float(seg["end"]) + t_ini
                timestamps.append({"start": start, "end": end, "text": texto})
                texto_final += texto + " "

            st.markdown(f"""
            <div class="success-card">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <h5 style="margin: 0; color: #155724;">✅ Parte {idx} concluída</h5>
                        <p style="margin: 0; color: #0c5460;">
                            Tempo: {tempo_parte:.1f}s | 
                            Trecho: {segs[0]['text'][:100]}...
                        </p>
                    </div>
                    <div class="status-success">
                        CONCLUÍDO
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-card">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <h5 style="margin: 0; color: #856404;">⚠️ Sem áudio detectado</h5>
                        <p style="margin: 0; color: #856404;">
                            Parte {idx} não contém áudio transcritível
                        </p>
                    </div>
                    <div class="status-warning">
                        SEM ÁUDIO
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        progresso = idx / total_partes
        progress_bar.progress(progresso)
        percent_text.markdown(f"**{progresso*100:.0f}%**")

    tempo_total = time.time() - inicio_geral
    return texto_final, timestamps, tempo_total, duracao_min, total_partes, tempos_partes

# =============================
# Sidebar – configurações modernas
# =============================
with st.sidebar:
    st.markdown("""
    <div style="padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; border-radius: 15px; margin-bottom: 2rem;">
        <h3 style="margin: 0;">⚙️ Configurações</h3>
        <p style="margin: 0; opacity: 0.9;">Ajuste os parâmetros de processamento</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Modelo Whisper")
    modelos = {
        "🧠 tiny – velocidade máxima": "tiny",
        "⚡ base – equilíbrio ideal": "base",
        "🎯 small – precisão superior": "small",
        "🏆 medium – qualidade premium": "medium",
        "👑 large-v3 – excelência máxima": "large-v3",
    }
    modelo_label = st.selectbox(
        "Selecione o modelo:",
        list(modelos.keys()),
        index=1
    )
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
        if not cpu_info or cpu_info == "":
            cpu_info = "Processador não identificado"
    except:
        cpu_info = "Processador não identificado"
    
    ram_total = psutil.virtual_memory().total / (1024**3)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Threads", num_threads)
        st.metric("RAM Total", f"{ram_total:.1f} GB")
    with col2:
        st.metric("PyTorch", torch.__version__[:6])
        st.metric("Sistema", platform.system())
    
    with st.expander("📋 Detalhes do Sistema"):
        st.write(f"**Processador:** {cpu_info}")
        st.write(f"**Arquitetura:** {platform.machine()}")
        st.write(f"**Python:** {platform.python_version()}")
        st.write(f"**Whisper:** {whisper.__version__ if hasattr(whisper, '__version__') else 'N/A'}")

        # Memória
        mem = psutil.virtual_memory()
        st.write(f"**RAM Usada:** {mem.percent}%")
        st.write(f"**RAM Disponível:** {mem.available / (1024**3):.1f} GB")

        # NPU
        tem_npu, desc_npu = detectar_npu(cpu_info)
        if tem_npu:
            st.write(f"**NPU:** {desc_npu}")
        else:
            st.write("**NPU:** não detectada")

        # GPU / CUDA e Placa de vídeo
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

        if placas_video:
            st.markdown("**Placa(s) de vídeo detectada(s):**")
            for nome in placas_video:
                st.write(f"• {nome}")
        else:
            st.write("**Placa de vídeo:** não identificada pelo sistema operacional")

# =============================
# Abas principais estilizadas - ORDEM AJUSTADA
# =============================
tab1, tab2, tab3 = st.tabs([
    "🎧 TRANSCREVER ÁUDIO",
    "📝 PÓS-PROCESSAMENTO",
    "📚 BIBLIOTECA DE CORREÇÕES"
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
                with st.spinner("🔧 Preparando áudio para processamento..."):
                    audio, sr_original = librosa.load(caminho_audio, sr=None, mono=True)

                    max_abs = max(1e-8, float(abs(audio).max()))
                    audio = audio / max_abs * 0.9

                    if sr_original != 16000:
                        audio = librosa.resample(
                            audio, orig_sr=sr_original, target_sr=16000
                        )
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
                ) = transcrever_com_whisper(
                    audio, sr, modelo_whisper, chunk_segundos
                )

                # Pós-processamento automático + parágrafos
                texto_pos = pos_processar_texto(texto_bruto)
                texto_corrigido = corrigir_pontuacao(capitalizar_frases(texto_pos))
                texto_paragrafado = organizar_paragrafos(texto_corrigido)

                st.session_state["texto_transcrito"] = texto_bruto.strip()
                st.session_state["texto_paragrafado"] = texto_paragrafado
                st.session_state["texto_pos_processado"] = texto_paragrafado

                if not texto_corrigido.strip():
                    st.error("❌ Nenhum texto final gerado. Verifique se o áudio tem fala clara.")
                else:
                    st.markdown("""
                    <div class="success-card" style="padding: 2rem;">
                        <div style="text-align: center;">
                            <h2 style="margin: 0; color: #155724;">🎉 Transcrição Concluída!</h2>
                            <p style="margin: 0; color: #0c5460;">Processamento finalizado com sucesso</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### 📈 Estatísticas de Processamento")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Duração Áudio</div>
                            <div class="metric-value">{formatar_tempo(duracao_min * 60)}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Tempo Process.</div>
                            <div class="metric-value">{formatar_tempo(tempo_proc)}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        velocidade_x = (duracao_min * 60) / tempo_proc if tempo_proc > 0 else 0
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Velocidade</div>
                            <div class="metric-value">{velocidade_x:.1f}x</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col4:
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
                        {preview_texto.replace("\n\n", "<br><br>")}
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
                        st.info("ℹ️ Nenhum timestamp disponível")
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
# Aba 2 – Pós-processamento
# =============================
with tab2:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>📝 Pós-processamento do Texto</h2>
        <p style="color: #666;">À esquerda o texto bruto. À direita o texto corrigido.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state["texto_transcrito"].strip():
        st.info("ℹ️ Ainda não há transcrição disponível. Faça uma transcrição na aba anterior.")
    else:
        # Inicializar texto corrigido se estiver vazio
        if not st.session_state["texto_pos_processado"].strip():
            st.session_state["texto_pos_processado"] = (
                st.session_state["texto_paragrafado"] or st.session_state["texto_transcrito"]
            )

        col_bruto, col_corr = st.columns(2)

        with col_bruto:
            st.markdown("#### 🎧 Texto bruto (saída direta do modelo)")
            st.text_area(
                "Texto bruto",
                value=st.session_state["texto_transcrito"],
                height=400,
                key="texto_bruto_view",
                disabled=True
            )

        with col_corr:
            st.markdown("#### ✨ Texto corrigido / revisado")
            texto_atual = st.text_area(
                "Texto corrigido",
                value=st.session_state["texto_pos_processado"],
                height=400,
                key="texto_pos_processado_area"
            )
            st.session_state["texto_pos_processado"] = texto_atual

        bcol1, bcol2 = st.columns(2)
        with bcol1:
            aplicar_corr = st.button(
                "⚙️ Aplicar biblioteca de correções no texto corrigido",
                use_container_width=True
            )
        with bcol2:
            limpar_btn = st.button(
                "🧹 Limpar texto corrigido",
                use_container_width=True
            )

        if aplicar_corr:
            texto_corr = pos_processar_texto(st.session_state["texto_pos_processado"])
            texto_corr = corrigir_pontuacao(capitalizar_frases(texto_corr))
            texto_corr = organizar_paragrafos(texto_corr)
            st.session_state["texto_pos_processado"] = texto_corr
            st.success("✅ Biblioteca de correções aplicada ao texto corrigido.")
            st.experimental_rerun()

        if limpar_btn:
            st.session_state["texto_pos_processado"] = ""
            st.experimental_rerun()

        st.markdown("### 📥 Download do Texto Corrigido")
        st.download_button(
            "📄 Baixar texto corrigido",
            data=st.session_state["texto_pos_processado"],
            file_name=f"texto_corrigido_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
            key="download_pos_processado"
        )

# =============================
# Aba 3 – Biblioteca de correções
# =============================
with tab3:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>📚 Biblioteca de Correções</h2>
        <p style="color: #666;">Gerencie as substituições automáticas aplicadas nas transcrições</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📋 Correções Ativas")
    dicionario_atual = get_correcoes_dicionario()
    
    if dicionario_atual:
        df_correcoes = pd.DataFrame([
            {"Original": k, "Substituir por": v}
            for k, v in dicionario_atual.items()
        ])
        
        st.dataframe(
            df_correcoes,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Original": st.column_config.TextColumn(
                    "Palavra Original",
                    help="Termo que será substituído"
                ),
                "Substituir por": st.column_config.TextColumn(
                    "Substituição",
                    help="Termo que substituirá o original"
                )
            }
        )
        
        st.markdown(f"""
        <div class="info-card">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h4 style="margin: 0;">📊 Resumo</h4>
                    <p style="margin: 0;">{len(dicionario_atual)} correções ativas</p>
                </div>
                <div class="status-success">
                    ATIVO
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-card">
            <div style="text-align: center; padding: 2rem;">
                <div style="font-size: 3rem;">📝</div>
                <h4>Nenhuma correção cadastrada</h4>
                <p>Adicione sua primeira correção abaixo</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### ➕ Adicionar Novas Correções")
    
    with st.form("form_add_correcoes"):
        st.markdown("""
        <div class="custom-card">
            <h4>Adicionar Múltiplas Regras de Correção</h4>
            <p style="color: #666; font-size: 0.9rem;">Preencha quantos campos desejar. Os campos vazios serão ignorados.</p>
        """, unsafe_allow_html=True)
        
        correcoes_inputs = []
        for i in range(8):
            col_orig, col_sub = st.columns([1, 1])
            with col_orig:
                original = st.text_input(
                    f"Original {i+1}",
                    placeholder="Ex: vc, tb, d+, etc.",
                    key=f"original_input_{i}"
                )
            with col_sub:
                substituir = st.text_input(
                    f"Substituir por {i+1}",
                    placeholder="Ex: você, também, muito, etc.",
                    key=f"substituir_input_{i}"
                )
            correcoes_inputs.append((original, substituir))
        
        submit_col1, submit_col2, submit_col3 = st.columns([2, 1, 1])
        with submit_col1:
            submitted = st.form_submit_button(
                "➕ Adicionar Todas as Correções",
                use_container_width=True,
                type="primary"
            )
        with submit_col2:
            add_selected = st.form_submit_button(
                "📝 Adicionar Selecionadas",
                use_container_width=True
            )
        with submit_col3:
            clear_all = st.form_submit_button(
                "🧹 Limpar Tudo",
                use_container_width=True,
                type="secondary"
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if submitted or add_selected:
            correcoes_adicionadas = []
            for original, substituir in correcoes_inputs:
                if original.strip() and substituir.strip():
                    chave = original.strip()
                    valor = substituir.strip()
                    st.session_state["correcoes_custom"][chave] = valor
                    correcoes_adicionadas.append(f"**'{chave}'** → **'{valor}'**")
            
            if correcoes_adicionadas:
                salvar_correcoes_custom(st.session_state["correcoes_custom"])
                if submitted:
                    st.success(f"✅ {len(correcoes_adicionadas)} correções adicionadas:")
                else:
                    st.success(f"✅ {len(correcoes_adicionadas)} correções selecionadas adicionadas:")
                for corr in correcoes_adicionadas:
                    st.markdown(f"- {corr}")
                st.experimental_rerun()
            else:
                st.warning("⚠️ Nenhuma correção válida para adicionar. Preencha pelo menos um par de campos.")
        
        if clear_all:
            st.session_state["correcoes_custom"] = {}
            salvar_correcoes_custom(st.session_state["correcoes_custom"])
            st.success("✅ Todas as correções personalizadas foram removidas")
            st.experimental_rerun()

# Fechar container principal
st.markdown('</div>', unsafe_allow_html=True)

# =============================
# Botão para voltar ao início - VERSÃO SIMPLES
# =============================
st.markdown("""
<div class="top-btn-container">
    <a href="#top" class="top-btn">↑</a>
</div>
""", unsafe_allow_html=True)

# Rodapé
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1.5rem;">
    <p style="font-size: 1.1rem; font-weight: 600;">🎯 Transcrição Inteligente - v4.2</p>
    <p style="color: #999; font-size: 0.9rem;">
        Whisper OpenAI • Processamento em tempo real • Correções automáticas • Interface moderna
    </p>
    <p style="color: #aaa; font-size: 0.8rem; margin-top: 1rem;">
        © 2024 • Para uso profissional • Desenvolvido com Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
