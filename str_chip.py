import os
import time
import warnings
import tempfile
import json
from pathlib import Path
from datetime import datetime

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
warnings.filterwarnings("ignore", message=".*huggingface_hub.*")

import torch

# Ajuste de threads para não brigar com o Streamlit
num_threads = os.cpu_count() or 4
try:
    torch.set_num_threads(num_threads)
except RuntimeError:
    pass
os.environ["OMP_NUM_THREADS"] = str(num_threads)

import librosa
import soundfile as sf
import streamlit as st
import pandas as pd
import numpy as np

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
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# =============================
# Arquivo de correções personalizadas
# =============================
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path(".")
CORRECOES_FILE = BASE_DIR / "correcoes_custom.json"

def carregar_correcoes_custom():
    """Carrega as correções personalizadas do arquivo JSON."""
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
    """Salva as correções personalizadas em arquivo JSON."""
    try:
        with open(CORRECOES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Erro ao salvar correções. {e}")

# =============================
# Estado da biblioteca de correções
# =============================
if "correcoes_custom" not in st.session_state:
    st.session_state["correcoes_custom"] = carregar_correcoes_custom()

# =============================
# Utilitários gerais
# =============================
BASE_PROMPT = (
    "Transcrição em português brasileiro formal, com pontuação correta, "
    "acentuação adequada e frases completas. Use nomes próprios, siglas e "
    "termos técnicos conforme aparecem no áudio. Evite inventar trechos."
)

def get_correcoes_dicionario():
    """Dicionário base somado às correções customizadas."""
    correcoes_base = {
        " pq ": " porque ",
        " tb ": " também ",
        " vc ": " você ",
        " d ": " de ",
        " q ": " que ",
        " ta ": " está ",
        " tava ": " estava ",
        " pra ": " para ",
        " ne ": " não é ",
        " naum ": " não ",
        " entao ": " então ",
        " tbm ": " também ",
        " obg ": " obrigado ",
        " vlw ": " valeu ",
        " blz ": " beleza ",
        " p ": " para ",
        " cm ": " com ",
        " td ": " tudo ",
        " qd ": " quando ",
        " qq ": " qualquer ",
    }
    correcoes_custom = st.session_state.get("correcoes_custom", {})
    correcoes = {}
    correcoes.update(correcoes_base)
    correcoes.update(correcoes_custom)
    return correcoes

def pos_processar_texto(texto: str) -> str:
    """Aplica a biblioteca de correções ao texto transcrito."""
    if not texto:
        return ""

    correcoes = get_correcoes_dicionario()

    texto = " " + texto + " "
    for errado, correto in correcoes.items():
        texto = texto.replace(errado, correto)

    while "  " in texto:
        texto = texto.replace("  ", " ")

    texto = (
        texto.replace(" .", ".")
        .replace(" ,", ",")
        .replace(" ?", "?")
        .replace(" !", "!")
    )

    return texto.strip()

def dividir_em_chunks(audio, sr, chunk_seg=120):
    partes = []
    tam = int(chunk_seg * sr)
    total = len(audio)
    for i in range(0, total, tam):
        parte = audio[i : i + tam]
        t_ini = i / sr
        t_fim = (i + len(parte)) / sr
        partes.append((parte, t_ini, t_fim))
    return partes

def formatar_tempo(segundos: float) -> str:
    """Converte segundos para formato MM:SS"""
    minutos = int(segundos // 60)
    seg = int(segundos % 60)
    return f"{minutos:02d}:{seg:02d}"

def formatar_timestamps(timestamps, max_chars=400):
    """Formata timestamps com limite de caracteres por item"""
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

    with st.spinner(f"📦 Carregando modelo Whisper {modelo_efetivo}..."):
        model = carregar_modelo_whisper(modelo_efetivo, device)

    partes = dividir_em_chunks(audio, sr, chunk_seg)
    total_partes = len(partes)
    
    # Contador de partes
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

    # Container de progresso
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
        
        # Status da parte atual
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

        # Atualizar progresso com percentual
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
    
    # Modelo Whisper
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
    
    # Configuração de chunks
    st.markdown("### 📊 Tamanho das Partes")
    chunk_segundos = st.slider(
        "Duração (segundos):",
        min_value=30,
        max_value=300,
        value=120,
        step=30,
        help="Partes menores = mais preciso\nPartes maiores = mais rápido"
    )
    
    # Info do sistema
    st.markdown("---")
    st.markdown("### 💻 Sistema")
    sys_col1, sys_col2 = st.columns(2)
    with sys_col1:
        st.metric("Threads", num_threads)
    with sys_col2:
        st.metric("PyTorch", torch.__version__[:6])

# =============================
# Abas principais estilizadas
# =============================
tab1, tab2 = st.tabs(["🎧 TRANSCREVER ÁUDIO", "📚 BIBLIOTECA DE CORREÇÕES"])

# =============================
# Aba 1 – Transcrição
# =============================
with tab1:
    # Uploader moderno
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>🎤 Envie seu Áudio</h2>
        <p style="color: #666;">Suporta MP3, WAV, M4A, OGG, FLAC, AAC, WMA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Área de upload
    audio_file = st.file_uploader(
        "Arraste e solte ou clique para selecionar",
        type=["mp3", "wav", "m4a", "ogg", "flac", "aac", "wma"],
        label_visibility="collapsed"
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

    # Botão de transcrição
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        transcribe_clicked = st.button(
            "🚀 INICIAR TRANSCRIÇÃO",
            disabled=(audio_file is None),
            use_container_width=True,
            type="primary"
        )

    if transcribe_clicked:
        if audio_file is None:
            st.error("⚠️ Por favor, envie um arquivo de áudio primeiro.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=audio_file.name) as tmp:
                tmp.write(audio_file.read())
                caminho_audio = tmp.name

            try:
                # Pré-processamento
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

                # Métricas iniciais
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

                # Processamento principal
                (
                    texto,
                    ts,
                    tempo_proc,
                    duracao_min,
                    total_partes,
                    tempos_partes,
                ) = transcrever_com_whisper(
                    audio, sr, modelo_whisper, chunk_segundos
                )

                texto = pos_processar_texto(texto)

                if not texto.strip():
                    st.error("❌ Nenhum texto final gerado. Verifique se o áudio tem fala clara.")
                else:
                    # Resultados
                    st.markdown("""
                    <div class="success-card" style="padding: 2rem;">
                        <div style="text-align: center;">
                            <h2 style="margin: 0; color: #155724;">🎉 Transcrição Concluída!</h2>
                            <p style="margin: 0; color: #0c5460;">Processamento finalizado com sucesso</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Métricas finais
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
                        palavras = len(texto.split())
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Palavras</div>
                            <div class="metric-value">{palavras}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Gráfico de desempenho
                    if tempos_partes:
                        st.markdown("### 📊 Desempenho por Parte")
                        df_tempos = pd.DataFrame({
                            "Parte": list(range(1, total_partes + 1)),
                            "Tempo (s)": tempos_partes,
                        })
                        st.bar_chart(df_tempos.set_index("Parte"))

                    # Preview do texto
                    st.markdown("### 🧾 Prévia da Transcrição")
                    st.markdown(f"""
                    <div class="text-preview">
                        {texto[:400]}...
                        {f"<br><br><small><i>Total de {len(texto)} caracteres</i></small>" if len(texto) > 400 else ""}
                    </div>
                    """, unsafe_allow_html=True)

                    # Timestamps
                    st.markdown("### ⏱️ Timestamps Detalhados")
                    if ts:
                        timestamps_html = formatar_timestamps(ts)
                        st.markdown(f"""
                        <div style="max-height: 400px; overflow-y: auto; padding: 1rem;">
                            {timestamps_html}
                        </div>
                        """, unsafe_allow_html=True)
                        texto_ts = "\n".join([f"[{formatar_tempo(t['start'])} - {formatar_tempo(t['end'])}] {t['text'][:400]}" 
                                            for t in ts])
                    else:
                        st.info("ℹ️ Nenhum timestamp disponível")
                        texto_ts = ""

                    # Botões de download
                    st.markdown("### 📥 Download dos Resultados")
                    nome_base = os.path.splitext(audio_file.name)[0]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        st.download_button(
                            "📄 Baixar Transcrição Completa",
                            data=texto,
                            file_name=f"transcricao_{nome_base}_{timestamp}.txt",
                            mime="text/plain",
                            use_container_width=True,
                            key="download_transcription"
                        )
                    with dl_col2:
                        if ts:
                            st.download_button(
                                "⏱️ Baixar Timestamps",
                                data=texto_ts,
                                file_name=f"timestamps_{nome_base}_{timestamp}.txt",
                                mime="text/plain",
                                use_container_width=True,
                                key="download_timestamps"
                            )

            finally:
                try:
                    os.unlink(caminho_audio)
                except Exception:
                    pass
    else:
        if not audio_file:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%); 
                      border-radius: 15px; margin: 2rem 0;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🎧</div>
                <h3 style="color: #667eea;">Faça o upload do áudio</h3>
                <p style="color: #666;">Arraste e solte ou clique para selecionar um arquivo</p>
                <p style="color: #999; font-size: 0.9rem;">Formatos suportados: MP3, WAV, M4A, OGG, FLAC, AAC, WMA</p>
            </div>
            """, unsafe_allow_html=True)

# =============================
# Aba 2 – Biblioteca de correções
# =============================
with tab2:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>📚 Biblioteca de Correções</h2>
        <p style="color: #666;">Gerencie as substituições automáticas aplicadas nas transcrições</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Correções em uso
    st.markdown("### 📋 Correções Ativas")
    dicionario_atual = get_correcoes_dicionario()
    
    if dicionario_atual:
        # Criar DataFrame para visualização
        df_correcoes = pd.DataFrame([
            {"Original": k.strip(), "Substituir por": v.strip()}
            for k, v in dicionario_atual.items()
        ])
        
        # Estilizar a tabela
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
        
        # Resumo
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
    
    # Adicionar nova correção
    st.markdown("### ➕ Adicionar Nova Correção")
    
    with st.form("form_add_correcao"):
        st.markdown("""
        <div class="custom-card">
            <h4>Nova Regra de Correção</h4>
        """, unsafe_allow_html=True)
        
        col_orig, col_arrow, col_sub = st.columns([5, 1, 5])
        with col_orig:
            original = st.text_input(
                "Palavra/Expressão original:",
                placeholder="Ex: vc, tb, d+, etc.",
                key="original_input"
            )
        with col_arrow:
            st.markdown("<div style='text-align: center; font-size: 2rem; margin-top: 1.5rem;'>→</div>", unsafe_allow_html=True)
        with col_sub:
            substituir = st.text_input(
                "Substituir por:",
                placeholder="Ex: você, também, muito, etc.",
                key="substituir_input"
            )
        
        submit_col1, submit_col2 = st.columns(2)
        with submit_col1:
            submitted = st.form_submit_button(
                "➕ Adicionar Correção",
                use_container_width=True,
                type="primary"
            )
        with submit_col2:
            clear_all = st.form_submit_button(
                "🧹 Limpar Tudo",
                use_container_width=True,
                type="secondary"
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if submitted:
            if not original.strip() or not substituir.strip():
                st.error("❌ Preencha ambos os campos antes de adicionar.")
            else:
                chave = f" {original.strip()} "
                valor = f" {substituir.strip()} "
                st.session_state["correcoes_custom"][chave] = valor
                salvar_correcoes_custom(st.session_state["correcoes_custom"])
                st.success(f"✅ Correção adicionada: **'{original.strip()}'** → **'{substituir.strip()}'**")
                st.rerun()
        
        if clear_all:
            st.session_state["correcoes_custom"] = {}
            salvar_correcoes_custom(st.session_state["correcoes_custom"])
            st.success("✅ Todas as correções personalizadas foram removidas")
            st.rerun()

# Fechar container principal
st.markdown('</div>', unsafe_allow_html=True)

# Rodapé
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1.5rem;">
    <p style="font-size: 1.1rem; font-weight: 600;">🎯 Transcrição Inteligente</p>
    <p style="color: #999; font-size: 0.9rem;">
        Whisper OpenAI • v2.0 • Processamento em tempo real • Correções automáticas • Interface moderna
    </p>
    <p style="color: #aaa; font-size: 0.8rem; margin-top: 1rem;">
        © 2024 • Para uso profissional • Desenvolvido com Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
