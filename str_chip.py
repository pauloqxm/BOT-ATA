import os
import time
import warnings
import tempfile
import json
from pathlib import Path
from datetime import datetime
import re

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
# Estado da aplicação
# =============================
if "correcoes_custom" not in st.session_state:
    st.session_state["correcoes_custom"] = carregar_correcoes_custom()

if "texto_transcrito" not in st.session_state:
    st.session_state["texto_transcrito"] = ""

if "texto_editado" not in st.session_state:
    st.session_state["texto_editado"] = ""

# =============================
# Utilitários gerais
# =============================
BASE_PROMPT = (
    "Transcrição em português brasileiro formal, com pontuação correta, "
    "acentuação adequada e frases completas. Use nomes próprios, siglas e "
    "termos técnicos conforme aparecem no áudio. Evite inventar trechos."
)

def get_correcoes_dicionario():
    """Dicionário base somado às correções customizadas (sem espaços extras)."""
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
    # Normaliza customizadas (remove espaços ao redor)
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
    """Aplica a biblioteca de correções ao texto transcrito (case-insensitive e com borda de palavra)."""
    if not texto:
        return ""

    correcoes = get_correcoes_dicionario()

    # Normaliza espaços múltiplos
    texto = re.sub(r"\s+", " ", texto)

    # Aplica cada correção com \b e ignore case
    for errado, correto in correcoes.items():
        padrao = r"\b{}\b".format(re.escape(errado))
        texto = re.sub(padrao, correto, texto, flags=re.IGNORECASE)

    # Ajusta espaço antes de pontuação
    texto = re.sub(r"\s+([.,!?])", r"\1", texto)

    return texto.strip()

def organizar_paragrafos(texto: str, max_caracteres=500) -> str:
    """Divide o texto em parágrafos com base na pontuação e limite de caracteres."""
    if not texto:
        return ""
    
    # Divide por pontos finais que não são abreviações
    frases = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+', texto)
    
    paragrafos = []
    paragrafo_atual = ""
    
    for frase in frases:
        if not frase.strip():
            continue
            
        # Se adicionar esta frase exceder o limite, fecha o parágrafo atual
        if len(paragrafo_atual) + len(frase) > max_caracteres and paragrafo_atual:
            paragrafos.append(paragrafo_atual.strip())
            paragrafo_atual = ""
        
        paragrafo_atual += frase + " "
    
    # Adiciona o último parágrafo
    if paragrafo_atual:
        paragrafos.append(paragrafo_atual.strip())
    
    # Junta parágrafos com quebra dupla de linha
    return "\n\n".join(paragrafos)

def capitalizar_frases(texto: str) -> str:
    """Capitaliza a primeira letra de cada frase."""
    if not texto:
        return ""
    
    # Divide o texto em frases
    frases = re.split(r'(?<=[.!?])\s+', texto)
    
    # Capitaliza cada frase
    frases_capitalizadas = []
    for frase in frases:
        if frase:
            frase = frase.strip()
            if frase:
                # Capitaliza primeira letra
                frase = frase[0].upper() + frase[1:]
                frases_capitalizadas.append(frase)
    
    return ' '.join(frases_capitalizadas)

def corrigir_pontuacao(texto: str) -> str:
    """Corrige problemas comuns de pontuação."""
    if not texto:
        return ""
    
    # Remove espaços antes de pontuação
    texto = re.sub(r'\s+([.,!?:;])', r'\1', texto)
    
    # Adiciona espaço após pontuação (exceto se for ponto final de abreviação)
    texto = re.sub(r'([.,!?:;])(?!\s|$)', r'\1 ', texto)
    
    # Corrige múltiplas pontuações
    texto = re.sub(r'([.,!?:;]){2,}', r'\1', texto)
    
    # Remove espaços duplicados
    texto = re.sub(r'\s+', ' ', texto)
    
    return texto.strip()

def formatar_ata(texto: str) -> str:
    """Formata texto para estrutura de ata formal."""
    if not texto:
        return ""
    
    # Adiciona cabeçalho se não existir
    if not texto.startswith("ATA DA REUNIÃO"):
        data_atual = datetime.now().strftime("%d/%m/%Y")
        texto = f"ATA DA REUNIÃO\nData: {data_atual}\n\n{texto}"
    
    # Adiciona rodapé se não existir
    if not "Encerramento" in texto and not "FIM DA ATA" in texto:
        texto += "\n\n---\nFIM DA ATA\n"
    
    return texto

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
# Abas principais estilizadas - AGORA COM 3 ABAS
# =============================
tab1, tab2, tab3 = st.tabs(["🎧 TRANSCREVER ÁUDIO", "📚 BIBLIOTECA DE CORREÇÕES", "✏️ EDITOR DE TEXTO"])

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
        label_visibility="collapsed",
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

    # Botão de transcrição
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

                # Aplica correções básicas
                texto = pos_processar_texto(texto)
                
                # Salva no estado da sessão
                st.session_state["texto_transcrito"] = texto
                st.session_state["texto_editado"] = texto  # Inicializa com o texto original

                if not texto.strip():
                    st.error("❌ Nenhum texto final gerado. Verifique se o áudio tem fala clara.")
                else:
                    # Resultados
                    st.markdown("""
                    <div class="success-card" style="padding: 2rem;">
                        <div style="text-align: center;">
                            <h2 style="margin: 0; color: #155724;">🎉 Transcrição Concluída!</h2>
                            <p style="margin: 0; color: #0c5460;">Processamento finalizado com sucesso</p>
                            <p style="margin: 1rem 0 0 0; font-size: 1.1rem;">
                                Acesse a aba <strong>✏️ EDITOR DE TEXTO</strong> para organizar e formatar o texto
                            </p>
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
                    preview_texto = texto[:500] + "..." if len(texto) > 500 else texto
                    st.markdown(f"""
                    <div class="text-preview">
                        {preview_texto}
                        <br><br><small><i>Total: {len(texto)} caracteres, {len(texto.split())} palavras</i></small>
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
                        texto_ts = "\n".join([
                            f"[{formatar_tempo(t['start'])} - {formatar_tempo(t['end'])}] {t['text'][:400]}" 
                            for t in ts
                        ])
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
                            "📄 Baixar Transcrição Bruta",
                            data=texto,
                            file_name=f"transcricao_bruta_{nome_base}_{timestamp}.txt",
                            mime="text/plain",
                            use_container_width=True,
                            key="download_raw_transcription"
                        )
                    with dl_col2:
                        if ts:
                            st.download_button(
                                "⏱️ Baixar Timestamps",
                                data=texto_ts,
                                file_name=f"timestamps_{nome_base}_{timestamp}.txt",
                                mime="text/plain",
                                use_container_width=True,
                                key="download_timestamps_tab1"
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
                key="original_input_tab2"
            )
        with col_arrow:
            st.markdown("<div style='text-align: center; font-size: 2rem; margin-top: 1.5rem;'>→</div>", unsafe_allow_html=True)
        with col_sub:
            substituir = st.text_input(
                "Substituir por:",
                placeholder="Ex: você, também, muito, etc.",
                key="substituir_input_tab2"
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
                chave = original.strip()
                valor = substituir.strip()
                st.session_state["correcoes_custom"][chave] = valor
                salvar_correcoes_custom(st.session_state["correcoes_custom"])
                st.success(f"✅ Correção adicionada: **'{chave}'** → **'{valor}'**")
                st.rerun()
        
        if clear_all:
            st.session_state["correcoes_custom"] = {}
            salvar_correcoes_custom(st.session_state["correcoes_custom"])
            st.success("✅ Todas as correções personalizadas foram removidas")
            st.rerun()

# =============================
# NOVA ABA 3 – Editor de Texto e Pós-Processamento COM DEEPSEEK
# =============================
with tab3:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>✏️ Editor de Texto com IA</h2>
        <p style="color: #666;">Organize, formate e refine sua transcrição com DeepSeek AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    texto_disponivel = st.session_state.get("texto_transcrito", "")
    
    if not texto_disponivel:
        st.markdown("""
        <div class="warning-card" style="text-align: center; padding: 3rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📝</div>
            <h3>Nenhuma transcrição disponível</h3>
            <p>Para usar o editor, primeiro transcreva um áudio na aba <strong>🎧 TRANSCREVER ÁUDIO</strong></p>
            <p style="color: #666; font-size: 0.9rem;">O texto transcrito aparecerá automaticamente aqui</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        texto_original = texto_disponivel

        # =============================
        # FUNÇÕES PARA API DEEPSEEK (COM SEGURANÇA)
        # =============================
        
        def obter_api_key():
            """Obtém a API key de forma segura."""
            # PRIMEIRO: Revogue a chave exposta e gere uma nova!
            # Chave exposta: sk-9a8fa1edd8cd4226a70cf968c68ee25d
            
            # Métodos seguros de obter a chave:
            try:
                # 1. Tenta do Streamlit Secrets (RECOMENDADO)
                if "DEEPSEEK_API_KEY" in st.secrets:
                    return st.secrets["DEEPSEEK_API_KEY"]
            except:
                pass
            
            try:
                # 2. Tenta de variável de ambiente
                import os
                if os.environ.get("DEEPSEEK_API_KEY"):
                    return os.environ.get("DEEPSEEK_API_KEY")
            except:
                pass
            
            # 3. Retorna None se não encontrar
            return None
        
        def validar_texto_para_api(texto: str, max_caracteres: int = 3000) -> tuple:
            """Valida e prepara texto para API, retornando (texto_valido, aviso)."""
            if not texto or not texto.strip():
                return "", "Texto vazio"
            
            # Remove caracteres problemáticos
            texto_limpo = texto.strip()
            
            # Limita tamanho para otimizar tokens
            if len(texto_limpo) > max_caracteres:
                aviso = f"Texto truncado para {max_caracteres} caracteres para otimizar tokens"
                return texto_limpo[:max_caracteres], aviso
            
            return texto_limpo, ""
        
        def corrigir_ortografia_deepseek(texto: str) -> str:
            """Corrige ortografia e gramática usando DeepSeek."""
            import requests
            import json
            import time
            
            api_key = obter_api_key()
            if not api_key:
                st.error("❌ API Key não configurada. Configure em Secrets ou variáveis de ambiente.")
                return texto
            
            # Valida e prepara texto
            texto_valido, aviso = validar_texto_para_api(texto)
            if aviso:
                st.info(f"ℹ️ {aviso}")
            
            if not texto_valido:
                return texto
            
            # Configuração da requisição
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Prompt otimizado para correção
            prompt = f"""Corrija a ortografia e gramática do seguinte texto em português brasileiro:

TEXTO PARA CORRIGIR:
{texto_valido}

REGRAS:
1. Corrija TODOS os erros de ortografia
2. Corrija concordância verbal e nominal
3. Ajuste pontuação (vírgulas, pontos, etc.)
4. Mantenha o significado original
5. Não altere nomes próprios ou termos técnicos
6. Mantenha a formalidade do texto
7. Retorne APENAS o texto corrigido, sem explicações

TEXTO CORRIGIDO:"""
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system", 
                        "content": "Você é um corretor ortográfico especializado em português brasileiro."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.1,  # Baixa para correções precisas
                "max_tokens": min(len(texto_valido) + 500, 2000),
                "top_p": 0.9
            }
            
            try:
                with st.spinner("🧠 Corrigindo com DeepSeek AI..."):
                    # Delay para respeitar rate limits
                    time.sleep(0.5)
                    
                    response = requests.post(
                        url, 
                        headers=headers, 
                        json=payload,
                        timeout=45
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        texto_corrigido = result["choices"][0]["message"]["content"].strip()
                        
                        # Log do uso (opcional)
                        if "usage" in result:
                            tokens = result["usage"]["total_tokens"]
                            st.session_state.setdefault("tokens_usados", 0)
                            st.session_state["tokens_usados"] += tokens
                        
                        return texto_corrigido
                    else:
                        error_msg = f"Erro {response.status_code}"
                        if response.text:
                            try:
                                error_data = response.json()
                                error_msg = error_data.get("error", {}).get("message", error_msg)
                            except:
                                error_msg = response.text[:200]
                        
                        st.error(f"❌ Erro na API: {error_msg}")
                        return texto
            
            except requests.exceptions.Timeout:
                st.error("⏰ Timeout na API. Texto muito longo ou servidor lento.")
                return texto
            except Exception as e:
                st.error(f"❌ Erro de conexão: {str(e)}")
                return texto
        
        def melhorar_clareza_deepseek(texto: str) -> str:
            """Melhora a clareza e fluência do texto."""
            import requests
            import json
            import time
            
            api_key = obter_api_key()
            if not api_key:
                return texto
            
            texto_valido, aviso = validar_texto_para_api(texto, 2500)
            if aviso:
                st.info(f"ℹ️ {aviso}")
            
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""Melhore a clareza e fluência deste texto em português brasileiro:

TEXTO ORIGINAL:
{texto_valido}

INSTRUÇÕES:
1. Torne as frases mais claras e diretas
2. Melhore a conexão entre ideias
3. Use vocabulário mais preciso quando necessário
4. Mantenha o tom formal apropriado para atas
5. Não altere fatos ou informações importantes
6. Organize em parágrafos lógicos quando possível
7. Retorne APENAS o texto melhorado

TEXTO MELHORADO:"""
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system", 
                        "content": "Você é um especialista em redação clara e objetiva para documentos formais."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": min(len(texto_valido) + 800, 3000)
            }
            
            try:
                with st.spinner("✨ Melhorando clareza com AI..."):
                    time.sleep(0.5)
                    response = requests.post(url, headers=headers, json=payload, timeout=60)
                    
                    if response.status_code == 200:
                        result = response.json()
                        return result["choices"][0]["message"]["content"].strip()
                    else:
                        return texto
            
            except:
                return texto
        
        def criar_resumo_ata_deepseek(texto: str) -> str:
            """Cria um resumo estruturado da ata."""
            import requests
            import json
            import time
            
            api_key = obter_api_key()
            if not api_key:
                return texto
            
            texto_valido, aviso = validar_texto_para_api(texto, 3500)
            if aviso:
                st.info(f"ℹ️ {aviso}")
            
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""Crie um resumo estruturado desta ata de reunião:

ATA COMPLETA:
{texto_valido}

FORMATO DO RESUMO:
1. DATA E HORÁRIO (extraia se mencionado)
2. PARTICIPANTES (liste se mencionados)
3. PONTOS PRINCIPAIS DISCUTIDOS (em tópicos)
4. DECISÕES TOMADAS (lista clara)
5. AÇÕES DEFINIDAS (com responsáveis se houver)
6. PRÓXIMOS PASSOS
7. PRÓXIMA REUNIÃO (se definida)

Se alguma informação não estiver no texto, indique "Não especificado".
Use formatação com marcadores (*) para melhor legibilidade.

RESUMO ESTRUTURADO:"""
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system", 
                        "content": "Você é um secretário especializado em resumir atas de reunião de forma estruturada."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 1500
            }
            
            try:
                with st.spinner("📋 Criando resumo estruturado..."):
                    time.sleep(0.5)
                    response = requests.post(url, headers=headers, json=payload, timeout=60)
                    
                    if response.status_code == 200:
                        result = response.json()
                        return result["choices"][0]["message"]["content"].strip()
                    else:
                        return texto
            
            except:
                return texto
        
        def formatar_ata_formal_deepseek(texto: str) -> str:
            """Formata o texto como uma ata formal completa."""
            import requests
            import json
            import time
            
            api_key = obter_api_key()
            if not api_key:
                return texto
            
            texto_valido, aviso = validar_texto_para_api(texto, 3000)
            if aviso:
                st.info(f"ℹ️ {aviso}")
            
            data_atual = datetime.now().strftime("%d de %B de %Y")
            
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""Transforme este texto em uma ata formal completa:

CONTEÚDO DA REUNIÃO:
{texto_valido}

ESTRUTURA DA ATA FORMAL:
1. TÍTULO: "ATA DA REUNIÃO"
2. DATA: {data_atual} (use esta data atual)
3. HORÁRIO: "Início: [hora] - Término: [hora]" (se não houver, use "Horário não especificado")
4. LOCAL: "Local não especificado" (a menos que mencionado)
5. PARTICIPANTES: Liste se mencionados, senão "Participantes não listados"
6. PAUTA: Resuma os principais tópicos
7. DISCUSSÃO: Organize o conteúdo em tópicos
8. DECISÕES: Destaque as decisões tomadas
9. ENCERRAMENTO: Inclua hora de término e data da próxima reunião se mencionada
10. ASSINATURAS: Linha para assinaturas

Use linguagem formal, parágrafos bem estruturados e formatação clara.
Não invente informações que não estão no texto.

ATA FORMAL COMPLETA:"""
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system", 
                        "content": "Você é um redator especializado em atas formais de reunião corporativa."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.4,
                "max_tokens": 3500
            }
            
            try:
                with st.spinner("🏢 Formatando como ata formal..."):
                    time.sleep(0.5)
                    response = requests.post(url, headers=headers, json=payload, timeout=90)
                    
                    if response.status_code == 200:
                        result = response.json()
                        return result["choices"][0]["message"]["content"].strip()
                    else:
                        return texto
            
            except:
                return texto

        # =============================
        # 1) Configuração da API (com aviso de segurança)
        # =============================
        with st.expander("🔐 Configuração da API DeepSeek", expanded=False):
            
            st.warning("""
            **⚠️ ATENÇÃO: SUA API KEY FOI EXPOSTA!**
            
            A chave `sk-9a8fa1edd8cd4226a70cf968c68ee25d` foi compartilhada publicamente.
            
            **AÇÕES URGENTES:**
            1. **Revogue imediatamente** esta chave no [DeepSeek Platform](https://platform.deepseek.com/api-keys)
            2. **Gere uma nova chave** segura
            3. **Configure no Streamlit Secrets** (recomendado) ou variáveis de ambiente
            """)
            
            # Opções de configuração
            config_method = st.radio(
                "Método de configuração:",
                ["Usar Streamlit Secrets (recomendado)", "Inserir manualmente (apenas para teste)"],
                horizontal=True
            )
            
            api_key = None
            
            if config_method == "Usar Streamlit Secrets (recomendado)":
                try:
                    api_key = st.secrets["DEEPSEEK_API_KEY"]
                    st.success("✅ API Key carregada do Secrets com segurança!")
                except:
                    st.error("""
                    **Secrets não configurado!**
                    
                    Para configurar, crie o arquivo `.streamlit/secrets.toml` com:
                    ```toml
                    DEEPSEEK_API_KEY = "sua-nova-chave-aqui"
                    ```
                    """)
            
            else:  # Manual (apenas para teste)
                api_key_input = st.text_input(
                    "API Key (não compartilhe!):",
                    type="password",
                    help="Cole sua nova API Key aqui (apenas para testes)"
                )
                if api_key_input:
                    api_key = api_key_input
                    st.warning("⚠️ Modo manual ativado - não recomendado para produção")
            
            # Verificador de status da API
            if api_key:
                col_status1, col_status2 = st.columns([3, 1])
                with col_status2:
                    if st.button("🔍 Testar Conexão", use_container_width=True):
                        try:
                            import requests
                            test_response = requests.get(
                                "https://api.deepseek.com/v1/models",
                                headers={"Authorization": f"Bearer {api_key}"},
                                timeout=10
                            )
                            if test_response.status_code == 200:
                                st.success("✅ Conexão com API OK!")
                            else:
                                st.error(f"❌ Erro: {test_response.status_code}")
                        except:
                            st.error("❌ Falha na conexão")
            
            # Monitor de uso (simples)
            if "tokens_usados" in st.session_state:
                tokens = st.session_state["tokens_usados"]
                percentual = min((tokens / 1000000) * 100, 100)  # 1M tokens free tier
                
                st.progress(percentual / 100)
                st.caption(f"📊 Tokens usados este mês: {tokens:,} / 1,000,000 ({percentual:.1f}%)")
            
            # Dicas de otimização
            with st.expander("💡 Dicas para otimizar tokens"):
                st.markdown("""
                **Para economizar tokens gratuitos:**
                1. **Corrija textos curtos** primeiro antes de textos longos
                2. **Use "Correção Rápida"** para pequenos ajustes
                3. **Limite textos** a 3000 caracteres por requisição
                4. **Cache resultados** para textos repetidos
                5. **Combine operações** usando "Aplicar Todas"
                """)

        # =============================
        # 2) Processa ações pendentes
        # =============================
        action = st.session_state.get("editor_action", None)

        if action is not None:
            texto_base = st.session_state.get("text_editor_area", texto_original)
            max_caracteres = st.session_state.get("max_caracteres_paragrafos", 500)
            aplicar_correcoes_cfg = st.session_state.get("aplicar_correcoes_editor", True)
            
            novo_texto = texto_base

            if action == "organizar":
                novo_texto = organizar_paragrafos(texto_base, max_caracteres=max_caracteres)

            elif action == "corrigir_palavras":
                novo_texto = pos_processar_texto(texto_base)

            elif action == "corrigir_deepseek":
                api_key = obter_api_key()
                if api_key:
                    novo_texto = corrigir_ortografia_deepseek(texto_base)
                else:
                    st.error("❌ Configure a API Key primeiro")
                    novo_texto = texto_base

            elif action == "melhorar_clareza":
                api_key = obter_api_key()
                if api_key:
                    novo_texto = melhorar_clareza_deepseek(texto_base)
                else:
                    st.error("❌ Configure a API Key primeiro")
                    novo_texto = texto_base

            elif action == "resumir_ata":
                api_key = obter_api_key()
                if api_key:
                    novo_texto = criar_resumo_ata_deepseek(texto_base)
                else:
                    st.error("❌ Configure a API Key primeiro")
                    novo_texto = texto_base

            elif action == "formatar_formal":
                api_key = obter_api_key()
                if api_key:
                    novo_texto = formatar_ata_formal_deepseek(texto_base)
                else:
                    st.error("❌ Configure a API Key primeiro")
                    novo_texto = texto_base

            elif action == "restaurar":
                novo_texto = texto_original

            elif action == "aplicar_todas":
                texto_processado = texto_original
                if aplicar_correcoes_cfg:
                    texto_processado = pos_processar_texto(texto_processado)
                texto_processado = organizar_paragrafos(texto_processado, max_caracteres=max_caracteres)
                texto_processado = capitalizar_frases(texto_processado)
                texto_processado = corrigir_pontuacao(texto_processado)
                
                # Adiciona DeepSeek se disponível
                api_key = obter_api_key()
                if api_key:
                    texto_processado = corrigir_ortografia_deepseek(texto_processado)
                
                novo_texto = texto_processado

            elif action == "limpar":
                novo_texto = ""

            st.session_state["text_editor_area"] = novo_texto
            st.session_state["texto_editado"] = novo_texto
            st.session_state["editor_action"] = None

        # Inicialização se necessário
        if "text_editor_area" not in st.session_state:
            st.session_state["text_editor_area"] = texto_original
        if "texto_editado" not in st.session_state:
            st.session_state["texto_editado"] = st.session_state["text_editor_area"]

        texto_editado = st.session_state.get("text_editor_area", texto_original)
        
        # =============================
        # 3) Estatísticas do texto original
        # =============================
        st.markdown("### 📊 Estatísticas do Texto Original")
        
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        with col_stats1:
            caracteres_orig = len(texto_original)
            st.metric("Caracteres", f"{caracteres_orig:,}")
        with col_stats2:
            palavras_orig = len(texto_original.split())
            st.metric("Palavras", f"{palavras_orig:,}")
        with col_stats3:
            linhas_orig = len(texto_original.split('\n'))
            st.metric("Linhas", linhas_orig)
        with col_stats4:
            paragrafos_orig = len([p for p in texto_original.split('\n\n') if p.strip()])
            st.metric("Parágrafos", paragrafos_orig)
        
        # =============================
        # 4) Ferramentas de IA
        # =============================
        st.markdown("### 🤖 Ferramentas Inteligentes de IA")
        
        # Verifica se API está configurada
        api_configurada = obter_api_key() is not None
        
        if not api_configurada:
            st.warning("""
            ⚠️ **API DeepSeek não configurada**
            
            Para usar as ferramentas de IA:
            1. **Revogue a chave exposta** e gere uma nova
            2. **Configure no Secrets** ou insira manualmente abaixo
            3. **Ative as ferramentas** de IA
            """)
        
        # Primeira linha - Correções avançadas
        col_ai1, col_ai2 = st.columns(2)
        
        with col_ai1:
            if st.button(
                "🔍 Correção Avançada (DeepSeek)",
                use_container_width=True,
                type="primary",
                disabled=not api_configurada,
                help="Correção ortográfica e gramatical avançada" if api_configurada else "Configure a API primeiro",
                key="btn_corrigir_deepseek"
            ):
                st.session_state["editor_action"] = "corrigir_deepseek"
                st.rerun()
        
        with col_ai2:
            if st.button(
                "✨ Melhorar Clareza",
                use_container_width=True,
                type="secondary",
                disabled=not api_configurada,
                help="Torna o texto mais claro e fluente" if api_configurada else "Configure a API primeiro",
                key="btn_melhorar_clareza"
            ):
                st.session_state["editor_action"] = "melhorar_clareza"
                st.rerun()
        
        # Segunda linha - Formatação avançada
        col_ai3, col_ai4 = st.columns(2)
        
        with col_ai3:
            if st.button(
                "📋 Resumo Estruturado",
                use_container_width=True,
                type="secondary",
                disabled=not api_configurada,
                help="Cria resumo em tópicos da ata" if api_configurada else "Configure a API primeiro",
                key="btn_resumir_ata"
            ):
                st.session_state["editor_action"] = "resumir_ata"
                st.rerun()
        
        with col_ai4:
            if st.button(
                "🏢 Ata Formal Completa",
                use_container_width=True,
                type="secondary",
                disabled=not api_configurada,
                help="Formata como ata corporativa formal" if api_configurada else "Configure a API primeiro",
                key="btn_formatar_formal"
            ):
                st.session_state["editor_action"] = "formatar_formal"
                st.rerun()
        
        # =============================
        # 5) Ferramentas básicas
        # =============================
        st.markdown("### ⚙️ Ferramentas Básicas")
        
        col_basic1, col_basic2, col_basic3 = st.columns(3)
        
        with col_basic1:
            if st.button(
                "📝 Organizar Parágrafos",
                use_container_width=True,
                type="primary",
                key="btn_organizar_paragrafos"
            ):
                st.session_state["editor_action"] = "organizar"
                st.rerun()
        
        with col_basic2:
            if st.button(
                "🔤 Correção Rápida",
                use_container_width=True,
                type="secondary",
                help="Usa dicionário local de correções",
                key="btn_corrigir_palavras"
            ):
                st.session_state["editor_action"] = "corrigir_palavras"
                st.rerun()
        
        with col_basic3:
            if st.button(
                "📌 Formatar Básico",
                use_container_width=True,
                type="secondary",
                help="Capitalização e pontuação básica",
                key="btn_formatar_basico"
            ):
                texto_processado = capitalizar_frases(texto_editado)
                texto_processado = corrigir_pontuacao(texto_processado)
                st.session_state["text_editor_area"] = texto_processado
                st.session_state["texto_editado"] = texto_processado
                st.success("✅ Formatação básica aplicada!")
                st.rerun()
        
        # =============================
        # 6) Configurações avançadas
        # =============================
        with st.expander("⚙️ Configurações Avançadas", expanded=False):
            col_adv1, col_adv2 = st.columns(2)
            
            with col_adv1:
                max_caracteres = st.slider(
                    "Máximo de caracteres por parágrafo:",
                    min_value=200,
                    max_value=1000,
                    value=st.session_state.get("max_caracteres_paragrafos", 500),
                    step=50,
                    help="Controla o tamanho máximo de cada parágrafo",
                    key="max_caracteres_paragrafos"
                )
            
            with col_adv2:
                aplicar_correcoes = st.checkbox(
                    "Aplicar correções automáticas em 'Aplicar Todas'",
                    value=st.session_state.get("aplicar_correcoes_editor", True),
                    help="Usa a biblioteca de correções quando você clicar em 'Aplicar Todas'",
                    key="aplicar_correcoes_editor"
                )
        
        # =============================
        # 7) Editor de texto
        # =============================
        st.markdown("### ✍️ Editor de Texto")
        
        col_view1, col_view2 = st.columns(2)
        
        with col_view1:
            st.markdown("#### 📋 Texto Original")
            preview_original = texto_original[:2000]
            if len(texto_original) > 2000:
                preview_original += f"\n\n[... texto truncado ...]\n\nTotal: {len(texto_original):,} caracteres"
            
            st.markdown(f"""
            <div class="text-editor" style="background: #f8f9fa; border-color: #dee2e6;">
                {preview_original}
            </div>
            """, unsafe_allow_html=True)
        
        with col_view2:
            st.markdown("#### 📝 Texto Editado")
            texto_editado_widget = st.text_area(
                "Edite seu texto:",
                value=st.session_state.get("text_editor_area", texto_original),
                height=350,
                label_visibility="collapsed",
                key="text_editor_area"
            )
            st.session_state["texto_editado"] = texto_editado_widget
        
        texto_editado = st.session_state.get("texto_editado", "")
        
        # =============================
        # 8) Estatísticas comparativas
        # =============================
        st.markdown("### 📈 Comparação e Análise")
        
        col_comp1, col_comp2, col_comp3, col_comp4 = st.columns(4)
        
        with col_comp1:
            caracteres_edit = len(texto_editado)
            delta_caracteres = caracteres_edit - len(texto_original)
            st.metric(
                "Caracteres",
                f"{caracteres_edit:,}",
                delta=f"{delta_caracteres:+d}"
            )
        
        with col_comp2:
            palavras_edit = len(texto_editado.split())
            delta_palavras = palavras_edit - palavras_orig
            st.metric(
                "Palavras",
                f"{palavras_edit:,}",
                delta=f"{delta_palavras:+d}"
            )
        
        with col_comp3:
            paragrafos_edit = len([p for p in texto_editado.split('\n\n') if p.strip()])
            delta_paragrafos = paragrafos_edit - paragrafos_orig
            st.metric(
                "Parágrafos",
                paragrafos_edit,
                delta=f"{delta_paragrafos:+d}"
            )
        
        with col_comp4:
            densidade_orig = palavras_orig / max(paragrafos_orig, 1)
            densidade_edit = palavras_edit / max(paragrafos_edit, 1)
            delta_densidade = densidade_edit - densidade_orig
            st.metric(
                "Densidade",
                f"{densidade_edit:.1f}",
                delta=f"{delta_densidade:+.1f}",
                help="Palavras por parágrafo"
            )
        
        # =============================
        # 9) Ações gerais
        # =============================
        st.markdown("### 💾 Ações")
        
        col_actions1, col_actions2, col_actions3 = st.columns(3)
        
        with col_actions1:
            if st.button("↩️ Restaurar Original", use_container_width=True, key="btn_restaurar_original"):
                st.session_state["editor_action"] = "restaurar"
                st.rerun()
        
        with col_actions2:
            if st.button("⚡ Aplicar Todas (Básicas)", use_container_width=True, type="primary", key="btn_aplicar_todas"):
                st.session_state["editor_action"] = "aplicar_todas"
                st.rerun()
        
        with col_actions3:
            if st.button("🗑️ Limpar Editor", use_container_width=True, type="secondary", key="btn_limpar_editor"):
                st.session_state["editor_action"] = "limpar"
                st.rerun()
        
        # =============================
        # 10) Download
        # =============================
        st.markdown("### 📥 Download do Texto Editado")
        
        if texto_editado:
            nome_base = "transcricao_editada"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            
            with col_dl1:
                st.download_button(
                    "📄 Baixar TXT",
                    data=texto_editado,
                    file_name=f"{nome_base}_{timestamp}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="download_txt"
                )
            
            with col_dl2:
                # Versão HTML bonita
                texto_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Transcrição Editada - {timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 30px; background: #f8f9fa; }}
        .container {{ background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        .meta {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; font-size: 0.9em; color: #7f8c8d; }}
        .paragraph {{ margin-bottom: 20px; padding: 15px; border-left: 4px solid #2ecc71; background: #f9f9f9; }}
        .highlight {{ background: #fffacd; padding: 2px 5px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📝 Transcrição Editada</h1>
        <div class="meta">
            <strong>📅 Data:</strong> {datetime.now().strftime("%d/%m/%Y %H:%M")}<br>
            <strong>📊 Caracteres:</strong> {len(texto_editado):,}<br>
            <strong>🔤 Palavras:</strong> {len(texto_editado.split()):,}<br>
            <strong>📋 Parágrafos:</strong> {len([p for p in texto_editado.split('\\n\\n') if p.strip()])}
        </div>"""
                
                # Processa parágrafos
                for i, paragrafo in enumerate(texto_editado.split('\n\n')):
                    if paragrafo.strip():
                        texto_html += f'\n        <div class="paragraph" id="p{i+1}">\n            {paragrafo.strip()}\n        </div>'
                
                texto_html += """
    </div>
</body>
</html>"""
                
                st.download_button(
                    "🌐 Baixar HTML",
                    data=texto_html,
                    file_name=f"{nome_base}_{timestamp}.html",
                    mime="text/html",
                    use_container_width=True,
                    key="download_html"
                )
            
            with col_dl3:
                # Versão Markdown para compatibilidade
                texto_md = f"""# Transcrição Editada

**Data de geração:** {datetime.now().strftime("%d/%m/%Y %H:%M")}  
**Caracteres:** {len(texto_editado):,}  
**Palavras:** {len(texto_editado.split()):,}  
**Parágrafos:** {len([p for p in texto_editado.split('\n\n') if p.strip()])}

---

{texto_editado}

---

*Documento gerado automaticamente por Transcrição Inteligente*
"""
                
                st.download_button(
                    "📝 Baixar Markdown",
                    data=texto_md,
                    file_name=f"{nome_base}_{timestamp}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="download_md"
                )
# Fechar container principal
st.markdown('</div>', unsafe_allow_html=True)

# Rodapé
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1.5rem;">
    <p style="font-size: 1.1rem; font-weight: 600;">🎯 Transcrição Inteligente - Editor Avançado</p>
    <p style="color: #999; font-size: 0.9rem;">
        Whisper OpenAI • v3.0 • Processamento em tempo real • Correções automáticas • Editor de texto avançado
    </p>
    <p style="color: #aaa; font-size: 0.8rem; margin-top: 1rem;">
        © 2024 • Para uso profissional • Desenvolvido com Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
