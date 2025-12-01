import os
import time
import warnings
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================
# Ajustes de ambiente
# =============================
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
warnings.filterwarnings("ignore", message=".*huggingface_hub.*")

import torch

# tentar espremer a CPU sem brigar com o Streamlit
num_threads = os.cpu_count() or 4
try:
    torch.set_num_threads(num_threads)
except RuntimeError:
    pass
os.environ["OMP_NUM_THREADS"] = str(num_threads)

import librosa
import soundfile as sf
import streamlit as st

# Whisper oficial (CPU/GPU NVIDIA)
import whisper

# Tentativa de OpenVINO
try:
    from optimum.intel.openvino import OVModelForCTC
    from transformers import AutoProcessor
    OPENVINO_OK = True
except ImportError:
    OPENVINO_OK = False

# =============================
# Config Streamlit + Tema
# =============================
st.set_page_config(
    page_title="Transcrição ATA – Whisper / OpenVINO",
    layout="wide",
)

st.markdown(
    """
<style>
/* Fundo geral */
.main, .block-container {
    background: linear-gradient(145deg, #0f172a 0%, #020617 40%, #020617 100%);
    color: #e5e7eb;
}

/* Títulos */
h1, h2, h3, h4 {
    color: #e5e7eb !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid #1f2937;
}
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] label, 
[data-testid="stSidebar"] p {
    color: #e5e7eb !important;
}

/* Cards */
.card {
    border-radius: 12px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.8rem;
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(148,163,184,0.25);
}
.card-soft {
    background: rgba(15,23,42,0.65);
}
.card-success {
    border-color: #22c55e80;
}
.card-warn {
    border-color: #eab30880;
}
.card-error {
    border-color: #ef444480;
}

/* Métricas */
[data-testid="stMetric"] {
    background: rgba(15,23,42,0.9);
    border-radius: 12px;
    padding: 0.8rem;
    border: 1px solid rgba(148,163,184,0.35);
}

/* Botão principal */
.stButton>button {
    width: 100%;
    border-radius: 999px;
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
    color: white !important;
    border: none;
    font-weight: 600;
    padding: 0.6rem 1rem;
}
.stButton>button:hover {
    filter: brightness(1.1);
}

/* Área de texto de timestamps */
textarea {
    background: rgba(15,23,42,0.9) !important;
    color: #e5e7eb !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("📝 Transcrição de Ata")
st.markdown(
    "<div class='card card-soft'>Transcrição em português brasileiro com foco em atas de reuniões, "
    "sessões de Câmara, audiências e falas formais.</div>",
    unsafe_allow_html=True,
)

# =============================
# Utilitários gerais
# =============================
BASE_PROMPT = (
    "Transcrição em português brasileiro formal, com pontuação correta, "
    "acentuação adequada e frases completas. Use nomes próprios, siglas e "
    "termos técnicos conforme aparecem no áudio. Evite inventar trechos."
)


def pos_processar_texto(texto: str) -> str:
    if not texto:
        return ""

    correcoes = {
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


def formatar_timestamps(timestamps):
    linhas = []
    for ts in timestamps:
        linhas.append(f"[{ts['start']:.1f}s - {ts['end']:.1f}s] {ts['text']}")
    return "\n".join(linhas)


# =============================
# Backend WHISPER (CPU / GPU NVIDIA)
# =============================
@st.cache_resource(show_spinner=True)
def carregar_modelo_whisper(nome_modelo: str, device: str):
    return whisper.load_model(nome_modelo, device=device)


def _transcrever_chunk_whisper(model, parte, t_ini, t_fim, idx):
    """Função auxiliar para rodar em threads (sem chamadas ao Streamlit aqui)."""
    inicio_parte = time.time()
    result = model.transcribe(
        parte,
        language="pt",
        task="transcribe",
        temperature=[0.0, 0.2],
        best_of=5,
        initial_prompt=BASE_PROMPT,
        fp16=False,  # seguro para CPU; se tiver GPU, PyTorch já otimiza por dentro
    )
    tempo_parte = time.time() - inicio_parte
    segs = result.get("segments", [])
    return {
        "idx": idx,
        "t_ini": t_ini,
        "t_fim": t_fim,
        "segments": segs,
        "tempo": tempo_parte,
    }


def transcrever_com_whisper(audio, sr, modelo_nome: str, chunk_seg: int):
    # Detecta device
    if torch.cuda.is_available():
        device = "cuda"
        device_msg = f"GPU NVIDIA detectada: {torch.cuda.get_device_name(0)}"
    else:
        device = "cpu"
        device_msg = "GPU NVIDIA não detectada. Usando CPU (Intel/AMD)."

    st.sidebar.markdown(
        f"<div class='card card-soft'><b>Backend:</b> Whisper<br><b>Device:</b> {device_msg}</div>",
        unsafe_allow_html=True,
    )

    duracao_min = len(audio) / sr / 60
    modelo_efetivo = modelo_nome
    if device == "cpu" and duracao_min > 20 and modelo_nome in ("small", "medium", "large-v3"):
        st.sidebar.warning(
            f"Áudio com {duracao_min:.1f} min e modelo '{modelo_nome}' em CPU "
            f"pode ficar muito lento. Usando 'base'."
        )
        modelo_efetivo = "base"

    st.markdown(
        f"<div class='card card-soft'>🎯 Whisper: modelo efetivo "
        f"<code>{modelo_efetivo}</code> em <code>{device}</code></div>",
        unsafe_allow_html=True,
    )

    with st.spinner(f"Carregando modelo Whisper {modelo_efetivo}..."):
        model = carregar_modelo_whisper(modelo_efetivo, device)

    partes = dividir_em_chunks(audio, sr, chunk_seg)
    total_partes = len(partes)
    st.markdown(
        f"<div class='card card-soft'>📦 Partes (Whisper): <b>{total_partes}</b> "
        f"de ~{chunk_seg}s</div>",
        unsafe_allow_html=True,
    )

    # Barra de progresso na sidebar
    sidebar_progress = st.sidebar.progress(0.0, text="Progresso geral dos chunks")
    sidebar_info = st.sidebar.empty()

    # Multithreading real nos chunks
    max_workers = min(4, os.cpu_count() or 2)
    resultados = {}
    inicio_geral = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, (parte, t_ini, t_fim) in enumerate(partes, start=1):
            futures.append(
                executor.submit(_transcrever_chunk_whisper, model, parte, t_ini, t_fim, idx)
            )

        for done, fut in enumerate(as_completed(futures), start=1):
            data = fut.result()
            idx = data["idx"]
            resultados[idx] = data

            sidebar_progress.progress(done / total_partes)
            sidebar_info.markdown(
                f"Processando chunk {done}/{total_partes} "
                f"(threads ativas: {max_workers})"
            )

    # Ordenar resultados pelo índice original
    texto_final = ""
    timestamps = []

    for idx in sorted(resultados.keys()):
        data = resultados[idx]
        t_ini = data["t_ini"]
        segs = data["segments"]
        tempo_parte = data["tempo"]

        janela_min = t_ini / 60
        janela_max = data["t_fim"] / 60

        st.markdown(
            f"<div class='card card-soft'>📝 Whisper – Parte {idx}/{total_partes} "
            f"({janela_min:.1f}–{janela_max:.1f} min) concluída em {tempo_parte:.1f}s</div>",
            unsafe_allow_html=True,
        )

        if segs:
            preview = segs[0]["text"][:120]
            st.markdown(
                f"<div class='card card-soft'>Prévia da parte {idx}: "
                f"<i>{preview}...</i></div>",
                unsafe_allow_html=True,
            )
            for seg in segs:
                texto = seg["text"]
                start = float(seg["start"]) + t_ini
                end = float(seg["end"]) + t_ini
                timestamps.append({"start": start, "end": end, "text": texto})
                texto_final += texto + " "
        else:
            st.warning(f"Nenhum texto detectado na parte {idx}.")

    tempo_total = time.time() - inicio_geral
    return texto_final, timestamps, tempo_total, duracao_min


# =============================
# Backend OPENVINO (NPU / GPU / CPU via AUTO)
# =============================
OV_MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese"


@st.cache_resource(show_spinner=True)
def carregar_modelo_openvino():
    processor = AutoProcessor.from_pretrained(OV_MODEL_ID)
    ov_model = OVModelForCTC.from_pretrained(
        OV_MODEL_ID,
        export=True,
        device="AUTO",
    )
    return processor, ov_model


def transcrever_com_openvino(audio, sr, chunk_seg: int):
    duracao_min = len(audio) / sr / 60

    if not OPENVINO_OK:
        st.error("OpenVINO / optimum-intel não instalados. Use o backend Whisper.")
        return None, None, 0.0, duracao_min

    st.sidebar.markdown(
        "<div class='card card-soft'><b>Backend:</b> OpenVINO<br>"
        "Device='AUTO' (tenta GPU/NPU/CPU Intel).</div>",
        unsafe_allow_html=True,
    )

    st.info("OpenVINO ativado. Device='AUTO' (tenta GPU/NPU/CPU Intel).")

    try:
        processor, ov_model = carregar_modelo_openvino()
    except Exception as e:
        st.error(
            "Falha ao inicializar o modelo OpenVINO "
            "(provável problema de encoding/pyctcdecode no Windows)."
        )
        st.text(str(e))
        st.warning("Voltando automaticamente para o backend Whisper.")
        return None, None, 0.0, duracao_min

    partes = dividir_em_chunks(audio, sr, chunk_seg)
    total_partes = len(partes)
    st.markdown(
        f"<div class='card card-soft'>📦 Partes (OpenVINO): <b>{total_partes}</b></div>",
        unsafe_allow_html=True,
    )

    sidebar_progress = st.sidebar.progress(0.0, text="Progresso geral dos chunks (OpenVINO)")
    sidebar_info = st.sidebar.empty()

    texto_final = ""
    timestamps = []
    inicio_geral = time.time()

    for idx, (parte, t_ini, t_fim) in enumerate(partes, start=1):
        janela_min = t_ini / 60
        janela_max = t_fim / 60
        st.markdown(
            f"<div class='card card-soft'>📝 OpenVINO – Parte {idx}/{total_partes} "
            f"({janela_min:.1f}–{janela_max:.1f} min)</div>",
            unsafe_allow_html=True,
        )

        parte_np = parte.astype("float32")

        inputs = processor(
            parte_np,
            sampling_rate=sr,
            return_tensors="pt",
            padding="longest",
        )

        with torch.no_grad():
            outputs = ov_model(**inputs)
            logits = outputs.logits

        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids)[0]

        texto_final += text + " "
        timestamps.append(
            {"start": float(t_ini), "end": float(t_fim), "text": text}
        )

        sidebar_progress.progress(idx / total_partes)
        sidebar_info.markdown(f"Chunk {idx}/{total_partes} concluído (OpenVINO).")

    tempo_total = time.time() - inicio_geral
    return texto_final, timestamps, tempo_total, duracao_min


# =============================
# Sidebar – escolha de backend
# =============================
st.sidebar.header("Configurações")

backend_options = ["Whisper (oficial)", "OpenVINO (experimental)"]
backend = st.sidebar.radio(
    "Motor de transcrição",
    backend_options,
    index=0,
    help="OpenVINO tenta usar NPU/GPU Intel. Se der erro, cai para Whisper automaticamente.",
)

chunk_segundos = st.sidebar.slider(
    "Duração de cada parte (segundos)",
    min_value=60,
    max_value=240,
    value=120,
    step=30,
)

modelos = {
    "tiny – mais rápido (menos preciso)": "tiny",
    "base – equilíbrio recomendado": "base",
    "small – mais preciso (mais pesado)": "small",
    "medium – alta precisão (pesado)": "medium",
    "large-v3 – máxima precisão (muito pesado)": "large-v3",
}
modelo_label = st.sidebar.selectbox(
    "Modelo Whisper (quando backend = Whisper)",
    list(modelos.keys()),
    index=1,
)
modelo_whisper = modelos[modelo_label]

if backend == "OpenVINO (experimental)" and not OPENVINO_OK:
    st.sidebar.error("OpenVINO / optimum-intel não instalados. Este backend não estará disponível.")

# =============================
# Upload de áudio
# =============================
st.markdown("<div class='card'>Envie o áudio da sessão ou reunião para iniciar a transcrição.</div>", unsafe_allow_html=True)

audio_file = st.file_uploader(
    "Arquivo de áudio",
    type=["mp3", "wav", "m4a", "ogg", "flac", "aac", "wma"],
)

if audio_file is not None:
    st.markdown(
        f"<div class='card card-soft'>Arquivo carregado: <b>{audio_file.name}</b><br>"
        f"Tamanho aproximado: {audio_file.size / 1024 / 1024:.2f} MB</div>",
        unsafe_allow_html=True,
    )

# =============================
# Botão principal
# =============================
if st.button("🚀 Transcrever agora", disabled=(audio_file is None)):
    if audio_file is None:
        st.warning("Envie um arquivo de áudio primeiro.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=audio_file.name) as tmp:
            tmp.write(audio_file.read())
            caminho_audio = tmp.name

        try:
            # Pré-processar áudio
            with st.spinner("🔧 Pré-processando áudio..."):
                audio, sr_original = librosa.load(caminho_audio, sr=None, mono=True)

                max_abs = max(1e-8, float(abs(audio).max()))
                audio = audio / max_abs * 0.9

                if sr_original != 16000:
                    audio = librosa.resample(audio, orig_sr=sr_original, target_sr=16000)
                    sr = 16000
                else:
                    sr = sr_original

            # Escolher backend
            if backend == "OpenVINO (experimental)" and OPENVINO_OK:
                texto, ts, tempo_proc, duracao_min = transcrever_com_openvino(
                    audio, sr, chunk_segundos
                )

                if texto is None:
                    st.markdown(
                        "<div class='card card-warn'>Usando backend Whisper como fallback.</div>",
                        unsafe_allow_html=True,
                    )
                    texto, ts, tempo_proc, duracao_min = transcrever_com_whisper(
                        audio, sr, modelo_whisper, chunk_segundos
                    )
            else:
                if backend == "OpenVINO (experimental)" and not OPENVINO_OK:
                    st.markdown(
                        "<div class='card card-warn'>OpenVINO não disponível. Usando Whisper.</div>",
                        unsafe_allow_html=True,
                    )
                texto, ts, tempo_proc, duracao_min = transcrever_com_whisper(
                    audio, sr, modelo_whisper, chunk_segundos
                )

            texto = pos_processar_texto(texto)

            if not texto.strip():
                st.markdown(
                    "<div class='card card-error'>Nenhum texto final gerado. "
                    "Verifique se o áudio tem fala clara.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='card card-success'>🎉 Transcrição concluída com sucesso!</div>",
                    unsafe_allow_html=True,
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Duração do áudio", f"{duracao_min:.1f} min")
                with col2:
                    st.metric("Tempo de processamento", f"{tempo_proc:.1f} s")

                st.subheader("🧾 Texto da Ata (Transcrição)")
                st.write(texto)

                st.subheader("⏱️ Timestamps")
                texto_ts = formatar_timestamps(ts)
                st.text(texto_ts)

                nome_base = os.path.splitext(audio_file.name)[0]
                st.download_button(
                    "📥 Baixar transcrição (.txt)",
                    data=texto,
                    file_name=f"TRANSCRICAO_{nome_base}.txt",
                    mime="text/plain",
                )
                st.download_button(
                    "📥 Baixar timestamps (.txt)",
                    data=texto_ts,
                    file_name=f"TIMESTAMPS_{nome_base}.txt",
                    mime="text/plain",
                )

        finally:
            try:
                os.unlink(caminho_audio)
            except Exception:
                pass
else:
    st.info("Envie o áudio e clique em Transcrever agora.")
