import os
import time
import warnings
import tempfile

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
    # se o PyTorch reclamar porque já tem thread rodando, ignora
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
# Config Streamlit
# =============================
st.set_page_config(
    page_title="Transcrição ATA – Whisper / OpenVINO",
    layout="wide",
)

st.title("📝 Transcrição de Ata – Whisper + OpenVINO (Intel)")
st.caption(
    "Tenta usar OpenVINO (NPU/GPU Intel) quando disponível. "
    "Se der erro, volta automaticamente para Whisper espremendo a CPU."
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


def transcrever_com_whisper(audio, sr, modelo_nome: str, chunk_seg: int):
    if torch.cuda.is_available():
        device = "cuda"
        fp16 = True
        device_msg = f"GPU NVIDIA detectada: {torch.cuda.get_device_name(0)}"
    else:
        device = "cpu"
        fp16 = False
        device_msg = "GPU NVIDIA não detectada. Usando CPU (Intel/AMD)."

    st.info(device_msg)

    duracao_min = len(audio) / sr / 60
    modelo_efetivo = modelo_nome
    if device == "cpu" and duracao_min > 20 and modelo_nome in ("small", "medium", "large-v3"):
        st.warning(
            f"Áudio com {duracao_min:.1f} min e modelo '{modelo_nome}' em CPU "
            f"pode ficar muito lento. Usando 'base'."
        )
        modelo_efetivo = "base"

    st.write(f"🎯 Whisper: modelo efetivo `{modelo_efetivo}` em `{device}`")

    with st.spinner(f"Carregando modelo Whisper {modelo_efetivo}..."):
        model = carregar_modelo_whisper(modelo_efetivo, device)

    partes = dividir_em_chunks(audio, sr, chunk_seg)
    total_partes = len(partes)
    st.write(f"📦 Partes (Whisper): **{total_partes}**")

    progresso = st.progress(0)
    texto_final = ""
    timestamps = []
    inicio_geral = time.time()

    for idx, (parte, t_ini, t_fim) in enumerate(partes, start=1):
        janela_min = t_ini / 60
        janela_max = t_fim / 60
        st.write(f"📝 Whisper – Parte {idx}/{total_partes} – {janela_min:.1f}–{janela_max:.1f} min")

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

        segs = result.get("segments", [])
        if segs:
            for seg in segs:
                texto = seg["text"]
                start = float(seg["start"]) + t_ini
                end = float(seg["end"]) + t_ini
                timestamps.append({"start": start, "end": end, "text": texto})
                texto_final += texto + " "

            st.write(f"✅ Parte {idx} concluída em {tempo_parte:.1f}s")
            st.write(f"Prévia: _{segs[0]['text'][:120]}..._")
        else:
            st.warning("⚠️ Nenhum texto detectado nesta parte.")

        progresso.progress(idx / total_partes)

    tempo_total = time.time() - inicio_geral
    return texto_final, timestamps, tempo_total, duracao_min


# =============================
# Backend OPENVINO (NPU / GPU / CPU via AUTO)
# =============================
OV_MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese"


@st.cache_resource(show_spinner=True)
def carregar_modelo_openvino():
    """
    Carrega modelo CTC em OpenVINO.
    Usa AUTO: tenta GPU/NPU/CPU automatizado.
    """
    processor = AutoProcessor.from_pretrained(OV_MODEL_ID)
    ov_model = OVModelForCTC.from_pretrained(
        OV_MODEL_ID,
        export=True,
        device="AUTO",  # AUTO tenta GPU/NPU/CPU
    )
    return processor, ov_model


def transcrever_com_openvino(audio, sr, chunk_seg: int):
    """
    Tenta usar OpenVINO. Se der erro (pyctcdecode, encoding etc),
    devolve texto=None para o chamador poder fazer fallback pro Whisper.
    """
    duracao_min = len(audio) / sr / 60

    if not OPENVINO_OK:
        st.error("OpenVINO / optimum-intel não instalados. Use o backend Whisper.")
        return None, None, 0.0, duracao_min

    st.info("OpenVINO ativado. Device='AUTO' (tenta GPU/NPU/CPU Intel).")

    # Tentativa protegida de carregar modelo/processador
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

    # Se chegou aqui, modelo carregou OK
    partes = dividir_em_chunks(audio, sr, chunk_seg)
    total_partes = len(partes)
    st.write(f"📦 Partes (OpenVINO): **{total_partes}**")

    progresso = st.progress(0)
    texto_final = ""
    timestamps = []
    inicio_geral = time.time()

    for idx, (parte, t_ini, t_fim) in enumerate(partes, start=1):
        janela_min = t_ini / 60
        janela_max = t_fim / 60
        st.write(f"📝 OpenVINO – Parte {idx}/{total_partes} – {janela_min:.1f}–{janela_max:.1f} min")

        # garantir float32 numpy
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

        st.write(f"✅ Parte {idx} concluída.")
        st.write(f"Prévia: _{text[:120]}..._")

        progresso.progress(idx / total_partes)

    tempo_total = time.time() - inicio_geral
    return texto_final, timestamps, tempo_total, duracao_min


# =============================
# Sidebar – escolha de backend
# =============================
st.sidebar.header("Backend de processamento")

backend_options = ["Whisper (oficial)", "OpenVINO (experimental)"]
backend = st.sidebar.radio(
    "Motor de transcrição",
    backend_options,
    index=0,
    help="OpenVINO tenta usar NPU/GPU Intel via device=AUTO. Se der erro, cai para Whisper.",
)

chunk_segundos = st.sidebar.slider(
    "Duração de cada parte (segundos)",
    min_value=60,
    max_value=240,
    value=120,
    step=30,
)

# configuração de modelo Whisper
modelos = {
    "tiny – mais rápido (menos preciso)": "tiny",
    "base – equilíbrio recomendado": "base",
    "small – mais preciso (mais pesado)": "small",
    "medium – alta precisão (pesado)": "medium",
    "large-v3 – máxima precisão (muito pesado)": "large-v3",
}
modelo_label = st.sidebar.selectbox(
    "Modelo Whisper (usado quando backend = Whisper)",
    list(modelos.keys()),
    index=1,
)
modelo_whisper = modelos[modelo_label]

if backend == "OpenVINO (experimental)" and not OPENVINO_OK:
    st.sidebar.error("OpenVINO / optimum-intel não instalados. Este backend não estará disponível.")


# =============================
# Upload de áudio
# =============================
audio_file = st.file_uploader(
    "Envie o arquivo de áudio da sessão/ata",
    type=["mp3", "wav", "m4a", "ogg", "flac", "aac", "wma"],
)

if audio_file is not None:
    st.success(f"Arquivo carregado: {audio_file.name}")
    st.write(f"Tamanho aproximado: {audio_file.size / 1024 / 1024:.2f} MB")


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
                # tenta OpenVINO
                texto, ts, tempo_proc, duracao_min = transcrever_com_openvino(
                    audio, sr, chunk_segundos
                )

                # se deu ruim (texto=None), faz fallback automático pro Whisper
                if texto is None:
                    st.info("Usando backend Whisper como fallback.")
                    texto, ts, tempo_proc, duracao_min = transcrever_com_whisper(
                        audio, sr, modelo_whisper, chunk_segundos
                    )
            else:
                if backend == "OpenVINO (experimental)" and not OPENVINO_OK:
                    st.warning("OpenVINO não disponível. Usando Whisper.")
                texto, ts, tempo_proc, duracao_min = transcrever_com_whisper(
                    audio, sr, modelo_whisper, chunk_segundos
                )

            texto = pos_processar_texto(texto)

            if not texto.strip():
                st.error("Nenhum texto final gerado. Verifique se o áudio tem fala clara.")
            else:
                st.success("🎉 Transcrição concluída com sucesso!")

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
    st.info("Envie o áudio e clique em '🚀 Transcrever agora'.")
