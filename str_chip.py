import os
import time
import warnings
import tempfile
import json
from pathlib import Path

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
import soundfile as sf  # mantido caso precise no futuro
import streamlit as st

# Whisper oficial
import whisper

# =============================
# Configuração Streamlit
# =============================
st.set_page_config(
    page_title="Transcrição ATA – Whisper oficial",
    layout="wide",
)

st.title("📝 Transcrição de Ata – Whisper oficial")
st.caption(
    "Usa exclusivamente o Whisper oficial da OpenAI. "
    "O modelo escolhido é mantido mesmo que o processamento fique mais lento."
)

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


def formatar_timestamps(timestamps):
    linhas = []
    for ts in timestamps:
        linhas.append(f"[{ts['start']:.1f}s - {ts['end']:.1f}s] {ts['text']}")
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
        device_msg = f"GPU NVIDIA detectada: {torch.cuda.get_device_name(0)}"
    else:
        device = "cpu"
        fp16 = False
        device_msg = "GPU NVIDIA não detectada. Usando CPU, pode ficar mais lento."

    st.info(device_msg)

    duracao_min = len(audio) / sr / 60
    modelo_efetivo = modelo_nome
    st.write(f"🎯 Whisper oficial usando o modelo `{modelo_efetivo}` em `{device}`")

    with st.spinner(f"Carregando modelo Whisper {modelo_efetivo}..."):
        model = carregar_modelo_whisper(modelo_efetivo, device)

    partes = dividir_em_chunks(audio, sr, chunk_seg)
    total_partes = len(partes)
    st.write(f"📦 Partes a processar: **{total_partes}**")

    progresso = st.progress(0)
    progresso_sidebar = st.sidebar.progress(0)

    texto_final = ""
    timestamps = []
    tempos_partes = []
    inicio_geral = time.time()

    for idx, (parte, t_ini, t_fim) in enumerate(partes, start=1):
        janela_min = t_ini / 60
        janela_max = t_fim / 60
        st.write(
            f"📝 Parte {idx}/{total_partes} "
            f"({janela_min:.1f}–{janela_max:.1f} min do áudio)"
        )

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

            st.write(f"✅ Parte {idx} concluída em {tempo_parte:.1f}s")
            st.write(f"Prévia: _{segs[0]['text'][:120]}..._")
        else:
            st.warning("⚠️ Nenhum texto detectado nesta parte.")

        progresso.progress(idx / total_partes)
        progresso_sidebar.progress(idx / total_partes)

    tempo_total = time.time() - inicio_geral
    return texto_final, timestamps, tempo_total, duracao_min, total_partes, tempos_partes


# =============================
# Sidebar – configurações
# =============================
st.sidebar.header("Configurações de processamento")

chunk_segundos = st.sidebar.slider(
    "Duração de cada parte em segundos",
    min_value=60,
    max_value=240,
    value=120,
    step=30,
)

modelos = {
    "tiny – mais rápido, menos preciso": "tiny",
    "base – equilíbrio recomendado": "base",
    "small – mais preciso, mais pesado": "small",
    "medium – alta precisão, pesado": "medium",
    "large-v3 – máxima precisão, muito pesado": "large-v3",
}
modelo_label = st.sidebar.selectbox(
    "Modelo Whisper oficial",
    list(modelos.keys()),
    index=1,
)
modelo_whisper = modelos[modelo_label]

# =============================
# Abas principais
# =============================
tab_transcricao, tab_biblioteca = st.tabs(
    ["🎧 Transcrição", "🧩 Biblioteca de correções"]
)

# =============================
# Aba 1 – Transcrição
# =============================
with tab_transcricao:
    audio_file = st.file_uploader(
        "Envie o arquivo de áudio da sessão ou ata",
        type=["mp3", "wav", "m4a", "ogg", "flac", "aac", "wma"],
    )

    if audio_file is not None:
        st.success(f"Arquivo carregado. Nome: {audio_file.name}")
        tamanho_mb = audio_file.size / 1024 / 1024
        st.write(f"Tamanho aproximado do arquivo: {tamanho_mb:.2f} MB")
    else:
        tamanho_mb = 0.0

    if st.button("🚀 Transcrever agora", disabled=(audio_file is None)):
        if audio_file is None:
            st.warning("Envie um arquivo de áudio primeiro.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=audio_file.name) as tmp:
                tmp.write(audio_file.read())
                caminho_audio = tmp.name

            try:
                with st.spinner("🔧 Pré processando áudio..."):
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

                st.markdown("### 📊 Visão geral do arquivo")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Duração do áudio", f"{duracao_min_pre:.1f} min")
                col_b.metric("Tamanho do arquivo", f"{tamanho_mb:.2f} MB")
                col_c.metric("Quantidade de partes", f"{total_partes_preview}")

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
                    st.error(
                        "Nenhum texto final gerado. Verifique se o áudio tem fala clara."
                    )
                else:
                    st.success("🎉 Transcrição concluída com sucesso.")

                    st.markdown("### 📈 Indicadores de processamento")
                    col1, col2 = st.columns(2)
                    col1.metric("Duração do áudio", f"{duracao_min:.1f} min")
                    col2.metric("Tempo total de processamento", f"{tempo_proc:.1f} s")

                    if tempos_partes:
                        import pandas as pd

                        st.markdown("### 📊 Desempenho por parte")
                        df_tempos = pd.DataFrame(
                            {
                                "Parte": list(range(1, total_partes + 1)),
                                "Tempo (s)": tempos_partes,
                            }
                        ).set_index("Parte")
                        st.bar_chart(df_tempos)

                    st.subheader("🧾 Texto da ata – prévia com 400 caracteres")
                    preview = texto[:400]
                    if len(texto) > 400:
                        preview += "..."
                    st.write(preview)

                    st.subheader("⏱️ Timestamps completos")
                    texto_ts = formatar_timestamps(ts)
                    st.text(texto_ts)

                    nome_base = os.path.splitext(audio_file.name)[0]
                    st.download_button(
                        "📥 Baixar transcrição completa em TXT",
                        data=texto,
                        file_name=f"TRANSCRICAO_{nome_base}.txt",
                        mime="text/plain",
                    )
                    st.download_button(
                        "📥 Baixar timestamps em TXT",
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

# =============================
# Aba 2 – Biblioteca de correções
# =============================
with tab_biblioteca:
    st.markdown("### 🧩 Palavras e expressões para correção automática")
    st.write(
        "Adicione aqui abreviações ou formas de fala que você quer que sejam "
        "corrigidas automaticamente na transcrição final."
    )
    st.write(
        "Exemplo. Original: vc. Substituir por. você. "
        "Essas correções valem para todas as sessões futuras."
    )

    st.markdown("#### Correções em uso (base mais personalizadas)")
    dicionario_atual = get_correcoes_dicionario()
    if dicionario_atual:
        orig = []
        novo = []
        for k, v in dicionario_atual.items():
            orig.append(k.strip())
            novo.append(v.strip())
        import pandas as pd

        st.table(pd.DataFrame({"Original": orig, "Substituir por": novo}))
    else:
        st.info("Nenhuma correção cadastrada ainda.")

    st.markdown("#### Adicionar nova correção personalizada")
    with st.form("form_add_correcao"):
        col1, col2 = st.columns(2)
        with col1:
            original = st.text_input("Original. palavra ou expressão")
        with col2:
            substituir = st.text_input("Substituir por")

        submitted = st.form_submit_button("Adicionar correção")
        if submitted:
            if not original.strip() or not substituir.strip():
                st.error("Preencha os dois campos antes de adicionar.")
            else:
                chave = f" {original.strip()} "
                valor = f" {substituir.strip()} "
                st.session_state["correcoes_custom"][chave] = valor

                salvar_correcoes_custom(st.session_state["correcoes_custom"])

                st.success(
                    f"Correção adicionada. '{original.strip()}' será trocado por "
                    f"'{substituir.strip()}' nas próximas transcrições."
                )

    if st.button("🧹 Limpar apenas correções personalizadas"):
        st.session_state["correcoes_custom"] = {}
        salvar_correcoes_custom(st.session_state["correcoes_custom"])
        st.success("Correções personalizadas limpas. As correções base continuam ativas.")
