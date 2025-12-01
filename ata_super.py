import os
import sys
import time
import warnings

# Ajustes de ambiente
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore", message=".*huggingface_hub.*")

import torch
import whisper
import librosa

# Opção de backend mais rápido em CPU (faster-whisper)
try:
    from faster_whisper import WhisperModel
    HAVE_FASTER_WHISPER = True
except Exception:
    HAVE_FASTER_WHISPER = False

# OpenVINO + Optimum (para usar GPU Intel / NPU)
try:
    from openvino import Core
    from optimum.intel.openvino import OVModelForSpeechSeq2Seq
    from transformers import AutoProcessor
    HAVE_OPENVINO = True
except Exception:
    HAVE_OPENVINO = False

# Se quiser testar o pré-ênfase em áudios muito abafados, mude para True
USE_PREEMPHASIS = False


class BarraProgresso:
    def __init__(self, total, descricao="", comprimento=40):
        self.total = max(1, total)
        self.descricao = descricao
        self.comprimento = comprimento
        self.atual = 0
        self.inicio_tempo = time.time()
    
    def atualizar(self, progresso=1):
        self.atual += progresso
        if self.atual > self.total:
            self.atual = self.total
        percentual = min(100, (self.atual / self.total) * 100)
        barras_preenchidas = int(self.comprimento * self.atual // self.total)
        barra = '█' * barras_preenchidas + '░' * (self.comprimento - barras_preenchidas)
        tempo_decorrido = time.time() - self.inicio_tempo
        if self.atual > 0:
            tempo_estimado = (tempo_decorrido / self.atual) * (self.total - self.atual)
        else:
            tempo_estimado = 0
        
        sys.stdout.write(
            f"\r{self.descricao} [{barra}] "
            f"{percentual:5.1f}% "
            f"(⏱ {tempo_decorrido:5.1f}s / ETA {tempo_estimado:5.1f}s)"
        )
        sys.stdout.flush()
        
        if self.atual == self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()


def aplicar_pre_enfase(audio, alpha=0.97):
    import numpy as np
    if not USE_PREEMPHASIS:
        return audio
    if len(audio) < 2:
        return audio
    pre = np.empty_like(audio)
    pre[0] = audio[0]
    pre[1:] = audio[1:] - alpha * audio[:-1]
    return pre


def normalizar_audio(audio):
    """Normalização mais suave e segura do áudio"""
    max_abs = max(1e-8, float(abs(audio).max()))
    return audio / max_abs * 0.9


def arquivo_local():
    print("\n📂 ARQUIVO LOCAL")
    print("=" * 30)
    print("Exemplos de caminho:")
    print("• C:\\Users\\Usuario\\Downloads\\audio.mp3")
    print("• audio.mp3 (se estiver na mesma pasta)")
    print("• ..\\pasta\\audio.wav")
    
    caminho = input("\n📁 Digite o caminho do arquivo: ").strip()
    caminho = caminho.replace('"', '').replace("'", '')
    
    if not caminho:
        print("❌ Nenhum caminho fornecido")
        return None
    
    if not os.path.exists(caminho):
        print(f"❌ Arquivo não encontrado: {caminho}")
        return None
    
    print(f"🎯 Arquivo selecionado: {caminho}")
    return caminho


def listar_arquivos():
    print("\n📂 ARQUIVOS NA PASTA ATUAL")
    print("=" * 30)
    
    extensoes = ('.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma')
    arquivos_audio = []
    
    for arquivo in os.listdir('.'):
        if arquivo.lower().endswith(extensoes):
            tamanho = os.path.getsize(arquivo) / 1024 / 1024
            arquivos_audio.append((arquivo, tamanho))
    
    if not arquivos_audio:
        print("❌ Nenhum arquivo de áudio encontrado")
        print("💡 Formatos suportados: MP3, WAV, M4A, OGG, FLAC, AAC, WMA")
        return None
    
    print("Arquivos de áudio encontrados:")
    for i, (arquivo, tamanho) in enumerate(arquivos_audio, 1):
        print(f"  {i}. {arquivo} ({tamanho:.1f} MB)")
    
    print(f"\nTotal: {len(arquivos_audio)} arquivo(s)")
    
    try:
        escolha = input(f"\n👉 Digite o número do arquivo (1-{len(arquivos_audio)}) ou Enter para voltar: ").strip()
        if not escolha:
            return None
        indice = int(escolha) - 1
        if 0 <= indice < len(arquivos_audio):
            arquivo_escolhido = arquivos_audio[indice][0]
            print(f"🎯 Arquivo selecionado: {arquivo_escolhido}")
            return arquivo_escolhido
        else:
            print("❌ Número inválido")
            return None
    except ValueError:
        print("❌ Por favor, digite um número válido")
        return None


def selecionar_arquivo():
    print("🎧 SELEÇÃO DE ÁUDIO")
    print("=" * 30)
    print("1. Digitar caminho de um arquivo")
    print("2. Listar arquivos de áudio da pasta atual")
    print("3. Sair")
    
    while True:
        opcao = input("\n👉 Escolha uma opção (1-3): ").strip()
        if opcao == "1":
            return arquivo_local()
        elif opcao == "2":
            return listar_arquivos()
        elif opcao == "3":
            print("👋 Até logo!")
            return None
        else:
            print("❌ Opção inválida. Tente novamente.")


def processar_segmentos(segments):
    textos = []
    timestamps = []
    
    if not segments:
        return "", []
    
    for seg in segments:
        texto = seg.get("text", "").strip()
        if texto:
            textos.append(texto)
            timestamps.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": texto
            })
    
    texto_final = " ".join(textos)
    return texto_final, timestamps


def pos_processar_texto(texto):
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
    
    for errado, correto in correcoes.items():
        texto = texto.replace(errado, correto)
    
    texto = texto.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
    
    while "  " in texto:
        texto = texto.replace("  ", " ")
    
    texto = texto.strip()
    if texto and len(texto) > 1:
        texto = texto[0].upper() + texto[1:]
    
    return texto


def salvar_resultados(texto, timestamps, nome_base, duracao):
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    barra_salvamento = BarraProgresso(2, "Salvando arquivos", 30)
    
    nome_principal = f"TRANSCRICAO_{nome_base}_{timestamp}.txt"
    with open(nome_principal, 'w', encoding='utf-8') as f:
        f.write("TRANSCRIÇÃO DE ALTA PRECISÃO - PORTUGUÊS BRASILEIRO\n")
        f.write(f"Arquivo: {nome_base}\n")
        f.write(f"Duração: {duracao/60:.1f} minutos\n")
        f.write(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
        f.write("="*50 + "\n\n")
        f.write(texto)
    barra_salvamento.atualizar(1)
    
    nome_timestamps = f"TIMESTAMPS_{nome_base}_{timestamp}.txt"
    with open(nome_timestamps, 'w', encoding='utf-8') as f:
        for ts in timestamps:
            f.write(f"[{ts['start']:.1f}s - {ts['end']:.1f}s] {ts['text']}\n")
    barra_salvamento.atualizar(1)
    
    palavras = len(texto.split())
    print(f"\n💾 ARQUIVOS SALVOS:")
    print(f"   📄 {nome_principal}")
    print(f"   ⏱️ {nome_timestamps}")
    print(f"\n📊 ESTATÍSTICAS:")
    print(f"   • Palavras: {palavras}")
    print(f"   • Caracteres: {len(texto)}")
    print(f"   • Segmentos: {len(timestamps)}")
    print(f"   • Duração áudio: {duracao/60:.1f} min")
    
    print(f"\n📄 PRÉVIA (primeiras 400 caracteres):")
    print("-" * 50)
    print(texto[:400] + "..." if len(texto) > 400 else texto)
    print("-" * 50)


BASE_PROMPT = (
    "Transcrição em português brasileiro formal, com pontuação correta, "
    "acentuação adequada e frases completas. Use nomes próprios, siglas e "
    "termos técnicos conforme aparecem no áudio. Evite inventar trechos. "
    "Mantenha siglas como foram faladas, como CMQ, AMTQ, STRAAF, Sintevitraver etc."
)


def escolher_modelo():
    print("\n🔧 Configurações de qualidade do modelo:")
    print("1. 🚀 Rápido        (tiny)")
    print("2. ⚖️ Balanceado    (base)")
    print("3. 🎯 Preciso       (small)")
    print("4. 🏆 Alta qualidade (medium)")
    print("5. 🏅 Máxima precisão (large-v3)")
    
    escolha = input("👉 Escolha o modelo (1-5, Enter = 4): ").strip() or "4"
    
    mapa = {
        "1": "tiny",
        "2": "base",
        "3": "small",
        "4": "medium",
        "5": "large-v3",
    }
    
    modelo = mapa.get(escolha, "medium")
    print(f"\n✅ Modelo selecionado: {modelo}")
    return modelo


def escolher_backend():
    print("\n⚙️ Backend de inferência:")
    print("   1) Whisper original (PyTorch)")
    print("   2) Faster-Whisper (CPU/CUDA, otimizado)")
    print("   3) OpenVINO (Intel GPU/NPU, quando disponível)")
    escolha = input("👉 Escolha o backend [1-3] (Enter = 2): ").strip() or "2"
    
    if escolha not in {"1", "2", "3"}:
        escolha = "2"
    
    if escolha == "2" and not HAVE_FASTER_WHISPER:
        print("⚠️ faster-whisper não está instalado. Rode: pip install faster-whisper")
        print("   Voltando para Whisper original.")
        return "whisper"
    
    if escolha == "3" and not HAVE_OPENVINO:
        print("⚠️ OpenVINO/Optimum não estão instalados.")
        print("   Rode: pip install 'optimum[openvino]' transformers openvino openvino-dev")
        print("   Voltando para faster-whisper (se disponível) ou whisper.")
        if HAVE_FASTER_WHISPER:
            return "faster"
        return "whisper"
    
    backend = {"1": "whisper", "2": "faster", "3": "openvino"}[escolha]
    print(f"\n✅ Backend selecionado: {backend}")
    return backend


def preparar_openvino_model(modelo_escolhido: str):
    # Por enquanto usamos sempre o tiny otimizado em OpenVINO
    model_id = "OpenVINO/whisper-tiny-fp16-ov"
    print(f"\n🔧 Carregando modelo OpenVINO: {model_id}")
    
    ie = Core()
    disponiveis = ie.available_devices
    print("   Dispositivos OpenVINO:", disponiveis)
    
    processor = AutoProcessor.from_pretrained(model_id)
    ov_model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, compile=False)
    
    # Prioridade: GPU > NPU > CPU
    if any("GPU" in d for d in disponiveis):
        device_target = "GPU"
    elif any("NPU" in d for d in disponiveis):
        device_target = "NPU"
    else:
        device_target = "CPU"
    
    print(f"   Enviando modelo para: {device_target}")
    ov_model.to(device_target)
    ov_model.compile()
    print(f"✅ Modelo OpenVINO compilado em {device_target}")
    
    return processor, ov_model, device_target


def transcrever_com_precisao():
    print("🎯 TRANSCRIÇÃO HIGH-ACCURACY PT-BR (GPU/Intel se disponível)")
    print("=" * 70)
    
    caminho_audio = selecionar_arquivo()
    if not caminho_audio:
        return
    
    if torch.cuda.is_available():
        DEVICE = "cuda"
        fp16 = True
        print("\n💻 PyTorch: usando GPU (CUDA)")
        print("   GPU:", torch.cuda.get_device_name(0))
    else:
        DEVICE = "cpu"
        fp16 = False
        print("\n💻 PyTorch: usando CPU (CUDA não disponível)")

    modelo_escolhido = escolher_modelo()
    backend = escolher_backend()
    
    # Configuração base para decodificação com foco em qualidade
    CONFIG_TRANSCRICAO_BASE = {
        "language": "pt",
        "task": "transcribe",
        "temperature": [0.0, 0.2, 0.4],
        "best_of": 5,
        "beam_size": None,          # beam search desativado quando usamos multi-temperatura
        "patience": None,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.3,
        "condition_on_previous_text": False,
        "initial_prompt": BASE_PROMPT,
        "fp16": fp16,
    }
    
    try:
        print(f"\n✅ Processando: {os.path.basename(caminho_audio)}")
        
        try:
            n_threads = max(1, os.cpu_count() or 4)
            torch.set_num_threads(n_threads)
            print(f"🧠 Threads CPU configuradas: {n_threads}")
        except Exception as e:
            print(f"⚠️ Não consegui ajustar threads da CPU: {e}")
        
        print("\n🔧 Pré-processando áudio...")
        audio, sr_original = librosa.load(caminho_audio, sr=None, mono=True)
        
        # Normalização mais suave e segura
        audio = normalizar_audio(audio)
        
        # Pré-ênfase opcional
        if USE_PREEMPHASIS:
            audio = aplicar_pre_enfase(audio)
        
        # Padronizar para 16 kHz
        if sr_original != 16000:
            audio = librosa.resample(audio, orig_sr=sr_original, target_sr=16000)
            sr = 16000
        else:
            sr = sr_original
        
        duracao_total = len(audio) / sr
        print(f"📊 Duração total: {duracao_total/60:.1f} minutos ({duracao_total:.1f} s)")
        
        CHUNK_DURACAO_SEG = 120
        amostras_por_chunk = int(CHUNK_DURACAO_SEG * sr)
        n_chunks = max(1, (len(audio) + amostras_por_chunk - 1) // amostras_por_chunk)
        
        print(f"🔪 Áudio será dividido em {n_chunks} parte(s) de até {CHUNK_DURACAO_SEG} segundos")
        
        # Carrega modelo conforme backend
        if backend == "openvino":
            processor_ov, ov_model, ov_device = preparar_openvino_model(modelo_escolhido)
            print(f"🔥 OpenVINO rodando em: {ov_device}")
            model = None  # só pra manter a variável
        elif backend == "faster":
            compute_type = "int8" if DEVICE == "cpu" else "float16"
            print(f"\n🔧 Carregando modelo faster-whisper {modelo_escolhido} em {DEVICE} ({compute_type})...")
            model = WhisperModel(modelo_escolhido, device=DEVICE, compute_type=compute_type)
            print("✅ Modelo faster-whisper carregado.")
        else:
            print(f"\n🔧 Carregando modelo Whisper {modelo_escolhido} em {DEVICE}...")
            model = whisper.load_model(modelo_escolhido, device=DEVICE)
            print("✅ Modelo Whisper carregado.")
        
        print("\n🎯 Iniciando transcrição em chunks de 2 minutos...")
        inicio = time.time()
        
        todos_segments = []
        total_amostras = len(audio)
        barra_chunks = BarraProgresso(total_amostras, "Transcrevendo áudio", 40)
        
        for i in range(n_chunks):
            start_sample = i * amostras_por_chunk
            end_sample = min(len(audio), (i + 1) * amostras_por_chunk)
            chunk_audio = audio[start_sample:end_sample]
            chunk_amostras = end_sample - start_sample
            
            offset_segundos = start_sample / sr
            inicio_parte = time.time()
            
            print(f"\n📝 Parte {i+1}/{n_chunks} "
                  f"({offset_segundos/60:.1f}–{min(duracao_total, offset_segundos + chunk_amostras/sr)/60:.1f} min)")
            
            if i == 0 or not todos_segments:
                config_chunk = CONFIG_TRANSCRICAO_BASE
            else:
                ultimo_contexto = " ".join(
                    seg.get("text", "").strip()
                    for seg in todos_segments[-12:]
                    if seg.get("text")
                ).strip()
                
                if ultimo_contexto:
                    contexto_prompt = (
                        BASE_PROMPT
                        + " Continue a transcrição a partir do seguinte contexto anterior, "
                          "mantendo a coerência e evitando repetições desnecessárias: "
                        + ultimo_contexto
                    )
                    config_chunk = {
                        **CONFIG_TRANSCRICAO_BASE,
                        "initial_prompt": contexto_prompt,
                    }
                else:
                    config_chunk = CONFIG_TRANSCRICAO_BASE
            
            # BACKEND: OPENVINO
            if backend == "openvino":
                # OpenVINO/Optimum não retorna segments com timestamps, então criamos 1 segmento por chunk
                inputs = processor_ov(
                    chunk_audio,
                    sampling_rate=16000,
                    return_tensors="pt",
                )
                input_features = inputs.input_features
                predicted_ids = ov_model.generate(input_features)
                texto_chunk = processor_ov.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )[0].strip()
                
                segments_chunk = []
                if texto_chunk:
                    segments_chunk.append({
                        "start": float(offset_segundos),
                        "end": float(offset_segundos + chunk_amostras / sr),
                        "text": texto_chunk,
                    })
            
            # BACKEND: FASTER-WHISPER
            elif backend == "faster":
                segments_chunk = []
                segments_iter, info = model.transcribe(
                    chunk_audio,
                    language=config_chunk.get("language", "pt"),
                    beam_size=config_chunk.get("beam_size", 5),
                    best_of=config_chunk.get("best_of", 5),
                    temperature=config_chunk.get("temperature", [0.0, 0.2, 0.4]),
                    initial_prompt=config_chunk.get("initial_prompt"),
                )
                for seg in segments_iter:
                    segments_chunk.append({
                        "start": float(seg.start + offset_segundos),
                        "end": float(seg.end + offset_segundos),
                        "text": seg.text.strip()
                    })
            
            # BACKEND: WHISPER PYTORCH
            else:
                result_chunk = model.transcribe(chunk_audio, **config_chunk)
                segments_raw = result_chunk.get("segments", [])
                segments_chunk = []
                for seg in segments_raw:
                    segments_chunk.append({
                        "start": float(seg["start"] + offset_segundos),
                        "end": float(seg["end"] + offset_segundos),
                        "text": seg["text"].strip()
                    })
            
            trecho_previa = " ".join(
                seg.get("text", "").strip()
                for seg in segments_chunk
                if seg.get("text")
            ).strip()
            print("   Prévia:", trecho_previa[:120] + ("..." if len(trecho_previa) > 120 else ""))
            
            todos_segments.extend(segments_chunk)
            
            barra_chunks.atualizar(chunk_amostras)
            print(f"   ⏱ Tempo da parte: {time.time() - inicio_parte:.1f}s")
        
        print("\n✅ Transcrição em chunks finalizada.")
        print(f"⏱ Tempo total: {time.time() - inicio:.1f}s")
        
        texto_bruto, timestamps = processar_segmentos(todos_segments)
        texto_final = pos_processar_texto(texto_bruto)
        
        if not texto_final.strip():
            print("⚠️ Nenhum texto reconhecido. Verifique o áudio.")
            return
        
        print("\n📄 Trecho inicial da transcrição:")
        print("-" * 50)
        print(texto_final[:500] + ("..." if len(texto_final) > 500 else ""))
        print("-" * 50)
        
        nome_base, _ = os.path.splitext(os.path.basename(caminho_audio))
        salvar_resultados(texto_final, timestamps, nome_base, duracao_total)
        
        print("\n🎉 Finalizado com sucesso!")
        print("   ✅ Alta precisão (beam search + prompt)")
        print("   ✅ Chunk de 2 minutos para não estourar memória")
        print("   ✅ Contexto entre chunks via prompt")
        print("   ✅ Arquivos salvos com timestamps")
        
        return texto_final
    
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    transcrever_com_precisao()
