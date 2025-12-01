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
        tempo_decorrido_str = self._formatar_tempo(tempo_decorrido)
        tempo_estimado_str = self._formatar_tempo(tempo_estimado)
        sys.stdout.write('\r')
        sys.stdout.write(
            f"{self.descricao} |{barra}| {percentual:.1f}% ({self.atual}/{self.total}) "
            f"[{tempo_decorrido_str}<{tempo_estimado_str}]"
        )
        sys.stdout.flush()
        if self.atual >= self.total:
            sys.stdout.write('\n')
    
    def _formatar_tempo(self, segundos):
        if segundos < 60:
            return f"{int(segundos)}s"
        elif segundos < 3600:
            return f"{int(segundos//60)}m{int(segundos%60)}s"
        else:
            return f"{int(segundos//3600)}h{int((segundos%3600)//60)}m{int(segundos%60)}s"


def selecionar_arquivo():
    print("\n📁 SELECIONAR ARQUIVO DE ÁUDIO")
    print("=" * 40)
    print("1. 📂 Usar arquivo local (digitar caminho)")
    print("2. 🗂️ Listar arquivos na pasta atual")
    print("3. 🚪 Sair")
    
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
        print("💡 Verifique se o caminho está correto")
        return None
    
    print(f"✅ Arquivo encontrado: {os.path.basename(caminho)}")
    return caminho


def listar_arquivos():
    print("\n🗂️ ARQUIVOS NA PASTA ATUAL")
    print("=" * 40)
    
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


def transcrever_com_precisao():
    print("🎯 TRANSCRIÇÃO HIGH-ACCURACY PT-BR (GPU se disponível)")
    print("=" * 70)
    
    caminho_audio = selecionar_arquivo()
    if not caminho_audio:
        return
    
    if torch.cuda.is_available():
        DEVICE = "cuda"
        fp16 = True
        print("\n💻 Usando GPU (CUDA)")
        print("   GPU:", torch.cuda.get_device_name(0))
    else:
        DEVICE = "cpu"
        fp16 = False
        print("\n💻 Usando CPU (CUDA não disponível)")
    
    try:
        print(f"\n✅ Processando: {os.path.basename(caminho_audio)}")
        
        try:
            n_threads = max(1, os.cpu_count() or 4)
            torch.set_num_threads(n_threads)
            print(f"🧠 Threads CPU configuradas: {n_threads}")
        except Exception as e:
            print(f"⚠️ Não consegui ajustar threads da CPU: {e}")
        
        print("\n🔧 Configurações de qualidade:")
        print("1. 🚀 Rápido (tiny)")
        print("2. ⚖️ Balanceado (base)")
        print("3. 🎯 Preciso (small)")
        print("4. 🏆 Alta qualidade (medium)")
        print("5. 🏅 Máxima precisão (large-v3)")
        
        modelo_opcao = input("Escolha o modelo (1-5, padrão=4): ").strip() or "4"
        modelos = {"1": "tiny", "2": "base", "3": "small", "4": "medium", "5": "large-v3"}
        MODELO = modelos.get(modelo_opcao, "medium")
        print(f"🎯 Usando modelo: {MODELO}")
        
        BASE_PROMPT = (
            "Transcrição em português brasileiro formal, com pontuação correta, "
            "acentuação adequada e frases completas. Use nomes próprios, siglas e "
            "termos técnicos conforme aparecem no áudio. Evite inventar trechos."
        )
        
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
            "condition_on_previous_text": False,  # vamos controlar contexto via prompt
            "verbose": False,
            "fp16": fp16,
            "initial_prompt": BASE_PROMPT,
        }
        
        print("\n🔧 Pré-processando áudio...")
        audio, sr_original = librosa.load(caminho_audio, sr=None, mono=True)
        
        # Normalização mais suave e segura
        max_abs = max(1e-8, float(abs(audio).max()))
        audio = audio / max_abs * 0.9
        
        # Pré-ênfase opcional
        if USE_PREEMPHASIS:
            audio = librosa.effects.preemphasis(audio, coef=0.97)
        
        # Padronizar para 16 kHz
        if sr_original != 16000:
            audio = librosa.resample(audio, orig_sr=sr_original, target_sr=16000)
            sr = 16000
        else:
            sr = sr_original
        
        duracao_total = len(audio) / sr
        print(f"📊 Duração total: {duracao_total/60:.1f} minutos")
        
        # Dividir em partes de 2 minutos
        CHUNK_DURACAO_SEG = 120
        amostras_por_chunk = int(CHUNK_DURACAO_SEG * sr)
        n_chunks = (len(audio) + amostras_por_chunk - 1) // amostras_por_chunk
        
        print(f"🔪 Áudio será dividido em {n_chunks} parte(s) de até {CHUNK_DURACAO_SEG} segundos")
        
        print(f"\n🔧 Carregando modelo {MODELO} em {DEVICE}...")
        model = whisper.load_model(MODELO, device=DEVICE)
        print("✅ Modelo carregado.")
        
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
            
            # Construir prompt com contexto dos últimos segmentos
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
                        + " Contexto anterior da conversa para manter coerência e nomes próprios: "
                        + ultimo_contexto[-400:]
                    )
                else:
                    contexto_prompt = BASE_PROMPT
                
                config_chunk = {
                    **CONFIG_TRANSCRICAO_BASE,
                    "initial_prompt": contexto_prompt,
                }
            
            result_chunk = model.transcribe(chunk_audio, **config_chunk)
            segments_chunk = result_chunk.get("segments", [])
            
            trecho_previa = " ".join(
                seg.get("text", "").strip()
                for seg in segments_chunk
                if seg.get("text")
            ).strip()
            if trecho_previa:
                print(f"   📄 Prévia: {trecho_previa[:120]}...")
            else:
                print("   📄 Prévia: [sem texto detectado]")
            
            for seg in segments_chunk:
                novo_seg = seg.copy()
                novo_seg["start"] = float(seg.get("start", 0.0)) + offset_segundos
                novo_seg["end"] = float(seg.get("end", 0.0)) + offset_segundos
                todos_segments.append(novo_seg)
            
            tempo_parte = time.time() - inicio_parte
            print(f"   ⏱️ Parte concluída em {tempo_parte:.1f}s")
            
            barra_chunks.atualizar(chunk_amostras)
        
        tempo_total = time.time() - inicio
        print(f"\n✅ Transcrição concluída em {tempo_total/60:.1f} minutos de processamento.")
        
        texto_completo, timestamps = processar_segmentos(todos_segments)
        
        print("\n🔧 Aplicando pós-processamento avançado...")
        barra_pos = BarraProgresso(3, "Pós-processamento", 30)
        
        texto_completo = pos_processar_texto(texto_completo)
        barra_pos.atualizar(1)
        time.sleep(0.2)
        
        texto_completo = pos_processar_texto(texto_completo)
        barra_pos.atualizar(1)
        time.sleep(0.2)
        
        texto_completo = texto_completo.strip()
        barra_pos.atualizar(1)
        
        nome_base = os.path.basename(caminho_audio).split('.')[0]
        salvar_resultados(texto_completo, timestamps, nome_base, duracao_total)
        
        print("\n🎉 TRANSCRIÇÃO CONCLUÍDA COM SUCESSO!")
        print("   ✅ Configurações focadas em qualidade para português brasileiro")
        print("   ✅ Contexto entre chunks via prompt")
        print("   ✅ Arquivos salvos com timestamps")
        
        return texto_completo
    
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    transcrever_com_precisao()
