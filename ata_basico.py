# transcrever_high_accuracy.py
import os
import sys
import tempfile
import time
import warnings

# Suprimir avisos específicos do huggingface
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
warnings.filterwarnings("ignore", message=".*huggingface_hub.*")

# FIX para OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from faster_whisper import WhisperModel
import librosa
import soundfile as sf

class BarraProgresso:
    """Classe para exibir barras de progresso no terminal"""
    
    def __init__(self, total, descricao="", comprimento=40):
        self.total = total
        self.descricao = descricao
        self.comprimento = comprimento
        self.atual = 0
        self.inicio_tempo = time.time()
    
    def atualizar(self, progresso=1):
        """Atualiza a barra de progresso"""
        self.atual += progresso
        percentual = min(100, (self.atual / self.total) * 100)
        
        # Calcula barras preenchidas
        barras_preenchidas = int(self.comprimento * self.atual // self.total)
        barra = '█' * barras_preenchidas + '░' * (self.comprimento - barras_preenchidas)
        
        # Calcola tempo decorrido e estimado
        tempo_decorrido = time.time() - self.inicio_tempo
        if self.atual > 0:
            tempo_estimado = (tempo_decorrido / self.atual) * (self.total - self.atual)
        else:
            tempo_estimado = 0
        
        # Formata tempo
        tempo_decorrido_str = self._formatar_tempo(tempo_decorrido)
        tempo_estimado_str = self._formatar_tempo(tempo_estimado)
        
        # Limpa linha e exibe progresso
        sys.stdout.write('\r')
        sys.stdout.write(f"{self.descricao} |{barra}| {percentual:.1f}% ({self.atual}/{self.total}) "
                        f"[{tempo_decorrido_str}<{tempo_estimado_str}]")
        sys.stdout.flush()
        
        if self.atual >= self.total:
            sys.stdout.write('\n')
    
    def _formatar_tempo(self, segundos):
        """Formata tempo em MM:SS ou HH:MM:SS"""
        if segundos < 60:
            return f"{int(segundos)}s"
        elif segundos < 3600:
            return f"{int(segundos//60)}m{int(segundos%60)}s"
        else:
            return f"{int(segundos//3600)}h{int((segundos%3600)//60)}m{int(segundos%60)}s"

def selecionar_arquivo():
    """
    Oferece opções para selecionar o arquivo de áudio
    """
    print("\n📁 SELECIONAR ARQUIVO DE ÁUDIO")
    print("=" * 40)
    print("1. 📂 Usar arquivo local (digitar caminho)")
    print("2. 📤 Fazer upload de arquivo")
    print("3. 🗂️ Listar arquivos na pasta atual")
    print("4. 🚪 Sair")
    
    while True:
        opcao = input("\n👉 Escolha uma opção (1-4): ").strip()
        
        if opcao == "1":
            return arquivo_local()
        elif opcao == "2":
            return upload_arquivo()
        elif opcao == "3":
            return listar_arquivos()
        elif opcao == "4":
            print("👋 Até logo!")
            return None
        else:
            print("❌ Opção inválida. Tente novamente.")

def arquivo_local():
    """
    Solicita o caminho do arquivo local
    """
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

def upload_arquivo():
    """
    Faz upload de arquivo usando interface do Jupyter
    """
    try:
        from IPython.display import display, FileUpload
        import io
        
        print("\n📤 UPLOAD DE ARQUIVO")
        print("=" * 30)
        print("1. Clique no botão 'Selecionar arquivo' abaixo")
        print("2. Escolha seu arquivo de áudio")
        print("3. Aguarde o upload completar")
        print("4. Volte aqui e pressione Enter")
        print("=" * 30)
        
        # Cria widget de upload
        uploader = FileUpload(
            accept='.mp3,.wav,.m4a,.ogg,.flac,.aac,.wma',
            multiple=False,
            description='Selecionar arquivo'
        )
        
        display(uploader)
        
        input("\n⏳ Após selecionar o arquivo, pressione Enter para continuar...")
        
        if not uploader.value:
            print("❌ Nenhum arquivo foi selecionado")
            return None
        
        # Processa o arquivo uploadado
        arquivo_info = list(uploader.value.values())[0]
        nome_arquivo = arquivo_info['metadata']['name']
        conteudo = arquivo_info['content']
        
        print(f"✅ Arquivo recebido: {nome_arquivo}")
        print(f"📊 Tamanho: {len(conteudo) / 1024 / 1024:.2f} MB")
        
        # Salva o arquivo temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{nome_arquivo}") as tmp:
            tmp.write(conteudo)
            caminho_temp = tmp.name
        
        print(f"💾 Arquivo salvo temporariamente: {caminho_temp}")
        return caminho_temp
        
    except ImportError:
        print("❌ IPython não disponível. Use a opção de arquivo local.")
        return None
    except Exception as e:
        print(f"❌ Erro no upload: {e}")
        return None

def listar_arquivos():
    """
    Lista arquivos de áudio na pasta atual
    """
    print("\n🗂️ ARQUIVOS NA PASTA ATUAL")
    print("=" * 40)
    
    # Extensões de áudio suportadas
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

def dividir_inteligente(audio, sr):
    """Divide o áudio em partes menores"""
    partes = []
    duracao_parte = 45 * sr  # 45 segundos (mais contexto)
    
    total_amostras = len(audio)
    for i in range(0, total_amostras, duracao_parte):
        parte = audio[i:i+duracao_parte]
        tempo_inicio = i / sr
        tempo_fim = (i + len(parte)) / sr
        partes.append((parte, f"{tempo_inicio/60:.1f}-{tempo_fim/60:.1f}min"))
    
    return partes

def processar_segmentos(segments):
    """Processa segmentos com validação robusta"""
    textos = []
    timestamps = []
    
    if segments is None:
        return "", []
    
    for segment in segments:
        if hasattr(segment, 'text') and segment.text:
            texto = segment.text.strip()
            if texto and len(texto) > 1:
                textos.append(texto)
                timestamps.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': texto
                })
    
    texto_final = " ".join(textos)
    return texto_final, timestamps

def pos_processar_texto(texto):
    """Aplica correções pós-transcrição"""
    correcoes = {
        " pq ": " porque ",
        " tb ": " também ",
        " vc ": " você ",
        " d ": " de ",
        " q ": " que ",
    }
    
    for errado, correto in correcoes.items():
        texto = texto.replace(errado, correto)
    
    # Limpar espaços duplos que podem aparecer
    while "  " in texto:
        texto = texto.replace("  ", " ")
    
    return texto

def salvar_resultados(texto, timestamps, nome_base, duracao):
    """Salva resultados com múltiplos formatos"""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Barra de progresso para salvamento
    barra_salvamento = BarraProgresso(2, "Salvando arquivos", 30)
    
    # Arquivo principal
    nome_principal = f"TRANSCRICAO_{nome_base}_{timestamp}.txt"
    with open(nome_principal, 'w', encoding='utf-8') as f:
        f.write(f"TRANSCRIÇÃO DE ALTA PRECISÃO\n")
        f.write(f"Arquivo: {nome_base}\n")
        f.write(f"Duração: {duracao/60:.1f} minutos\n")
        f.write(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
        f.write("="*50 + "\n\n")
        f.write(texto)
    barra_salvamento.atualizar(1)
    
    # Arquivo com timestamps
    nome_timestamps = f"TIMESTAMPS_{nome_base}_{timestamp}.txt"
    with open(nome_timestamps, 'w', encoding='utf-8') as f:
        for ts in timestamps:
            f.write(f"[{ts['start']:.1f}s - {ts['end']:.1f}s] {ts['text']}\n")
    barra_salvamento.atualizar(1)
    
    # Estatísticas
    palavras = len(texto.split())
    print(f"\n💾 ARQUIVOS SALVOS:")
    print(f"   📄 {nome_principal}")
    print(f"   ⏱️ {nome_timestamps}")
    print(f"\n📊 ESTATÍSTICAS:")
    print(f"   • Palavras: {palavras}")
    print(f"   • Caracteres: {len(texto)}")
    print(f"   • Segmentos: {len(timestamps)}")
    print(f"   • Duração áudio: {duracao/60:.1f} min")
    
    # Preview
    print(f"\n📄 PRÉVIA (primeiras 400 caracteres):")
    print("-" * 50)
    print(texto[:400] + "..." if len(texto) > 400 else texto)
    print("-" * 50)

def transcrever_com_precisao():
    """
    Versão com máxima acurácia e interface de seleção de arquivo
    """
    print("🎯 TRANSCRIÇÃO HIGH-ACCURACY COM UPLOAD")
    print("=" * 55)
    
    # Selecionar arquivo
    caminho_audio = selecionar_arquivo()
    if not caminho_audio:
        return
    
    # Verificar se é arquivo temporário de upload
    arquivo_temporario = "temp" in caminho_audio.lower()
    
    try:
        print(f"\n✅ Processando: {os.path.basename(caminho_audio)}")
        
        # CONFIGURAÇÕES DE PRECISÃO
        print("\n🔧 Configurações de qualidade:")
        print("1. 🚀 Rápido (tiny) - Menos preciso")
        print("2. ⚖️ Balanceado (base) - Bom equilíbrio")  
        print("3. 🎯 Preciso (small) - Alta qualidade")
        print("4. 🏆 Máxima precisão (medium) - Melhor qualidade")
        
        modelo_opcao = input("Escolha o modelo (1-4, padrão=4): ").strip() or "4"
        
        modelos = {
            "1": "tiny",
            "2": "base", 
            "3": "small",
            "4": "medium"
        }
        
        MODELO = modelos.get(modelo_opcao, "medium")
        print(f"🎯 Usando modelo: {MODELO}")
        
        # CONFIGURAÇÕES OTIMIZADAS PARA CPU/QUALIDADE
        CONFIG_TRANSCRICAO = {
            "language": "pt",
            "task": "transcribe",
            "initial_prompt": "Transcrição em português do Brasil, com frases completas e pontuação adequada.",
            "beam_size": 6,
            "best_of": 6,
            "patience": 3,
            "temperature": [0.0, 0.2],
            "compression_ratio_threshold": 2.6,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.35,
            "condition_on_previous_text": False,
            "vad_filter": True,
            "vad_parameters": {
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 300
            },
        }
        
        # 1. CARREGAR E PRÉ-PROCESSAR ÁUDIO
        print("\n🔧 Pré-processando áudio...")
        audio, sr_original = librosa.load(caminho_audio, sr=None, mono=True)
        
        # Normalização de volume (evita fala muito baixa)
        max_abs = max(1e-8, float(abs(audio).max()))
        audio = audio / max_abs * 0.9
        
        if sr_original != 16000:
            audio = librosa.resample(audio, orig_sr=sr_original, target_sr=16000)
            sr = 16000
        else:
            sr = sr_original
        
        duracao_total = len(audio) / sr
        print(f"📊 Duração: {duracao_total/60:.1f} minutos")
        
        # 2. CARREGAR MODELO PRECISO
        print(f"\n🔧 Carregando modelo {MODELO}...")
        print("⏳ Isso pode levar alguns minutos...")
        
        # Barra de progresso para carregamento do modelo
        barra_carregamento = BarraProgresso(100, "Carregando modelo", 30)
        for i in range(100):
            time.sleep(0.02)
            barra_carregamento.atualizar(1)
        
        model = WhisperModel(
            MODELO,
            device="cpu",
            compute_type="float32",
            cpu_threads=os.cpu_count() or 4
        )
        
        # 3. DIVISÃO INTELIGENTE
        print("✂️ Dividindo em partes...")
        partes = dividir_inteligente(audio, sr)
        print(f"📦 Partes criadas: {len(partes)}")
        
        # 4. TRANSCRIÇÃO DE PRECISÃO
        print("\n🎯 Iniciando transcrição...")
        texto_completo = ""
        timestamps = []
        
        # Barra de progresso principal
        barra_principal = BarraProgresso(len(partes), "Transcrevendo partes", 40)
        
        offset_global = 0.0  # em segundos
        
        for i, (parte_audio, parte_info) in enumerate(partes, 1):
            print(f"\n📝 Parte {i}/{len(partes)} - {parte_info}")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                sf.write(tmp.name, parte_audio, sr)
                temp_path = tmp.name
            
            try:
                inicio_parte = time.time()
                
                # TRANSCRIÇÃO
                segments, info = model.transcribe(temp_path, **CONFIG_TRANSCRICAO)
                texto_parte, timestamps_parte = processar_segmentos(segments)
                
                # Ajusta timestamps para o tempo global do áudio
                for ts in timestamps_parte:
                    ts['start'] += offset_global
                    ts['end'] += offset_global
                
                tempo_parte = time.time() - inicio_parte
                
                print(f"  ✅ Concluída em {tempo_parte:.1f}s")
                if texto_parte.strip():
                    print(f"  📄 {texto_parte[:80]}...")
                else:
                    print(f"  📄 Sem texto detectado")
                
                texto_completo += texto_parte + " "
                timestamps.extend(timestamps_parte)
                
                # Atualiza offset global (duração desta parte em segundos)
                offset_global += len(parte_audio) / sr
                
                # Atualiza barra de progresso
                barra_principal.atualizar(1)
                
            except Exception as e:
                print(f"  ⚠️ Erro na parte {i}: {e}")
                continue
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        # 5. PÓS-PROCESSAMENTO
        print("\n🔧 Aplicando pós-processamento...")
        barra_pos_processamento = BarraProgresso(3, "Pós-processamento", 30)
        texto_completo = pos_processar_texto(texto_completo)
        barra_pos_processamento.atualizar(1)
        
        time.sleep(0.5)
        barra_pos_processamento.atualizar(1)
        
        time.sleep(0.5)
        barra_pos_processamento.atualizar(1)
        
        # 6. SALVAR COM METADADOS
        nome_base = os.path.basename(caminho_audio).split('.')[0]
        salvar_resultados(texto_completo, timestamps, nome_base, duracao_total)
        
        print(f"\n🎉 TRANSCRIÇÃO CONCLUÍDA COM SUCESSO!")
        
        return texto_completo
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Limpar arquivo temporário se foi upload
        if arquivo_temporario and os.path.exists(caminho_audio):
            try:
                os.unlink(caminho_audio)
                print("🧹 Arquivo temporário removido")
            except:
                pass

if __name__ == "__main__":
    transcrever_com_precisao()
