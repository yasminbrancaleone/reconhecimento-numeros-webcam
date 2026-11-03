import cv2
import numpy as np
import tensorflow as tf

# ==============================================================================
# 1. CARREGAMENTO DO MODELO TREINADO
# ==============================================================================
nome_modelo = 'meu_modelo_numeros.keras'
print(f"Carregando o modelo '{nome_modelo}'...")
modelo = tf.keras.models.load_model(nome_modelo)
print("Modelo carregado! Abrindo a webcam...")

# ==============================================================================
# 2. INICIALIZAÇÃO DA CÂMERA
# ==============================================================================
captura = cv2.VideoCapture(0)

while True:
    sucesso, frame = captura.read()
    if not sucesso:
        print("Erro ao acessar a webcam.")
        break

    # ATENÇÃO: NÃO aplicamos cv2.flip aqui. 
    # O dígito precisa ser lido na orientação original (não-espelhada).
    
    altura, largura, _ = frame.shape

    # Define uma caixa quadrada de captura no centro da imagem (ROI)
    tamanho_box = 200
    x1 = int((largura - tamanho_box) / 2)
    y1 = int((altura - tamanho_box) / 2)
    x2 = x1 + tamanho_box
    y2 = y1 + tamanho_box

    # Extrai apenas a região interna da caixa
    roi = frame[y1:y2, x1:x2]

    # ==============================================================================
    # 3. PRÉ-PROCESSAMENTO DO DÍGITO
    # ==============================================================================
    # Converte para escala de cinza
    cinza = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Suaviza ruídos da câmera
    borrado = cv2.GaussianBlur(cinza, (5, 5), 0)

    # Binarização com Otsu: isola o traço escuro da caneta do fundo claro do papel
    # e inverte para traço branco em fundo preto (padrão de treinamento)
    _, binario = cv2.threshold(borrado, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Redimensiona para 28x28 (tamanho esperado pela CNN)
    redimensionado = cv2.resize(binario, (28, 28), interpolation=cv2.INTER_AREA)

    # Adiciona as dimensões de lote e canal: (1, 28, 28, 1)
    tensor_entrada = np.expand_dims(redimensionado, axis=(0, -1)).astype(np.float32)

    # ==============================================================================
    # 4. INFERÊNCIA MULTICLASSE
    # ==============================================================================
    previsoes = modelo.predict(tensor_entrada, verbose=0)
    classe_predita = int(np.argmax(previsoes[0]))
    confianca = float(np.max(previsoes[0])) * 100.0

    # ==============================================================================
    # 5. RETORNO VISUAL NA TELA
    # ==============================================================================
    # Desenha a caixa central onde o número deve ser apontado
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Posicione o numero aqui", (x1 - 20, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Exibe o resultado da predição
    texto_resultado = f"Digito: {classe_predita} ({confianca:.1f}%)"
    cv2.putText(frame, texto_resultado, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 3)

    cv2.putText(frame, "Pressione 'Q' para sair", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Exibe o frame da webcam e uma mini janela com a imagem processada em 28x28
    cv2.imshow('Classificador de Numeros - Webcam', frame)
    cv2.imshow('Visao da Rede (28x28)', cv2.resize(binario, (140, 140)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()