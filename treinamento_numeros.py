import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report

# ==============================================================================
# 1. CARREGAMENTO COM TRATAMENTO DO CANAL ALFA (32 BITS)
# ==============================================================================
diretorio_base = './dataset_formatado'
tamanho_imagem = (28, 28)
tamanho_lote = 128

print("\nCarregando dados com canal RGBA (4 canais)...")
# Carregamos com RGBA para capturar os 8 bits do canal Alfa (transparência)
raw_treino = tf.keras.utils.image_dataset_from_directory(
    diretorio_base,
    validation_split=0.2,
    subset="training",
    seed=123,
    color_mode="rgba",
    image_size=tamanho_imagem,
    batch_size=tamanho_lote
)

raw_validacao = tf.keras.utils.image_dataset_from_directory(
    diretorio_base,
    validation_split=0.2,
    subset="validation",
    seed=123,
    color_mode="rgba",
    image_size=tamanho_imagem,
    batch_size=tamanho_lote
)

nomes_classes = raw_treino.class_names
print(f"Classes detectadas: {nomes_classes}")

# Função para tratar a transparência:
# Se o traço estiver no canal alfa, ele é extraído; se for fundo transparente,
# ele aplica a transparência sobre um fundo padrão para gerar 1 canal nítido em escala de cinza.
def tratar_rgba(imagens, rotulos):
    rgb = imagens[..., :3]
    alfa = imagens[..., 3:] / 255.0
    
    # Se a imagem tiver traço baseado em transparência (comum em 32-bit):
    # Mescla o RGB com o canal Alfa
    cinza = tf.image.rgb_to_grayscale(rgb) * alfa
    
    # Caso a imagem original fosse com traço escuro em fundo transparente ou vice-versa:
    # Se a média for quase nula, o desenho estava 100% no próprio canal alfa:
    canal_alfa = imagens[..., 3:]
    imagem_final = tf.where(tf.reduce_mean(cinza) < 1.0, canal_alfa, cinza)
    
    return imagem_final, rotulos

dataset_treino = raw_treino.map(tratar_rgba)
dataset_validacao = raw_validacao.map(tratar_rgba)

AUTOTUNE = tf.data.AUTOTUNE
dataset_treino = dataset_treino.cache().prefetch(buffer_size=AUTOTUNE)
dataset_validacao = dataset_validacao.cache().prefetch(buffer_size=AUTOTUNE)

# ==============================================================================
# 2. ARQUITETURA DA CNN MULTICLASSE (1 CANAL - 28x28x1)
# ==============================================================================
print("\nConstruindo a arquitetura da Rede Neural Convolucional (CNN)...")
modelo = models.Sequential([
    # Rescaling na entrada monocromática de 28x28x1
    layers.Rescaling(1./255, input_shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),

    layers.Dense(10, activation='softmax')
])

modelo.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

modelo.summary()

# ==============================================================================
# 3. TREINAMENTO
# ==============================================================================
epocas = 4
print(f"\nIniciando o treinamento por {epocas} épocas...")

historico = modelo.fit(
    dataset_treino,
    validation_data=dataset_validacao,
    epochs=epocas
)

nome_modelo = 'meu_modelo_numeros.keras'
modelo.save(nome_modelo)
print(f"\nTreinamento concluído! Modelo salvo como '{nome_modelo}'.")

# ==============================================================================
# 4. GRÁFICOS DE ACURÁCIA E LOSS
# ==============================================================================
print("\nGerando gráficos de desempenho...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(historico.history['accuracy'], label='Treinamento', marker='o')
plt.plot(historico.history['val_accuracy'], label='Validação', marker='o')
plt.title('Acurácia ao longo das Épocas')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(historico.history['loss'], label='Treinamento', marker='o')
plt.plot(historico.history['val_loss'], label='Validação', marker='o')
plt.title('Erro (Loss) ao longo das Épocas')
plt.xlabel('Época')
plt.ylabel('Erro')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('grafico_desempenho_numeros.png', dpi=300)
plt.show()

# ==============================================================================
# 5. MATRIZ DE CONFUSÃO E RELATÓRIO
# ==============================================================================
print("\nCalculando a Matriz de Confusão no conjunto de validação...")

rotulos_reais = []
predicoes_prob = []

for imagens, rotulos in dataset_validacao:
    rotulos_reais.extend(rotulos.numpy())
    preds = modelo.predict(imagens, verbose=0)
    predicoes_prob.extend(preds)

rotulos_reais = np.array(rotulos_reais)
predicoes_prob = np.array(predicoes_prob)

classes_preditas = np.argmax(predicoes_prob, axis=1)

matriz = confusion_matrix(rotulos_reais, classes_preditas)

plt.figure(figsize=(8, 7))
sns.heatmap(
    matriz, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=nomes_classes,
    yticklabels=nomes_classes,
    cbar=False
)
plt.title('Matriz de Confusão - Reconhecimento de Dígitos (0-9)', fontsize=13)
plt.xlabel('Classe Prevista pela Rede', fontsize=11)
plt.ylabel('Classe Real (Gabarito)', fontsize=11)
plt.tight_layout()
plt.savefig('matriz_confusao_numeros.png', dpi=300)
plt.show()

print("\n--- Relatório Detalhado de Métricas ---")
print(classification_report(rotulos_reais, classes_preditas, target_names=nomes_classes))