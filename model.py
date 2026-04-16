"""
CNN para Classificação de Imagens
Guia didático — TensorFlow/Keras
Jovem Mestre Renan-kun | CHLOE Assistant
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras import layers

from utils.class_manager import carregar_classes, salvar_classes
from utils.plot import plotar_historico, visualizar_matriz_confusao

# ─────────────────────────────────────────
# 0. CONFIGURAÇÕES GLOBAIS
# ─────────────────────────────────────────

DATASET_DIR = "dataset"  # pasta raiz com subpastas por classe
IMG_SIZE = (128, 128)  # todas as imagens viram esse tamanho
BATCH_SIZE = 32  # quantas imagens processa por vez
EPOCHS = 30  # quantas vezes percorre o dataset completo
SEED = 42  # garante reprodutibilidade
MODEL_PATH = "modelo_cnn.keras"  # nome do arquivo de exportação


# ─────────────────────────────────────────
# 1. INGESTÃO E DIVISÃO DOS DADOS
# ─────────────────────────────────────────

# Keras lê a pasta e divide automaticamente em treino/validação
# validation_split=0.2 → 80% treino, 20% validação
# subset="training" / "validation" seleciona qual parte pegar

train_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,  # redimensiona todas para o mesmo tamanho
    batch_size=BATCH_SIZE,
    label_mode="categorical",  # one-hot: [1,0,0,0] em vez de só 0
)

val_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    color_mode="rgb",  # ← e aqui também
)

# Detecta as classes automaticamente (nome das subpastas)
CLASS_NAMES = train_ds.class_names
NUM_CLASSES = len(CLASS_NAMES)
print(f"Classes encontradas: {CLASS_NAMES}")
print(f"Total de classes: {NUM_CLASSES}")


# ─────────────────────────────────────────
# 2. PRÉ-PROCESSAMENTO E PERFORMANCE
# ─────────────────────────────────────────

# AUTOTUNE: deixa o TensorFlow decidir quantos dados pré-carregar
# Isso evita que a GPU fique esperando o disco
AUTOTUNE = tf.data.AUTOTUNE


def preparar_dataset(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Otimiza o carregamento dos dados."""
    return (
        ds.cache()  # guarda na memória após 1ª leitura (mais rápido)
        .shuffle(1000)  # embaralha para não aprender a ordem das imagens
        .prefetch(AUTOTUNE)  # prepara o próximo batch enquanto treina o atual
    )


train_ds = preparar_dataset(train_ds)
val_ds = val_ds.cache().prefetch(AUTOTUNE)  # validação não precisa shuffle


# ─────────────────────────────────────────
# 3. DATA AUGMENTATION (anti-overfitting)
# ─────────────────────────────────────────
# Quando temos poucas imagens, criamos variações artificiais durante o treino.
# Isso ensina o modelo que um gato de cabeça pra baixo ainda é um gato.
# IMPORTANTE: augmentation só é aplicado no treino, NUNCA na validação.

augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),  # espelha horizontalmente
        layers.RandomRotation(0.15),  # gira até 15%
        layers.RandomZoom(0.1),  # zoom de até 10%
        layers.RandomTranslation(0.1, 0.1),  # desloca levemente
        layers.RandomBrightness(0.1),  # varia brilho levemente
    ],
    name="data_augmentation",
)


# ─────────────────────────────────────────
# 4. CONSTRUÇÃO DO MODELO CNN
# ─────────────────────────────────────────
# Arquitetura: Augmentation → Normalização → Blocos Conv → Classificador


def criar_modelo(num_classes: int) -> keras.Model:
    """
    Cria a CNN.
    Entrada: imagem 128x128x3 (RGB)
    Saída:   probabilidade para cada classe
    """

    inputs = keras.Input(shape=(*IMG_SIZE, 3))  # 128x128 pixels, 3 canais RGB

    # ── Pré-processamento dentro do modelo ──────────────────────────────────
    x = augmentation(inputs)  # aplica variações aleatórias
    x = layers.Rescaling(1.0 / 255)(x)  # normaliza pixels: 0–255 → 0.0–1.0

    # ── Bloco Conv 1: detecta bordas e formas simples ───────────────────────
    # Conv2D: filtros que "varrem" a imagem procurando padrões
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)  # estabiliza o aprendizado
    x = layers.MaxPooling2D()(x)  # reduz o tamanho pela metade
    x = layers.Dropout(0.2)(x)  # desliga 20% dos neurônios → evita overfitting

    # ── Bloco Conv 2: detecta texturas e padrões mais complexos ─────────────
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)

    # ── Bloco Conv 3: detecta partes do objeto (orelhas, rodas, etc.) ───────
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    # ── Classificador final ──────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D()(x)  # resume cada mapa de feature em 1 número
    x = layers.Dense(256, activation="relu")(x)  # camada densa para combinar tudo
    x = layers.Dropout(0.4)(x)  # dropout maior antes da saída
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    # softmax → probabilidades que somam 1.0: ex: [0.7, 0.1, 0.1, 0.1]

    return keras.Model(inputs, outputs, name="cnn_classificador")


modelo = criar_modelo(NUM_CLASSES)
modelo.summary()  # exibe a arquitetura completa


# ─────────────────────────────────────────
# 5. COMPILAÇÃO DO MODELO
# ─────────────────────────────────────────
# optimizer: como o modelo aprende (Adam é o mais confiável para iniciantes)
# loss: como medimos o erro (categorical porque usamos one-hot)
# metrics: o que queremos monitorar durante o treino

modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",  # função de erro para múltiplas classes
    metrics=["accuracy"],
)


# ─────────────────────────────────────────
# 6. CALLBACKS (auxiliares do treino)
# ─────────────────────────────────────────
# Callbacks = "assistentes" que observam o treino e agem automaticamente

callbacks = [
    # Salva o melhor modelo automaticamente
    keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        save_best_only=True,  # só salva se melhorou
        monitor="val_accuracy",  # monitora a acurácia na validação
        verbose=1,
    ),
    # Para o treino se não melhorar por N épocas seguidas
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=7,  # aguarda 7 épocas sem melhora antes de parar
        restore_best_weights=True,  # volta para os pesos da melhor época
        verbose=1,
    ),
    # Reduz o learning rate quando o treino estagna
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,  # divide o lr por 2
        patience=3,  # após 3 épocas sem melhora
        min_lr=1e-6,
        verbose=1,
    ),
]


# ─────────────────────────────────────────
# 7. TREINAMENTO
# ─────────────────────────────────────────

print("\n🚀 Iniciando treinamento...\n")

historico = modelo.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)


# ─────────────────────────────────────────
# 8. VISUALIZAÇÃO DO TREINAMENTO
# ─────────────────────────────────────────

plotar_historico(historico)


# ─────────────────────────────────────────
# 9. AVALIAÇÃO DETALHADA
# ─────────────────────────────────────────

print("\n📋 Avaliação no conjunto de validação:")
loss, acc = modelo.evaluate(val_ds, verbose=0)
print(f"   Acurácia: {acc:.2%}")
print(f"   Perda:    {loss:.4f}")

# Relatório por classe (precision, recall, f1)
y_true, y_pred = [], []

for imagens, labels in val_ds:
    previsoes = modelo.predict(imagens, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))  # classe real
    y_pred.extend(np.argmax(previsoes, axis=1))  # classe prevista

print("\n📊 Relatório por Classe:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Matriz de Confusão — mostra onde o modelo erra mais
visualizar_matriz_confusao(y_true, y_pred, CLASS_NAMES)


# ─────────────────────────────────────────
# 10. EXPORTAÇÃO PARA REUSO
# ─────────────────────────────────────────
# O ModelCheckpoint já salvou o melhor modelo durante o treino.
# Vamos também salvar as classes para usar na hora de prever.
salvar_classes(CLASS_NAMES)
print(f"\n✅ Modelo salvo em: {MODEL_PATH}")
print("✅ Classes salvas em: classes.json")


# ─────────────────────────────────────────
# 11. EXEMPLO DE INFERÊNCIA (uso do modelo)
# ─────────────────────────────────────────


def prever_imagem(caminho_imagem: str) -> None:
    """
    Carrega o modelo salvo e classifica uma nova imagem.
    Use esta função depois do treino para testar o modelo.
    """
    # Carrega modelo e classes
    modelo_salvo = keras.models.load_model(MODEL_PATH)
    classes = carregar_classes()
    # Carrega e prepara a imagem
    img = keras.utils.load_img(caminho_imagem, target_size=IMG_SIZE)
    arr = keras.utils.img_to_array(img)  # converte para array numérico
    arr = np.expand_dims(arr, axis=0)  # adiciona dimensão de batch: (1, 128, 128, 3)
    # Nota: não normaliza aqui pois o Rescaling está DENTRO do modelo

    # Previsão
    previsao = modelo_salvo.predict(arr, verbose=0)[0]
    classe_idx = np.argmax(previsao)  # índice da maior probabilidade
    confianca = previsao[classe_idx]

    print(f"\n🔍 Imagem: {caminho_imagem}")
    print(f"   Classe prevista : {classes[classe_idx]}")
    print(f"   Confiança       : {confianca:.2%}")
    print("\n   Probabilidades por classe:")
    for nome, prob in zip(classes, previsao):
        barra = "█" * int(prob * 20)
        print(f"   {nome:15s} {prob:.2%}  {barra}")
    return classes[classe_idx], confianca


# Descomente para testar após o treino:
# prever_imagem("minha_foto.jpg")
