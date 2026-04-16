# ─────────────────────────────────────────
# 8. VISUALIZAÇÃO DO TREINAMENTO
# ─────────────────────────────────────────


import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def plotar_historico(hist):
    """Plota acurácia e perda ao longo das épocas."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epocas = range(1, len(hist.history["accuracy"]) + 1)

    # Gráfico de Acurácia
    ax1.plot(epocas, hist.history["accuracy"], label="Treino")
    ax1.plot(epocas, hist.history["val_accuracy"], label="Validação")
    ax1.set_title("Acurácia por Época")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Acurácia")
    ax1.legend()
    ax1.grid(True)

    # Gráfico de Perda
    ax2.plot(epocas, hist.history["loss"], label="Treino")
    ax2.plot(epocas, hist.history["val_loss"], label="Validação")
    ax2.set_title("Perda por Época")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Perda")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle("Histórico de Treinamento", fontsize=14)
    plt.tight_layout()
    plt.savefig("historico_treino.png", dpi=150)
    plt.show()
    print("📊 Gráfico salvo em historico_treino.png")


def visualizar_matriz_confusao(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Matriz de Confusão")
    plt.ylabel("Real")
    plt.xlabel("Previsto")
    plt.tight_layout()
    plt.savefig("matriz_confusao.png", dpi=150)
    plt.show()
    print("📊 Matriz salva em matriz_confusao.png")
