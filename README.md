# CNN para Classificação de Imagens

Uma implementação educacional completa de uma Rede Neural Convolucional (CNN) usando TensorFlow e Keras para classificação de imagens. Projeto didático que abrange desde o carregamento de dados até a avaliação do modelo com métricas detalhadas.

## 🎯 Características

- **Arquitetura CNN**: Modelo sequencial com camadas convolucionais, pooling e densa
- **Pré-processamento de Dados**: Redimensionamento, normalização e augmentação de imagens
- **Divisão Train/Validation**: Separação automática de dados (80/20)
- **Avaliação Completa**: Classification Report, Confusion Matrix e métricas de desempenho
- **Visualizações**: Histórico de treinamento e matriz de confusão
- **Código Didático**: Comentários detalhados em cada seção explicando cada passo

## 📋 Requisitos

- Python >= 3.12
- pip ou conda

## 🚀 Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/Renan-RodriguesDEV/classification-images-with-tensorflow.git
cd cnn-model
```

### 2. Crie um ambiente virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### Dependências principais

- **TensorFlow 2.17.0**: Framework de Deep Learning
- **NumPy 2.4.4**: Computação numérica
- **Scikit-learn 1.8.0**: Métricas e avaliação
- **Matplotlib 3.10.8**: Visualizações
- **Seaborn 0.13.2**: Gráficos estatísticos

## 📁 Estrutura do Projeto

```
cnn-model/
├── main.py                 # Script principal
├── model.py                # Implementação do modelo CNN
├── requirements.txt        # Dependências do projeto
├── pyproject.toml         # Configuração do projeto
├── README.md              # Este arquivo
├── LICENSE                # Licença MIT
├── dataset/               # Pasta com dados de treinamento
│   └── classe_1/          # Subpasta de cada classe
│   └── classe_2/
│   └── ...
├── notebooks/             # Jupyter notebooks para análise
└── utils/
    └── plot.py            # Funções de visualização
```

## 📚 Estrutura do Dataset

O dataset deve estar organizado em subpastas, uma para cada classe:

```
dataset/
├── gatos/
│   ├── imagem1.jpg
│   ├── imagem2.jpg
│   └── ...
├── cachorros/
│   ├── imagem1.jpg
│   ├── imagem2.jpg
│   └── ...
└── passaros/
    ├── imagem1.jpg
    └── ...
```

## 💻 Como Usar

### 1. Prepare seus dados

Organize suas imagens na pasta `dataset/` com subpastas para cada classe.

### 2. Configure os parâmetros (opcional)

Edite `model.py` para ajustar:

```python
IMG_SIZE = (128, 128)      # Tamanho das imagens
BATCH_SIZE = 32            # Tamanho do lote
EPOCHS = 30                # Número de épocas
SEED = 42                  # Seed para reprodutibilidade
```

### 3. Treine o modelo

```bash
python model.py
```

O modelo será treinado e salvo em `modelo_cnn.keras`

### 4. Use o script principal

```bash
python main.py
```

## 📊 Saída Esperada

Após o treinamento, você terá:

1. **Arquivo do Modelo**: `modelo_cnn.keras` - Modelo treinado salvo
2. **Gráficos de Desempenho**:
   - Histórico de perda e acurácia
   - Matriz de confusão
3. **Métricas de Avaliação**:
   - Precision, Recall e F1-Score por classe
   - Acurácia geral
   - Relatório de classificação completo

## 🔧 Detalhes da Implementação

### Pipeline de Dados

```
Imagens Brutas
    ↓
Redimensionamento (IMG_SIZE)
    ↓
Normalização
    ↓
Divisão Train/Validation (80/20)
    ↓
Batch Processing (BATCH_SIZE)
```

### Arquitetura do Modelo

O modelo sequencial inclui:

1. **Camadas Convolucionais**: Extração de features
2. **Max Pooling**: Redução de dimensionalidade
3. **Flatten**: Conversão para vetor 1D
4. **Camadas Densas**: Classificação
5. **Dropout**: Prevenção de overfitting
6. **Softmax**: Normalização de probabilidades

### Métricas de Avaliação

- **Acurácia**: Porcentagem de predições corretas
- **Precision**: Corretude das predições positivas
- **Recall**: Cobertura de instâncias positivas
- **F1-Score**: Média harmônica entre Precision e Recall

## 🎓 Conteúdo Educacional

Este projeto foi desenvolvido como guia didático abrangendo:

- Carregamento e pré-processamento de imagens
- Construção de modelos CNN do zero
- Treinamento e validação
- Avaliação com métricas detalhadas
- Visualização de resultados
- Boas práticas em Deep Learning

## 🚨 Troubleshooting

### Erro: "Module 'tensorflow' has no attribute 'keras'"

```bash
pip install --upgrade tensorflow
```

### Erro: Dataset não encontrado

Certifique-se de que a pasta `dataset/` existe e contém subpastas com as imagens.

### GPU não está sendo usada

Instale a versão com suporte CUDA:

```bash
pip install tensorflow[and-cuda]
```

## 📝 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👤 Autor

**Renan Rodrigues** - [GitHub](https://github.com/Renan-RodriguesDEV)

## 🤝 Contribuições

Contribuições são bem-vindas! Para contribuir:

1. Faça um Fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## ⭐ Se este projeto foi útil, considere deixar uma estrela!

---

**Versão**: 0.1.0  
**Última atualização**: Abril de 2026
