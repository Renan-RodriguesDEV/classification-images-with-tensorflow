import json


def salvar_classes(class_names):
    # Salva os nomes das classes em ordem
    with open("classes.json", "w") as f:
        json.dump(class_names, f)


def carregar_classes():
    # Carrega os nomes das classes
    with open("classes.json") as f:
        classes = json.load(f)
    return classes
