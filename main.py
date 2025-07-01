import os
import zipfile
import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Força uso do backend gráfico Tkinter no Linux/Ubuntu
import matplotlib.pyplot as plt
import cv2
import time
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from PIL import Image

# CONFIGURAÇÕES
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
DATASET_BASE_DIR = '.'  # pasta base para procurar dataset
MODEL_FILENAME = 'garbage_classification_model.h5'


def setup_kaggle_api():
    print("\n--- Configurando API do Kaggle ---")
    try:
        import kaggle
        print("Biblioteca 'kaggle' já está instalada.")
    except ImportError:
        print("Instalando biblioteca 'kaggle'...")
        os.system('pip install kaggle')
        import kaggle

    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')
    os.makedirs(kaggle_dir, exist_ok=True)

    if os.path.exists(kaggle_json_path):
        print("Credenciais do Kaggle encontradas.")
    else:
        print("⚠️  Arquivo 'kaggle.json' não encontrado.")
        print("Baixe em: https://www.kaggle.com/account -> 'Create New API Token'")
        username = input("Digite seu username do Kaggle: ")
        key = input("Digite sua API key do Kaggle: ")
        with open(kaggle_json_path, 'w') as f:
            f.write(f'{{"username":"{username}","key":"{key}"}}')
        os.chmod(kaggle_json_path, 0o600)
        print("Credenciais salvas com sucesso.")


def download_dataset():
    print("\n--- Verificando Dataset ---")
    dataset_path = find_dataset_path()
    if dataset_path:
        print(f"✅ Dataset já encontrado em: {dataset_path}")
        return True

    print("Baixando dataset do Kaggle...")
    os.makedirs('dataset_download_temp', exist_ok=True)
    os.system(f'kaggle datasets download -d sumn2/garbage-classification-v2 -p dataset_download_temp')

    zip_path = os.path.join('dataset_download_temp', 'garbage-classification-v2.zip')
    if not os.path.exists(zip_path):
        print("❌ Erro ao baixar o dataset.")
        return False

    print("Extraindo o dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('dataset_download_temp')

    # Mover para pasta padrão (garbage_dataset)
    if not os.path.exists('garbage_dataset'):
        os.rename(os.path.join('dataset_download_temp', 'Garbage Classification V2'), 'garbage_dataset')
    else:
        print("Pasta 'garbage_dataset' já existe. Verifique manualmente.")

    # Limpar pasta temporária
    for f in os.listdir('dataset_download_temp'):
        path_f = os.path.join('dataset_download_temp', f)
        if os.path.isfile(path_f):
            os.remove(path_f)
        elif os.path.isdir(path_f):
            shutil.rmtree(path_f)
    os.rmdir('dataset_download_temp')

    print("✅ Dataset extraído com sucesso.")
    return True


def find_dataset_path():
    """
    Procura automaticamente o caminho do dataset dentro da pasta base,
    procurando pastas que contenham as categorias esperadas.
    """
    expected_categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    for root, dirs, _ in os.walk(DATASET_BASE_DIR):
        dirs_lower = [d.lower() for d in dirs]
        # Se encontrar todas (ou pelo menos 3) categorias dentro desse diretório
        match_count = sum(cat in dirs_lower for cat in expected_categories)
        if match_count >= 3:
            return root
    return None


def visualize_samples(dataset_path, num_samples=3):
    print("\n--- Visualizando Amostras ---")
    categories = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    fig, axes = plt.subplots(len(categories), num_samples, figsize=(15, 3 * len(categories)))

    for i, category in enumerate(categories):
        cat_path = os.path.join(dataset_path, category)
        images = [f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        samples = random.sample(images, min(num_samples, len(images)))
        for j, img_name in enumerate(samples):
            img_path = os.path.join(cat_path, img_name)
            img = Image.open(img_path)
            ax = axes[i, j] if len(categories) > 1 else axes[j]
            ax.imshow(img)
            ax.set_title(category)
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def train_model(dataset_path):
    print("\n--- Treinando Modelo ---")
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    class_names = list(train_gen.class_indices.keys())
    with open('class_names.txt', 'w') as f:
        f.writelines([f"{c}\n" for c in class_names])

    base_model = applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)
    model.save(MODEL_FILENAME)
    print(f"✅ Modelo salvo como {MODEL_FILENAME}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Perda')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model, class_names


def run_webcam_classification(model=None, class_names=None):
    print("\n--- Classificação em Tempo Real ---")

    # Tenta carregar o modelo se não veio por parâmetro
    if model is None:
        if os.path.exists(MODEL_FILENAME):
            try:
                model = load_model(MODEL_FILENAME)
                print(f"Modelo '{MODEL_FILENAME}' carregado com sucesso.")
            except Exception as e:
                print(f"Erro ao carregar o modelo: {e}")
                return
        else:
            print(f"Arquivo do modelo '{MODEL_FILENAME}' não encontrado. Treine o modelo primeiro.")
            return

    # Tenta carregar as classes se não vieram por parâmetro
    if class_names is None:
        if os.path.exists('class_names.txt'):
            with open('class_names.txt') as f:
                class_names = [line.strip() for line in f]
            print("Nomes das classes carregados.")
        else:
            print("Arquivo 'class_names.txt' não encontrado. Treine o modelo primeiro.")
            return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao acessar a webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar frame da webcam.")
            break

        display = cv2.resize(frame, (640, 480))
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = np.expand_dims(img, axis=0)

        start = time.time()
        pred = model.predict(img, verbose=0)[0]
        end = time.time()

        idx = np.argmax(pred)
        label = class_names[idx]
        conf = pred[idx] * 100
        fps = 1 / (end - start)

        cv2.putText(display, f"{label} ({conf:.1f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Classificador de Lixo", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    print("=== CLASSIFICADOR DE LIXO - VISÃO COMPUTACIONAL ===")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU detectada: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("⚠️  Nenhuma GPU detectada. Usando CPU.")

    while True:
        print("\n=== MENU ===")
        print("1. Configurar API do Kaggle")
        print("2. Baixar Dataset")
        print("3. Visualizar Amostras")
        print("4. Treinar Modelo")
        print("5. Classificar com Webcam")
        print("6. Executar Processo Completo")
        print("0. Sair")

        opcao = input("Escolha: ")

        if opcao == '1':
            setup_kaggle_api()
        elif opcao == '2':
            download_dataset()
        elif opcao == '3':
            path = find_dataset_path()
            if path:
                visualize_samples(path)
            else:
                print("❌ Dataset não encontrado. Baixe-o primeiro com a opção 2.")
        elif opcao == '4':
            path = find_dataset_path()
            if path:
                train_model(path)
            else:
                print("❌ Dataset não encontrado. Baixe-o primeiro com a opção 2.")
        elif opcao == '5':
            run_webcam_classification()
        elif opcao == '6':
            setup_kaggle_api()
            if download_dataset():
                path = find_dataset_path()
                if path:
                    visualize_samples(path)
                    model, class_names = train_model(path)
                    run_webcam_classification(model, class_names)
                else:
                    print("❌ Dataset não encontrado após download.")
        elif opcao == '0':
            print("Encerrando...")
            break
        else:
            print("Opção inválida.")


if __name__ == "__main__":
    main()
