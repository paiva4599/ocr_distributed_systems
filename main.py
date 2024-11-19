import time
start_time = time.time()
import os
import cv2
import pytesseract
import functions
from pyspark import SparkContext
from difflib import SequenceMatcher


# Configuração do pytesseract para o idioma português
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Caminho para o executável do Tesseract em Linux
caminho = r"C:\Users\paiva\OneDrive\Área de Trabalho\sistemasDistribuidos\tessdata"
tessdata_dir_config = f'--oem 3 --psm 6 --tessdata-dir {caminho}'  
# Caminho para os dados de treinamento
language = 'por+lat'  # Configura o idioma como português


# Inicializar contexto do Spar
# sc = SparkContext("spark://IP", "OCR Preprocessing")
sc = SparkContext("local", "OCR Preprocessing")


# Caminho para a pasta com imagens
image_folder = r"C:\Users\paiva\OneDrive\Área de Trabalho\sistemasDistribuidos\non-processed"


# Caminho para a pasta que armazena os textos de referência esperados para cada imagem
reference_text_folder = r"C:\Users\paiva\OneDrive\Área de Trabalho\sistemasDistribuidos\_textsreference"
# Caminho para o modelo de super-resolução ESPCN
super_res_model_path = r"C:\Users\paiva\OneDrive\Área de Trabalho\sistemasDistribuidos\models\LapSRN_x2.pb"  # Altere para o caminho do seu modelo


# Função para calcular a similaridade entre dois textos
def calculate_accuracy(extracted_text, reference_text):
    return SequenceMatcher(None, extracted_text, reference_text).ratio()


# Função para aplicar super-resolução usando OpenCV
def apply_super_resolution(image, model_path):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("lapsrn", 2)  # x2 é a escala do modelo
    upscaled_image = sr.upsample(image)
    return upscaled_image


# Função para pré-processar e executar OCR em uma imagem
def preprocess_and_ocr(image_path):
    # Carregar a imagem em escala de cinza
    image = cv2.imread(image_path)
    upscale = apply_super_resolution(image, super_res_model_path)
   
    cv2.imwrite(fr"C:\Users\paiva\OneDrive\Área de Trabalho\sistemasDistribuidos\processed\1upscale_{os.path.basename(image_path)}", upscale)
   
    cinza = functions.get_grayscale(upscale)
    cv2.imwrite(fr"C:\Users\paiva\OneDrive\Área de Trabalho\sistemasDistribuidos\processed\2cinza_{os.path.basename(image_path)}", cinza)
       
    # Executar o OCR com Tesseract para o idioma português e configurações personalizadas
    extracted_text = pytesseract.image_to_string(cinza, lang=language, config=tessdata_dir_config)
   
    # Carregar texto de referência correspondente
    reference_text_path = os.path.join(reference_text_folder, os.path.splitext(os.path.basename(image_path))[0] + '.txt')
    if os.path.exists(reference_text_path):
        with open(reference_text_path, 'r', encoding='utf-8') as ref_file:
            reference_text = ref_file.read()
    else:
        reference_text = ""  # Se não houver texto de referência, deixa como string vazia

    # Calcular a acurácia do OCR
    accuracy = calculate_accuracy(extracted_text, reference_text)
   
    # Retornar o caminho da imagem processada, o texto extraído e a acurácia
    return (image_path, extracted_text, accuracy)

# Listar todas as imagens na pasta
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

# Carregar lista de imagens em um RDD (Resilient Distributed Dataset)
images_rdd = sc.parallelize(image_paths)

# Aplicar a função de pré-processamento e OCR em paralelo usando Map
results = images_rdd.map(preprocess_and_ocr).collect()

# Mostrar os caminhos das imagens pré-processadas, o texto extraído e as acurácias calculadas
for processed_image, extracted_text, accuracy in results:
    print(f"Imagem: {processed_image}\nTexto Extraído:\n{extracted_text}\nAcurácia: {accuracy:.2f}\n")

execution_time = time.time() - start_time
print("--- %s segundos ---" % execution_time)
