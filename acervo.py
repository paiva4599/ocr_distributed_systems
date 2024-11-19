from difflib import SequenceMatcher

# Função para carregar o conteúdo de um arquivo de texto
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


# Função para calcular a similaridade entre dois textos
def compare_texts(text1, text2):
    matcher = SequenceMatcher(None, text1, text2)
    return matcher.ratio()  # Retorna um valor entre 0 e 1, onde 1 significa texto idêntico


# Caminhos para os arquivos de texto
text_path_1 = '/root/wendy-projects/ocr/acervo/text2.txt'
text_path_2 = '/root/wendy-projects/ocr/reference_texts/text2.txt'


# Carregar os textos dos arquivos
text1 = load_text(text_path_1)
text2 = load_text(text_path_2)


# Calcular a similaridade
similarity = compare_texts(text1, text2)


# Exibir o resultado
print(f"Acurácia acervo: {similarity:.2f}")