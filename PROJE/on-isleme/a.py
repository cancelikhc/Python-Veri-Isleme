import re
from zemberek import TurkishMorphology, TurkishSentenceNormalizer

morphology = TurkishMorphology.create_with_defaults()

def normalize_turkish_text(text):
    normalizer = TurkishSentenceNormalizer(morphology)
    normalized_text = normalizer.normalize(text)
    return normalized_text

def extract_root_neg_sentence(sentence, morphology):
    words = sentence.split()
    roots = []
    for word in words:
        analyses = morphology.analyze(word)
        root = ""
        for analysis in analyses:
            root = analysis.get_stem()
            for morpheme in analysis.get_morphemes():
                if morpheme.name == "Negative":
                    root += "NEG"
        roots.append(root)
    return ' '.join(roots)

def detect_emotion_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002600-\U000027BF"
                               u"\U0001f300-\U0001f64F"
                               u"\U0001f680-\U0001f6FF"
                               u"\U0001f900-\U0001f9ff"
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'POSEMOTION ', text)
    return text


def preprocess_text(text):
    text = normalize_turkish_text(text)
    text = detect_emotion_emojis(text)
    text = remove_punctuation(text)
    text = remove_extra_spaces(text)
    text = extract_root_neg_sentence(text, morphology)
    return text

def process_and_save_texts(source_file, target_file):
    with open(source_file, 'r', encoding='utf-8') as file:
        texts = file.readlines()

    with open(target_file, 'w', encoding='utf-8') as file:
        for text in texts:
            cleaned_text = preprocess_text(text)
            file.write(cleaned_text + '\n')

def remove_extra_spaces(text):
    return ' '.join(text.split())

def remove_punctuation(text):
    punctuation_marks = ['.', ',', ';', '?', '!']
    for mark in punctuation_marks:
        text = text.replace(mark, '')
    return text


# Paths for the source and target files
source_file_path = r"C:/Users/LENOVO\Desktop/ÖDEV-1/veri seti/pozitif_yorumlar.txt"
target_file_path = r"C:/Users/LENOVO\Desktop/ÖDEV-1/yeni_pozitif"
process_and_save_texts(source_file_path, target_file_path)
print("pozitif işlem tamamlandı")

source_file_path = r"C:/Users/LENOVO\Desktop/ÖDEV-1/veri seti/negatif_yorumlar.txt"
target_file_path = r"C:/Users/LENOVO\Desktop/ÖDEV-1/yeni_negatif.txt"
process_and_save_texts(source_file_path, target_file_path)
print("pozitif işlem tamamlandı")