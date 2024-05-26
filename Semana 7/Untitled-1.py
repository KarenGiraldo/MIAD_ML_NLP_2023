
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset

import nltk
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
import re

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

import spacy

# Descargar las stopwords si es la primera vez que usas nltk
nltk.download('stopwords')

# Lista de stop words 
stop_words = set(stopwords.words('english'))

# Función para eliminar stop words
def remove_stop_words(text):
    # Dividir el texto en palabras
    words = re.findall(r'\b\w+\b', text.lower())
    # Filtrar las palabras que no están en las stop words
    filtered_words = [word for word in words if word not in stop_words]
    # Unir las palabras filtradas de nuevo en una oración
    return ' '.join(filtered_words)

# Descargar recursos de nltk
nltk.download('punkt')


# Descargar los recursos necesarios
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')

# Cargar el modelo en inglés de spaCy
nlp = spacy.load("en_core_web_sm")

# Función para lematizar una frase
def lemmatize_sentence(sentence, nlp):
    # Procesar la frase con spaCy
    doc = nlp(sentence)
    # Extraer las lemas
    lemmas = [token.lemma_ for token in doc]
    # Unir las lemas en una frase
    lemmatized_sentence = " ".join(lemmas)
    return lemmatized_sentence


# Cargar el dataframe
allData = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTraining.zip', encoding='UTF-8', index_col=0)

# Tokenizar las tramas de las películas
#allData['plot'] = allData['plot'].apply(lambda x: ' '.join(word_tokenize(x.lower())))
# Aplicar la función a la columna 'plot'
#allData['plot'] = allData['plot'].apply(remove_stop_words)
#allData['plot'] = allData['plot'].apply(lambda x: lemmatize_sentence(x, nlp))

# Preparar los datos
X = allData['plot'].tolist()
y = allData['genres'].map(lambda x: eval(x))  # Asegurarse de que los géneros están en formato de lista



# Convertir las etiquetas a formato binarizado
mlb = MultiLabelBinarizer()
y_binarized = mlb.fit_transform(y)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_binarized, test_size=0.01, random_state=42)

# Definir una clase de Dataset para la compatibilidad con Hugging Face
class MovieGenreDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(labels, dtype=torch.float)
        return item

# Preparar el tokenizer y el dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = MovieGenreDataset(X_train, y_train, tokenizer, max_length=128)
test_dataset = MovieGenreDataset(X_test, y_test, tokenizer, max_length=128)

# Configurar el modelo
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(mlb.classes_))

# Configurar el Trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Entrenar el modelo
trainer.train()

# Evaluar el modelo
#predictions, labels, _ = trainer.predict(test_dataset)
#predictions = torch.sigmoid(torch.tensor(predictions)).numpy()

# Calcular el AUC macro
#auc_macro = roc_auc_score(y_test, predictions, average='macro')
#print(f'Macro AUC: {auc_macro}')

# transformación variables predictoras X del conjunto de test
dataTesting = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTesting.zip', encoding='UTF-8', index_col=0)

class MovieGenreTestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        return item

# Tokenizar las tramas de las películas
Xp = dataTesting['plot'].tolist()
Xp_dataset = MovieGenreTestDataset(Xp, tokenizer, max_length=128)

# Evaluar el modelo
predictions, labels, _ = trainer.predict(Xp_dataset)
predictions = torch.sigmoid(torch.tensor(predictions)).numpy()

cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']

# Guardar predicciones en formato exigido en la competencia de kaggle
from google.colab import files
res = pd.DataFrame(predictions, index=dataTesting.index, columns=cols)
res.to_csv('/content/pred_genres_text_RF_Bert_E03.csv', index_label='ID')
files.download('/content/pred_genres_text_RF_Bert_E03.csv')
res.head()