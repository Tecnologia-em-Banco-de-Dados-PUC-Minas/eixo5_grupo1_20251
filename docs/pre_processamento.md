# ETAPA 3 - Pré-Processamento

Nesta etapa do trabalho, realizaremos uma coleta de dados de imagens de RM para, por meio do machine learning, buscar identificar padrões complexos que possam indicar a presença de tumores cerebrais e auxiliar na classificação em diferentes tipos.

O objetivo é avaliar a eficácia das técnicas e ferramentas de aprendizado de máquina na previsão de diagnósticos, buscando demonstrar a precisão e a rapidez no processo de detecção de achados críticos nas imagens.

Uma das etapas necessárias para utilizar o algoritmo de machine learning em imagens é o pré-processamento. Como as imagens, em sua essência, são compostas por pixels, é necessário convertê-los em representações numéricas adequadas.
Esse processo envolve diversas técnicas, como normalização, redimensionamento e outras transformações que visam otimizar os dados para o treinamento dos modelos.

Para esta etapa do processo, utilizamos a ferramenta TensorFlow, que oferece um módulo dedicado ao pré-processamento de imagens, automatizando as etapas de conversão e preparação dos dados de imagem.
O módulo de pré-processamento do TensorFlow nos permite alimentar diretamente as imagens de RM em nossos modelos de machine learning, com a garantia de que foram aplicadas as transformações necessárias para a sua correta interpretação numérica.



```#Importando as bibliotecas básicas

import tensorflow as tf
import pandas as pd
import numpy as np
import os

#Importando bibliotecas especificas do tensorflow
from tensorflow.keras.preprocessing import image_dataset_from_directory

DIR_BASE = '/kaggle/input/brain-tumor-mri-dataset'
DIR_TREINO = os.path.join(DIR_BASE, 'Training')
DIR_TESTE = os.path.join(DIR_BASE, 'Testing')
classes = ['glioma', 'meningioma', 'pituitary', 'notumor']

TAMANHO_IMG = (224, 224)
TAMANHO_BATCH = 32


#Utilizando biblioteca do keras para pré-processamento das imagens e transformar em datasets

ds_treino = tf.keras.preprocessing.image_dataset_from_directory(
    DIR_TREINO,
    image_size=TAMANHO_IMG,
    batch_size=TAMANHO_BATCH,
    validation_split=0.15,  # Vamos separar 15% do dataset para validação do treino
    subset="training",
    seed=42, # Colocando um seed fixo para garantir a divisão correta
    shuffle=True  
)

ds_validacao = tf.keras.preprocessing.image_dataset_from_directory(
    DIR_TREINO,
    image_size=TAMANHO_IMG,
    batch_size=TAMANHO_BATCH,
    validation_split=0.15, 
    subset="validation",
    seed=42, 
    shuffle=True
)

ds_teste = tf.keras.preprocessing.image_dataset_from_directory(
    DIR_TESTE,
    image_size=TAMANHO_IMG,
    batch_size=TAMANHO_BATCH,
    shuffle=False  
)

nomes_classes = ds_treino.class_names
print("Nome das classes:", nomes_classes)```
