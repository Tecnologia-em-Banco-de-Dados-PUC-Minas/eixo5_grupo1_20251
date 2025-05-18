# Etapa 4: Aprendizagem de Máquina

Nesta etapa iniciamos a aplicação dos modelos de aprendizagem de máquina para analisar as imagens de ressonância já pre-processados na etapa anterior.

Na nossa pesquisa, identificamos que um dos melhores algoritmos para realizar a análise de imagens são as redes neurais convolucionais, tambem chamadas de CNN.
A escolha se deu pelas CNNs terem dentro das suas caracterisicas as camadas de convoluções, que permite que o modelo detecte caracteristicas como bordas, texturas e formas. 
Também por ser formados por camadas, as CNNs permitem que se desenvolvam diversos tipos de modelo dentro dela, uma vez que as camadas funcionam de forma hierarquica.
Por isso, decidimos desenvolver duas CNNs diferentes: uma elaborada manualmente, e uma usando um modelo já estabelecido conhecido como VGG16.

No modelo manual, escolhemos utilizar 14 camadas na rede neural como demostrado no código abaixo. As duas primeiras camadas são apenas para configurar o input e fazer um redimensionamento da imagem. 
Depois, as próximas 8 camadas são constituídas por uma camada de convolução, que busca identificar os padrões da imagem como explicado anteriormente, seguida de uma camada chamada de pooling.
Essa camada de pooling pega o resultado da camada mapeada pela convolução e reduz o mapa de caracteristicas, selecionando apenas os pontos com maior valor da região mapeada.
Intercalando essas duas camadas permite que o modelo foque cada vez mais em caracteriscias das ressonâncias que sirvam para identificar os tumores.
Após essas 8 etapas, usamos a camada de flatten para transformar essa martiz em um vetor unidimensional, seguida de uma camada densa para fazer uma análise em cima desse vetor.
De modo a evitar um possível overfitting, tambem incluímos uma camada de dropout no final, que desativa aleatoriamente unidades em uma camada, e depois mais uma camada densa para finalizar.

```
# Definindo o Modelo de forma Manual
model = Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1./255),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(4, activation="softmax")
])

# Usando o optimizador Adam do Keras
optimizer = tf.keras.optimizers.Adam()

# Compilando o modelo
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

```

Já o VGG16 é um CNN desenvolvido pelo Visual Geometry Group (VGG) da Universidade de Oxford em 2014. Ele se destaca por ter uma arquitetura simples e homogenea, o que o torna uma das melhores CNNs para análise de imagem.
Ela é composta por 16 camadas, sendo 13 convolucionais e 3 totalmente conectadas. essa profundidade  permite a extração de características hierárquicas, desde bordas simples até padrões complexos, ideal para nosso caso.
Uma das limitações do modelo VGG, é que ele exige muita potência computacional, então, para conseguir rodar nosso modelo, decidimos deixar apenas as ultimas 3 camadas treinaveis.

```
# Arquitetura do modelo
IMAGE_SIZE = 128  # Image size 
base_model = VGG16(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')

# Congelar todas as camadas
for layer in base_model.layers:
    layer.trainable = False

# Habilitando apenas as 3 ultimas para serem treinadas
base_model.layers[-2].trainable = True
base_model.layers[-3].trainable = True
base_model.layers[-4].trainable = True

# Modelo final
model2 = Sequential()
model2.add(Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))  
model2.add(base_model)  # Colocando o VVG16 como a camadas
model2.add(Flatten())  
model2.add(Dropout(0.3))  
model2.add(Dense(128, activation='relu')) 
model2.add(Dropout(0.2))  
model2.add(Dense(len(os.listdir(train_dir)), activation='softmax'))  

# Compilando com Adam
model2.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
```

Para ambos os modelos, colocamos as mesmas funções de paradas antecipada, salvar melhor resultado e redução da taxa de aprendizagem. 
Como o modelo de VGG também exige um poder computacional maior e já é amplamente testado, reduzimos o número de epocas para 5 ao inves de 20.

```
# Definindo função de parada caso o modelo identifique que não está evoluindo entre épocas
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    mode='max',
    verbose=1
)

# Callback para salvar melhor modelo após concluir o treinamento
model_checkpoint = ModelCheckpoint(
    filepath='/kaggle/working/best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=0
)

# Reduz a taxa de aprendizagem quando uma metrica não evolui mais
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=0.00001
)

#MODELO MANUAL
history = model.fit(
    train_ds,
    epochs=20,
    validation_data=val_ds,
    callbacks=[model_checkpoint, early_stopping, lr_schedule],
    verbose=1
)

#MODELO VGG16
history2 = model2.fit(
    train_ds,
    epochs=5,
    validation_data=val_ds,
    callbacks=[model_checkpoint, early_stopping, lr_schedule],
    verbose=1
)
```

Após treinar os dois modelos, realizamos as predições no modelo de testes e calculamos a acuracia, recall, precisão e score F1 de cada modelo para avaliar a sua qualidade.

```
#MODELO MANUAL
y_true = tf.concat([y for _, y in test_ds], axis=0)
y_pred_probs = model.predict(test_ds)  # Probabilities
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert to class labels

test_accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, digits=4, target_names=test_ds.class_names, output_dict=True)

# Extraindo méticas
test_accuracy = report["accuracy"]
precision = report["weighted avg"]["precision"]
recall = report["weighted avg"]["recall"]
f1_score = report["weighted avg"]["f1-score"]

print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")


#MODELO VGG
y_true = tf.concat([y for _, y in test_ds], axis=0)
y_pred_probs = model.predict(test_ds)  # Probabilities
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert to class labels

test_accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, digits=4, target_names=test_ds.class_names, output_dict=True)

# Extraindo méticas
test_accuracy = report["accuracy"]
precision = report["weighted avg"]["precision"]
recall = report["weighted avg"]["recall"]
f1_score = report["weighted avg"]["f1-score"]

print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")

```

Os resultados obtidos foram os seguintes:

Modelo Manual:
- Accuracy: 0.9558
- Precision: 0.9556
- Recall: 0.9558
- F1 Score: 0.9554

Modelo VGG:
- Accuracy: 0.9611
- Precision: 0.9622
- Recall: 0.9611
- F1 Score: 0.9613

Pelas métricas, podemos ver que o modelo VGG performou melhor do que o nosso modelo manual, por uma diferença de aproximadamente 0.006 em todas as metricas. 
Isso tambem mostra a potência de um modelo já testado de mercado como o VGG, que conseguiu um resultado significativo mesmo não utilizando seu máximo potencial e tendo menos épocas de treino.
Porém, o desenvolvimento do nosso modelo manual também se provou bastante efetivo, chegando próximo ao resultado de um modelo amplamente testado, mostrando que sua efetivade foi satisfatória.
