# Лабораторная работа №4

## Классификация бабочек с использованием пуллинга и аугментации

Этот проект выполняет задачу классификации изображений бабочек с использованием современных методов машинного обучения, включая пуллинг и аугментацию данных.

[Ссылка на датасет](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification/data)

### Используемые методы
- **Свёрточные нейросети (CNN)**
- **Макспуллинг (MaxPooling)**
- **Аугментация данных** (вращение, сдвиги, масштабирование и отражение)
- **Использование GPU для ускорения обучения**

---

## Основные этапы

### 1. Исследование и подготовка данных
- Загрузка данных и их анализ
- Визуализация распределения классов
- Применение методов предобработки изображений

### 2. Аугментация данных
- Вращение, сдвиги, масштабирование, отражение
- Использование `ImageDataGenerator` для расширения обучающего набора

### 3. Создание и обучение модели
- Использование `TensorFlow` и `Keras`
- Архитектура модели с сверточными слоями, пуллингом и плотными слоями
- Компиляция и обучение модели с функцией потерь `categorical_crossentropy`

### 4. Оценка модели
- Построение графиков точности и потерь
- Визуализация матрицы ошибок
- Анализ ошибок классификации

---

## Зависимости

```bash
pip install tensorflow numpy pandas seaborn matplotlib scikit-learn
```

Используемые библиотеки:
- `numpy`, `pandas` — работа с данными
- `seaborn`, `matplotlib` — визуализация данных
- `tensorflow`, `keras` — построение и обучение нейросети
- `sklearn` — предобработка данных и оценка модели

---

## Архитектура модели

```python
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(data['label'].nunique(), activation='softmax')
])
```

Компиляция модели:

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## Обучение модели

```python
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    steps_per_epoch=int(train_data.n//32),
    validation_steps=int(val_data.n//32)
)
```

Графики обучения:

```python
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
```

---

## Оценка модели

Матрица ошибок:

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

Примеры предсказаний:

```python
num_samples = 25
random_indices = random.sample(range(len(val_data.filenames)), num_samples)

plt.figure(figsize=(15, num_samples * 3))
for i, idx in enumerate(random_indices):
    img_path = os.path.join('/content/butterfly/train', val_data.filenames[idx])
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)
    pred_class = np.argmax(model.predict(img_array_exp, verbose=0))
    true_class = y_true[idx]
    pred_label = labels[pred_class]
    true_label = labels[true_class]
    plt.subplot(num_samples, 3, i * 3 + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Pred: {pred_label}\nTrue: {true_label}', color='green' if pred_class == true_class else 'red')
plt.tight_layout()
plt.show()
```

---

