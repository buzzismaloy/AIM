# Лабораторная работа 2

## Классификация кардиотокографии плода с использованием нейронных сетей

Этот проект выполняет многоклассовую классификацию данных кардиотокографии плода (Fetal CTG) для предсказания состояния здоровья. Используется нейронная сеть, обучаемая на PyTorch.

[Ссылка на Kaggle](https://www.kaggle.com/datasets/akshat0007/fetalhr)

---

### Основные этапы

1. **Загрузка и предобработка данных**
   - Очистка данных (удаление ненужных столбцов, заполнение пропусков, удаление дубликатов)
   - Анализ корреляций между признаками
   - Визуализация распределения классов

2. **Подготовка данных**
   - Разделение на обучающую и тестовую выборки
   - Нормализация признаков
   - Балансировка классов с **SMOTE** (для устранения дисбаланса)

3. **Модель нейронной сети**
   - Простая **Feedforward**-сеть с одним скрытым слоем
   - Активация **ReLU**
   - Выходной слой с **Softmax** для трех классов

4. **Обучение и оценка модели**
   - Использование **CrossEntropyLoss** и **Adam-оптимизатора**
   - Метрики: **Accuracy, F1-score, ROC-AUC**
   - Построение **матрицы ошибок** и **ROC-кривых**

---

### Зависимости

Проект использует следующие библиотеки:

- `pandas`, `numpy` — для работы с данными
- `matplotlib`, `seaborn` — для визуализации
- `scikit-learn` — для предобработки данных и расчёта метрик
- `torch`, `torch.nn`, `torch.optim` — для построения и обучения нейросетей

---

### Анализ данных

**1. Распределение классов:**

```python
sns.countplot(data=df, x='NSP')
plt.show()
```

**2. Корреляция признаков:**

```python
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()
```

**3. Распределение ключевых признаков:**

```python
plt.figure(figsize=(15, 10))
for i, feature in enumerate(selected_features[:6]):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df[feature], kde=True, bins=30)
    plt.title(f'Распределение {feature}')
plt.tight_layout()
plt.show()
```

---

### Архитектура модели

```python
class FetalHealthModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

---

### Обучение модели

```python
def train_model(model, train_loader, criterion, optimizer, epochs=100):
    losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / len(train_loader))

    plt.plot(range(1, epochs + 1), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()
```

---

### Оценка модели

#### Метрики качества:

```python
def evaluate_model(model, test_loader, class_labels):
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(y_batch.tolist())
            y_pred.extend(predicted.tolist())
            y_scores.extend(torch.softmax(outputs, dim=1).numpy())

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_scores, multi_class='ovr')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1-score: {f1:.4f}')
    print(f'ROC-AUC: {roc_auc:.4f}')
```

#### Матрица ошибок:

```python
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

#### ROC-кривые:

```python
plt.figure(figsize=(8, 6))
for i in range(3):
    fpr, tpr, _ = roc_curve([1 if y == i else 0 for y in y_true], [y[i] for y in y_scores])
    plt.plot(fpr, tpr, label=f'{class_labels[i]}')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()
```

---

### Балансировка классов (SMOTE)

Так как классы несбалансированы, применяем **SMOTE** для генерации дополнительных примеров:

```python
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

---
