# Лабораторная работа 1

## Прогнозирование выживаемости пассажиров Титаника с помощью нейронных сетей

Что ожидается в лабе:
1. EDA (разведочный анализ данных).
2. Попробовать использовать однослойную и многослойную структуру перцептрона (сравнить результаты).
3. Отобразить графики обучения модели и протестировать модели на тестовой выборке.

[Ссылка на Kaggle](https://www.kaggle.com/c/titanic/overview)

### Основные этапы

1. **Загрузка и анализ данных**  
   - Чтение данных из `train.csv`
   - Исследование структуры и первичный анализ с `pandas`
   - Визуализация распределений (`matplotlib`, `seaborn`)

2. **Предобработка данных**  
   - Заполнение пропусков (например, `Age` медианным значением)
   - Кодирование категориальных переменных (`Sex`)
   - Нормализация данных (`StandardScaler`)
   - Разделение на тренировочную и тестовую выборки

3. **Построение нейросетевых моделей**  
   - **Однослойный перцептрон** (простая полносвязная сеть)
   - **Многослойный перцептрон** (глубокая нейросеть с ReLU-активацией)

4. **Обучение моделей**  
   - Использование **Binary Cross Entropy Loss** и **Adam-оптимизатора**
   - Запись значений функции потерь на каждой эпохе

5. **Оценка моделей**  
   - **F1-score** и **ROC-AUC** для сравнения качества
   - Построение **ROC-кривых** для визуального анализа

### Зависимости

Проект использует следующие библиотеки:

- `pandas`, `numpy` — для работы с данными
- `matplotlib`, `seaborn` — для визуализации
- `scikit-learn` — для предобработки данных и расчёта метрик
- `torch`, `torch.nn`, `torch.optim` — для построения и обучения нейросетей

### Анализ данных

Для понимания данных строятся графики:

- **Распределение выживаемости пассажиров**
- **Влияние класса билета на выживаемость**
- **Влияние пола на выживаемость**
- **Распределение возраста среди пассажиров**

Пример кода для одной из визуализаций:

```python
sns.countplot(x="Survived", data=df, palette="Set2")
plt.show()
```

### Архитектура моделей

#### Однослойный перцептрон (Single Layer NN)

```python
class SingleLayerNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(len(features), 1)
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))
```

#### Многослойный перцептрон (Multi Layer NN)

```python
class MultiLayerNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(len(features), 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
```

### Обучение моделей

Обучение проходит в **100 эпох** с оптимизатором **Adam** и функцией потерь **BCELoss**:

```python
def train_model(model, X_train, y_train, X_test, y_test, epochs=100, lr=0.01):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
```

### Оценка моделей

Оценка качества с метриками **F1-score** и **ROC-AUC**:

```python
def calculate_metrics(model, X_test, y_test, threshold=0.75):
    with torch.no_grad():
        probs = model(X_test).numpy()
        preds = (probs >= threshold).astype(int)
        return f1_score(y_test, preds), roc_auc_score(y_test, probs)
```

#### ROC-кривая

```python
plt.plot(fpr, tpr, label=f"Model (AUC = {roc_auc_score(y_test, probabilities):.4f})")
```
