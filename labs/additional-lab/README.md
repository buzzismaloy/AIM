# Дополнительная лабораторная работа

## Анализ данных опоссумов: классификация пола и предсказание длины головы

Этот проект выполняет анализ данных о мордочках опоссумов.
Основные задачи:
1. **Классификация пола опоссума** (мужской/женский) с помощью нейронной сети.
2. **Регрессионное предсказание длины головы** на основе других признаков.

[Ссылка на датасет](https://www.kaggle.com/datasets/abrambeyer/openintro-possum)

---

### Основные этапы

1. Разведочный анализ данных (EDA)
    - Загрузка и предварительная очистка данных (`pandas`).
    - Визуализация распределений (`matplotlib`, `seaborn`).
    - Построение тепловой карты корреляции.

2. Предобработка данных
    - Заполнение пропусков и кодирование категориальных переменных (`LabelEncoder`).
    - Нормализация признаков (`StandardScaler`).
    - Разделение на обучающую и тестовую выборки.

3. Классификация пола (Neural Network)
    - Архитектура модели: **3 полносвязных слоя** с `ReLU`-активацией и `Sigmoid` на выходе.
    - Функция потерь: `BCELoss` (бинарная кроссэнтропия).
    - Оптимизатор: `Adam` (скорость обучения = 0.001).
    - Метрики качества: `accuracy_score`, `F1-score`, `ROC-AUC`.
    - Построение **ROC-кривой** и анализ качества модели.

4. Регрессионное предсказание длины головы
    - Архитектура модели: **3 полносвязных слоя** с `ReLU`-активацией.
    - Функция потерь: `MSELoss` (среднеквадратичная ошибка).
    - Метрики: `MSE`, `MAE`, `R²`.
    - Визуализация **кривой обучения** модели.

---

### Зависимости

Проект использует следующие библиотеки:

- **Обработка данных**: `pandas`, `numpy`
- **Машинное обучение**: `scikit-learn`
- **Визуализация**: `matplotlib`, `seaborn`
- **Глубокое обучение**: `torch`, `torch.nn`, `torch.optim`

---

### Анализ данных

Для понимания данных строятся графики:

- **Распределение возраста особей опоссумов**.
- **Распределение полов** (мужской/женский).
- **Распределение популяций** (разделение на группы).
- **Тепловая карта корреляции признаков** для понимания взаимосвязей.

Пример кода для одной из визуализаций:

```python
sns.histplot(data['age'], bins=10, kde=True, ax=axes[0])
axes[0].set_title('Распределение возраста особей')
axes[0].set_xlabel('Возраст')
axes[0].set_ylabel('Количество')
```

---

### Модель классификации пола опоссумов

Для задачи классификации пола опоссумов используется простая нейронная сеть с тремя слоями:

1. Входной слой — принимает данные с размерностью входных признаков.
2. Два скрытых слоя — с использованием ReLU активации для нелинейности.
3. Выходной слой — с сигмоидной функцией активации, которая преобразует выход в значение от 0 до 1 для бинарной классификации.

#### Код модели:

```python
class PossumClassifier(nn.Module):
    def __init__(self, input_dim):
        super(PossumClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
```

---

### Модель регрессии мордочки опоссумов

Для задачи регрессии мордочки опоссумов также используется нейронная сеть, но с целью предсказания непрерывного значения (длины головы). Модель состоит из трех слоев с ReLU активацией на каждом скрытом слое.

#### Код модели:

```python
class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

---

### Обучение моделей

#### Процесс обучения

Для обеих задач обучения используется оптимизатор **Adam** и функция потерь в зависимости от задачи:

- Для классификации: **BCELoss** (Binary Cross-Entropy Loss).
- Для регрессии: **MSELoss** (Mean Squared Error Loss).

Модели обучаются на протяжении **100 эпох**. За каждый шаг расчёт ошибки и корректировка параметров модели через оптимизатор.

#### Код обучения:

```python
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    model.train()
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Оценка loss на тестовой выборке
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        model.train()

    return train_losses, test_losses
```

---

### Оценка моделей

#### Оценка классификационной модели

Для оценки качества модели классификации используется несколько метрик:

- **Accuracy** — точность классификации.
- **F1-Score** — гармоническое среднее точности и полноты.
- **ROC-AUC** — площадь под ROC-кривой для оценки качества бинарного классификатора.

##### Код оценки классификации:

```python
def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()
            preds = (outputs >= 0.5).float()
            all_labels.extend(labels.numpy())
            all_preds.extend(outputs.numpy())
    accuracy = accuracy_score(all_labels, (np.array(all_preds) >= 0.5).astype(int))
    f1 = f1_score(all_labels, (np.array(all_preds) >= 0.5).astype(int))
    roc_auc = roc_auc_score(all_labels, all_preds)
    return accuracy, f1, roc_auc, all_labels, all_preds
```

---

#### Оценка регрессионной модели

Для регрессии модель оценивается с использованием метрик:

- **Mean Squared Error (MSE)** — среднеквадратичная ошибка, измеряет, насколько точны предсказания.
- **Mean Absolute Error (MAE)** — средняя абсолютная ошибка, более устойчива к выбросам.
- **R² (коэффициент детерминации)** — процент объяснённой дисперсии.

##### Код оценки регрессии:

```python
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()  # Предсказания модели
        predictions = scaler_y.inverse_transform(predictions)  # Обратное преобразование
        y_test_original = scaler_y.inverse_transform(y_test)  # Обратное преобразование истинных значений
        mse = mean_squared_error(y_test_original, predictions)
        mae = mean_absolute_error(y_test_original, predictions)
        r2 = r2_score(y_test_original, predictions)
        print(f'MSE on test data: {mse:.4f}')
        print(f'MAE on test data: {mae:.4f}')
        print(f'R^2 on test data: {r2:.4f}')
        return mse, mae, r2, predictions
```

---

### Визуализация

#### Кривая обучения

Для отслеживания прогресса обучения модели строится кривая потерь на обучающих и тестовых данных.

##### Код визуализации:

```python
plt.figure(figsize=(10, 5))
plt.plot(range(1, 101), train_losses, label='Training Loss', color='blue')
plt.plot(range(1, 101), test_losses, label='Test Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss Curve')
plt.legend()
plt.show()
```

#### ROC-Кривая

Для модели классификации строится ROC-кривая, чтобы оценить её способность различать классы.

##### Код визуализации:

```python
fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
```
