# Dex Pilot (user_sat)

Приложение для распознавания эмоций в реальном времени с веб-камеры. Система детектирует лицо, классифицирует эмоцию и считает метрику **удовлетворённости** пользователя на основе накопленных данных за сессию.

## Возможности

- Захват видео с камеры и отображение кадра
- Детекция лица и построение face mesh через MediaPipe
- Классификация 7 эмоций: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`
- Отображение текущей эмоции и уверенности модели на видеопотоке
- Накопительный анализ сессии: позитив, негатив, итоговая удовлетворённость
- Поддержка GPU: Apple MPS, CUDA или CPU

## Архитектура

```
Веб-камера (OpenCV)
        ↓
   Кадр BGR → flip → RGB
        ↓
MediaPipe Face Landmarker  →  landmarks
        ↓
crop_face (копия ROI)  →  MobileNetV3-Small  →  softmax → эмоция
        ↓
draw_mesh / UI overlay (только для отображения)
        ↓
PyQt6 интерфейс
```

### Расчёт удовлетворённости

Для каждого кадра (не чаще 4 раз в секунду) считается:

| Эмоция   | Валентность |
|----------|-------------|
| happy    | +1.0        |
| surprise | +0.5        |
| neutral  |  0.0        |
| fear     | -0.7        |
| disgust  | -0.8        |
| sad      | -1.0        |
| angry    | -1.0        |

- **Позитив** — сумма вероятностей `happy` + `surprise`
- **Негатив** — сумма вероятностей `angry`, `disgust`, `fear`, `sad`
- **Удовлетворённость кадра** = `50 + 50 × valence`, где valence — взвешенная сумма по всем классам
- **Итог сессии** — среднее по всем обработанным кадрам

## Структура проекта

```
user_sat/
├── src/
│   ├── main.py          # модель, MediaPipe, инференс, точка входа
│   ├── interface.py     # PyQt6 UI
│   └── train.py         # обучение модели на датасете
├── models/
│   ├── emotion_model.pth   # веса MobileNetV3 (нужны для запуска)
│   └── classes.json        # список классов эмоций
├── face_landmarker.task    # модель MediaPipe (скачать отдельно)
├── requirements.txt
└── README.md
```

## Установка

1. Клонируйте репозиторий:

```bash
git clone https://github.com/artemkuzmin72/user_sat
cd user_sat
```

2. Создайте и активируйте виртуальное окружение:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

3. Установите зависимости:

```bash
pip install -r requirements.txt
```

4. Убедитесь, что в корне проекта лежат:
   - `face_landmarker.task`
   - `models/emotion_model.pth`
   - `models/classes.json`

## Запуск

Из корня проекта:

```bash
python src/main.py
```

## Стек технологий

| Библиотека | Назначение |
|------------|------------|
| **PyTorch** + **torchvision** | Классификация эмоций (MobileNetV3-Small) |
| **MediaPipe** | Детекция лица и landmarks |
| **OpenCV** | Захват видео, обработка кадров |
| **PyQt6** | Десктопный интерфейс |
