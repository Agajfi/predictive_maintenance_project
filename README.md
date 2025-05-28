# Проект: Предиктивное обслуживание оборудования

## Описание
Streamlit-приложение для прогнозирования отказов оборудования на основе датасета [Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/predictive+maintenance+dataset).

## Установка
```bash
git clone https://github.com/ваш-username/predictive_maintenance_project.git
cd predictive_maintenance_project
pip install -r requirements.txt
```

## Запуск
```bash
streamlit run app.py
```

## Структура проекта
- `app.py` - основной скрипт приложения
- `analysis_and_model.py` - страница анализа и модели
- `presentation.py` - презентация проекта
- `requirements.txt` - зависимости
- `data/` - данные (опционально)
- `video/` - видео-демонстрация

## Использование
1. На главной странице выберите "Анализ и модель"
2. Выберите источник данных:
   - Загрузите CSV-файл
   - Используйте демо-данные из интернета
   - Используйте локальный файл (data/predictive_maintenance.csv)
3. Модель автоматически обучится и покажет результаты
4. Используйте форму для прогнозирования на новых данных

## Видео-демонстрация
Смотрите видео-демонстрацию работы приложения в файле `video/demo.mp4`
