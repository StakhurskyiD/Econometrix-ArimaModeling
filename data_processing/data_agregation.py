
import pandas as pd

def aggregate_inflation_data(file_path, start_date, output_path):
    # Завантаження даних з файлу
    data = pd.read_csv(file_path, names=['Data'], header=None, skiprows=1)

    # Перетворення значень інфляції у числовий формат
    data['Data'] = data['Data'].str.replace(',', '.').astype(float)

    # Створення індексу дати
    dates = pd.date_range(start=start_date, periods=len(data), freq='M')
    data.index = pd.DatetimeIndex(dates)

    # Агрегація даних по кварталах
    quarterly_data = data.resample('Q').mean()

    # Збереження результату у новий CSV файл
    quarterly_data.to_csv(output_path)

    return quarterly_data

if __name__ == '__main__':
    # Параметри скрипта
    start_date = '2010-01-01'  # Дата початку даних
    file_path = '/Users/dstakhurskyi/Downloads/Macro Draft Data - Аркуш1-3.csv'

    output_path = '/Users/dstakhurskyi/PycharmProjects/arimaModeling/output/result.csv'
    # Виконання агрегації
    quarterly_data = aggregate_inflation_data(file_path, start_date, output_path)

    print("Перші кілька рядків агрегованих даних:")
    print(quarterly_data.head())



