import pandas as pd


def interpolate_annual_to_quarterly_data(file_path, start_date, output_path):
    # Завантаження даних з файлу
    data = pd.read_csv(file_path, names=['Data'], header=None, skiprows=1)
    data['Data'] = data['Data'].str.replace(',', '.').astype(float)

    # Створення індексу дати з річною частотою
    dates = pd.date_range(start=start_date, periods=len(data), freq='A')
    data.index = pd.DatetimeIndex(dates)

    # Розширення індексу до квартальних значень
    quarterly_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='Q')
    data = data.reindex(quarterly_index)

    # Інтерполяція для отримання квартальних значень
    data['Data'] = data['Data'].interpolate()

    # Збереження результату у новий CSV файл
    data.to_csv(output_path)

    return data


if __name__ == '__main__':
    # Параметри скрипта
    start_date = '2010-01-01'  # Дата початку даних
    file_path = '/Users/dstakhurskyi/Downloads/Macro Draft Data - Аркуш1-4.csv'
    output_path = '/Users/dstakhurskyi/PycharmProjects/arimaModeling/output/result.csv'

    # Виконання інтерполяції
    interpolated_data = interpolate_annual_to_quarterly_data(file_path, start_date, output_path)

    print("Перші кілька рядків інтерпольованих квартальних даних:")
    print(interpolated_data.head())
