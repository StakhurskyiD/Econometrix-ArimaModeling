import pandas as pd

def interpolate_annual_to_quarterly_data(input_file_path, output_file_path):
    # Load the data from CSV
    data = pd.read_csv(input_file_path, header=None, names=['Year', 'Data'])

    # Clean and convert 'Data' column to float
    data['Data'] = data['Data'].str.replace(' ', '').str.replace(',', '.').astype(float)

    # Convert 'Year' to datetime format representing the end of each year
    data['Year'] = pd.to_datetime(data['Year'].astype(str) + '-12-31')

    # Set 'Year' as the index
    data.set_index('Year', inplace=True)

    # Generate a complete date range that includes all quarters
    complete_dates = pd.date_range(start=data.index.min(), end=data.index.max(), freq='Q')

    # Reindex the original data to this complete range, introducing NaNs for new dates
    complete_data = data.reindex(complete_dates)

    # Now interpolate the NaNs linearly
    interpolated_data = complete_data.interpolate(method='linear')

    # Convert the index back to a more readable date format
    interpolated_data.index = interpolated_data.index.strftime('%Y-%m-%d')

    # Save the interpolated data to CSV
    interpolated_data.to_csv(output_file_path, index_label='Date')

    print(f"Interpolated data saved to {output_file_path}")

if __name__ == "__main__":
    input_file_path = '/Users/dstakhurskyi/Downloads/Data - Аркуш1-2.csv'
    output_file_path = '/Users/dstakhurskyi/PycharmProjects/arimaModeling/output/result.csv'

    interpolate_annual_to_quarterly_data(input_file_path, output_file_path)

