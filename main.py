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






# def interpolate_annual_to_quarterly_data(file_path, output_path):
#     # Load the data
#     data = pd.read_csv(file_path)
#
#     # Convert the 'Data' column to numeric, coercing errors to NaN
#     data['Data'] = pd.to_numeric(data['Data'], errors='coerce')
#
#     # Try to parse the 'Date' column automatically
#     data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
#
#     # If automatic parsing fails, specify the date format directly
#     # data['Date'] = pd.to_datetime(data['Date'], format='DATE_FORMAT', errors='coerce')
#
#     # Set the 'Date' column as the index
#     data.set_index('Date', inplace=True)
#
#     # Drop rows with NaN dates
#     data = data.dropna(subset=['Date'])
#
#     # Reindex the DataFrame to include all end-of-quarter dates and then interpolate
#     quarterly_data = data.resample('Q').asfreq()
#     quarterly_data['Data'] = quarterly_data['Data'].interpolate()
#
#     # Save the interpolated data
#     quarterly_data.to_csv(output_path, index_label='Date')
#
#     return quarterly_data
#
#
# if __name__ == '__main__':
#     file_path = '/Users/dstakhurskyi/Downloads/Data - Аркуш1-2.csv'
#     output_path = '/Users/dstakhurskyi/PycharmProjects/arimaModeling/output/result.csv'
#
#     # Perform the interpolation
#     interpolated_data = interpolate_annual_to_quarterly_data(file_path, output_path)
#
#     # Print to verify
#     print("First few rows of interpolated quarterly data:")
#     print(interpolated_data.head())

# import pandas as pd
#
#
# def interpolate_annual_to_quarterly_data(file_path, start_year, output_path):
#     data = pd.read_csv(file_path)
#
#     # Convert the 'Data' column to numeric, coercing errors to NaN
#     data['Data'] = pd.to_numeric(data['Data'], errors='coerce')
#
#     # Check if the conversion resulted in any data to work with
#     if data['Data'].isna().all():
#         print("Warning: No valid numeric data to interpolate.")
#         return pd.DataFrame()  # Return an empty DataFrame or handle as needed
#
#     dates = pd.date_range(start=f'{start_year}-01-01', periods=len(data), freq='YS')
#     data['Date'] = dates
#     data.set_index('Date', inplace=True)
#
#     quarterly_data_frames = []
#
#     for year in range(start_year, start_year + len(data)):
#         quarterly_dates = pd.date_range(start=f'{year}-01-01', periods=4, freq='QS')
#         next_year_start = pd.Timestamp(year=year + 1, month=1, day=1)
#         annual_data_value = data.loc[f'{year}-01-01', 'Data']
#
#         temp_df = pd.DataFrame(index=quarterly_dates.union([next_year_start]), columns=['Data'])
#         temp_df.loc[next_year_start, 'Data'] = annual_data_value
#
#         # Ensure 'Data' is treated as numeric for interpolation
#         temp_df['Data'] = pd.to_numeric(temp_df['Data'], errors='coerce')
#
#         # Perform the interpolation
#         temp_df = temp_df.interpolate().iloc[:-1]
#
#         quarterly_data_frames.append(temp_df)
#
#     quarterly_data = pd.concat(quarterly_data_frames)
#     quarterly_data.to_csv(output_path, index_label='Date')
#
#     return quarterly_data
#
#
# if __name__ == '__main__':
#     file_path = '/Users/dstakhurskyi/Downloads/Macro Draft Data - Аркуш1.csv'
#     output_path = '/Users/dstakhurskyi/PycharmProjects/arimaModeling/output/result.csv'
#     start_year = 2010
#
#     interpolated_data = interpolate_annual_to_quarterly_data(file_path, start_year, output_path)
#     print("First few rows of interpolated quarterly data:")
#     print(interpolated_data.head())
