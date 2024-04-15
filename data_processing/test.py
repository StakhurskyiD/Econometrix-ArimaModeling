import pandas as pd

# Example data
data = {
    "Year": [2010, 2011],
    "Data": [1713.9, 1661.9]
}

# Convert to DataFrame and set up the datetime index
df = pd.DataFrame(data)
df['Year'] = pd.to_datetime(df['Year'], format='%Y') + pd.offsets.YearEnd()
df.set_index('Year', inplace=True)

# Generate a quarterly date range and reindex the DataFrame
quarterly_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='Q')
df_reindexed = df.reindex(quarterly_range)

# Perform linear interpolation
df_interpolated = df_reindexed.interpolate(method='linear')

# Export the interpolated DataFrame to a CSV file
output_file_path = 'interpolated_export_data.csv'  # Define your output file path here
df_interpolated.to_csv(output_file_path, index_label='Date')

print(f"Interpolated data exported to {output_file_path}")
