import pandas as pd

# Step 2: Read the CSV file
file_path = 'example_test.csv'  # Change this to the path of your CSV file
df = pd.read_csv(file_path)

# Step 3: Drop a certain column
# Replace 'column_to_drop' with the name of the column you want to remove
df_modified = df.drop('Label', axis=1)

# Step 4: Write the modified DataFrame to a new CSV file
output_file_path = 'new_example_test.csv'  # Change this to your desired output file path
df_modified.to_csv(output_file_path, index=False)  # Set index=False to not write row indices
