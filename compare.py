import pandas as pd

# Function to compare two CSV files
def compare_csv(file1, file2):
    # Read CSV files into DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Check column names
    if list(df1.columns) != list(df2.columns):
        print("Column names are different:")
        print("File 1 columns:", list(df1.columns))
        print("File 2 columns:", list(df2.columns))
        return

    print("Column names are the same.")

    # Check values
    different_values = []
    for col in df1.columns:
        if not df1[col].equals(df2[col]):
            different_values.append(col)

    if different_values:
        print("Columns with different values:")
        for col in different_values:
            print(col)
    else:
        print("All values are the same.")

# Example usage
compare_csv("nor_oneH.csv", "nor_oneH2.csv")
