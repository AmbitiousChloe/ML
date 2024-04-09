import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("clean_dataset.csv")
    df_test = df.sample(frac=1, random_state=42)
    print(list(df_test["Label"]))
    del df_test["Label"]
    df_test.to_csv("example_test.csv", index=False)