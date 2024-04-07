import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("clean_dataset.csv")
    del df["Label"]
    df_test = df.sample(frac=1, random_state=311)[:100]
    print(df_test)
    df_test.to_csv("example_test.csv", index=False)