import random
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

file_name = "clean_dataset.csv"


"""###############################
Helper functions for reading data:
"""
def get_number_list(s):
    """Get a list of integers contained in string `s`
    """
    return [int(n) for n in re.findall("(\d+)", str(s))]


def get_number(s, na):
    """Get the first number contained in string `s`.

    If `s` does not contain any numbers, return -1.
    """
    n_list = get_number_list(s)
    return n_list[0] if len(n_list) >= 1 else na


def cat_in_s(s, cat):
    """Return if a category is present in string `s` as an binary integer.
    """
    return int(cat in s) if not pd.isna(s) else 0


def to_numeric(s):
    """Converts string `s` to a float.

    Invalid strings and NaN values will be converted to float('nan').
    """

    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
    return float(s)


def radm_dict(d):
    value_to_keys = {}
    for key, value in d.items():
        if value in value_to_keys:
            value_to_keys[value].append(key)
        else:
            value_to_keys[value] = [key]

    new_dict = {}
    for value, keys in value_to_keys.items():
        if len(keys) > 1:
            key_to_keep = random.choice(keys)
            new_dict[key_to_keep] = value
        else:
            new_dict[keys[0]] = value

    return new_dict


def to_dict_complete(s):
    samples = s.split(",")
    result_dict = {}
    for sample in samples:
        key, value = sample.split('=>')
        if value == "":
            break
        result_dict[key.strip()] = int(value.strip())
    return result_dict


def process_data(filename: str) -> pd.DataFrame:
   # read the data as pandas dataframe
   df = pd.read_csv(file_name)

   # pre process Q1 - Q4
   na_14 = 2
   df["Q1"] = df["Q1"].apply(lambda x: get_number(x, na_14))
   df["Q2"] = df["Q2"].apply(lambda x: get_number(x, na_14))
   df["Q3"] = df["Q3"].apply(lambda x: get_number(x, na_14))
   df["Q4"] = df["Q4"].apply(lambda x: get_number(x, na_14))
   
    # pre process Q5
   q5_category = ["Partner", "Friends", "Siblings", "Co-worker"]

   # create one hot features for 4 categories
   for cat in q5_category:
    cat_name = f"{cat}"
    df[cat_name] = df["Q5"].apply(lambda s: cat_in_s(s, cat))


   # pre process Q7 - Q9
   df["Q7"] = df["Q7"].apply(to_numeric)
   df["Q8"] = df["Q8"].apply(to_numeric)
   df["Q9"] = df["Q9"].apply(to_numeric)

   return df


if __name__ == "__main__":
    df = process_data(file_name)

#     Q1_na = df['Q1'].isna().sum()
#     Q2_na = df['Q2'].isna().sum()
#     Q3_na = df['Q3'].isna().sum()
#     Q4_na = df['Q4'].isna().sum()
#     Q5_na = df['Q5'].isna().sum()
#     Q6_na = df['Q6'].isna().sum()
#     Q7_na = df['Q7'].isna().sum()
#     Q8_na = df['Q8'].isna().sum()
#     Q9_na = df['Q9'].isna().sum()
#     Q10_na = df['Q10'].isna().sum()

#     q7_min = -50
#     q7_max = 50
#     q89_min = 1
#     q89_max = 15

#     Q7_out = ((df['Q7'] < q7_min) | (df['Q7'] > q7_max)).sum()
#     Q8_out = ((df['Q8'] < q89_min) | (df['Q8'] > q89_max)).sum()
#     Q9_out = ((df['Q9'] < q89_min) | (df['Q9'] > q89_max)).sum()

    # df.loc[(df['Q7'] < q7_min), 'Q7'] = q7_min
    # df.loc[(df['Q7'] > q7_max), 'Q7'] = q7_max
    # df.loc[(df['Q8'] < q89_min), 'Q8'] = q89_min
    # df.loc[(df['Q8'] > q89_max), 'Q8'] = q89_max
    # df.loc[(df['Q9'] < q89_min), 'Q9'] = q89_min
    # df.loc[(df['Q9'] > q89_max), 'Q9'] = q89_max

#     df["Q6"].fillna("Skyscrapers=>3.5,Sport=>3.5,Art and Music=>3.5,Carnival=>3.5,Cuisine=>3.5,Economic=>3.5", 
#                         inplace = True)
#     df["Q6_Skyscr"] = df["Q6"].apply(lambda x: to_dict_complete(x).get("Skyscrapers", 3.5))
#     df["Q6_Sport"] = df["Q6"].apply(lambda x: to_dict_complete(x).get("Sport", 3.5))
#     df["Q6_AM"] = df["Q6"].apply(lambda x: to_dict_complete(x).get("Art and Music", 3.5))
#     df["Q6_Carnival"] = df["Q6"].apply(lambda x: to_dict_complete(x).get("Carnival", 3.5))
#     df["Q6_Cuisine"] = df["Q6"].apply(lambda x: to_dict_complete(x).get("Cuisine", 3.5))
#     df["Q6_Eco"] = df["Q6"].apply(lambda x: to_dict_complete(x).get("Economic", 3.5))

#     # Data for the table
#     columns = ['Missing Value count', 'Outlier count']
#     rows = ['Popular Score', 'Efficient Score', 'Uniqueness Score', 'Enthusiasm Score', 
#             'Traveler Choice', 'Key word Rank', 'Avg Temp', 'Language count', 'Style Count', 'Quote']

#     data = [[Q1_na, 0],  # Data for Column1
#             [Q2_na, 0],
#             [Q3_na, 0],
#             [Q4_na, 0],
#             [Q5_na, 0],
#             [Q6_na, 0],
#             [Q7_na, Q7_out],
#             [Q8_na, Q8_out],
#             [Q9_na, Q9_out],
#             [Q10_na, 0]]   # Data for Column2

#     # Create a figure and a grid to display the table
#     fig, ax = plt.subplots()

#     # Hide the axes
#     ax.axis('tight')
#     ax.axis('off')

#     table = ax.table(cellText=data, 
#                     rowLabels=rows, 
#                     colLabels=columns, 
#                     cellLoc='center', 
#                     loc='center')
#     plt.savefig('./preprocessor/plots/miss_out Count.png', bbox_inches='tight', dpi=150)

    # na_14 = 2
    # q5_category = ["Partner", "Friends", "Siblings", "Co-worker"]

    # df["Q1"].fillna(na_14, inplace = True)
    # df["Q2"].fillna(na_14, inplace = True)
    # df["Q3"].fillna(na_14, inplace = True)
    # df["Q4"].fillna(na_14, inplace = True)
    # df["Q5"].fillna(random.choice(q5_category))
    # df["Q7"].fillna(df["Q7"].mean(), inplace = True)
    # df["Q8"].fillna(df["Q8"].mean(), inplace = True)
    # df["Q9"].fillna(df["Q9"].mean(), inplace = True)

    # # Q1
    # sns.boxplot(x='Q1', y='Label', data=df)
    # plt.title('Q1 distribution')
    # plt.xlabel('Popular score')
    # plt.ylabel('City')
    # plt.savefig("./preprocessor/plots/Q1_box.png")
    # plt.clf()

    # # Q2
    # sns.boxplot(x='Q2', y='Label', data=df)
    # plt.title('Q2 distribution')
    # plt.xlabel('Efficient score')
    # plt.ylabel('City')
    # plt.savefig("./preprocessor/plots/Q2_box.png")
    # plt.clf()

    # # Q3
    # sns.boxplot(x='Q3', y='Label', data=df)
    # plt.title('Q3 distribution')
    # plt.xlabel('Uniqueness score')
    # plt.ylabel('City')
    # plt.savefig("./preprocessor/plots/Q3_box.png")
    # plt.clf()

    # # Q4
    # sns.boxplot(x='Q4', y='Label', data=df)
    # plt.title('Q4 distribution')
    # plt.xlabel('Enthusiasm score')
    # plt.ylabel('City')
    # plt.savefig("./preprocessor/plots/Q4_box.png")
    # plt.clf()

    # # Q7
    # sns.boxplot(x='Q7', y='Label', data=df)
    # plt.title('Q7 distribution')
    # plt.xlabel('Temperature')
    # plt.ylabel('City')
    # plt.savefig("./preprocessor/plots/Q7_box.png")
    # plt.clf()
    
    # # Q8
    # sns.boxplot(x='Q8', y='Label', data=df)
    # plt.title('Q8 distribution')
    # plt.xlabel('Languages counts')
    # plt.ylabel('City')
    # plt.savefig("./preprocessor/plots/Q8_box.png")
    # plt.clf()
    
    # # Q9
    # sns.boxplot(x='Q9', y='Label', data=df)
    # plt.title('Q9 distribution')
    # plt.xlabel('Fashion styles count')
    # plt.ylabel('City')
    # plt.savefig("./preprocessor/plots/Q9_box.png")
    # plt.clf()

    # Q5
    aggregated_data = df.groupby('Label').sum().reset_index()
    melted_data = aggregated_data.melt(id_vars='Label', value_vars=["Partner", "Friends", "Siblings", "Co-worker"],  var_name='Choice', value_name='Count')

    sns.barplot(x='Label', y='Count', hue='Choice', data=melted_data)
    plt.title('Side by Side Compare of Traveler Choice Count by City')
    plt.xlabel('City')
    plt.ylabel('Count')
    plt.legend(title='Column')
    plt.savefig("./preprocessor/plots/Q5_bar.png")
    plt.clf()

    # # Q6_Skyscr
    # sns.boxplot(x='Q6_Skyscr', y='Label', data=df)
    # plt.title('Q6_Skyscr distribution')
    # plt.xlabel('Skyscarapers score')
    # plt.ylabel('City')
    # plt.savefig("./preprocessor/plots/Q6_Skyscr.png")
    # plt.clf()

    # # Q6_Sport
    # sns.boxplot(x='Q6_Sport', y='Label', data=df)
    # plt.title('Q6_Sport distribution')
    # plt.xlabel('Sport score')
    # plt.ylabel('City')
    # plt.savefig("./preprocessor/plots/Q6_Sport.png")
    # plt.clf()

    # # Q6_AM
    # sns.boxplot(x='Q6_AM', y='Label', data=df)
    # plt.title('Q6_AM distribution')
    # plt.xlabel('Art and Music score')
    # plt.ylabel('City')
    # plt.savefig("./preprocessor/plots/Q6_AM.png")
    # plt.clf()

    # # Q6_Carnival
    # sns.boxplot(x='Q6_Carnival', y='Label', data=df)
    # plt.title('Q6_Carnival distribution')
    # plt.xlabel('Carnival score')
    # plt.ylabel('City')
    # plt.savefig("./preprocessor/plots/Q6_Carnival.png")
    # plt.clf()

    # # Q6_Cuisine
    # sns.boxplot(x='Q6_Cuisine', y='Label', data=df)
    # plt.title('Q6_Cuisine distribution')
    # plt.xlabel('Cuisine score')
    # plt.ylabel('City')
    # plt.savefig("./preprocessor/plots/Q6_Cuisine.png")
    # plt.clf()

    # # Q6_Eco
    # sns.boxplot(x='Q6_Eco', y='Label', data=df)
    # plt.title('Q6_Eco distribution')
    # plt.xlabel('Economic score')
    # plt.ylabel('City')
    # plt.savefig("./preprocessor/plots/Q6_Eco.png")
    # plt.clf()