import pandas as pd
all_data = pd.read_csv('./data/result.csv', encoding='utf-8', header=1, on_bad_lines='skip', sep=';')
all_data = list(all_data.sort_values(by=all_data.columns[2]).values)

for doc in all_data:
    print(type(list(doc[5])))
