import pandas as pd


# file paths
data_path = 'autoencoder/data/sgp'


full_df = pd.read_csv(f'{data_path}/training_file_8_bit.csv')

print(full_df.shape)

uclass_data = []
notUclass_data = []

for row in full_df.values:
    if row[1] == 2:
        uclass_data.append(row)
    else:
        notUclass_data.append(row)

uclass_df = pd.DataFrame(data=uclass_data, columns=full_df.columns)
notUclass_df = pd.DataFrame(data=notUclass_data, columns=full_df.columns)

uclass_df.to_csv(f'{data_path}/uclass_8_bit.csv', sep=',', index=False)
notUclass_df.to_csv(f'{data_path}/notUclass_8_bit.csv', sep=',', index=False)
