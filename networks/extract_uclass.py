import pandas as pd

# extract all undertext pixels from training_file_8_bit.csv

# file paths
data_path = 'networks/data/sgp'


full_df = pd.read_csv(f'{data_path}/training_file_8_bit.csv')

print(full_df.shape)


uclass_df = full_df.loc[full_df['class_name'] == 2]
print(uclass_df.shape)

notUclass_df = full_df.loc[full_df['class_name'] != 2]
print(notUclass_df.shape)


# save in csv
uclass_df.to_csv(f'{data_path}/uclass_8_bit.csv', sep=',', index=False)
notUclass_df.to_csv(f'{data_path}/notUclass_8_bit.csv', sep=',', index=False)
