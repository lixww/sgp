import pandas as pd


# file paths
data_path = 'autoencoder/data/sgp'

full_df = pd.read_csv(f'{data_path}/training_file_8_bit.csv')

print(full_df.shape)

target_folio = ['024r-029v', '102v-107r', '214v-221r']


folio_df = full_df.loc[full_df['folio_name'].isin(target_folio)]

print(folio_df.shape)

folio_df.to_csv(f'{data_path}/folio_8_bit.csv', sep=',', index=False)