import numpy as np
import pandas as pd
from os import walk


from utils import load_raw_images_data


# file paths
data_path = 'networks/data/sgp'


def get_filenames(file_path):
    files = []
    remove_list = ['color', 'sharpie', 'csharpie', 'pseudo', 'DS_Store']
    for (dirpath, dirnames, filenames) in walk(f'{file_path}/'):
        for filename in filenames:
            if not any(substring in filename for substring in remove_list):
                files.append(file_path + filename)
    files.sort()

    return files

def get_window(radius, x=0, y=0):
    window = []
    for i in range(-radius, radius+1):
        for j in range(-radius, radius+1):
            window.append((j+x,i+y))

    return window


## get folionames
center_pxls = pd.read_csv(f'{data_path}/folio_8_bit.csv')

center_pxl_num = len(center_pxls)
# add new column: center_pxl_id
center_pxls['center_pxl_id'] = pd.Series(range(1, center_pxl_num+1), 
                                         index=center_pxls.index)

target_folio = center_pxls['folio_name'].unique()

extend_pxls = pd.DataFrame(columns=center_pxls.columns)

for folio_name in target_folio:
    print('process on', folio_name, '..')
    # load images
    file_path = f'~/Desktop/sgp-imgs/{folio_name}/tif/'
    filenames = get_filenames(file_path)
    imgs = load_raw_images_data(filenames, preserve_range_after_rescale=True)
    channel_len = len(imgs)
    # find neighbors: [(5x5): radius-2, (3x3): radius-1]
    radius = 1
    pxls_index = center_pxls.loc[center_pxls['folio_name'] == folio_name].index
    for i in pxls_index:
        label = center_pxls.loc[i]['class_name']
        x_l = center_pxls.loc[i]['x_loc'] - 1
        y_l = center_pxls.loc[i]['y_loc'] - 1
        center_id = center_pxls.loc[i]['center_pxl_id']
        neighbors_locs = get_window(radius, x=x_l, y=y_l)
        for (neigh_x, neigh_y) in neighbors_locs:
            # intensity data
            channel_data = []
            for c in range(channel_len):
                channel_data.append(imgs[c][neigh_y][neigh_x])
            # add pixel data to new extended dataframe
            df_index = 0 if pd.isnull(extend_pxls.index.max()) else extend_pxls.index.max()+1
            extend_pxls.loc[df_index] = [folio_name] + [label] + \
                                        [neigh_x+1] + [neigh_y+1] + \
                                        channel_data + \
                                        [center_id]

# save extended pixels
print('extended count:', extend_pxls.shape)
extend_pxls.to_csv(f'{data_path}/folio_8_bit_extended_3x3.csv', sep=',', index=False)