import json
from os import walk
from os import rename



file_path = '~/Desktop/sgp-imgs/214v-221r/'
folder_name = 'jpg'


filenames = []
relates = {}
with open(f'{file_path}/imgs.json', 'r') as json_f:
    json_o = json.load(json_f)

    for url in json_o[0]['jpg_urls']:
        filenames.append(url.split('/')[-1])

    for item in json_o[0]['jpg_imgs']:
        relates[item['path'].split('/')[-1]] = item['url'].split('/')[-1]


files = []
for (dirpath, dirnames, filenames) in walk(f'{file_path}/{folder_name}/'):
    files.extend(filenames)
    break


filedir = f'{file_path}/{folder_name}/'
for f in filenames:
    if f in relates.keys():
        rename(filedir+f, filedir+relates[f])