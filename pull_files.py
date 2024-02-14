import eumdac
import numpy as np

consumer_key = '9Kplg3cMsnqL3Uzo_aZrhUqwOu8a'
consumer_secret = 'Hf9qKiMD3UWKOoMV2_uku0fdHv8a'

credentials = (consumer_key, consumer_secret)

token = eumdac.AccessToken(credentials)
datastore = eumdac.DataStore(token)
selected_collection = datastore.get_collection('EO:EUM:DAT:MSG:HRSEVIRI')

import shutil
import pandas as pd
import os

df = pd.read_excel('../../other_data/list_of_cases.xlsx', parse_dates=[0])
df.h = df.h.astype('int')
datetimes = [date.replace(hour=hour) for date, hour in zip(df.dates, df.h)]
ddir = r"C:\Users\sw825517\OneDrive - University of Reading\research\code\eumetsat"

for start in datetimes:
    if start.year == 2023:
        dt = np.timedelta64(15, 'm')
        dir_name = 'data/' + start.strftime('%Y-%m-%d_%H')

        if os.path.exists(dir_name):
            continue
        else:
            os.makedirs(dir_name)

        if not os.listdir(dir_name):
            print(f"Directory {dir_name} is empty")

            # Retrieve datasets that match our filter
            products = selected_collection.search(
                dtstart=start,
                dtend=start + dt)

            for product in products:
                print(str(product))

                with product.open() as fsrc, \
                        open(fsrc.name, mode='wb') as fdst:
                    shutil.copyfileobj(fsrc, fdst)
                    print(f'Download of product {product} finished.')
                    print(f'Unpacking {fdst.name}')
                    shutil.unpack_archive(fdst.name, extract_dir=ddir + '/data/' + start.strftime('%Y-%m-%d_%H'))

                os.remove(fdst.name)
                print(f'Removed {fdst.name}')

        else:
            pass
    else:
        pass

print('All downloads are finished.')
