import os
import shutil

import eumdac
import numpy as np
import pandas as pd

consumer_key = '9Kplg3cMsnqL3Uzo_aZrhUqwOu8a'
consumer_secret = 'Hf9qKiMD3UWKOoMV2_uku0fdHv8a'

credentials = (consumer_key, consumer_secret)

token = eumdac.AccessToken(credentials)
datastore = eumdac.DataStore(token)
selected_collection = datastore.get_collection('EO:EUM:DAT:MSG:HRSEVIRI')

df = pd.read_excel('../list_of_cases.xlsx', parse_dates=[0])
datetimes = [date.replace(hour=int(hour)) for date, hour in zip(df.dates, df.h) if date.year == 2023]
ddir = '/storage/silver/metstudent/phd/sw825517/'

for start in datetimes:
    if start.year == 2023:
        dt = np.timedelta64(15, 'm')
        dir_name = ddir + 'seviri_data/' + start.strftime('%Y-%m-%d_%H')

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
                    fdst.close()
                    print(f'Unpacking {fdst.name}')
                    shutil.unpack_archive(fdst.name, extract_dir=ddir + '/seviri_data/' + start.strftime('%Y-%m-%d_%H'))

                os.remove(fdst.name)
                print(f'Removed {fdst.name}')

        else:
            pass
    else:
        pass

print('All downloads are finished.')
