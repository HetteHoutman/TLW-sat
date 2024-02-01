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
datetimes = [date.replace(hour=hour) for date, hour in zip(df.dates[df.selected == 'x'], df.h[df.selected == 'x'])]
ddir = r"C:\Users\sw825517\OneDrive - University of Reading\research\code\eumetsat"

for start in datetimes:
    dt = np.timedelta64(15, 'm')

    # Retrieve datasets that match our filter
    products = selected_collection.search(
        dtstart=start,
        dtend=start + dt)

    for product in products:
        print(str(product))

        if os.path.exists('data/' + start.strftime('%Y-%m-%d_%H')):
            continue
        else:
            os.makedirs('data/' + start.strftime('%Y-%m-%d_%H'))

        with product.open() as fsrc, \
                open(fsrc.name, mode='wb') as fdst:
            shutil.copyfileobj(fsrc, fdst)
            print(f'Download of product {product} finished.')
            print(f'Unpacking {fdst.name}')
            shutil.unpack_archive(fdst.name, extract_dir=ddir + '/data/' + start.strftime('%Y-%m-%d_%H'))

        os.remove(fdst.name)
        print(f'Removed {fdst.name}')

print('All downloads are finished.')
