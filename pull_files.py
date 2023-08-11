import eumdac
consumer_key = '9Kplg3cMsnqL3Uzo_aZrhUqwOu8a'
consumer_secret = 'Hf9qKiMD3UWKOoMV2_uku0fdHv8a'

credentials = (consumer_key, consumer_secret)

token = eumdac.AccessToken(credentials)
datastore = eumdac.DataStore(token)
selected_collection = datastore.get_collection('EO:EUM:DAT:MSG:HRSEVIRI')

import datetime
import shutil

start = datetime.datetime(2023, 5, 30, 10, 45)
dt = datetime.timedelta(minutes=15)

# Retrieve datasets that match our filter
products = selected_collection.search(
    dtstart=start,
    dtend=start+dt)

for product in products:
    print(str(product))

for product in products:
    with product.open() as fsrc, \
            open(fsrc.name, mode='wb') as fdst:
        shutil.copyfileobj(fsrc, fdst)
        print(f'Download of product {product} finished.')
print('All downloads are finished.')