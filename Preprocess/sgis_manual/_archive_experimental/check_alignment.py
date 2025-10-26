import pandas as pd

sgis = pd.read_csv('sgis_improved_final.csv')
labels = pd.read_csv('../../Data/Preprocessed_data/Dong/AirBnB_raw_embedding.csv')

print('SGIS:')
print(f'  Shape: {sgis.shape}')
print(f'  Unique dongs: {sgis["Dong_name"].nunique()}')
print(f'  Date range: {sgis["Reporting Month"].min()} to {sgis["Reporting Month"].max()}')
print(f'  Unique months: {sgis["Reporting Month"].nunique()}')

print('\nLabels (raw embedding):')
print(f'  Shape: {labels.shape}')
print(f'  Unique dongs: {labels["Dong_name"].nunique()}')
print(f'  Date range: {labels["Reporting Month"].min()} to {labels["Reporting Month"].max()}')
print(f'  Unique months: {labels["Reporting Month"].nunique()}')

missing_in_sgis = set(labels['Dong_name'].unique()) - set(sgis['Dong_name'].unique())
print(f'\nDongs in labels but not in SGIS: {len(missing_in_sgis)}')
if len(missing_in_sgis) > 0:
    print('Sample missing dongs:', list(missing_in_sgis)[:10])

print(f'\nAlignment check: {"GOOD" if len(missing_in_sgis) == 0 else "NEEDS FIXING"}')
