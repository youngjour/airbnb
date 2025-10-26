import pandas as pd

df = pd.read_csv('sgis_improved_final.csv')

print(f'Total rows: {len(df)}')
print(f'Columns: {df.columns.tolist()}')

# Check for duplicates
dups = df.groupby(['Dong_name', 'Reporting Month']).size()
dups_count = (dups > 1).sum()

print(f'\nDuplicate (Dong, Month) combinations: {dups_count}')

if dups_count > 0:
    print('\nSample duplicates (showing first 20):')
    duplicate_combos = dups[dups > 1].head(20)
    print(duplicate_combos)

    # Show actual rows for first duplicate
    if len(duplicate_combos) > 0:
        first_dup = duplicate_combos.index[0]
        dong_name, month = first_dup
        print(f'\nActual rows for {dong_name}, {month}:')
        print(df[(df['Dong_name'] == dong_name) & (df['Reporting Month'] == month)])
