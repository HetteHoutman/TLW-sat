import pandas as pd
import sys

from miscellaneous import check_argv_num

check_argv_num(sys.argv, 1, '(new format file)')
df = pd.read_csv(sys.argv[1], parse_dates=[0])

df.drop_duplicates(['date', 'region', 'h'], inplace=True)
df['lambda_min'] = df['lambda'] - 0.01
df['lambda_max'] = df['lambda'] + 0.01
df['theta_min'] = df['theta'] - 0.01
df['theta_max'] = df['theta'] + 0.01

df.to_csv(sys.argv[1][:-4] + '_oldformat.csv', index=False)

