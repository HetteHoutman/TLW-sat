import sys

import pandas as pd

df1 = pd.read_csv(sys.argv[1], index_col=[0, 1, 2], parse_dates=[0])

case_list = pd.read_excel('../list_of_cases.xlsx', parse_dates=[0])
case_list = case_list.assign(regions=case_list.regions.str.split('/')).explode('regions')

case_list = case_list[case_list.dates.dt.year == 2023]

case_list.rename(columns={'dates': 'date', 'regions': 'region'}, inplace=True)
case_list.set_index(['date', 'region', 'h'], inplace=True)
full = case_list.merge(df1, how='outer', left_index=True, right_index=True)

print(full[full.theta.isna()])
print(full.theta.isna().sum(), ' NaNs found')
