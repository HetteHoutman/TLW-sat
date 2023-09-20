import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('../../other_data/sat_vs_ukv_results.xlsx', header=1).dropna(subset=['sat_lambda']).reset_index(
    drop=True)

xy_line = [df[['sat_lambda', 'ukv_lambda']].min(axis=None), df[['sat_lambda', 'ukv_lambda']].max(axis=None)]

plt.plot(xy_line, xy_line, 'k--', zorder=0)
for i in df.index:
    plt.errorbar(df.sat_lambda[i], df.ukv_lambda[i],
                 xerr=[
                     [df.sat_lambda[i] - 2 * np.pi / (2 * np.pi / df.sat_lambda[i] + 0.1)],
                     [2 * np.pi / (2 * np.pi / df.sat_lambda[i] - 0.1) - df.sat_lambda[i]]],
                 yerr=[
                     [df.ukv_lambda[i] - 2 * np.pi / (2 * np.pi / df.ukv_lambda[i] + 0.1)],
                     [2 * np.pi / (2 * np.pi / df.ukv_lambda[i] - 0.1) - df.ukv_lambda[i]]],
                 label=f'{df.date[i].date()} {df.region[i]}', marker='o', capsize=3)
plt.legend(loc='upper left')
plt.xlabel('Satellite wavelength (km)')
plt.ylabel('UKV wavelength (km)')
plt.savefig('plots/lambda_comparison.png', dpi=300)
plt.show()

xy_line = [df[['sat_theta', 'ukv_theta']].min(axis=None), df[['sat_theta', 'ukv_theta']].max(axis=None)]

plt.plot(xy_line, xy_line, 'k--', zorder=0)
for i in df.index:
    plt.errorbar(df.sat_theta[i], df.ukv_theta[i], xerr=5, yerr=5,
                 label=f'{df.date[i].date()} {df.region[i]}', marker='o', capsize=3)
plt.legend(loc='lower right')
plt.xlabel('Satellite wavevector direction from north (deg)')
plt.ylabel('UKV wavevector direction from north (deg)')
plt.savefig('plots/theta_comparison.png', dpi=300)
plt.show()
