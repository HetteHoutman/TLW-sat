import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('../../other_data/sat_vs_ukv_results.xlsx').reset_index(
    drop=True)

xy_line = [df[['sat_lambda', 'ukv_lambda', 'sat_lambda_ellipse', 'ukv_lambda_ellipse', 'ukv_radsim_lambda_ellipse']].min(axis=None),
           df[['sat_lambda', 'ukv_lambda', 'sat_lambda_ellipse', 'ukv_lambda_ellipse', 'ukv_radsim_lambda_ellipse']].max(axis=None)]
plt.plot(xy_line, xy_line, 'k--', zorder=0)

for i in df.index:
    # noinspection PyTypeChecker
    plt.errorbar(df.sat_lambda[i], df.ukv_lambda[i],
                 xerr=[
                     [df.sat_lambda[i] - 2 * np.pi / (2 * np.pi / df.sat_lambda[i] + 0.1)],
                     [2 * np.pi / (2 * np.pi / df.sat_lambda[i] - 0.1) - df.sat_lambda[i]]],
                 yerr=[
                     [df.ukv_lambda[i] - 2 * np.pi / (2 * np.pi / df.ukv_lambda[i] + 0.1)],
                     [2 * np.pi / (2 * np.pi / df.ukv_lambda[i] - 0.1) - df.ukv_lambda[i]]],
                 label=f'{df.date[i]} {df.region[i]}', marker='o', capsize=3, c=plt.cm.tab20(i))

for i in df.index:
    # noinspection PyTypeChecker
    plt.errorbar(df.sat_lambda_newres[i], df.ukv_lambda_newres[i],
                 xerr=[
                     [df.sat_lambda_newres[i] - 2 * np.pi / (2 * np.pi / df.sat_lambda_newres[i] + 0.05)],
                     [2 * np.pi / (2 * np.pi / df.sat_lambda_newres[i] - 0.05) - df.sat_lambda_newres[i]]],
                 yerr=[
                     [df.ukv_lambda_newres[i] - 2 * np.pi / (2 * np.pi / df.ukv_lambda_newres[i] + 0.05)],
                     [2 * np.pi / (2 * np.pi / df.ukv_lambda_newres[i] - 0.05) - df.ukv_lambda_newres[i]]],
                 marker='s', capsize=3, c=plt.cm.tab20(i))

for i in df.index:
    # noinspection PyTypeChecker
    upper_sat_error = [max(df.sat_lambda_ellipse_min[i], df.sat_lambda_ellipse_max[i]) - df.sat_lambda_ellipse[i]]
    lower_sat_error = [df.sat_lambda_ellipse[i] - min(df.sat_lambda_ellipse_min[i], df.sat_lambda_ellipse_max[i])]

    upper_ukv_error = [max(df.ukv_lambda_ellipse_min[i], df.ukv_lambda_ellipse_max[i]) - df.ukv_lambda_ellipse[i]]
    lower_ukv_error = [df.ukv_lambda_ellipse[i] - min(df.ukv_lambda_ellipse_min[i], df.ukv_lambda_ellipse_max[i])]

    plt.errorbar(df.sat_lambda_ellipse[i], df.ukv_lambda_ellipse[i],
                 xerr=[lower_sat_error, upper_sat_error], yerr=[lower_ukv_error, upper_ukv_error],
                 marker='p', capsize=3, c=plt.cm.tab20(i))

for i in df.index:
    # noinspection PyTypeChecker
    upper_sat_error = [max(df.sat_lambda_ellipse_min[i], df.sat_lambda_ellipse_max[i]) - df.sat_lambda_ellipse[i]]
    lower_sat_error = [df.sat_lambda_ellipse[i] - min(df.sat_lambda_ellipse_min[i], df.sat_lambda_ellipse_max[i])]

    upper_ukv_radsim_error = [max(df.ukv_radsim_lambda_ellipse_min[i], df.ukv_radsim_lambda_ellipse_max[i]) - df.ukv_radsim_lambda_ellipse[i]]
    lower_ukv_radsim_error = [df.ukv_radsim_lambda_ellipse[i] - min(df.ukv_radsim_lambda_ellipse_min[i], df.ukv_radsim_lambda_ellipse_max[i])]

    plt.errorbar(df.sat_lambda_ellipse[i], df.ukv_radsim_lambda_ellipse[i],
                 xerr=[lower_sat_error, upper_sat_error], yerr=[lower_ukv_radsim_error, upper_ukv_radsim_error],
                 marker='v', capsize=3, c=plt.cm.tab20(i))

plt.legend(loc='best')
plt.xlabel('Satellite wavelength (km)')
plt.ylabel('UKV wavelength (km)')
plt.savefig('plots/lambda_comparison.png', dpi=300)
plt.show()

xy_line = [df[['sat_theta', 'ukv_theta', 'sat_theta_ellipse', 'ukv_theta_ellipse', 'ukv_radsim_theta_ellipse']].min(axis=None),
           df[['sat_theta', 'ukv_theta', 'sat_theta_ellipse', 'ukv_theta_ellipse', 'ukv_radsim_theta_ellipse']].max(axis=None)]
plt.plot(xy_line, xy_line, 'k--', zorder=0)

for i in df.index:
    plt.errorbar(df.sat_theta[i], df.ukv_theta[i], xerr=5, yerr=5,
                 label=f'{df.date[i]} {df.region[i]}', marker='o', capsize=3, c=plt.cm.tab20(i))
    plt.errorbar(df.sat_theta_newres[i], df.ukv_theta_newres[i], xerr=5, yerr=5,
                 marker='s', capsize=3, c=plt.cm.tab20(i))

    upper_sat_error = [df.sat_theta_ellipse_max[i] - df.sat_theta_ellipse[i]]
    lower_sat_error = [df.sat_theta_ellipse[i] - df.sat_theta_ellipse_min[i]]
    if lower_sat_error[0] < 0:
        lower_sat_error[0] += 360

    upper_ukv_error = [df.ukv_theta_ellipse_max[i] - df.ukv_theta_ellipse[i]]
    lower_ukv_error = [df.ukv_theta_ellipse[i] - df.ukv_theta_ellipse_min[i]]
    if lower_ukv_error[0] < 0:
        lower_ukv_error[0] += 360

    plt.errorbar(df.sat_theta_ellipse[i], df.ukv_theta_ellipse[i],
                 xerr=[lower_sat_error, upper_sat_error], yerr=[lower_ukv_error, upper_ukv_error],
                 marker='p', capsize=3, c=plt.cm.tab20(i))

    upper_ukv_radsim_error = [df.ukv_radsim_theta_ellipse_max[i] - df.ukv_radsim_theta_ellipse[i]]
    lower_ukv_radsim_error = [df.ukv_radsim_theta_ellipse[i] - df.ukv_radsim_theta_ellipse_min[i]]
    if lower_ukv_radsim_error[0] < 0:
        lower_ukv_radsim_error[0] += 360

    plt.errorbar(df.sat_theta_ellipse[i], df.ukv_radsim_theta_ellipse[i],
                 xerr=[lower_sat_error, upper_sat_error], yerr=[lower_ukv_radsim_error, upper_ukv_radsim_error],
                 marker='v', capsize=3, c=plt.cm.tab20(i))

plt.legend(loc='best')
plt.xlabel('Satellite wavevector direction from north (deg)')
plt.ylabel('UKV wavevector direction from north (deg)')
plt.savefig('plots/theta_comparison.png', dpi=300)
plt.show()
