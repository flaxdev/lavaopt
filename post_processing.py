import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.stats import gaussian_kde

def post_process(n_vent, flux_pct_res):
	df_list = []
	df_h2o = []
	df_flux = []

	for run_dir in glob.glob('run_*'):
		run = int(run_dir[4:])
		name = f'{run_dir}/accepted_{run}.csv'
		try:
			df = pd.read_csv(name)
			print((run, run_dir, name))
			df_list.append(df)
		except FileNotFoundError:
			print(f'File accepted_{run}.csv not found')

	
	merged_df = pd.concat(df_list)
	
	grouped_df = merged_df.groupby(merged_df.columns.tolist(), as_index=False).size().reset_index()
	i_max_fit = grouped_df['fit'].idxmax()
	row_with_max_fit = grouped_df.iloc[i_max_fit]
	print(row_with_max_fit)
	
	name_out = "accepted.csv"
	grouped_df.to_csv(name_out, index=False)

	sel_row = data.iloc[i_max_fit]
	
	headers_list = ["lon_1", "lat_1", "flux_pct_1", "ph2o"]
	
	data_selected = data[headers_list]
	
	lon = sel_row[headers_list[0]]
	lat = sel_row[headers_list[1]]
	flux = sel_row[headers_list[2]]
	h2o = sel_row[headers_list[3]]

	pairplot = sns.pairplot(data_selected, kind = 'kde')#kiag_kind='hist', kind = 'kde')
	axes = pairplot.axes

	ax = axes[1, 1]
	ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)  if x % 200 == 0 else ''))
	ax = axes[1, 0]
	ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))


	axes[0, 0].axvline(lon, color='red', linestyle='--')
	axes[1, 1].axvline(lat, color='red', linestyle='--')
	axes[2, 2].axvline(flux, color='red', linestyle='--')
	axes[3, 3].axvline(h2o, color='red', linestyle='--')

	axes[1, 0].plot(lon, lat, 0.005, marker='o', markersize=8, color='red')
	axes[2, 0].plot(lon, flux, 0.005, marker='o', markersize=8, color='red')
	axes[3, 0].plot(lon, h2o, 0.005, marker='o', markersize=8, color='red')
	axes[0, 1].plot(lat, lon, 0.005, marker='o', markersize=8, color='red')
	axes[2, 1].plot(lat, flux, 0.005, marker='o', markersize=8, color='red')
	axes[3, 1].plot(lat, h2o, 0.005, marker='o', markersize=8, color='red')
	axes[0, 2].plot(flux, lon, 0.005, marker='o', markersize=8, color='red')
	axes[1, 2].plot(flux, lat, 0.005, marker='o', markersize=8, color='red')
	axes[3, 2].plot(flux, h2o, 0.005, marker='o', markersize=8, color='red')
	axes[0, 3].plot(h2o, lon, 0.005, marker='o', markersize=8, color='red')
	axes[1, 3].plot(h2o, lat, 0.005, marker='o', markersize=8, color='red')
	axes[2, 3].plot(h2o, flux, 0.005, marker='o', markersize=8, color='red')
	plt.savefig('mtx_corr.jpg')	

if __name__ == "__main__":
	pass
