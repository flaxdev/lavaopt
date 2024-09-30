#!/usr/bin/python3

import numpy as np
import pandas as pd
import os
import random
import glob
import shutil
from osgeo import gdal
import sys
from create_input_files import create_inputfiles
from run import run_sim
from post_processing import post_process
import copy
import shutil

#### Global Variables  ####

from input_parameters import par_vents
from input_parameters import vents
from input_parameters import fluxrate
from input_parameters import emiss
from input_parameters import dtsave_list
from input_parameters import fixed_vents

n_samples = 200
main_dir = os.getcwd()
row_fits = {}
old_vents = []
old_seq_vents = []

# parameters which define the range within search e sample the proposal
dist_max = 180      # max distance within search the vents
x_maxs = [v[0] + dist_max for v in vents.values()]
x_mins = [v[0] - dist_max for v in vents.values()]
y_maxs = [v[1] + dist_max for v in vents.values()]
y_mins = [v[1] - dist_max for v in vents.values()]
x_max = x_maxs[0]
x_min = x_mins[0]
y_max = y_maxs[0]
y_min = y_mins[0]
dist = 50           # distance within a new sample is defined through the jump distribution
res = 10            # DEM resolution to discretize the sampling
n_cell = int(dist/res)
jump_dist_array = np.linspace(-n_cell, n_cell, n_cell*2+1) #values of te samples of the jumping distribution es. .. -20, -10, 0, 10, 20... in meters
p_vent = [((n_cell+1)-abs(r))/((n_cell+1)**2) for r in jump_dist_array] #probability jumping distribution
h2o_min = -2 
h2o_max = -0.5
flux_pct_min = 0.5
flux_pct_max = 1.5
jump_h2o_array = [-0.2, -0.1, 0, 0.1, 0.2]
flux_pct_res = 0.1
p_h2o = p_flux_pct = [0.1, 0.25, 0.3, 0.25, 0.1]
dx = (x_max - x_min)/5
dy = (y_max - y_min)/5
dh2o = (h2o_max - h2o_min)/5
dflux = (flux_pct_max - flux_pct_min)/5

#intervals used to built the array and matrix for the initial samples using Hypercube
x_interval_array = [[x_min + k*dx, x_min + (k+1)*dx] for k in range(0, 5)]
y_interval_array = [[y_min + k*dy, y_min + (k+1)*dy] for k in range(0, 5)]
h2o_interval_array = [[h2o_min + k*dh2o, h2o_min + (k+1)*dh2o] for k in range(0, 5)]
flux_interval_array = [[flux_pct_min + k*dflux, flux_pct_min + (k+1)*dflux] for k in range(0, 5)]
mtx_interval = np.ones((5,5)) 


### function which creates input files and run simulation, if a configuration has been already sampled the code jump it but still count has a new proposal ###
def calc_fit(init_row, df, n_vent, i_sim):
	configuration = df.iloc[-1]
	key = tuple(configuration)
	if key in row_fits:
		i_org, fit = row_fits[key]
		print(f"Simulation {i_sim} for {key} reuses fit from {i_org}")
		return fit
	print(f"Running simulation {i_sim} for {key}")
	create_inputfiles(init_row, configuration, n_vent, i_sim, main_dir)
	fit = float(run_sim(i_sim))
	row_fits[key] = (i_sim, fit)
	return fit


### function which calculates the proposed values ###
def calculate_proposed(current, array, p, r, vmin, vmax):
	while True:
		proposed = current + np.random.choice(array, 1, p)[0]*r
		if proposed >= vmin and proposed <= vmax: break
	return proposed


### function which controls if two vents have the same coordinates ###
def vent_control(vents_p):
	k = list(vents_p.keys())
	n_k = len(k)
	for j in range(n_k - 1):
		for z in range(j +1, n_k):
			if vents_p[k[j]] == vents_p[k[z]]:
				return True 
	return False

### a new row with proposed parameters is added in the df
def add_row(df, i, run_seq, x, y, flux_pct, tstart, tend, dtsave, fluxrate,  emiss, h2o, log_h2o):
	new_row = [i, run_seq, x, y, flux_pct, *tstart, *tend, dtsave, fluxrate,  emiss, h2o, log_h2o] 
	new_df = pd.DataFrame([new_row], columns=df.columns)                     #creation of the second dataframe
	df = pd.concat([df, new_df], ignore_index=True)                          #the second dataframe is joined with the first dataframe
	return df


### function that perform the simulations and the metropolis algorithm
def metropolis(init_row, run_seq, samples_csv, n_vent, x_current, y_current, vents_current, flux_pct_current, tstart, tend, dtsave, fluxrate, emiss, h2o_current, log_h2o_current, fit_list, index_accepted, count_list, jump_flux_pct_array):
	
	while True:
		indices_mtx = np.where(mtx_interval == 1)
		random_index = np.random.choice(len(indices_mtx[0]))
		r_i_x, r_i_y = indices_mtx[0][random_index], indices_mtx[1][random_index]		
		selected_interval_y = y_interval_array[r_i_y]
		selected_interval_x = x_interval_array[r_i_x]
		x_current = (random.randint(selected_interval_x[0]//res, selected_interval_x[1]//res))*res
		y_current = (random.randint(selected_interval_y[0]//res, selected_interval_y[1]//res))*res
		x_current = 500000
		y_current = 4177450
		i = 0
		colonne = [
			'i',
			'run_seq',
			*(f'lon_{v}' for v in range(n_vent)),
			*(f'lat_{v}' for v in range(n_vent)),
			*(f'flux_pct_{v}' for v in range(n_vent)),
			*(f'tstart_{v}' for v in range(n_vent)),
			*(f'tend_{v}' for v in range(n_vent)),
			'dtsave', 'fluxrate', 'emiss', 'ph2o', 'log_h2o'
		]

		df = pd.DataFrame(columns=colonne)  # creation of the dataframe
		df = add_row(df, i, run_seq, x_current, y_current, flux_pct_current, tstart, tend, dtsave, fluxrate, emiss, h2o_current, log_h2o_current) 
		df.to_csv(samples_csv, index=False)     #the file csv is updated

		fit_current = calc_fit(init_row, df, n_vent, i)   #function which create input filles and run the simulation with proposed values
		
		if fit_current == 0:
			mtx_interval[r_i_x, r_i_y] = 0 
			shutil.rmtree ("sim_00000")
			continue
		else: 
			mtx_interval[r_i_x,:] = 0
			mtx_interval[:,r_i_y] = 0
			#x_interval_array.pop(random_i_x)
			#y_interval_array.pop(random_i_y)
			break

	fit_list.append(fit_current)
	count = 1    		#a variable which count how many times is accepted the current vent

	print ("i : 0")
	print ("fit: ", fit_current)
	print ("counting : 1")
	print ()

	#vents_proposed = copy.deepcopy(vents_current)
	x_proposed = np.zeros(n_vent)
	y_proposed = np.zeros(n_vent)
	flux_pct_proposed = np.zeros(n_vent)

	##### start of the Loop of the current sequence #####

	for i in range(1, n_samples):
		
		print(i)
		### calculation of proposed values for h2o, flux and vents position ###

		log_h2o_proposed = round(calculate_proposed(log_h2o_current, jump_h2o_array, p_h2o, 1, h2o_min, h2o_max), 1)
		h2o_proposed = round(10**log_h2o_proposed, 2)
	
		x_proposed = calculate_proposed(x_current, jump_dist_array, p_vent, res, x_min, x_max)
		y_proposed = calculate_proposed(y_current, jump_dist_array, p_vent, res, y_min, y_max)
		flux_pct_proposed = round(calculate_proposed(flux_pct_current, jump_flux_pct_array, p_flux_pct, 1, flux_pct_min, flux_pct_max), 1)
		flux_pct_proposed = round(flux_pct_proposed, 1)
			
		df = add_row(df, i, run_seq, x_proposed, y_proposed, flux_pct_proposed, tstart, tend, dtsave, fluxrate, emiss, h2o_proposed, log_h2o_proposed) 
		df.to_csv(samples_csv, index=False)                                                                                           

		fit_proposed = calc_fit(init_row, df, n_vent, i)
		fit_list.append(fit_proposed)

		acceptance_ratio = fit_proposed/fit_current
		fit_best = np.max(fit_list)
		acceptance_probability = min(1, acceptance_ratio)

		is_accepted = False

		if fit_proposed >= fit_current or np.random.uniform(0, 1) < acceptance_probability:   # acceptance criterio
			is_accepted = True

		if is_accepted:     #if accepted start a new count and the proposal became the current
			count = 1
			fit_current = fit_proposed
			x_current = x_proposed
			y_current = y_proposed
			log_h2o_current = log_h2o_proposed
			flux_pct_current = flux_pct_proposed
			if i > int(n_samples/3):			#warm-up: the first 30% of proposed are not included in the accepted
				index_accepted.append(i)
				count_list.append(count) 
		else:
			count += 1  
			if i > int(n_samples/3) and len(count_list) > 0:	#if the proposed is not accepted the count of the current is updated but not when the first proposed after the warm-up is not accepted
				count_list[-1] = count

		print ("i :", i)
		print ("fit proposed: ", fit_proposed)
		print ("fit current: ", fit_current)
		print ("best fit: ", fit_best)
		print ("acceptance_ratio: ", acceptance_ratio)
		print ("counting: ", count)
		print ()
	
	#### end of the loop  ####

	df['fit'] = fit_list
	df.to_csv(samples_csv, index=False)

	df_accepted = df.loc[index_accepted].copy()
	df_accepted['counting'] = count_list
	filename_accepted = f'accepted_{run_seq}.csv'

	df_accepted.to_csv(filename_accepted, index=False)


###  main function that develop the variuos steps of the workflow and call the metropolis algorithm
def main():
	
	for parameters in par_vents.items():
		
	# coordinates and time start and end of the observed vents
	tstart = parameters["tstart"]
	tend = parameters["tend"]
	dtsave = parameters["dtsave"]
	vents_current = copy.deepcopy(vents)
	
	n_vent = len(seq_vents)    # total number of active vents considering each time interval
	
	n_run = (max_n_vents * 3) +1 +1      # define the number of starting points
		

	### run of the sequences starting from n points ###
	
	for run_seq in range(n_run):

		# creation of the dir with the block simulation
		run_dir = f'run_{run_seq}'
		os.mkdir(run_dir)
		os.chdir(run_dir)

		# definition of the initial conditions
		count_list = []
		index_accepted = []
		fit_list = []
		samples_csv = f'samples_{run_seq}.csv'
		
		random_i_h2o = random.randint(0, len(h2o_interval_array) - 1)
		selected_interval = h2o_interval_array[random_i_h2o]
		log_h2o_current = round(random.uniform(selected_interval[0], selected_interval[1]), 1)
		log_h2o_current = -0.5
		h2o_current = round(10**log_h2o_current, 2)
		h2o_interval_array.pop(random_i_h2o)
		flux_pct_res_per_vent = 0.1 
		jump_flux_pct_array = [-2*flux_pct_res_per_vent, -flux_pct_res_per_vent, 0, flux_pct_res_per_vent, 2*flux_pct_res_per_vent]
		random_i_flux = random.randint(0, len(flux_interval_array) - 1)
		selected_interval = flux_interval_array[random_i_flux]
		flux_pct_current = round(random.uniform(selected_interval[0], selected_interval[1]), 1)
		flux_interval_array.pop(random_i_flux)
		flux_pct_current  = round(np.random.uniform(0.5, 1.5), 1))
		x_current = np.zeros(n_vent)
		y_current = np.zeros(n_vent)
		
		metropolis(init_row, run_seq, samples_csv, n_vent, x_current, y_current, vents_current, flux_pct_current, tstart, tend, dtsave, fluxrate, emiss, h2o_current, log_h2o_current, fit_list, index_accepted, count_list, jump_flux_pct_array)

		post_process(n_vent, flux_pct_res)
		os.chdir(main_dir)

if __name__ == "__main__":
	main()
