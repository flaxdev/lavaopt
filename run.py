import subprocess
import sys
import glob
import os
import numpy as np
from osgeo import gdal, gdal_array
from scipy.spatial.distance import cdist
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation, binary_erosion
from input_parameters import dem
from input_parameters import reale

def salva_raster(output, raster_ds):
	
	array = raster_ds.GetRasterBand(1).ReadAsArray()
	array = np.where(array > 0, 1, 0)
	array = array.astype(np.int8)
	array = binary_fill_holes(array, structure=np.ones((3, 2)))
	array = np.logical_xor(binary_dilation(array), binary_erosion(array))

	driver = gdal.GetDriverByName("GTiff")
	nuovo_ds = driver.Create(output, raster_ds.RasterXSize, raster_ds.RasterYSize, 1, gdal.GDT_Byte)
	nuovo_ds.SetGeoTransform(raster_ds.GetGeoTransform())
	nuovo_ds.SetProjection(raster_ds.GetProjection())
	nuovo_ds.GetRasterBand(1).WriteArray(array)

	nuovo_ds = None
	
def get_coordinates_with_value(raster_ds, value):
	
	transform = raster_ds.GetGeoTransform()
	indices = np.argwhere(raster_ds.GetRasterBand(1).ReadAsArray() == value)
	coordinates = [((transform[0] + transform[1] * col + transform[2] * row),
		(transform[3] + transform[4] * col + transform[5] * row))
		for row, col in indices]

	return np.array(coordinates)

def calculate_distance(coordinates1, coordinates2):
	
	distanze = cdist(coordinates1, coordinates2, 'euclidean')
	distanze_minime = np.min(distanze, axis=1)
	distanza_massima = np.max(distanze_minime)
	return distanza_massima


def run_sim(i:
	current_dir =  os.getcwd()
	outdir = "sim_" + "{:05d}".format(i)
	working_dir = os.path.join(current_dir, outdir)

	init = working_dir+"/init.txt"
	vent = working_dir+"/vent.txt"

	run = subprocess.run(['../gpuflow', dem, init, vent, 'state', '--single-state', '-d', outdir], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  #run the simulation

	os.chdir(working_dir)	  	#enter in the dir where simulation is run

	reale_frontiera = 'path to the raster representing the border of the real lava flow'
	
	if os.path.exists(reale_frontiera):
		pass
	else:
		reale_ds = gdal.Open('path to the raster representing the reference lava flow')
		salva_raster(reale_frontiera, reale_ds) #save the raster of the border of the real lava flow
	
	fit = 0
	for fname in glob.glob('*final.bsq'):  #look for the raster created at end fo simulation to use for the fit

		#run the calc-fit to calculate the Jaccard Index
		fit = float(subprocess.run(['../../calc-fit', fname, reale[step-1], '-q'], capture_output=True, text=True).stdout.splitlines()[-1])
		
		sim_ds = gdal.Open(fname)
		reale_ds = gdal.Open('path to the raster representing the border of the real lava flow')
		salva_raster('sim_frontiera.tif', sim_ds)
		sim_ds = gdal.Open('sim_frontiera.tif')
		coordinates_reale = get_coordinates_with_value(reale_ds, 1)
		coordinates_sim = get_coordinates_with_value(sim_ds, 1)
		max_distances_1 = calculate_distance(coordinates_reale, coordinates_sim)
		max_distances_2 = calculate_distance(coordinates_sim, coordinates_reale)
		
		#calculate the Hausdorff distance
		if max_distances_1 > max_distances_2: h = max_distances_1
		else: h = max_distances_2
		d = 200 #10% of the total lava lenght (2 km)
		p = 2
		likelihood = fit*np.exp(-(h/d)**p)

	if fit == 0:
		likelihood = 0


	os.chdir(current_dir)

	return likelihood

if __name__ == "__main__":
	pass   
