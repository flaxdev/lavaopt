import os
import shutil

#global variables
vent_header = "# MAGFLOW vent file v2\n"
vent_block = """location: {LON} {LAT}
temperature: 1360
tstart: {TSTART}
tend: {TEND}
width: 10
flux_pct: {FLUX_PCT}
fluxrate: {FLUXRATE}

"""

init_block = """density:	2600.0			// densita (kg.m-3)
Tground:	300			// temperatura suolo
Tlsolid:	1223			// temperatura solidificazione lava
Tlliquid:	1360			// temperatura lava stato liquido
Tclinker:	1303			// temperatura lava stato liquido
emissivity:	{EMISS}			// emissivita
crad:		1			// coeff. correttivo perdite per radiazione
hrad:		0.2			// altezza minima radiante
hcapacity:	840.0			// capacita termica
visclaw:        giordano_dingwell
ph2o:           {H2O}
avisc:		24.4226			// coeff. legge viscosita
bvisc:		-0.0166			// 0.7 % di acqua
ays:		13.9997			// coeff. legge yield strength
bys:		-0.0089			//
tend:		{TEND}			// fine della simulazione (s)
dt1:		2			// passo temporale durante l'eruzione (s)
dt2:		10			// passo temporale dopo l'eruzione (s)
dtsave:		{DTSAVE}			// tempo fra scrittura risultati (s)
niter:		1			// numero iterazioni Monte Carlo
qmax:		100
cooling:	1
dtadapt:	1
"""

def create_files(init_row, row, n_vent):

	fname_init = "init.txt"
	fname_vent = "vent.txt"

	emiss_ = row['emiss']
	ph2o = row['ph2o']
	tstart_ = row['tstart']
	tend_ = row['tend']
	dtsave_ = row['dtsave']
	fluxrate_ = row['fluxrate']
	
	with open(fname_vent, 'w') as vent:
		vent.write(vent_header)

		for n in range(n_vent):
			latitude = row[f'lat_{n}']
			longitude = row[f'lon_{n}']
			flux_pct = row[f'flux_pct_{n}']

			vent.write(vent_block.format(
				LON=longitude, LAT=latitude,
				TSTART=tstart, TEND=tend,
				FLUXRATE=fluxrate_, FLUX_PCT=flux_pct))	

	with open(fname_init, 'w') as init:
		init.write(init_block.format(EMISS=emiss_, H2O=ph2o, TEND=dtsave_, DTSAVE=dtsave_))


def create_inputfiles(init_row, configuration, n_vent, i, main_dir):

	current_dir =  os.getcwd()

	working_dir = "sim_"+"{:05d}".format(i)

	try:
		os.mkdir(working_dir)
		working_dir = os.path.join(current_dir,working_dir)
		os.chdir(working_dir)
	except FileExistsError:
		working_dir = os.path.join(current_dir, working_dir)
		os.chdir(working_dir)

	create_files(init_row, configuration, n_vent)

	file_fluxrate = configuration['fluxrate']

	src = os.path.join(main_dir,file_fluxrate)
	dest = os.path.join(current_dir,file_fluxrate)

	shutil.copy2(src, dest)
	os.chdir(current_dir) 

if __name__ == "__main__":
    pass
