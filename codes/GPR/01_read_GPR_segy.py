#-----------------------------------------------------------------------------------------#
import class_read_GPR as C
#-----------------------------------------------------------------------------------------#

# NOTE Load the SEGY file
segy_file = 'data/segy_profiles_migrated_final/Shipyard_Lane_migrated.sgy'
percent_clip = 90

#-----------------------------------------------------------------------------------------#

# NOTE Plot the SEGY file
plotter = C.SEGYDataPlotter(segy_file)
plotter.plot(percent_clip)  

#-----------------------------------------------------------------------------------------#