#-----------------------------------------------------------------------------------------#
import class_read_GPR as C
#-----------------------------------------------------------------------------------------#

gssi_file = 'data/DZT_files/PONGT002.DZT'
percent_clip = 100

#-----------------------------------------------------------------------------------------#

plotter = C.GPRDataPlotter(gssi_file)
plotter.plot(percent_clip)  

#-----------------------------------------------------------------------------------------#