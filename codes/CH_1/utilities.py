#-----------------------------------------------------------------------------------------#
import matplotlib
import matplotlib.pyplot as plt
import os
#-----------------------------------------------------------------------------------------#
params = {
	'savefig.dpi': 300,  
	'figure.dpi' : 300,
	'axes.labelsize':10,  
	'axes.titlesize':10,
	'axes.titleweight': 'bold',
	'legend.fontsize': 8,
	'xtick.labelsize':8,
	'ytick.labelsize':8,
	'font.family': 'serif',
}
matplotlib.rcParams.update(params)
#-----------------------------------------------------------------------------------------#

if not os.path.exists('save_figures'):
	os.makedirs('save_figures')

#-----------------------------------------------------------------------------------------#

def gravity_acceleration(width, hight, depth, univeral_gravity, density, distance):
	volume = width*hight*depth
	return (univeral_gravity*density*volume) / pow(distance, 2)

def scatter_plot(stations, width, hight, depth, univeral_gravity, density):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_subplot(111)
    for i in range (0, len(stations)):
        g = gravity_acceleration(width, hight, depth, univeral_gravity, density, stations[i][2])
        g_mgal = g * 1e5
        scatter = ax.scatter(stations[i][0], g_mgal, s=60, linewidths=2, edgecolors='black',
                             color='tab:orange')
    ax.set_title('Gravity Acceleration per Station')
    ax.set_xlabel('Station Index')
    ax.set_ylabel('Gravity Acceleration (mGal)')
    plt.savefig('save_figures/' + 'ex_01_ans' + '.webp', format='webp', dpi=100,
                bbox_inches='tight', transparent=False, pad_inches=0)
    plt.show()