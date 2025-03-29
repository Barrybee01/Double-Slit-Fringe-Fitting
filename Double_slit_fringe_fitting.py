import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


##################################################################################
#Determine the observed intensity distribution

img_data = pd.read_csv('Image7 intensity profile.csv') #chamge file name as needed

pixels = np.array(img_data['Distance_(_)'])
counts = np.array(img_data['Gray_Value'])

norm_intensity = counts / np.linalg.norm(counts)

plt.figure(figsize=(10,6))
plt.plot(pixels, norm_intensity, c='k')
plt.ylabel('Intensity (a.u.)', fontsize=12)
plt.show()
#################################################################################
#Initiating system params and outputting ideal case plots
def keV2m(keV):
    """Converts Photon Energy [keV] to wavelength [m].
    .. note:: Calculation in Vacuum.
    """
    wl = 1./(keV*1000)*4.1356*(10**(-7))*2.9998
    return wl

ener = 6.9 #keV
I_0 = 1
a = 3.2e-6 # slit width in  m
d = 101e-6 # slit seperation in m
L = 6.0 # slit to detec dist in m
detec_pixel_size =0.72e-6 #pixel size

x = np.arange(-500,500)*detec_pixel_size + 1e-9

def coh_inter_patt(x):
    lam = keV2m(ener)
    sinq_part = ((np.sin((np.pi*x*a)/(lam*L))/((np.pi*x*a)/(lam*L)))**2)
    sinq_part = np.nan_to_num(sinq_part, nan=1)
    inter_pat = I_0*(np.cos((np.pi*x*d)/(lam*L))**2)*sinq_part
    return inter_pat

def coh_inter_patt_sft(x,detec_pixel_size,sft=10):
    lam = keV2m(ener)
    sinq_part = ((np.sin((np.pi*(x+(sft*detec_pixel_size))*a)/(lam*L))/((np.pi*(x+(sft*detec_pixel_size))*a)/(lam*L)))**2)
    sinq_part = np.nan_to_num(sinq_part, nan=1)
    inter_pat = I_0*(np.cos((np.pi*(x+(sft*detec_pixel_size))*d)/(lam*L))**2)*sinq_part
    print(sft) #debugging
    return inter_pat


inter_pat = coh_inter_patt(x)
inter_pat_sft = coh_inter_patt_sft(x,detec_pixel_size,sft=10)

max_vals = np.sort(argrelextrema(inter_pat, np.greater)[0])
min_vals = np.sort(argrelextrema(inter_pat, np.less)[0])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x,inter_pat,label='central')
ax1.plot(x[max_vals],inter_pat[max_vals],label = 'maxima')
ax1.plot(x[min_vals],inter_pat[min_vals],label = 'minima')
#ax1.plot(inter_pat_sft,label='shifted')
ax1.legend()
##############################################################################
# 'Blurred' patterns, curve fitting procedure
extension = 7.8  #start with value from goodness of fit calc and go from there
blur_patt  = np.zeros_like(coh_inter_patt(x))

for i in np.arange(0, extension, 0.05):
    blur_patt += coh_inter_patt_sft(x,detec_pixel_size,sft=i)
    blur_patt += coh_inter_patt_sft(x,detec_pixel_size,sft=-i)

blur_patt = blur_patt/np.linalg.norm(blur_patt)
max_vals = np.sort(argrelextrema(inter_pat, np.greater)[0])
min_vals = np.sort(argrelextrema(inter_pat, np.less)[0])


blur_max_vals = np.sort(argrelextrema(blur_patt, np.greater)[0])
blur_min_vals = np.sort(argrelextrema(blur_patt, np.less)[0])

fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(111)
ax1.plot(pixels, norm_intensity, color='red', label='Observed')
ax1.plot(blur_patt,label='Analytical', color='k')
ax1.set_ylabel('Intensity (a.u.)', fontsize=12)
ax1.set_xlabel('Pixel Number', fontsize=12)
plt.show()
###############################################################################
#Goodness of fit calculation
#Calculates the R^2 value over a range of source extension values
def Blurred_fringes(N):
  extension = N
  blur_patt  = np.zeros_like(coh_inter_patt(x))

  for i in np.arange(0,extension,0.1):

    blur_patt += coh_inter_patt_sft(x,detec_pixel_size,sft=i)
    blur_patt += coh_inter_patt_sft(x,detec_pixel_size,sft=-i)

  blur_patt = blur_patt/np.linalg.norm(blur_patt)
  blur_max_vals = np.sort(argrelextrema(blur_patt, np.greater)[0])
  return blur_patt

def R_squared(N):
  fit = Blurred_fringes(N)
  ss_res = np.sum(np.square(norm_intensity[350:650] - fit[350:650]))

  data_mean = np.mean(norm_intensity[350:650])
  ss_tot = np.sum(np.square(norm_intensity[350:650] - data_mean))

  R2 = 1 - (ss_res / ss_tot)
  return R2

r2_scan = [R_squared(j) for j in np.arange(1,30, 0.1)]
N_range = [j for j in np.arange(1,30,0.1)]

r2_arr = np.array(r2_scan)
r2_max = np.max(r2_arr)

best_fit = max(r2_scan)
bestfit_index = r2_scan.index(r2_max)


print(f'The best fit occurs at N={r2_scan.index(r2_max)+1}')
print(f'The R2 of the best fit is {r2_max}')
print(f'R2 of preferred fit is {R_squared(12)}')

plt.plot(N_range, r2_scan)
plt.xlabel('N')
#plt.yscale('log')
plt.ylabel(r'$R^{2}$')
plt.show()
###########################################################################
#Visibility Calculation
def calculate_visibility(norm_intensity):
    minima = np.sort(argrelextrema(norm_intensity, np.less, order=6)[0]) #6 neighbouring points seems to work
    maxima = np.sort(argrelextrema(norm_intensity, np.greater, order=6)[0])

    central_index = 500

    closest_max_idx = maxima[np.argmin(np.abs(maxima - central_index))]
    closest_min_idx = minima[np.argmin(np.abs(minima - central_index))]

    if abs(closest_max_idx - central_index) < abs(closest_min_idx - central_index):
        max_intensity = norm_intensity[closest_max_idx]
        min_intensity = norm_intensity[minima[np.argmin(np.abs(minima - closest_max_idx))]]
    else:
        min_intensity = norm_intensity[closest_min_idx]
        max_intensity = norm_intensity[maxima[np.argmin(np.abs(maxima - closest_min_idx))]]

    visibility = (max_intensity - min_intensity) / (max_intensity + min_intensity)

    plt.plot(pixels, norm_intensity)
    plt.plot(pixels[maxima], norm_intensity[maxima], 'ro', label='maxima')
    plt.plot(pixels[minima], norm_intensity[minima], 'go', label='minima')
    plt.legend(loc='upper right')
    return visibility

V = calculate_visibility(norm_intensity)
print(f'The visibility is {V}')