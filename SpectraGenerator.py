#Function generates optical emission spectra based on given elemnt, temperature and pressure and electron density
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from LIBSmethods import voigt, partition_function
import h5py
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
"""def generate_spectra(element, temperature, electron_density):
    #Calculate the energy of the electron in the ground state
    energy_ground_state = const.Rydberg * const.Z**2 / (1 + const.Z**2)
    #Calculate the energy of the electron in the excited state
    energy_excited_state = const.Rydberg * const.Z**2 / (1 + const.Z**2)
    #Calculate the energy of the electron in the ionized state
    energy_ionized_state = const.Rydberg * const.Z**2 / (1 + const.Z**2)
    #Calculate the energy of the electron in the metastable state
    energy_metastable_state = const.Rydberg * const.Z**2 / (1 + const.Z**2)
"""

#Constants========================================================
kb = const.k*10**6 #erg/K
h = const.h*10**6 #erg*s
c = const.c #m/s
e = const.e #C
me = const.electron_mass*1000 #g
#-----------------------------------------------------------------
Te = 10000 #K
Ne = 1.79e+18 #cm^-3
l = 1.4e-01
N=1
C= 1

print(f"kb: {kb}")
print(f"h: {h}")
print(f"c: {c}")
print(f"e: {e}")
print(f"me: {me}")

#read sample wavelengths

file_path = '/home/LIBS/prochazka/data/Running_projects/25_0069_3D_chemical_imaging/Measurements/mandible 266nm/mandible_266_v1.h5'
with h5py.File(file_path, 'r') as file:
    wavelength = file['measurements/Measurement_1/libs/calibration'][:]

element = 'Cu'

conn = sqlite3.connect('/home/LIBS/prochazka/data/Running_projects/24_0057_LIBSdata_processing/Methods/Mapping/Java/QuantParam.db')
cursor = conn.cursor()  
# Get the partition function for the element elem
cursor.execute("SELECT Elem_name, ion_state, Wavelength, Ei, Ek, gi, gk, Ak FROM QuantParam WHERE Elem_name = ?", (element,))
QuantParam = pd.DataFrame(cursor.fetchall(), columns=['Elem_name', 'ion_state', 'Wavelength', 'Ei', 'Ek', 'gi', 'gk', 'Ak'])
conn.commit()
conn.close()

print(f"QuantParam: {QuantParam.head()}")

PF_I, PF_II = partition_function(element, Te)
S10 = (((2*PF_II)/(Ne*PF_I))*((me*kb*Te)/((h**2)/(2*np.pi)))**(1.5))*np.exp(-(QuantParam['Ek']*1.60217e-12)/(kb*Te)) if not QuantParam.empty else 1
kt = ((QuantParam['Wavelength']**4)/(8*np.pi*c)) * (QuantParam['Ak']*QuantParam['gk']*np.exp(-(QuantParam['Ei']*1.60217e-12)/(kb*Te))) * (1-np.exp(-1.60217e-12*(QuantParam['Ek']-QuantParam['Ei'])/(kb*Te))) / np.where(QuantParam['ion_state']=="I", PF_I, PF_II)
ri = np.where(QuantParam['ion_state']=="I", 1/(1+S10), S10/(1+S10))
ion_state = np.where(QuantParam['ion_state']=="I", 1, 0)
Lp = ((8*np.pi*h*c)/(10*QuantParam['Wavelength']**3))*N*np.exp((-1.60217e-12*(QuantParam['Ek']-QuantParam['Ei']))/(kb*Te))*(QuantParam['gk']/QuantParam['gi'])
tau = C*N*ri*l*kt
Ifin = Lp*(1-np.exp(-tau))

print(f"Lp: {Lp.shape}")
print(f"Ifin: {Ifin}")
print(f'Ifin shape: {Ifin.shape}')
print(f'Max intensity: {np.max(Ifin)}')



plt.plot(QuantParam['Wavelength'], Lp)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (a.u.)')
plt.title('Spectrum')
plt.savefig('SpectraGenerator.png')
plt.close()