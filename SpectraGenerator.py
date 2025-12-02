#Function generates optical emission spectra based on given elemnt, temperature and pressure and electron density
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from LIBSmethods import voigt, partition_function

#Constants
kB = const.k*10**6 #erg/K
h = const.h*10**6 #erg*s
c = const.c #m/s
e = const.e #C
m_e = const.m_e*10**3 #g
m_p = const.m_p
m_n = const.m_n
m_u = const.m_u
m_u = const.m_u

Te = 12000 #K
Ne = 1.79e+18 #cm^-3

print(f"kB: {kB}")
print(f"h: {h}")
print(f"c: {c}")
print(f"e: {e}")
print(f"m_e: {m_e}")
print(f"m_p: {m_p}")
print(f"m_n: {m_n}")
print(f"m_u: {m_u}")



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