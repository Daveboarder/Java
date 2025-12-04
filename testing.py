from LIBSmethods import voigt, triangular_function, partition_function
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def print_hdf5_tree(name, obj, prefix="", is_last=True):
    """Print HDF5 structure as a tree"""
    connector = "└── " if is_last else "├── "
    print(f"{prefix}{connector}{name.split('/')[-1]}")
    
    if isinstance(obj, h5py.Group):
        items = list(obj.items())
        for i, (key, value) in enumerate(items):
            is_last_item = (i == len(items) - 1)
            extension = "    " if is_last else "│   "
            print_hdf5_tree(f"{name}/{key}", value, prefix + extension, is_last_item)
    
#file_path='/home/LIBS/prochazka/data/Running_projects/24_0057_LIBSdata_processing/Methods/Mapping/Java/output_c1.h5' 
file_path = '/home/LIBS/prochazka/data/Running_projects/25_0069_3D_chemical_imaging/Measurements/mandible 266nm/mandible_266_v1.h5'
with h5py.File(file_path, 'r') as file:
    wavelength = file['measurements/Measurement_1/libs/calibration'][:]
    data = file['/measurements/Measurement_1/libs/data'][:]
    x_pos = file['measurements/Measurement_1/libs/metadata/X_pos'][:]
    y_pos = file['measurements/Measurement_1/libs/metadata/Y_pos'][:]
    #write a structured list of all variables in the hdf5 file to a terminal
    print("HDF5 File Structure:")
    print("=" * 50)
    for key in file.keys():
        print_hdf5_tree(key, file[key], "", True)

    b1_w = 280.1
    b2_w = 280.6
    b1 = np.argmin(np.abs(wavelength - b1_w))
    b2 = np.argmin(np.abs(wavelength - b2_w))
    gamma = 0.1
    sigma = 0.1
    x = wavelength[b1:b2]
    data_slice = data[:,b1:b2]

    print(f"data slice: {data_slice.shape}")
    max_data = np.max(data_slice, axis=0)
    
    a1 = b1 + np.argmax(max_data)
    x0 = wavelength[a1]
    print(f"x0: {x0}")

    triangle = triangular_function(b1, a1, b2)
    print(f"triangle: {triangle}")


    selected_spectrum = 45580

    basis = voigt(x, x0, np.max(max_data), gamma, sigma)
    denom = np.dot(basis, basis)
    amplitudes = data_slice @ basis / denom
    popt, pcov = curve_fit(voigt, x, max_data, p0=[x0, np.max(max_data), gamma, sigma])
    print(f"popt: {popt}")
    try:
        test_popt, _ = curve_fit(voigt, x, data_slice[selected_spectrum,:], p0=[x0, np.max(data_slice[selected_spectrum,:]), gamma, sigma])
        print(f"test_popt: {test_popt}")
    except RuntimeError:
        print("RuntimeError: Failed to fit voigt function")
        test_popt = [x0, np.max(data_slice[selected_spectrum,:]), gamma, sigma]
        
    U_I, U_II = partition_function('C', 10000)
    print(f"U_I: {U_I}")
    print(f"U_II: {U_II}")
    test_real_fit = voigt(x, *test_popt)    #test the fit on a single spectrum
    #correlation coefficient between max_data and real_fit from pcov
    std = np.sqrt(np.diag(pcov))
    correlation = pcov / (std[0] * std[1])
    print(f"correlation: {correlation}")
    print(f"std: {std}")
    real_fit=voigt(x, *popt)
    intensity = np.trapz(real_fit, x)
    RMSE = np.sqrt(np.mean((max_data - real_fit)**2))
    NRMSE = (RMSE / np.max(max_data)) * 100
    R2 = 1 - (np.sum((max_data - real_fit)**2) / np.sum((max_data - np.mean(max_data))**2))
    R2_test = 1 - (np.sum((data_slice[selected_spectrum,:] - test_real_fit)**2) / np.sum((data_slice[selected_spectrum,:] - np.mean(data_slice[selected_spectrum,:]))**2))
    print(f"RMSE: {RMSE}")
    print(f"NRMSE: {NRMSE}")
    print(f"R²: {R2}")
    print(f"pcov: {pcov}")
    print(f"intensity: {intensity}")
    print(f"amplitudes: {amplitudes.shape}")
    print(f"denom: {denom}")
    print(f"basis: {basis}")
    print(f"max data: {max_data.shape}")
    print(f"real fit: {popt}")
    
    plt.plot(x, max_data, label='Max Data')
    plt.plot(x, real_fit, label=f'Real Fit,R2:{R2}')
    #plt.plot(x, basis*np.max(max_data), label='Basis')
    plt.plot(x, test_real_fit, label=f'Test Real Fit,R2:{R2_test}')
    plt.plot(x, data_slice[selected_spectrum,:], label='Data_test')
    plt.legend()
    plt.savefig('real_fit.png')
    plt.close()
