import numpy as np
from scipy.signal import correlate
from scipy.stats import pearsonr
from scipy.special import wofz
from scipy.optimize import curve_fit
import pandas as pd

PARTITION_FUNCTION_PATH = "/home/LIBS/prochazka/data/Running_projects/24_0011_CF_spark/CF SPARK/Methods/Git/CF_OES/PartF_var.txt"
EION_PATH = "/home/LIBS/prochazka/data/Running_projects/24_0011_CF_spark/CF SPARK/Methods/Git/CF_OES/E_ion.txt"

def snv(data):
    """
    Standard Normal Variate (SNV) normalization.
    Normalizes each row to have zero mean and unit variance.
    
    Handles zero variance cases (rows with all identical values) by adding
    a small epsilon to prevent division by zero, which would produce inf/nan values.
    """
    data_mean = np.mean(data, axis=1, keepdims=True)
    data_std = np.std(data, axis=1, keepdims=True)
    
    # Add small epsilon to prevent division by zero when variance is zero
    # When std is zero, mean-centered data is already zero, so result will be zero
    epsilon = 1e-10
    data_std = np.maximum(data_std, epsilon)
    
    data_snv = (data - data_mean) / data_std
    return data_snv

def triangular_function(b1,a1,b2):
    left = np.linspace(0, 1, a1-b1)
    right = np.linspace(1, 0, b2-a1)
    return np.concatenate((left, right))

def calculate_signal_area(data, b1,b2):
    baseline = np.trapezoid([data[b1], data[b2]])
    signal_area = np.trapezoid(data[b1:b2]) - baseline
    return signal_area

def voigt(x, x0_fit, amplitude_fit, gamma_fit, sigma_fit):
    z = (x - x0_fit + 1j * gamma_fit) / (sigma_fit * np.sqrt(2))
    voigt_fit = amplitude_fit * wofz(z).real / (sigma_fit * np.sqrt(2 * np.pi))
    return voigt_fit

def peak_intensity(data, wavelengths, b1_w, b2_w):
    # Find the indices of the wavelengths
    b1 = np.argmin(np.abs(wavelengths - b1_w))
    #a1 = np.argmin(np.abs(wavelengths - a1_w))
    b2 = np.argmin(np.abs(wavelengths - b2_w))
    max_data = np.max(data[:,b1:b2], axis=0)
    a1 = b1 + np.argmax(max_data)

    
    # Create triangular function
    triangle = triangular_function(b1, a1, b2)

    # Calculate the correlation
    correlation = np.apply_along_axis(correlate,1,data[:,b1:b2], triangle, mode='valid') #correlate(data[b1:b2], triangle, mode='valid')

    # Calculate the pearsor correlation coefficient and p-value
    def pearsonr_wrapper(row, triangle):
        r, p = pearsonr(row, triangle)
        return r, p

    results = np.apply_along_axis(pearsonr_wrapper,1,data[:,b1:b2], triangle)
    r_value = results[:,0]
    p_value = results[:,1]

    # Calculate the signal area
    signal_area = np.apply_along_axis(calculate_signal_area,1,data,b1,b2) #calculate_signal_area(data, b1, b2)

    # Determine the peak intensity on significance level
    correlation = np.array(correlation)
    signal_area = np.array(signal_area)
    peak_intensity = correlation[:,0] * signal_area
    peak_intensity[(r_value < -0.2) & (p_value > 0.1)] = 0
    print (f"r_value max: {np.max(r_value)}")
    print (f"p_value max: {np.max(p_value)}")
    print (f"r_valu_min: {np.min(r_value)}")
    print (f"p_value_min: {np.min(p_value)}")
    
    return peak_intensity

def simple_sum(data, wavelengths, b1_w, b2_w):
    # Find the indices of the wavelengths
    b1 = np.argmin(np.abs(wavelengths - b1_w))
    b2 = np.argmin(np.abs(wavelengths - b2_w))

    signal_area = np.apply_along_axis(calculate_signal_area, 1, data, b1, b2)
    return signal_area

def voigt_fit(data, wavelengths, b1_w, b2_w):
    # Find the indices of the wavelengths
    b1 = np.argmin(np.abs(wavelengths - b1_w))
    b2 = np.argmin(np.abs(wavelengths - b2_w))
    gamma = 0.1
    sigma = 0.006
    x = wavelengths[b1:b2]
    data_slice = data[:,b1:b2]

    max_data = np.max(data_slice, axis=0)
    a1 = b1 + np.argmax(max_data)
    x0 = wavelengths[a1]
    
    def voigt_fit_wrapper(vector, x, x0, gamma, sigma):
        try:
            popt, _ = curve_fit(voigt, x, vector, p0=[x0, np.max(vector), gamma, sigma])
            # Calculate fitted curve
            real_fit = voigt(x, *popt)
            
            # 6. Check RMSE and normalized RMSE (fit quality)
            RMSE = np.sqrt(np.mean((vector - real_fit)**2))
            
            # Calculate normalized RMSE (NRMSE) as percentage of data range
            data_range = np.max(vector) - np.min(vector)
            if data_range > 0:
                NRMSE = (RMSE / data_range) * 100  # Percentage
            else:
                NRMSE = float('inf')  # No variation in data
            
            # Calculate R² (coefficient of determination)
            ss_res = np.sum((vector - real_fit)**2)
            ss_tot = np.sum((vector - np.mean(vector))**2)
            if ss_tot > 0:
                r_squared = 1 - (ss_res / ss_tot)
            else:
                r_squared = -float('inf')
            
            print(f"R²: {r_squared:.4f}")
            
            if r_squared < 0.85:
                return 0
            else:# Calculate signal area and return R²
                signal_area = np.trapz(real_fit, x)
                return signal_area
        except (RuntimeError, ValueError, TypeError):
            return 0
    signal_area = np.apply_along_axis(voigt_fit_wrapper, 1, data_slice, x, x0, gamma, sigma)
    return signal_area

def simple_voigt_fit(data, wavelengths, b1_w, b2_w):
    # Find the indices of the wavelengths
    b1 = np.argmin(np.abs(wavelengths - b1_w))
    b2 = np.argmin(np.abs(wavelengths - b2_w))
    gamma = -2
    sigma = 0.6
    x = wavelengths[b1:b2]
    data_slice = data[:,b1:b2]
    max_data = np.max(data_slice, axis=0)
    a1 = b1 + np.argmax(max_data)
    x0 = wavelengths[a1]
    basis = voigt(x, x0, np.max(max_data), gamma, sigma)
    denom = np.dot(basis, basis)
    amplitudes = data_slice @ basis / denom
    signal_area = amplitudes * np.trapz(basis, x)
    return signal_area  # Return the signal area for each row

def partition_function(elem, T):
    df = pd.read_csv(PARTITION_FUNCTION_PATH, sep='\t', decimal='.')
    kb_eV = 8.617333262e-5  # eV/K

    df_I = df[(df['Element'] == elem) & (df['ionState'] == 'I')]
    df_II = df[(df['Element'] == elem) & (df['ionState'] == 'II')]

    U_I = np.sum(df_I['gi'] * np.exp(-df_I['Ei'] / (kb_eV * T)))
    U_II = np.sum(df_II['gi'] * np.exp(-df_II['Ei'] / (kb_eV * T)))
    #Change U_I and U_II to float
    U_I = float(U_I)
    U_II = float(U_II)

    return U_I, U_II