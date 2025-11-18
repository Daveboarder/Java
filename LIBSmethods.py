import numpy as np
from scipy.signal import correlate
from scipy.stats import pearsonr
from scipy.special import wofz
from scipy.optimize import curve_fit


def calculate_snr(data):
    #normalize the data to standard normal distribution
    data_snr = (data - np.mean(data)) / np.std(data)
    return data_snr

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
    gamma = -2
    sigma = 0.6
    x = wavelengths[b1:b2]
    data_slice = data[:,b1:b2]

    max_data = np.max(data_slice, axis=0)
    a1 = b1 + np.argmax(max_data)
    x0 = wavelengths[a1]
    
    def voigt_fit_wrapper(vector, x, x0, gamma, sigma):
        try:
            popt, _ = curve_fit(voigt, x, vector, p0=[x0, np.max(vector), gamma, sigma])
            real_fit=voigt(x, *popt)
            RMSE = np.sqrt(np.mean((vector - real_fit)**2))
            print(f"RMSE: {RMSE}")
            if RMSE > 15:
                signal_area = 0
                return signal_area
            else:
                signal_area = np.trapz(real_fit, x)
                return signal_area
        except RuntimeError:
            signal_area = 0
            return 0
    signal_area = np.apply_along_axis(voigt_fit_wrapper, 1, data_slice, x, x0, gamma, sigma)
    return signal_area  # Return the signal area for each row

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