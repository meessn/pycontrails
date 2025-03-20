import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# Define the data
data = {
    "Engine": ["PW1127G", "PW1127G", "PW1127G", "PW1127G",
               "PW1124G", "PW1124G", "PW1124G", "PW1124G",
               "PW1129G", "PW1129G", "PW1129G", "PW1129G",
               "PW1133G", "PW1133G", "PW1133G", "PW1133G"],
    "LTO": ["T/O", "C/O", "APPR", "IDLE",
            "T/O", "C/O", "APPR", "IDLE",
            "T/O", "C/O", "APPR", "IDLE",
            "T/O", "C/O", "APPR", "IDLE"],
    "TT3": [835, 805, 643, 485, 810, 780, 629, 467, 852, 820, 657, 488, 878, 843, 663, 493],
    "PT3": [32.2, 27.8, 10.8, 3.9, 29.1, 25.1, 9.4, 3.5, 34.7, 29.8, 10.9, 3.9, 39.0, 33.5, 12.6, 4.1],
    "FAR": [0.02741, 0.02535, 0.01905, 0.01476, 0.02523, 0.02363, 0.02005, 0.01482,
            0.02864, 0.02678, 0.02251, 0.01645, 0.03137, 0.02879, 0.02025, 0.01602],
    "TT4": [1719, 1636, 1312, 1037, 1637, 1565, 1331, 1023, 1767, 1689, 1430, 1096, 1863, 1764, 1365, 1086],
    "nvPM_mass_ICAO": [36.3, 26.3, 0.6, 7.8, 30.1, 19.6, 0.6, 8.7, 40.1, 31.6, 0.6, 7.1, 44.9, 38.7, 0.9, 6],
    "nvPM_number_ICAO": [1.45E+15, 1.60E+15, 3.85E+14, 5.78E+15,
                         1.58E+15, 1.51E+15, 4.36E+14, 6.41E+15,
                         1.26E+15, 1.58E+15, 3.78E+14, 5.31E+15,
                         8.72E+14, 1.38E+15, 4.30E+14, 4.51E+15],
    "number/mass": [3.99E+13, 6.08E+13, 6.42E+14, 7.41E+14,
                    5.25E+13, 7.70E+13, 7.27E+14, 7.37E+14,
                    3.14E+13, 5.00E+13, 6.30E+14, 7.48E+14,
                    1.94E+13, 3.57E+13, 4.78E+14, 7.52E+14]
}

# data = {
#     "Engine": ["PW1127G", "PW1127G", "PW1127G", "PW1127G",
#                "PW1124G", "PW1124G", "PW1124G", "PW1124G",
#                "PW1129G", "PW1129G", "PW1129G", "PW1129G",
#                "PW1133G", "PW1133G", "PW1133G"],
#     "LTO": ["T/O", "C/O", "APPR", "IDLE",
#             "T/O", "C/O", "APPR", "IDLE",
#             "T/O", "C/O", "APPR", "IDLE",
#              "C/O", "APPR", "IDLE"],
#     "TT3": [835, 805, 643, 485, 810, 780, 629, 467, 852, 820, 657, 488,  843, 663, 493],
#     "PT3": [32.2, 27.8, 10.8, 3.9, 29.1, 25.1, 9.4, 3.5, 34.7, 29.8, 10.9, 3.9,  33.5, 12.6, 4.1],
#     "FAR": [0.02741, 0.02535, 0.01905, 0.01476, 0.02523, 0.02363, 0.02005, 0.01482,
#             0.02864, 0.02678, 0.02251, 0.01645,  0.02879, 0.02025, 0.01602],
#     "TT4": [1719, 1636, 1312, 1037, 1637, 1565, 1331, 1023, 1767, 1689, 1430, 1096,  1764, 1365, 1086],
#     "nvPM_mass_ICAO": [36.3, 26.3, 0.6, 7.8, 30.1, 19.6, 0.6, 8.7, 40.1, 31.6, 0.6, 7.1,  38.7, 0.9, 6],
#     "nvPM_number_ICAO": [1.45E+15, 1.60E+15, 3.85E+14, 5.78E+15,
#                          1.58E+15, 1.51E+15, 4.36E+14, 6.41E+15,
#                          1.26E+15, 1.58E+15, 3.78E+14, 5.31E+15,
#                           1.38E+15, 4.30E+14, 4.51E+15],
#     "number/mass": [3.99E+13, 6.08E+13, 6.42E+14, 7.41E+14,
#                     5.25E+13, 7.70E+13, 7.27E+14, 7.37E+14,
#                     3.14E+13, 5.00E+13, 6.30E+14, 7.48E+14,
#                      3.57E+13, 4.78E+14, 7.52E+14]
# }

# Create DataFrame
df = pd.DataFrame(data)



# Extract the required data from the DataFrame
TT3 = df["TT3"].values
nvPM_mass_ICAO = df["nvPM_mass_ICAO"].values
nvPM_number_ICAO = df["nvPM_number_ICAO"].values
number_mass = df["number/mass"].values

### Sixth-order Polynomial Fit for nvPM_mass_ICAO vs. TT3 ###
# Fit a 6th order polynomial
poly_coeffs = np.polyfit(TT3, nvPM_mass_ICAO, 6)
poly_fit = np.poly1d(poly_coeffs)

t = (TT3 - 694.4) / 151.5
poly_coeffs_t = np.polyfit(t, nvPM_mass_ICAO, 6)
poly_fit_t = np.poly1d(poly_coeffs_t)
print(poly_coeffs_t)
# Generate smooth curve
TT3_smooth = np.linspace(min(TT3), max(TT3), 100)
nvPM_mass_fit = poly_fit(TT3_smooth)

# Plot the polynomial fit
plt.figure(figsize=(6, 4))
plt.scatter(TT3, nvPM_mass_ICAO, label="ICAO", color='blue')
plt.plot(TT3_smooth, nvPM_mass_fit, 'r-', label="P3T3 Correlation Saluja (2023)")
plt.xlabel("TT3")
plt.ylabel("nvPM mass ICAO")
plt.title("EI_nvPM_mass,GR")
plt.legend()
plt.grid()
plt.show()
# Compute v = nvPM_number_ICAO / nvPM_mass_ICAO
v = nvPM_number_ICAO / nvPM_mass_ICAO  # This is the dependent variable
v_large = v
# Compute little t = (TT3 - 694.4) / 151.5
t = (TT3 - 694.4) / 151.5

# Scale v to avoid large numbers
v = v / 1e14


# Define the piecewise function using np.where
def piecewise_exp(t, A, B, m, c):
    """
    A, B: Parameters for the exponential decay (for t > t_threshold)
    m, c: Parameters for the linear function (for t <= t_threshold)
    """
    t_threshold = (636 - 694.4) / 151.5  # Convert TT3 = 629 to corresponding t value
    c = (A + 12.2274) * np.exp(B * t_threshold) - m * t_threshold
    # Apply the piecewise function using np.where
    return np.where(
        t <= t_threshold,
        m * t + c,  # Linear part for t <= t_threshold
        (A + 12.2274) * np.exp(B * t)  # Exponential part for t > t_threshold
    )

# Initial guesses for parameters
initial_guess = [-4.0, -1.2, 0, max(v)]  # A, B (exponential), m, c (linear)

# Perform curve fitting
params, covariance = curve_fit(piecewise_exp, t, v, p0=initial_guess)
print(params)
# Generate smooth fitted curve
t_smooth = np.linspace(min(t), max(t), 100)
v_fit = piecewise_exp(t_smooth, *params) * 1e14  # Rescale v back
t_smooth_TT3 = t_smooth * 151.5 + 694.4
# Reference Equation Parameters
A_ref = -4.0240*np.exp(1)
B_ref = -1.2210
C_ref = 12.2274
scale_factor = 1e14  # Scale factor for v

# Compute reference v values using the given equation
v_reference = (A_ref + C_ref) * np.exp(B_ref * t_smooth) * scale_factor

# poly_coeffs = np.polyfit(TT3, v_large, 1000)
# poly_fit = np.poly1d(poly_coeffs)
# v_poly6_fit = poly_fit(t_smooth_TT3)
# Plot the original data, fitted curve, and reference equation
plt.figure(figsize=(6, 4))
plt.scatter(t, v * 1e14, label="Data", color='blue')
plt.plot(t_smooth, v_reference, 'g--', label="P3T3 Correlation Saluja (2023)")
plt.plot(t_smooth, v_fit, 'r-', label="P3T3 Piecewise Curve Fit")
plt.xlabel("t (normalized TT3)")
plt.ylabel("v (number/mass)")
plt.title("Comparison of Fitted vs. Reference Equation")
plt.legend()
plt.grid()

plt.figure(figsize=(6, 4))
plt.scatter(TT3, v * 1e14, label="ICAO", color='blue')
plt.plot(t_smooth_TT3, v_reference, 'g--', label="P3T3 Correlation Saluja (2023)")
plt.plot(t_smooth_TT3, v_fit, 'r-', label="P3T3 Piecewise Curve Fit")
# plt.plot(t_smooth_TT3, v_poly6_fit, 'r-', label="P3T3 6th order polyfit")
plt.xlabel("TT3")
plt.ylabel("v (number/mass)")
plt.title("v Curve Fit")
plt.legend()
plt.grid()
plt.show()

