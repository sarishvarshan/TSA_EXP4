# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES


### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
data=pd.read_csv('tsla_2014_2023.csv')

# Declare required variables and set figure size, and visualise the data
N=1000
plt.rcParams['figure.figsize'] = [12, 6]
X=data['close'] # Use the 'close' column

# Visualise the data
plt.plot(X)
plt.title('Original Data (TSLA Close Price)')
plt.show()

# Plot ACF and PACF for Original Data
num_lags = 40 # Capped lags for better visualization
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plot_acf(X, lags=num_lags, ax=plt.gca())
plt.title('Original Data ACF')
plt.subplot(2, 1, 2)
plot_pacf(X, lags=num_lags, ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()

# Fitting the ARMA(1,1) model and deriving parameters
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

# Simulate ARMA(1,1) Process
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

# Plot ARMA(1,1)
plt.plot(ARMA_1)
plt.title(f'Simulated ARMA(1,1) Process: $\\phi_1={phi1_arma11:.4f}$, $\\theta_1={theta1_arma11:.4f}$')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF for ARMA(1,1)
plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()

# Fitting the ARMA(2,2) model and deriving parameters
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])

# Simulate ARMA(2,2) Process
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N * 10)

# Plot ARMA(2,2)
plt.plot(ARMA_2)
plt.title(f'Simulated ARMA(2,2) Process: $\\phi_1={phi1_arma22:.4f}, \\phi_2={phi2_arma22:.4f}, \\theta_1={theta1_arma22:.4f}, \\theta_2={theta2_arma22:.4f}$')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF for ARMA(2,2)
plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()
~~~

OUTPUT:
Original Data:
<img width="1973" height="1070" alt="Screenshot 2025-09-30 085246" src="https://github.com/user-attachments/assets/e8caad19-7a1b-4d3e-929d-a85ac4226e03" />

Original Data ACF:
<img width="2376" height="788" alt="Screenshot 2025-09-30 085258" src="https://github.com/user-attachments/assets/da6021d2-5418-45e7-bddf-3b5a8425ed1d" />

Original Data PACF:
<img width="2377" height="805" alt="Screenshot 2025-09-30 085311" src="https://github.com/user-attachments/assets/8ec791bf-4e4f-42d4-a997-6a092aec84e3" />

SIMULATED ARMA(1,1) PROCESS:
<img width="2000" height="1074" alt="Screenshot 2025-09-30 085406" src="https://github.com/user-attachments/assets/4864d5ba-07ac-4668-a52f-95cd22559b6c" />

Partial Autocorrelation:
<img width="1988" height="1049" alt="Screenshot 2025-09-30 085456" src="https://github.com/user-attachments/assets/ad906683-e29e-4405-be76-acbe3500b4a8" />

Autocorrelation:
<img width="1996" height="1053" alt="Screenshot 2025-09-30 085434" src="https://github.com/user-attachments/assets/fd2376a7-6b5a-402e-ab2d-c983b70a203e" />

SIMULATED ARMA(2,2) PROCESS:
<img width="2001" height="1091" alt="Screenshot 2025-09-30 085520" src="https://github.com/user-attachments/assets/5c87ee7b-967b-4fb9-bccd-e1f19bac2636" />

Partial Autocorrelation:
<img width="2008" height="1053" alt="Screenshot 2025-09-30 085614" src="https://github.com/user-attachments/assets/416024d3-c500-4547-a7df-11e0dbea28f5" />

Autocorrelation:
<img width="2003" height="1044" alt="Screenshot 2025-09-30 085545" src="https://github.com/user-attachments/assets/ec7e6d5b-4610-472a-8506-cd674ca192bf" />

RESULT:
Thus, a python program is created to fir ARMA Model successfully.
