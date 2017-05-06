# Student: Kaue Ueda Silveira
# NUSP: 7987498

import numpy as np
from scipy import signal as signalib
from scipy.fftpack import fft, ifft
from matplotlib import pyplot as plt

# 1. Parameter Input
filename = str(input())
s = float(input())
c = float(input())
show_flag = int(input())

# 2. Read the input image
sigf = np.fromfile(filename, dtype=np.int32)
if sigf is None:
    print("Could not load file.")
    exit()

# 3. Apply Gaussian Windowing
# Find window
window = signalib.gaussian(sigf.size, std=(sigf.size/s))
# Then multiply by signal
sigg = np.multiply(sigf, window)

# 4. Apply FFT f -> F
sigF = fft(sigf)

# 5. Apply FFT g -> G
sigG = fft(sigg)

# 6. Show Plots

# Compute NFS for each signal
normsigF = np.divide(np.abs(sigF), 2*sigF.size)
normsigG = np.divide(np.abs(sigG), 2*sigG.size)
# Then plot as instructed
if (show_flag == 1):
    plt.subplot(221), plt.plot(sigf)
    plt.subplot(222), plt.plot(normsigF)
    plt.subplot(223), plt.plot(sigg)
    plt.subplot(224), plt.plot(normsigG)
    plt.show()

# 7. Apply Low Pass Filter

# Find treshold
treshold = c*np.argmax(np.abs(sigG))

filteredsig = []
for i, f in enumerate(sigF):
    if (i < treshold):
        filteredsig.append(f)
    else:
        filteredsig.append(0)

# 8. Compute IFFT FH-> fh
sigRes = ifft(filteredsig)

# 9. Show Plots
if (show_flag == 1):
    plt.subplot(211), plt.plot(sigf)
    plt.subplot(212), plt.plot(sigRes)
    plt.show()

# 10. Outputs
print(np.argmax(np.abs(sigF)))
print(np.argmax(np.abs(sigG)))
print(sigf.max())
print(int(np.real(sigRes.max())))