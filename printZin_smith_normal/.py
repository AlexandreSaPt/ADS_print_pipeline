import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 1. Define Constants
Z0 = 50.0                   
ZL = 100.0 + 50j  # Load impedance (complex)      
beta = 2 * np.pi            

# 2. Define the independent variable (l)
l = np.linspace(0, 1, 1000)

# 3. Calculate Zin, Gamma_in, and Gamma_L
numerator = ZL + 1j * Z0 * np.tan(beta * l)
denominator = Z0 + 1j * ZL * np.tan(beta * l)
Z_in = Z0 * (numerator / denominator)

Gamma_in = (Z_in - Z0) / (Z_in + Z0)
Gamma_L = (ZL - Z0) / (ZL + Z0) # Calculate the reflection coefficient at the load

# 4. Initialize the plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: Normal Cartesian Graph ---
ax1.plot(l, np.real(Z_in), label='Real($Z_{in2}$)', color='blue')
ax1.plot(l, np.imag(Z_in), label='Imaginary($Z_{in2}$)', color='red')
ax1.set_title('Input Impedance ($Z_{in2}$) vs Length ($l$)')
ax1.set_xlabel('Length $l$ (m)')
ax1.set_ylabel('Impedance ($\Omega$)')
ax1.legend()
ax1.grid(True)
# ax1.set_ylim(-200, 300) # Uncomment to bound asymptotes if needed

# --- Plot 2: Smith Diagram Overlay ---
ax2.set_aspect('equal')

# Define your custom extent here. 
# Because we removed ax2.set_xlim() and ax2.set_ylim(), Matplotlib will 
# now automatically scale the window to fit whatever extent you define here 
# without cutting off the image.
SMITH_EXTENT = [-1.18423, 1.18980, -1.18918, 1.18485]

try:
    # Referencing the exact filename requested (corrected from 'diagram.png')
    img = mpimg.imread('diagram.png')
    ax2.imshow(img, extent=SMITH_EXTENT, zorder=0)
except FileNotFoundError:
    print("Warning: 'diagram.png' not found in the current directory.")

# Plot the calculated Gamma values over the diagram
ax2.plot(np.real(Gamma_in), np.imag(Gamma_in), color='blue', linewidth=2, label='$Z_{in2}$ Path')

# Plot ZL with a red cross
ax2.plot(np.real(Gamma_L), np.imag(Gamma_L), marker='x', color='red', markersize=10, markeredgewidth=2, linestyle='None', label='$Z_L$')

ax2.set_title('Smith Diagram of $Z_{in2}$')

# Add legend to Smith chart to identify the cross
ax2.legend(loc='upper right')

# Completely disables the Matplotlib axes, ticks, and bounding box
ax2.axis('off') 

plt.tight_layout()
plt.show()