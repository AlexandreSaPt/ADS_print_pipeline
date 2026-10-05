import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 1. Define Constants (using specific, non-zero values for this example)
Z0 = 50.0                  # Characteristic impedance in Ohms
ZL = 100        # Load impedance in Ohms
beta = 2 * np.pi           # Phase constant (normalized to wavelength, so 1 lambda is unit 1 for l)

# 2. Define data: Length 'l' over one full rotation (0.5 lambda)
# Normalized length l/lambda from 0 to 0.5.
l_normalized = np.linspace(0, 0.5, 500)
# 'l' values used in the function (corresponds to beta*l from 0 to pi)
l_for_function = l_normalized 

# 3. Calculate complex vectors
# Transmission Line Input Impedance Function
numerator = ZL + 1j * Z0 * np.tan(beta * l_for_function)
denominator = Z0 + 1j * ZL * np.tan(beta * l_for_function)
Z_in = Z0 * (numerator / denominator)

# Complex Reflection Coefficient vector
Gamma_in = (Z_in - Z0) / (Z_in + Z0)

# 4. Set up the plotting figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- Plot 1: Normal Cartesian Graph ---
# Plot Real and Imaginary Zin vs normalized length
ax1.plot(l_normalized, np.real(Z_in), color='blue', label='Real($Z_{in}$)')
ax1.plot(l_normalized, np.imag(Z_in), color='green', label='Imaginary($Z_{in}$)')
ax1.set_title('Input Impedance ($Z_{in}$) vs Length ($l$)')
ax1.set_xlabel('Length ($l/\lambda$)')
ax1.set_ylabel('Impedance ($\Omega$)')
ax1.grid(True)
ax1.legend(loc='upper right')

# --- Plot 2: Smith Chart Overlay (Fixed Positioning) ---
# Create a robust complex plane for the Smith chart data

# 1. Set the aspect ratio of this subplot to equal to prevent distortion
ax2.set_aspect('equal')

SMITH_EXTENT = [-1.18423, 1.18980, -1.18918, 1.18485]
# 2. Load the Smith Diagram image (handle missing file gracefully)
try:
    img = mpimg.imread('diagram.png')
    # 3. CRITICAL positioning of the image:
    # Set the extent to cover [-1, 1] on both the x and y data axes.
    # This aligns the pixel data with the complex Gamma coordinate system.
    # We set zorder to 0 to put it behind the plot lines.
    ax2.imshow(img, extent=SMITH_EXTENT, zorder=0, origin='lower')
except FileNotFoundError:
    print("Warning: 'diagram.png' not found. Drawing a simple unit circle fallback.")
    # Standard unit circle fallback
    circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--')
    ax2.add_patch(circle)

# 4. Plot the data (Gamma_in trace) over the image in red, on top.
ax2.plot(np.real(Gamma_in), np.imag(Gamma_in), color='red', linewidth=2.5, zorder=1)

# 5. Turn off Matplotlib's coordinate axes, relying entirely on the PNG's grid.
ax2.axis('off')

# 6. Add markers with offset for specific points: start (l=0) and end (l=lambda/2)
# Using 'scatter' and 'text' with offsets to show the labels clearly.
ax2.scatter(np.real(Gamma_in[0]), np.imag(Gamma_in[0]), s=100, color='lime', marker='x', label='$l=0$', zorder=2)
ax2.text(np.real(Gamma_in[0]) + 0.08, np.imag(Gamma_in[0]) + 0.05, '$l=0$', color='lime', fontsize=12, fontweight='bold', zorder=2)

ax2.scatter(np.real(Gamma_in[-1]), np.imag(Gamma_in[-1]), s=100, color='lime', marker='x', label='$l=\lambda/2$', zorder=2)
ax2.text(np.real(Gamma_in[-1]) + 0.08, np.imag(Gamma_in[-1]) + 0.05, '$l=\lambda/2$', color='lime', fontsize=12, fontweight='bold', zorder=2)

# Set final axis properties
ax2.set_xlim([-1.1, 1.1])
ax2.set_ylim([-1.1, 1.1])
ax2.set_title('Smith Chart Overlay with Correct Image Position')

# Optional verification lines (uncomment to verify image alignment):
# ax2.axhline(0, color='orange', linestyle='-', zorder=1)
# ax2.axvline(0, color='orange', linestyle='-', zorder=1)
# ax2.add_patch(plt.Circle((0, 0), 1, color='magenta', fill=False, linestyle='-', zorder=1))

plt.tight_layout()
plt.show()