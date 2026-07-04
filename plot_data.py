import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("xy_data.csv")
x = data['x'].values
y = data['y'].values

# Plot
plt.figure(figsize=(10, 7))
plt.scatter(x, y, color='steelblue', s=8, alpha=0.5, label='Data points (xy_data.csv)')

plt.title('X vs Y Data Points', fontsize=15, fontweight='bold')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig("xy_plot.png", dpi=150)
print("Plot saved to xy_plot.png")
plt.show()
