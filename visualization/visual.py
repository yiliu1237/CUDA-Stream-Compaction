import matplotlib.pyplot as plt
import pandas as pd

# === CONFIGURATION ===
filename = "C:\\Users\\thero\\OneDrive\\Documents\\GitHub\\Project2-Stream-Compaction\\visualization\\scan_timing_results.csv"
save_path = "C:\\Users\\thero\\OneDrive\\Documents\\GitHub\\Project2-Stream-Compaction\\visualization\\scan_performance_plot.png"

# === READ CSV ===
df = pd.read_csv(filename)

# Skip rows that aren't numeric
df = df[pd.to_numeric(df['N'], errors='coerce').notnull()]
df['N'] = df['N'].astype(int)

# === Extract data ===
x = df['N']
cpu = df['CPU(ms)']
naive = df['Naive(ms)']
efficient = df['Efficient(ms)']
# thrust = df['Thrust(ms)']  # Uncomment if thrust values are added later

# === Plot ===
plt.figure(figsize=(10, 6))
plt.plot(x, cpu, 'o-', label='CPU')
plt.plot(x, naive, 'o-', label='Naive GPU')
plt.plot(x, efficient, 'o-', label='Efficient GPU')
# plt.plot(x, thrust, 'o-', label='Thrust GPU')

plt.xscale('log', base=2)
plt.xlabel("Input Size (N)")
plt.ylabel("Time (ms)")
plt.title("Scan Performance Comparison")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

# === Save and Show ===
plt.savefig(save_path, dpi=300)
plt.show()
