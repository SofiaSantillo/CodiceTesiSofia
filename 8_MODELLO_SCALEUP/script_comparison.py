import matplotlib.pyplot as plt

scripts = ["Refined Model", "Script1", "Script2", "Script3", "Script4", "Script5"]
r2_size = [-0.28, 0.2329, 0.2175, 0.08, 0.2511, 0.4195]
r2_pdi = [-0.4131, 0.5576, 0.5431, -1.08, 0.7487, 0.7302]

fig, axs = plt.subplots(2, 1, figsize=(10, 8)) 

axs[0].plot(scripts, r2_size, marker='o', linestyle='-', color='blue')
axs[0].set_title('R2 SIZE ')
axs[0].set_ylabel('R2 SIZE')
axs[0].set_ylim(min(r2_size) - 0.1, max(r2_size) + 0.1)
axs[0].grid(True)

axs[1].plot(scripts, r2_pdi, marker='s', linestyle='-', color='green')
axs[1].set_title('R2 PDI ')
axs[1].set_ylabel('R2 PDI')
axs[1].set_ylim(min(r2_pdi) - 0.2, max(r2_pdi) + 0.2)
axs[1].grid(True)

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('8_MODELLO_SCALEUP/_plot/R2_script_comparison.png', dpi=300)
plt.show()
