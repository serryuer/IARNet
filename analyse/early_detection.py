import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4]
wb_y = {
    "PPC": [0.919, 0.9201, 0.9195, 0.9199, 0.9205],
    "GLAN": [0.938, 0.947, 0.942, 0.937, 0.939],
    "GRU-RNN": [0.759, 0.820, 0.847, 0.873, 0.907],
    "DTR": [0.690, 0.708, 0.715, 0.736, 0.790],
    "HGAT": [0.945, 0.948, 0.958, 0.962, 0.968]
}

tw15_y = {
    "RvNN": [0.41, 0.573, 0.628, 0.658, 0.720],
    "GLAN": [0.889, 0.891, 0.899, 0.896, 0.900],
    "GRU-RNN": [0.410, 0.518, 0.523, 0.594, 0.596],
    "DTR": [0.350, 0.412, 0.387, 0.375, 0.298],
    "PPC": [0.835, 0.835, 0.836, 0.835, 0.836],
    "HGAT": [0.915, 0.918, 0.928, 0.922, 0.928]
}
tw16_y = {
    "PPC": [0.858, 0.859, 0.858, 0.858, 0.860],
    "GLAN": [0.887, 0.900, 0.886, 0.8865, 0.8869],
    "GRU-RNN": [0.400, 0.445, 0.528, 0.594, 0.587],
    "DTR": [0.400, 0.429, 0.409, 0.403, 0.4082],
    "RvNN": [0.42, 0.53, 0.617, 0.653, 0.689],
    "HGAT": [0.915, 0.918, 0.928, 0.922, 0.925]
}
y = [wb_y, tw15_y, tw16_y]

colors = {
    "PPC": 'dodgerblue',
    "GLAN": 'brown',
    "GRU-RNN": 'greenyellow',
    "DTR": 'gold',
    "HGAT": 'red',
    'RvNN': 'blue'
}
fig, axes = plt.subplots(1, 3)
axes[0].set_yticks([0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1])
axes[0].set_ylim([0.65, 1.05])
axes[1].set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
axes[1].set_ylim([0.4, 1.01])
axes[2].set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
axes[2].set_ylim([0.4, 1.01])

for i in range(3):
    for key, spine in axes[i].spines.items():
        if key == 'right' or key == 'top':
            spine.set_visible(False)
    axes[i].set_xticklabels(['-1', '0', '4', '8', '12', '24'])
    axes[i].set_xlabel('Detection deadline(h)')
    axes[i].set_ylabel('Accuracy')
    for key in y[i].keys():
        axes[i].plot(x, y[i][key], linestyle='dashed', marker='o', color=colors[key], label=key)
    axes[i].legend(loc='upper center', ncol=3)

fig.show()
print()
