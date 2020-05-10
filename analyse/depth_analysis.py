import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
wb_y = {
    "IARNet-GloVe": [0.949, 0.956, 0.965, 0.962],
    "IARNet-BERT": [0.955, 0.963, 0.969, 0.966]
}

fk_y = {
    "IARNet-GloVe": [0.951, 0.953, 0.960, 0.958],
    "IARNet-BERT": [0.953, 0.958, 0.964, 0.962]
}

y = [wb_y, fk_y]

colors = {
    "IARNet-GloVe": '#3C3CA3',
    'IARNet-BERT': '#6AA46A'
}
fig, axes = plt.subplots(1, 1)
# axes[0].set_yticks([0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1])
# axes[0].set_ylim([0.65, 1.05])
# axes[1].set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# axes[1].set_ylim([0.3, 1.01])
"""
top=0.835,
bottom=0.11,
left=0.11,
right=0.9,
hspace=0.2,
wspace=0.2
"""
font1 = {'family': 'Times New Roman',
         # 'weight': 'normal',
         'size': 16,
         }

font2 = {'family': 'Times New Roman',
         # 'weight': 'normal',
         'size': 18,
         }

# for i in range(1):
for i in range(1, 2):
    for key, spine in axes.spines.items():
        # if key == 'right' or key == 'top':
        spine.set_visible(False)
    # axes.set_title('Weibo', font2)
    axes.set_title('Fakeddit', font2)
    axes.set_xticks([1, 2, 3, 4])
    axes.set_xlabel('Number of layers', font2)
    axes.set_ylabel('Accuracy', font2)
    axes.set_facecolor('#E5E5E5')
    axes.grid(color='#FFFFFF')
    # foey in y[i].keys():
    axes.plot(x, y[i]['IARNet-GloVe'], linestyle='-.', marker='o', color=colors['IARNet-GloVe'], label='IARNet-GloVe')
    axes.plot(x, y[i]['IARNet-BERT'], linestyle='-', marker='*', color=colors['IARNet-BERT'], label='IARNet-BERT')
    axes.legend(loc='lower right', ncol=1, prop=font1)

plt.show()
print()
