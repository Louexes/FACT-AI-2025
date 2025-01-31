import seaborn as sns
import matplotlib.pyplot as plt
# sns.set_style("white")
sns.set(font_scale=1.6)


def save_matrix(matrix, names, boundary=False, fmt='.2f', vmin=0, vmax=None,
                annotsize=20, labelsize=18, xlabel='Predicted', ylabel='Actual', filename='matrix.png'):
    
    if boundary:
        matrix = matrix.copy()
        matrix[matrix == 0.00] = float('nan')
    
    plt.figure(figsize=(10, 8))  # Create figure with larger size
    ax = sns.heatmap(
        matrix,
        annot=True, annot_kws=dict(size=annotsize), fmt=fmt, vmin=vmin, vmax=vmax, linewidth=1,
        cmap=sns.color_palette("light:b", as_cmap=True),
        xticklabels=names,
        yticklabels=names,
    )
    ax.set_facecolor('white')
    ax.tick_params(axis='x', labelsize=labelsize, rotation=0)
    ax.tick_params(axis='y', labelsize=labelsize, rotation=0)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()  # Adjust layout to fit all elements
    plt.savefig(filename, bbox_inches='tight')  # Save with tight bounding box
    plt.close()