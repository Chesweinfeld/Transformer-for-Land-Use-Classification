import matplotlib.pyplot as plt


# models = ['BiT', 'ViT-B/16', 'ViT-B/32', 'ViT-L/16', 'ViT-L/32']
# results = [0.8656, 0.8800, 0.7807, 0.8870, 0.7819]
# plt.figure()


# plt.plot(models, results, marker="o")
# plt.ylabel("Validation Accuracy")
# plt.xlabel("Model")
# plt.title("ViT  vs ResNet")

# plt.savefig("vit_scaling_plot.png", dpi=300, bbox_inches="tight")

# plt.show()

plots = {
    "heads": (['8','12','16','24','32'], [0.8507,0.8437,0.8452,0.8426,0.8493], "# of Heads"),
    "patch_size": (['8','16','32'], [0.8507, 0.7911, 0.7052], "Patch Size"),
    "pos_embed": (['learned', 'none', '2d'], [0.8507, 0.8470, 0.7915], "Positional Embedding"),
    "token": (['Class Token', 'Token Average'], [0.8507, 0.8559], "Classfication")
}

for name, (x, y, xlabel) in plots.items():
    plt.figure()
    plt.scatter(x, y, marker="o")
    plt.ylabel("Validation Accuracy")
    plt.xlabel(xlabel)
    plt.title(f"Variation of {xlabel}")
    
    plt.savefig(f"{name}_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

heads = ['8', '12', '16', '24', '32']
results = [0.8507, 0.8437, 0.8452, 0.8426, 0.8493]

