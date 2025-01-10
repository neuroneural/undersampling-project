import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the PNG image
image_path = 'ica-comps.png'  # Path to your image
img = mpimg.imread(image_path)

# Create a figure
fig, ax = plt.subplots(figsize=(10, 8))  # Adjust figsize as needed

# Display the image
ax.imshow(img)
ax.axis('off')  # Turn off axes for cleaner display

# Add the caption
caption = (
    "Figure 2: Brain networks extracted using constrained ICA for the fast rate (TR=100ms) and slow rate (TR=2150ms) OULU subjects."
)
plt.figtext(0.5, 0.12, caption, ha='center', va='top', fontsize=10, wrap=True)

# Adjust layout to avoid cutting off the caption
plt.tight_layout(rect=[0, 0.05, 1, 1])

# Save the figure with the embedded caption
output_path = 'ica-comps_with_caption.png'
plt.savefig(output_path, bbox_inches='tight')
print(f"Image with caption saved to {output_path}")

# Display the image
plt.savefig('comps-capt.png')
