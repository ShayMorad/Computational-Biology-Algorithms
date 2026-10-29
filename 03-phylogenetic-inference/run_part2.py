import os
import subprocess

# The sequence lengths based on your data folder
lengths = [50, 100, 250, 500, 750, 1000, 1250, 1500]
algorithms = ["NJ", "UPGMA"]

# Ensure an output folder exists so we don't clutter your main directory
os.makedirs("tree_images", exist_ok=True)

for length in lengths:
    fasta_path = f"data/descendants_{length}.fasta"

    for algo in algorithms:
        print(f"Generating {algo} tree for length {length}...")

        # Build the command just like you would type in the terminal
        command = [
            "python", "ex3.py",
            "--fasta", fasta_path,
            "--part", "build_tree",
            "--tree_type", algo
        ]

        # Run the command
        subprocess.run(command, check=True)

        # Rename and move the generated image
        new_image_name = f"tree_images/{algo}_tree_{length}.png"

        if os.path.exists("cool_tree.png"):
            os.replace("cool_tree.png", new_image_name)
            print(f"Saved -> {new_image_name}")
        else:
            print(f"Error: cool_tree.png was not generated for {algo} {length}")

print("\nAll 16 trees generated successfully! Check the 'tree_images' folder.")