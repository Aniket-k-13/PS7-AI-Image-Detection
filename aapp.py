import nbformat

file_path = r"C:\Users\anike\Downloads\COLAB_Training.ipynb"

nb = nbformat.read(file_path, as_version=4)

# Remove widget metadata ONLY
if "widgets" in nb.metadata:
    del nb.metadata["widgets"]

# Save cleaned notebook
nbformat.write(nb, file_path)

print("✅ Fixed notebook without removing outputs!")