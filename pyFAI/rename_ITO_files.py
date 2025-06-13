import os

folder = r"D:\printz_Apr2024\MAPI_YL"

for filename in os.listdir(folder):
    full_path = os.path.join(folder, filename)
    if os.path.isfile(full_path) and '.' not in filename:
        new_name = filename + '.tif'
        new_path = os.path.join(folder, new_name)
        os.rename(full_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")