import os
import shutil
import stat

def on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

base_dir = "main_results_figures"

for root, dirs, files in os.walk(base_dir):
    if "climate" in dirs:
        climate_path = os.path.join(root, "climate")
        shutil.rmtree(climate_path, onerror=on_rm_error)
        print(f"Deleted: {climate_path}")