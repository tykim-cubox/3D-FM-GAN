import shutil
import subprocess
import os

nvidia_smi_path = shutil.which("nvidia-smi")
if nvidia_smi_path is None:
    raise FileNotFoundError("nvidia-smi: command not found")
result = subprocess.run(
    [nvidia_smi_path, "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
    encoding="utf-8",
    capture_output=True,
    check=True,
)

# Convert lines into a dictionary
gpu_memory = [float(x) for x in result.stdout.strip().split(os.linesep)]
gpu_memory_map = {f"gpu_id: {gpu_id}/memory.used (MB)": memory for gpu_id, memory in enumerate(gpu_memory)}
print(gpu_memory_map)