import re
import simplejson as json
import os
import time
import torch
from diffusers import  FluxPipeline
import diffusers

_flux_rope = diffusers.models.transformers.transformer_flux.rope
def new_flux_rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."
    if pos.device.type == "mps":
        return _flux_rope(pos.to("cpu"), dim, theta).to(device=pos.device)
    else:
        return _flux_rope(pos, dim, theta)

def get_time_delta_str(start_time: float, end_time: float) -> str:
    delta = end_time - start_time
    hours, rem = divmod(delta, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

diffusers.models.transformers.transformer_flux.rope = new_flux_rope

# get user input for a prompt and file name with validation that the file name doesn't exist
prompt = input("Enter a prompt: ")
num_steps = input("Enter number of interfence steps (default 50): ")
print("Enter guidance scale. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.")
guidance = input("Guidance scale (default 7.0): ")

file_name = input("Enter file name: ")

# check if the file name already exists
path_exists = os.path.exists(file_name + ".png")

# validate the file name for special characters
valid_file_name = re.match("^[a-zA-Z0-9_-]+$", file_name) is not None

while path_exists or not valid_file_name:
    if path_exists:
        print("File name already exists. Please enter a different file name.")
        file_name = input("Enter file name: ")
        path_exists = os.path.exists(file_name)

    if not valid_file_name:
        print("File name contains invalid characters. Please enter a valid file name.")
        file_name = input("Enter file name: ")
        valid_file_name = valid_file_name = re.match("^[a-zA-Z0-9_-]+$", file_name) is not None


# if guidance is empty, set it to 7.0
guidance = 7.0 if not guidance else float(guidance)

# if num_steps is empty, set it to 50
num_steps = 50 if not num_steps else int(num_steps)

# keep a text file log of past generated images with their prompt, guidance scale, and number of steps
# start timestamp, end timestamp, and time taken to generate the image
start_time = time.time()

log = {
    "prompt": prompt,
    "guidance_scale": guidance,
    "num_inference_steps": num_steps,
    "file_name": file_name + ".png",
    "Success": True,
    "Error": None
}


try:
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("mps")

    out = pipe(
        prompt=prompt,
        guidance_scale=guidance,
        height=1024,
        width=1024,
        num_inference_steps=num_steps,
        max_sequence_length=512,
    ).images[0]
    out.save(f"./pictures/{file_name}.png")

    end_time = time.time()
    log["time_taken"] = get_time_delta_str(start_time, end_time)

    # jsonify the log and write it to a file
    json_log = json.dumps(log, indent=4)

    with open("log.txt", "a") as f:
        f.write(str(json_log) + "\n\n")

except KeyboardInterrupt:
    end_time = time.time()
    log["time_taken"] = get_time_delta_str(start_time, end_time)
    log["Success"] = False
    log["Error"] = "Keyboard interrupt while generating " + file_name + ".png"

    json_log = json.dumps(log, indent=4)
    with open("log.txt", "a") as f:
        f.write(json_log + "\n\n")
except Exception as e:
    end_time = time.time()
    log["time_taken"] = get_time_delta_str(start_time, end_time)
    log["Success"] = False
    log["Error"] = str(e)

    json_log = json.dumps(log, indent=4)
    with open("log.txt", "a") as f:
        f.write(json_log + "\n\n")


