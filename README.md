This script makes it easier to run the Flux-1.DEV model using HuggingFace.

*Features:*
- Maintains a log.txt of all attempts to generate images
- Prompts for parameters: num_inference_steps, guidance_scale, output filename, and prompt.
- Outputs images into a pictures folder

Instructions for running:

1. Set up virtual environment
   ```
   python -m venv .env
   ```
2. Enter virtual environment (read venv docs for OS specific instructions)
3. Install dependencies
   ```
   pip install -r ./requirements.txt
   ```
4. Run script
   ```
   python hf-interactive.py
   ```
