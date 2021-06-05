# CT2021
## Code Structure
- setup.sh: create a conda environment, and install required packages.
- train.sh, eval.sh: entry points to train and evaluate the model respectively.
- Dockerfile, Dockerfile.develop: to create image with necessary packages and source code; the later not require to copy source code to the image, and thus make it more flexible in developing the model.
- requirements.cpu.txt, requirements.gpu.txt: specify packages to train with only either cpu or gpu.  


## How to train and eval the models
- For development, follow the steps:
-- Run setup.sh
-- Make an artifacts directory within the project directory, or make a soft link if having an existing one
-- Run train.sh or eval.sh
- For deployment, follow the steps:
-- Create one of the provided Dockerfile
-- Call "docker run" with command similar as "python -W ignore "/home/CONTAINER_USER/projects/ct2021/src/train.py" --option_file OPTIONAL_FILE
