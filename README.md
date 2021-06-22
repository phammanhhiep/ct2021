# CT2021
## Code Structure
- script/setup.sh: create a conda environment, and install required packages.
- script/train.sh, script/eval.sh: entry points to train and evaluate the model respectively.
- Dockerfile, Dockerfile.develop: to create image with necessary packages and source code; the later not require to copy source code to the image, and thus make it more flexible in developing the model.
- requirements.cpu.txt, requirements.gpu.txt: specify packages to train with only either cpu or gpu.

## Major change from v.1
- Use pytorch lightning