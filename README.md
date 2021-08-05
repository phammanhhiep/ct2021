# CT2021
## Summary
The project is based on the research "FaceShifter: Towards High Fidelity And Occlusion Aware Face Swapping (Li et al)", with the goal to replicate the research results and improve if possible.

## Code Structure
- script/setup.sh: create a conda environment, and install required packages.
- script/train.sh, script/eval.sh: entry points to train and evaluate the model respectively.
- Dockerfile, Dockerfile.develop: to create image with necessary packages and source code; the later not require to copy source code to the image, and thus make it more flexible in developing the model.
- requirements.cpu.txt, requirements.gpu.txt: specify packages to train with only either cpu or gpu.