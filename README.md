# HateSpeechProject: MemeCLIP setup

> ⚠️ **Before following this tutorial**, make sure to read the original MemeCLIP README located inside the `MemeCLIP_marc` directory.

---

## 🧩 Dependencies

### ✅ Option 1: Download a New Image from Scratch

Add the following to your `requirements.txt`:

``` bash
yacs
torchmetrics
pytorch_lightning
transformers
git+https://github.com/openai/CLIP.git
```

Then build the image as explained in the course tutorial.

---

### ✅ Option 2: Append Dependencies to an Existing Docker Image _(Recommended)_

> These are the exact steps I followed to append the dependencies without building a new image from scratch:

1. **Open Docker Desktop**

2. **Open a terminal (e.g., PowerShell)**

3. **Run the following commands:**

```bash
# Start a container from your existing image
docker run -it --name memeclip-dev registry.rcp.epfl.ch/ee-559-mmouawad/my-toolbox:v0.2 bash

# Inside the container, install the dependencies
pip install yacs torchmetrics pytorch_lightning transformers
pip install git+https://github.com/openai/CLIP.git

# Commit the container to save the changes
docker commit memeclip-dev registry.rcp.epfl.ch/ee-559-mmouawad/my-toolbox:v0.2-clip

# Push the updated image to the registry
docker push registry.rcp.epfl.ch/ee-559-mmouawad/my-toolbox:v0.2-clip
```

### 🧪 Running the Code
▶️ For Training:
In configs.py, set the following:
```bash
cfg.test_only = False  # Line 16
cfg.reproduce = False  # Line 43
```
⚠️ Make sure to save a copy of model.ckpt, since training will overwrite it.


🧪 For Testing:
```bash
cfg.test_only = True   # Line 16
cfg.reproduce = True   # Line 43
```
Also in case your are running this on your desktop with a CPU: 

Change line 49 in main.py to:
```bash
trainer = Trainer(accelerator='cpu', devices=1, max_epochs=cfg.max_epochs, callbacks=[checkpoint_callback], deterministic=False)
```
Otherwise, you can run it on the GPU with:
```bash
trainer = Trainer(accelerator='gpu', devices=cfg.gpus, max_epochs=cfg.max_epochs, callbacks=[checkpoint_callback], deterministic=False)
```
Moreover:

You can download the dataset and store it inside MemeCLIP_marc 
(<a href="https://drive.google.com/file/d/17WozXiXfq44Z6kkWsPPDHRzqIH2daUaQ/view?usp=sharing">link</a>)

Create a directory inside the code folder called: checkpoints
Inside this directory you can put the model checkpoint from the original MemeCLIP repo, name it model.ckpt

(<a href="https://drive.google.com/file/d/1sUlHw5fSvzPRnMu_K4uzHQY-df3E2pSi/view?usp=sharing">link</a>)
 
