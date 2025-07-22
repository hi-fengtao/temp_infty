<div align="center">
<img src="https://github.com/hi-fengtao/temp_infty/blob/main/img/test7.png"/ width=600>  
</div>


<div align="center">
✨ <em>"The path to AGI is continual, INFTY paves the way."<em> ✨ </strong><br />
<em>— INFTY × AGI</em>
</div>

-----------

<p align="center">
  <a href="https://github.com/hi-fengtao/temp_infty/"><img src="https://img.shields.io/badge/Documentation-Read%20the%20Docs-blue.svg" alt="Documentation"></a>
  <a href="https://arxiv.org/abs/2501.01045"><img src="https://img.shields.io/badge/arXiv-2507.12345-b31b1b.svg" alt="arXiv"></a>
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
  <a href="https://pypi.org/project/numpy/"><img src="https://img.shields.io/pypi/v/numpy.svg?label=numpy" alt="numpy"></a>
  <a href="https://pypi.org/project/matplotlib/"><img src="https://img.shields.io/pypi/v/matplotlib.svg?label=matplotlib" alt="matplotlib"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
</a>
</p>

<div align="center">
  <center><h1><span style="color:#7DB9B6">INFTY </span> Engine: An Optimization Toolkit to Support Continual AI</h1></center>
</div>

# What is INFTY?

**INFTY**, a flexible and user-friendly optimization engine tailored for Continual AI. INFTY includes a suite of built-in optimization algorithms that directly tackle core challenges (e.g., the stability–plasticity dilemma, generalization) in Continual AI. And INFTY supports plug-and-play and theoretical analysis utilities, compatible with: i) various Continual AI, e.g., PTM-based CL, and Continual PEFT, Continual Diffusion, and Lifelong RL etc. ii) diverse models, e.g., ResNet, ViT, CLIP, LLM, and Diffusion etc. INFTY provides a unified optimization solution in Continual AI, can serve as infrastructure for broad deployment.

<div align="center">
<img src="https://github.com/hi-fengtao/temp_infty/blob/main/img/INFTY_demo.gif" alt="animated" />
</div>
</br>


# Navigation
- [Overview](#Overview)
- [Navigation](#Navigation)
- [Features](#Features)
- [Algorithms](#Algorithms)
- [Installation](#Installation)
  - [Using pip](##Using-pip)
  - [Developer installation](##Developer-installation)
- [Quick start](#Quick-start)
- [Custom usage](#Custom-usage)
  - [Optimizers](##Optimizers)
  - [Visualization plots](##Visualization-plots)
- [Citation](#Citation)
- [Acknowledgements](#Acknowledgements)
- [Contact us](#Contact-us)
- [License](#License)



# Features
- **Generality**: Built-in CL–friendly optimization algorithms, supporting a wide range of scenarios, models, methods, and learning paradigms.
    
- **Usability**: Portable, plugin-style design, enabling easy replacement of fixed options within existing pipelines.
    
- **Utilities**: Built-in tools for theoretical analysis and visualization, facilitating investigation and diagnostic insight into optimization behavior.

# Algorithms
INFTY has implemented three mainstream algorithms currently:

<div align="center">
<img src="https://github.com/hi-fengtao/temp_infty/blob/main/img/overview.png"/>  
</div>

# Installation

## Using pip

```bash
pip install infty
```
## Developer installation
```bash
git clone https://github.com/xxx.git

cd src/infty && pip install -e .
```

# Quick start
Thanks to the open-source PILOT repo, we provide an example showcasing INFTY optimizers.
Hyperparameters for specific methods are configured in `../infty_configs/`
```
cd examples/PILOT

python main.py --config=exps/ease.json --inftyopt=c_flat
python main.py --config=exps/ease.json --inftyopt=zo_sgd_conserve
python main.py --config=exps/memo_scr.json --inftyopt=pcgrad
```


# Custom usage
## Optimizers
Step 1. Wrap your base optimizer with an INFTY optimizer
```
from infty import optim as infty_optim

base_optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()), 
                lr=self.args['lrate'], 
                momentum=0.9, 
                weight_decay=self.args['weight_decay']
            )
optimizer = infty_optim.C_Flat(params=self._network.parameters(), base_optimizer=base_optimizer, model=self._network, args=self.args)
```
Step 2. Implement the create_loss_fn function
```
def create_loss_fn(self, model, inputs, targets):
    """
    Create a closure to calculate the loss
    """
    def loss_fn():
        outputs = model(inputs)
        logits = outputs["logits"]
        loss_clf = F.cross_entropy(logits, targets)
        return logits, [loss_clf]
    return loss_fn
```
Step 3. Use the loss_fn to calculate the loss and backward
```
loss_fn = self.create_loss_fn(self._network, inputs, targets)
optimizer.set_closure(loss_fn)
logits, loss_list = optimizer.step()

losses += sum(loss_list)
losses_clf += loss_list[0]
```

## Visualization plots
INFTY includes built-in visualization tools for inspecting optimization behavior:
- [x] **Loss Landscape**: visualize sharpness around local minima
- [x] **Hessian ESD**: curvature analysis via eigenvalue spectrum density
- [x] **Conflict Curves**: quantify gradient interference (supports PCGrad, UniGrad, CAGrad)
- [x] **Optimization Trajectory**: observe optimization directions under gradient shifts with a toy example
```
from infty import plot as infty_plot

infty_plot.visualize_landscape(self._network, self.create_loss_fn, train_loader, self._cur_task, self._device)
infty_plot.visualize_esd(self._network, self.create_loss_fn, train_loader, self._cur_task, self._device)
infty_plot.visualize_conflicts()
infty_plot.visualize_trajectory(optim="c_flat")
```

# Citation
xxxx

# Acknowledgements
xxxx

# Contact us
If you have any questions, feel free to open an issue or contact the corresponding author: Tao Feng (fengtao.hi@gmail.com).

# License
This project is licensed under the MIT License.
