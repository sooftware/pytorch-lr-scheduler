# pytorch-lr-scheduler
PyTorch implementation of some learning rate schedulers for deep learning researcher.
  
## Usage
  
### [`ReduceLROnPlateauScheduler`](https://github.com/sooftware/pytorch-lr-scheduler/blob/main/lr_scheduler/reduce_lr_on_plateau_lr_scheduler.py)  
  
- Visualize
  
<img src="https://github.com/sooftware/pytorch-lr-scheduler/blob/main/images/ReduceLROnPlateauScheduler.png" width=400>
  
- Example code
  
```python
import torch

from lr_scheduler.reduce_lr_on_plateau_lr_scheduler import ReduceLROnPlateauScheduler

if __name__ == '__main__':
    max_epochs, steps_in_epoch = 10, 10000

    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = torch.optim.Adam(model, 1e-4)

    scheduler = ReduceLROnPlateauScheduler(optimizer, patience=1, factor=0.3)

    for epoch in range(max_epochs):
        for timestep in range(steps_in_epoch):
            ...
            ...
        
        val_loss = validate()
        scheduler.step(val_loss)
```
  
  
### [`TransformerLRScheduler`](https://github.com/sooftware/pytorch-lr-scheduler/blob/main/lr_scheduler/transformer_lr_scheduler.py)
  
- Visualize
  
<img src="https://github.com/sooftware/pytorch-lr-scheduler/blob/main/images/TransformerLRScheduler.png" width=400>
  
- Example code
  
```python
import torch

from lr_scheduler.transformer_lr_scheduler import TransformerLRScheduler

if __name__ == '__main__':
    max_epochs, steps_in_epoch = 10, 10000

    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = torch.optim.Adam(model, 1e-10)

    scheduler = TransformerLRScheduler(
        optimizer=optimizer, 
        init_lr=1e-10, 
        peak_lr=0.1,
        final_lr=1e-4, 
        final_lr_scale=0.05,
        warmup_steps=3000, 
        decay_steps=17000,
    )

    for epoch in range(max_epochs):
        for timestep in range(steps_in_epoch):
            ...
            ...
            scheduler.step()
```
  

### [`TriStageLRScheduler`](https://github.com/sooftware/pytorch-lr-scheduler/blob/main/lr_scheduler/tri_stage_lr_scheduler.py)
  
- Visualize
  
<img src="https://github.com/sooftware/pytorch-lr-scheduler/blob/main/images/TriStageLRScheduler.png" width=400>
  
- Example code
  
```python
import torch

from lr_scheduler.tri_stage_lr_scheduler import TriStageLRScheduler

if __name__ == '__main__':
    max_epochs, steps_in_epoch = 10, 10000

    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = torch.optim.Adam(model, 1e-10)

    scheduler = TriStageLRScheduler(
        optimizer, 
        init_lr=1e-10, 
        peak_lr=1e-4, 
        final_lr=1e-7, 
        init_lr_scale=0.01, 
        final_lr_scale=0.05,
        warmup_steps=30000, 
        hold_steps=70000, 
        decay_steps=100000,
        total_steps=200000,
    )

    for epoch in range(max_epochs):
        for timestep in range(steps_in_epoch):
            ...
            ...
            scheduler.step()
```
  
  
### [`WarmupReduceLROnPlateauScheduler`](https://github.com/sooftware/pytorch-lr-scheduler/blob/main/lr_scheduler/warmup_reduce_lr_on_plateau_scheduler.py)
  
- Visualize
  
<img src="https://github.com/sooftware/pytorch-lr-scheduler/blob/main/images/WarmupReduceLROnPlateauScheduler.png" width=400>
  
- Example code
```python
import torch

from lr_scheduler.warmup_reduce_lr_on_plateau_scheduler import WarmupReduceLROnPlateauScheduler

if __name__ == '__main__':
    max_epochs, steps_in_epoch = 10, 10000

    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = torch.optim.Adam(model, 1e-10)

    scheduler = WarmupReduceLROnPlateauScheduler(
        optimizer, 
        init_lr=1e-10, 
        peak_lr=1e-4, 
        warmup_steps=30000, 
        patience=1,
        factor=0.3,
    )

    for epoch in range(max_epochs):
        for timestep in range(steps_in_epoch):
            ...
            ...
            if timestep < warmup_steps:
                scheduler.step()
                
        val_loss = validate()
        scheduler.step(val_loss)
```
  
  
### [`WarmupLRScheduler`](https://github.com/sooftware/pytorch-lr-scheduler/blob/main/lr_scheduler/warmup_lr_scheduler.py)
  
- Visualize
  
<img src="https://github.com/sooftware/pytorch-lr-scheduler/blob/main/images/WarmupLRScheduler.png" width=400>
  
- Example code
  
```python
import torch

from lr_scheduler.warmup_lr_scheduler import WarmupLRScheduler

if __name__ == '__main__':
    max_epochs, steps_in_epoch = 10, 10000

    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = torch.optim.Adam(model, 1e-10)

    scheduler = WarmupLRScheduler(
        optimizer, 
        init_lr=1e-10, 
        peak_lr=1e-4, 
        warmup_steps=4000,
    )

    for epoch in range(max_epochs):
        for timestep in range(steps_in_epoch):
            ...
            ...
            scheduler.step()
```
  
  
## Troubleshoots and Contributing
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/sooftware/pytorch-lr-scheduler/issues) on Github.   
  
I appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.
  
## Code Style
I follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) for code style. Especially the style of docstrings is important to generate documentation. 
  
## License
This project is licensed under the MIT LICENSE - see the [LICENSE.md](https://github.com/sooftware/pytorch-lr-scheduler/blob/master/LICENSE) file for details
