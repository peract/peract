# Perceiver-Actor

[**Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation**](https://arxiv.org/abs/2209.05451)  
[Mohit Shridhar](https://mohitshridhar.com/), [Lucas Manuelli](http://lucasmanuelli.com/), [Dieter Fox](https://homes.cs.washington.edu/~fox/)  
[CoRL 2022](https://www.robot-learning.org/) 

PerAct is an end-to-end behavior cloning agent that learns to perform a wide variety of language-conditioned manipulation tasks. PerAct uses a Transformer that exploits the 3D structure of _voxel patches_ to learn policies with just a few demonstrations per task.

![](media/sim_tasks.gif)

The best entry-point for understanding PerAct is [this Colab Tutorial](https://colab.research.google.com/drive/1wpaosDS94S0rmtGmdnP0J1TjS7mEM14V?usp=sharing). If you just want to apply PerAct to your problem, then start with the notebook, otherwise this repo is for mostly reproducing  RLBench results from the paper. 

For the latest updates, see: [peract.github.io](https://peract.github.io)


## Guides

- Getting Started: [Installation](#installation), [Quickstart](#quickstart), [Download](#download)
- Data Generation: [Data Generation](#data-generation)
- Training & Evaluation: [Multi-Task Training and Evaluation](#training-and-evaluation)
- Miscellaneous: [Notebooks](#notebooks), [Disclaimers](#disclaimers-and-limitations), [FAQ](#faq), [Docker Guide](#docker-guide), [Licenses](#licenses)
- Acknowledgements: [Acknowledgements](#acknowledgements), [Citations](#citations)

## Installation

### Prerequisites

PerAct is built-off the [ARM repository](https://github.com/stepjam/ARM) by James et al. The prerequisites are the same as ARM. 

#### 1. Environment

```bash
# setup a virtualenv with whichever package manager you prefer
virtualenv -p $(which python3.8) --system-site-packages peract_env  
source peract_env/bin/activate
pip install --upgrade pip
```

#### 2. PyRep and Coppelia Simulator

Follow instructions from the official [PyRep](https://github.com/stepjam/PyRep) repo; reproduced here for convenience:

PyRep requires version **4.1** of CoppeliaSim. Download: 
- [Ubuntu 16.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu16_04.tar.xz)
- [Ubuntu 18.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz)
- [Ubuntu 20.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)

Once you have downloaded CoppeliaSim, you can pull PyRep from git:

```bash
cd <install_dir>
git clone https://github.com/stepjam/PyRep.git
cd PyRep
```

Add the following to your *~/.bashrc* file: (__NOTE__: the 'EDIT ME' in the first line)

```bash
export COPPELIASIM_ROOT=<EDIT ME>/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this.

**Warning**: CoppeliaSim might cause conflicts with ROS workspaces. 

Finally install the python library:

```bash
pip install -r requirements.txt
pip install .
```

You should be good to go!
You could try running one of the examples in the *examples/* folder.

If you encounter errors, please use the [PyRep issue tracker](https://github.com/stepjam/PyRep/issues).

#### 3. RLBench

PerAct uses my [RLBench fork](https://github.com/MohitShridhar/RLBench/tree/peract). Follow the instructions from the official [RLBench repo](https://github.com/stepjam/RLBench); reproduced here for convenience:

```bash
cd <install_dir>
git clone -b peract https://github.com/MohitShridhar/RLBench.git # note: 'peract' branch

cd RLBench
pip install -r requirements.txt
python setup.py develop
```

For [running in headless mode](https://github.com/MohitShridhar/RLBench/tree/peract#running-headless), tasks setups, and other issues, please refer to the [official repo](https://github.com/stepjam/RLBench).

#### 4. YARR

PerAct uses my [YARR fork](https://github.com/MohitShridhar/YARR/tree/peract). Follow the instructions from the official [YARR repo](https://github.com/stepjam/YARR); reproduced here for convenience:

```bash
cd <install_dir>
git clone -b peract https://github.com/MohitShridhar/YARR.git # note: 'peract' branch

cd YARR
pip install -r requirements.txt
python setup.py develop
```

### PerAct Repo
Clone:
```bash
cd <install_dir>
git clone https://github.com/peract/peract.git
```

Install:
```bash
cd peract
pip install -r requirements.txt

export PERACT_ROOT=$(pwd)  # mostly used as a reference point for tutorials
python setup.py develop
```

**Note**: You might need versions of `torch==1.7.1` and `torchvision==0.8.2` that are compatible with your CUDA and hardware. Later versions should also be fine (in theory). 

### Gotchas

Later, if you see GL errors, it's probably the PyRender voxel visualizer. See this [issue](https://github.com/mmatl/pyrender/issues/86) for reference. You might have to set the following environment variables depending on your setup:

```bash
export DISPLAY=:0
export MESA_GL_VERSION_OVERRIDE=4.1
export PYOPENGL_PLATFORM=egl
```

## Quickstart

A quick tutorial on evaluating a pre-trained multi-task agent.

Download a [pre-trained PerAct checkpoint](https://drive.google.com/file/d/1vc_IkhxhNfEeEbiFPHxt_AsDclDNW8d5/view?usp=share_link) trained with 100 demos per task (18 tasks in total):
```bash
cd $PERACT_ROOT
python script/quickstart_download.py 
# if this throws PermissionErrors, see https://github.com/wkentaro/gdown/issues/43
```

Generate a small `val` set of 10 episodes for `open_drawer` inside `$PERACT_ROOT/data`:
```bash
cd <install_dir>/RLBench/tools
python dataset_generator.py --tasks=open_drawer \
                            --save_path=$PERACT_ROOT/data/val \
                            --image_size=128,128 \
                            --renderer=opengl \
                            --episodes_per_task=10 \
                            --processes=1 \
                            --all_variations=True
```
This will take a few minutes to finish.

Evaluate the pre-trained PerAct agent:
```bash
cd $PERACT_ROOT
CUDA_VISIBLE_DEVICES=0 python eval.py \
    rlbench.tasks=[open_drawer] \
    rlbench.task_name='multi' \
    rlbench.demo_path=$PERACT_ROOT/data/val \
    framework.gpu=0 \
    framework.logdir=$PERACT_ROOT/ckpts/ \
    framework.start_seed=0 \
    framework.eval_envs=1 \
    framework.eval_from_eps_number=0 \
    framework.eval_episodes=10 \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    framework.eval_type='last' \
    rlbench.headless=False
```
If you are on a headless machine turn off the visualization with `headless=True`.

You can evaluate the same agent on other tasks. First generate a validation dataset like above (or [download a pre-generated dataset](#download)) and then run `eval.py`.

**Note:** The dowloaded checkpoint might not necessarily be the best one for a given task, it's simply the last checkpoint from training.

## Download

### Pre-Generated Datasets

We provide [pre-generated RLBench demonstrations](https://drive.google.com/drive/folders/1nXgp0G7jqFwJiOljS3o7DG9uURhN0mK9?usp=sharing) for train (100 episodes), validation (25 episodes), and test (25 episodes) splits used in the paper. If you directly use these datasets, you don't need to run `tools/data_generator.py` from RLBench. Using these datasets will also help reproducibility since each scene is randomly sampled in `data_generator.py`.

Is there one big zip file with all splits and tasks instead of individual files? No. My gDrive account will get rate-limited if everyone is directly downloading huge files. I recommend downloading through [rclone](https://rclone.org/drive/) with Google API Console enabled. The full dataset of zip files is ~116GB. 

### Pre-Trained Checkpoints

#### [PerAct (600K)](https://drive.google.com/file/d/1vc_IkhxhNfEeEbiFPHxt_AsDclDNW8d5/view?usp=share_link)
- Num Tasks: 18
- Train Demos: 100 episodes per task
- Voxel Size: 100x100x100
- Cameras: `front`, `left_shoulder`, `right_shoulder`, `wrist`
- Latents: 2048
- Self-Attention Layers: 6
- Voxel Feature Dim: 64
- Data Augmentation: 45 deg yaw perturbations

Todo: add more checkpoints ...

## Data Generation

Data generation is pretty similar to the [ARM setup](https://github.com/stepjam/RLBench), except you use `--all_variations=True` to sample all task variations:

```bash
cd <install_dir>/RLBench/tools
python dataset_generator.py --tasks=open_drawer \
                            --save_path=$PERACT_ROOT/data/train \
                            --image_size=128,128 \
                            --renderer=opengl \
                            --episodes_per_task=100 \
                            --processes=1 \
                            --all_variations=True
```

You can run these in parallel for multiple tasks. Here is a list of 18 tasks used in the paper: 

```
close_jar
insert_onto_square_peg
light_bulb_in
meat_off_grill
open_drawer
place_cups
place_shape_in_shape_sorter
push_buttons
put_groceries_in_cupboard
put_item_in_drawer
put_money_in_safe
reach_and_drag
stack_blocks
stack_cups
turn_tap
place_wine_at_rack_location
slide_block_to_color_target
sweep_to_dustpan_of_size
```

You can probably train PerAct on more RLBench tasks. These 18 tasks were hand-selected for their diversity in task variations and language instruction. 

**Warning**: Each scene generated with `data_generator.py` will use a different random seed to configure objects and states in the scene. This means you will get very different train, val, and test sets to the pre-generated ones. This should be fine for PerAct, but you will likely see small differences in evaluation performances. It's recommended to use the pre-generated datasets for reproducibility. Using larger test sets will also help. 

## Training and Evaluation

The following is a guide for training everything from scratch. All tasks follow a 4-phase workflow:
 
1. Generate `train`, `val`, `test` datasets with `data_generator.py` or download pre-generated datasets. 
2. Train agent with `train.py` and save 10K iteration checkpoints.
3. Run validation with `eval.py` with `framework.eval_type=missing` to find the best checkpoint on `val` tasks and save results in `eval_data.csv`.
4. Evaluate the best checkpoint in `eval_data.csv` on `test` tasks with `eval.py` and `framework.eval_type=best`. Save final results to `test_data.csv`. 


Make sure you have a `train`, `val`, and `test` set with sufficient demos for the tasks you want to train and evaluate on. 

### Training

Train a `PERACT_BC` agent with `100` demos per task for 600K iterations with 8 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    method=PERACT_BC \
    rlbench.tasks=[close_jar,insert_onto_square_peg,light_bulb_in,meat_off_grill,open_drawer,place_cups,place_shape_in_shape_sorter,push_buttons,put_groceries_in_cupboard,put_item_in_drawer,put_money_in_safe,reach_and_drag,stack_blocks,stack_cups,turn_tap,place_wine_at_rack_location,slide_block_to_color_target,sweep_to_dustpan_of_size] \
    rlbench.task_name='multi_18T' \
    rlbench.cameras=[front,left_shoulder,right_shoulder,wrist] \
    rlbench.demos=100 \
    rlbench.demo_path=$PERACT_ROOT/data/train \
    replay.batch_size=1 \
    replay.path=/tmp/replay \
    replay.max_parallel_processes=32 \
    method.voxel_sizes=[100] \
    method.voxel_patch_size=5 \
    method.voxel_patch_stride=5 \
    method.num_latents=2048 \
    method.transform_augmentation.apply_se3=True \
    method.transform_augmentation.aug_rpy=[0.0,0.0,45.0] \
    method.pos_encoding_with_lang=True \
    framework.training_iterations=600000 \
    framework.num_weights_to_keep=60 \
    framework.start_seed=0 \
    framework.log_freq=1000 \
    framework.save_freq=10000 \
    framework.logdir=$PERACT_ROOT/logs/ \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    ddp.num_devices=8
```

Make sure there is enough disk-space for `replay.path` and `framework.logdir`. Adjust `replay.max_parallel_processes` to fill the replay buffer in parallel based on your resources. You can also train on fewer GPUs, but training will take a long time to converge. 

Use `tensorboard` to monitor training progress with logs inside `framework.logdir`.

### Validation

Evaluate `PERACT_BC` seed0 on 18 `val` tasks sequentially (slow!):

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
    rlbench.tasks=[close_jar,insert_onto_square_peg,light_bulb_in,meat_off_grill,open_drawer,place_cups,place_shape_in_shape_sorter,push_buttons,put_groceries_in_cupboard,put_item_in_drawer,put_money_in_safe,reach_and_drag,stack_blocks,stack_cups,turn_tap,place_wine_at_rack_location,slide_block_to_color_target,sweep_to_dustpan_of_size] \
    rlbench.task_name='multi_18T' \
    rlbench.demo_path=$PERACT_ROOT/data/val \
    framework.logdir=$PERACT_ROOT/logs/ \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    framework.eval_envs=1 \
    framework.start_seed=0 \
    framework.eval_episodes=25 \
    framework.eval_type='missing' \
    rlbench.headless=True
```

This script will slowly go through each 10K interval checkpoint and save success rates in `eval_data.csv`.

### Testing

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
    rlbench.tasks=[close_jar,insert_onto_square_peg,light_bulb_in,meat_off_grill,open_drawer,place_cups,place_shape_in_shape_sorter,push_buttons,put_groceries_in_cupboard,put_item_in_drawer,put_money_in_safe,reach_and_drag,stack_blocks,stack_cups,turn_tap,place_wine_at_rack_location,slide_block_to_color_target,sweep_to_dustpan_of_size] \
    rlbench.task_name='multi_18T' \
    rlbench.demo_path=$PERACT_ROOT/data/test \
    framework.logdir=$PERACT_ROOT/logs/ \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    framework.eval_envs=1 \
    framework.start_seed=0 \
    framework.eval_episodes=25 \
    framework.eval_type='best' \
    rlbench.headless=True
```

The final results will be saved in `test_data.csv`.

### Baselines and Ablations

All agents reported in the paper are [here](https://github.com/peract/peract/tree/main/agents) along with their respective [config files](https://github.com/peract/peract/tree/main/conf/method):

|    **Code Name**   | **Paper Name** |
|:------------------:|:--------------:|
|      `PERACT_BC`   |     PerAct     |
|`C2FARM_LINGUNET_BC`|    C2FARM-BC   |
|    `VIT_BC_LANG`   | Image-BC (VIT) |
|      `BC_LANG`     | Image-BC (CNN) |

PerAct ablations are set with:

```bash
method.no_skip_connection: False
method.no_perceiver: False
method.no_language: False
method.keypoint_method: 'heuristic'
```

## Disclaimers and Limitations

- **Code quality level**: Desperate grad student. 
- **Why isn't your code more modular?**: My code, like this project, is end-to-end. 
- **Small test set**: The test should be larger than just 25 episodes. If you parallelize the evaluation, you can easily evaluate on larger test sets.
- **Parallelization**: A lot of things (data generation, evaluation) are slow because everything is done serially. Parallelizing these processes will save you a lot of time. 
- **Impossible tasks**: Some tasks like `push_buttons` are not solvable by PerAct since it doesn't have any memory.
- **Switch from DP to DDP**: For the paper submission, I was using PyTorch DataParallel for multi-gpu training. For this code release, I switched to DistributedDataParallel. Hopefully, I didn't introduce any new bugs. 
- **Collision avoidance**: All simulated evaluations use V-REP's internal motion-planner with collision avoidance. For real-world experiments, you have to setup MoveIt to use the voxel grid for avoiding occupied voxels. 
- **YARR Modifications**: My changes to the YARR repo are a total mess. Sorry :(
- **Other limitations**: See Appendix L of the paper for more details.

## FAQ

#### How much training data do I need for real-world tasks?

It depends on the complexity of the task. With 10-20 demonstrations the agent should start to do something useful, but it will often make mistakes by picking the wrong object. For robustness you probably need 50-100 demostrations. A good way to gauge how much data you might need is to setup a simulated version of the problem and evaluate agents trained with 10, 100, 250 demonstrations.  

#### Why doesn't the agent follow my language instruction?

This means either there is some sort of bias in the dataset that the agent is exploiting (e.g. always 'blue blocks'), or you don't have enough training data. Also make sure that the task is doable - if a referred attribute is barely legible in the voxel grid, then it's going to be hard for agent to figure out what you mean. 

#### How to pick the best checkpoint for real-robot tasks?

Ideally, you should create a validation set with heldout instances and then choose the checkpoint with the lowest translation and rotation errors. You can also reuse the training instances but swap the language instructions with unseen goals. But all real-world experiments in the paper simply chose the last checkpoint. 

#### Can you replace the motion-planner with a learnable module?

Yes, see [C2FARM+LPR](https://github.com/stepjam/ARM) by James et al. 


## Docker Guide

Coming soon...
 

## Notebooks

- [Colab Tutorial](https://colab.research.google.com/drive/1wpaosDS94S0rmtGmdnP0J1TjS7mEM14V?usp=sharing): This tutorial is a good starting point for understanding the data-loading and training pipeline.
- Dataset Visualizer: Coming soon ... see [Colab](https://colab.research.google.com/drive/1wpaosDS94S0rmtGmdnP0J1TjS7mEM14V?usp=sharing) for now.
- Q-Prediction Visualizer:  Coming soon ... see [Colab](https://colab.research.google.com/drive/1wpaosDS94S0rmtGmdnP0J1TjS7mEM14V?usp=sharing) for now.
- Results Notebook: Coming soon ...


## Hardware Requirements

PerAct agents for the paper were trained with 8 P100 cards with 16GB of memory each. You can use fewer GPUs, but training will take a long time to converge.

Tested with:
- **GPU** - NVIDIA P100
- **CPU** - Intel Xeon (Quad Core)
- **RAM** - 32GB
- **OS** - Ubuntu 16.04, 18.04

For inference, a single GPU is sufficient.

## Acknowledgements

This repository uses code from the following open-source projects:

#### ARM 
Original:  [https://github.com/stepjam/ARM](https://github.com/stepjam/ARM)  
License: [ARM License](https://github.com/stepjam/ARM/LICENSE)    
Changes: Data loading was modified for PerAct. Voxelization code was modified for DDP training.

#### PerceiverIO
Original: [https://github.com/lucidrains/perceiver-pytorch](https://github.com/lucidrains/perceiver-pytorch)   
License: [MIT](https://github.com/lucidrains/perceiver-pytorch/blob/main/LICENSE)   
Changes: PerceiverIO adapted for 6-DoF manipulation.

#### ViT
Original: [https://github.com/lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)     
License: [MIT](https://github.com/lucidrains/vit-pytorch/blob/main/LICENSE)   
Changes: ViT adapted for baseline.   

#### LAMB Optimizer
Original: [https://github.com/cybertronai/pytorch-lamb](https://github.com/cybertronai/pytorch-lamb)   
License: [MIT](https://github.com/cybertronai/pytorch-lamb/blob/master/LICENSE)   
Changes: None.

#### OpenAI CLIP

Original: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)  
License: [MIT](https://github.com/openai/CLIP/blob/main/LICENSE)  
Changes: Minor modifications to extract token and sentence features.

Thanks for open-sourcing! 

## Licenses
- [PerAct License (Apache 2.0)](LICENSE) - Perceiver-Actor Transformer
- [ARM License](ARM_LICENSE) - Voxelization and Data Preprocessing 
- [YARR Licence (Apache 2.0)](https://github.com/stepjam/YARR/blob/main/LICENSE)
- [RLBench Licence](https://github.com/stepjam/RLBench/blob/master/LICENSE)
- [PyRep License (MIT)](https://github.com/stepjam/PyRep/blob/master/LICENSE)
- [Perceiver PyTorch License (MIT)](https://github.com/lucidrains/perceiver-pytorch/blob/main/LICENSE)
- [LAMB License (MIT)](https://github.com/cybertronai/pytorch-lamb/blob/master/LICENSE)

## Release Notes

**Update 23-Nov-2022**
- I ditched PyTorch Lightning and implemented multi-gpu trying directly with Pytorch DDP. I could have introduced some bugs during this transition and from refactoring the repo in general. 

**Update 31-Oct-2022**:
- I have pushed my changes to [RLBench](https://github.com/MohitShridhar/RLBench/tree/peract) and [YARR](https://github.com/MohitShridhar/YARR/tree/peract). The data generation is pretty similar to [ARM](https://github.com/stepjam/ARM#running-experiments), except you run `data_generator.py` with `--all_variations=True`. You should be able to use these generated datasets with the [Colab](https://colab.research.google.com/drive/1wpaosDS94S0rmtGmdnP0J1TjS7mEM14V?usp=sharing) code.  
- For the paper, I was using PyTorch DataParallel to train on multiple GPUs. This made the code very messy and brittle. I am currently [stuck](https://github.com/Lightning-AI/lightning/issues/10098) cleaning this up with DDP and PyTorch Lightning. So the code release might be a bit delayed. Apologies.


## Citations 

**PerAct**
```
@inproceedings{shridhar2022peract,
  title     = {Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation},
  author    = {Shridhar, Mohit and Manuelli, Lucas and Fox, Dieter},
  booktitle = {Proceedings of the 6th Conference on Robot Learning (CoRL)},
  year      = {2022},
}
```

**C2FARM**
```
@inproceedings{james2022coarse,
  title={Coarse-to-fine q-attention: Efficient learning for visual robotic manipulation via discretisation},
  author={James, Stephen and Wada, Kentaro and Laidlow, Tristan and Davison, Andrew J},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13739--13748},
  year={2022}
}
```

**PerceiverIO**
```
@article{jaegle2021perceiver,
  title={Perceiver io: A general architecture for structured inputs \& outputs},
  author={Jaegle, Andrew and Borgeaud, Sebastian and Alayrac, Jean-Baptiste and Doersch, Carl and Ionescu, Catalin and Ding, David and Koppula, Skanda and Zoran, Daniel and Brock, Andrew and Shelhamer, Evan and others},
  journal={arXiv preprint arXiv:2107.14795},
  year={2021}
}
```


**RLBench**
```
@article{james2020rlbench,
  title={Rlbench: The robot learning benchmark \& learning environment},
  author={James, Stephen and Ma, Zicong and Arrojo, David Rovick and Davison, Andrew J},
  journal={IEEE Robotics and Automation Letters},
  volume={5},
  number={2},
  pages={3019--3026},
  year={2020},
  publisher={IEEE}
}
```