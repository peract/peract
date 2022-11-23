# Model Card: Perceiver-Actor

Following [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993) and [Lessons from Archives (Jo & Gebru)](https://arxiv.org/pdf/1912.10389.pdf) we provide additional information on PerAct.

## Model Details


### Overview
- Developed by Shridhar et al. at University of Washington and NVIDIA. PerAct is an end-to-end behavior cloning agent that learns to perform a wide variety of language-conditioned manipulation tasks. PerAct uses a Transformer that exploits the 3D structure of _voxel patches_ to learn policies with just a few demonstrations per task.
- Architecture: Transformer trained from scratch with end-to-end supervised learning.
- Trained for 6-DoF manipulation tasks with objects that appear in tabletop scenes.

### Model Date

Nov 2022

### Documents

- [PerAct Paper](https://peract.github.io/paper/peract_corl2022.pdf)
- [PerceiverIO Paper](https://arxiv.org/abs/2107.14795)
- [C2FARM Paper](https://arxiv.org/abs/2106.12534)


## Model Use

- **Primary intended use case**: PerAct is intended for robotic manipulation research. We hope the benchmark and pre-trained models will enable researchers to study the capabilities of Transformers for end-to-end 6-DoF Manipulation. Specifically, we hope the setup serves a reproducible framework for evaluating  robustness and scaling capabilities of manipulation agents. 
- **Primary intended users**: Robotics researchers. 
- **Out-of-scope use cases**: Deployed use cases in real-world autonomous systems without human supervision during test-time is currently out-of-scope. Use cases that involve manipulating novel objects and observations with people, are not recommended for safety-critical systems. The agent is also intended to be trained and evaluated with English language instructions.

## Data

- Pre-training Data for CLIP's language encoder: See [OpenAI's Model Card](https://github.com/openai/CLIP/blob/main/model-card.md#data) for full details. **Note:** We do not use CLIP's vision encoders for any agents in the repo. 
- Manipulation Data for PerAct: The agent was trained with expert demonstrations. In simulation, we use oracle agents and in real-world we use human demonstrations. Since the agent is used in few-shot settings with very limited data, the agent might exploit intended and un-intented biases in the training demonstrations. Currently, these biases are limited to just objects that appear on tabletops.


## Limitations

- Depends on a sampling-based motion planner.
- Hard to extend to dexterous and continuous manipulation tasks.
- Lacks memory to solve tasks with ordering and history-based sequencing. 
- Exploits biases in training demonstrations.
- Needs good hand-eye calibration.
- Doesn't generalize to novel objects.
- Struggles with grounding complex spatial relationships. 
- Does not predict task completion.

See Appendix L in the [paper](https://peract.github.io/paper/peract_corl2022.pdf) for an extended discussion.