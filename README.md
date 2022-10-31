### Perceiver-Actor

This will be the official code repository for the [PerAct](https://peract.github.io/) paper. 

We are still in the process of cleaning up, but this [**Colab Tutorial**](https://colab.research.google.com/drive/1wpaosDS94S0rmtGmdnP0J1TjS7mEM14V?usp=sharing) contains everything you need for training PerAct from scratch. The only things missing from this tutorial are our modifications to [RLBench](https://github.com/stepjam/RLBench) and [YARR](https://github.com/stepjam/YARR) for multi-task data-generation, training, and evaluation. 

**Update 31-Oct-2022**: 
- I have pushed my changes to [RLBench](https://github.com/MohitShridhar/RLBench/tree/peract) and [YARR](https://github.com/MohitShridhar/YARR/tree/peract). The data generation is pretty similar to [ARM](https://github.com/stepjam/ARM#running-experiments), except you run `data_generator.py` with `--all_variations=True`. You should be able to use these generated datasets with the [Colab](https://colab.research.google.com/drive/1wpaosDS94S0rmtGmdnP0J1TjS7mEM14V?usp=sharing) code.  
- For the paper, I was using PyTorch DataParallel to train on multiple GPUs. This made the code very messy and brittle. I am currently [stuck](https://github.com/Lightning-AI/lightning/issues/10098) cleaning this up with DDP and PyTorch Lightning. So the code release might be a bit delayed. Apologies.
