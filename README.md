# A Study on Hybrid Quantum-Classical Convolutional Neural Networks and its Quantum Image Encoding Methods for Image Classification

#### Abstract
This dissertation explores the innovative intersection of quantum computing and classical convolutional neural networks through the development of a Hybrid Quantum-Classical Convolutional Neural Network (HQCCNN). By integrating quantum image encoding methods—specifically the Enhanced Novel Enhanced Quantum Representation (ENEQR) and Enhanced Flexible Representation of Quantum Images (EFRQI)—this research pushes the boundaries of traditional image classification techniques. The quantum encoding methods utilized demonstrate a unique ability to enhance image classification tasks by leveraging quantum mechanics' complex and high-dimensional space. Experimental results reveal that HQCCNNs incorporating these quantum encoding strategies surpass conventional models in accuracy and efficiency, particularly when processing intricate image datasets. This study not only highlights the potential of quantum technologies to revolutionize fields reliant on image classification but also sets the stage for future advancements in quantum artificial intelligence, suggesting a pathway towards more sophisticated quantum-enhanced computational models.

## Requirements
is set in `main.py`.

```docker-compose-gpu.yml``` for simply running the experiments with the default parameters under utilization of a GPU.

```docker-compose-dev-gpu.yml``` same parameters but also makes the code available within the container to allow for easy execution of modified the code (no rebuild of the image necessary).

<br/>

In the top-level `code` directory run the following commands to set up the container and start bash:

```setup
docker-compose -f docker-compose.yml up -d
docker-compose exec experiments bash
```

From now on all commands are run in this bash shell, unless stated otherwise.

## Training

To train the models in the paper, run the following procedure.

First reproduce our model parameters in yaml files:
```setup_yamls
python generate_experiments.py
```
Then you can run the training procedure with the yaml file corresponding to the desired kernel size and random seed:
```train
python main.py experiments_seed_0_filtersize_2x2_imgsize_14.yaml
```
The code will generate entries in the `save` directory, from which you can inspect the parameters and the performance of the models and reload the pre-trained models. 

## Evaluation
You can observe the performance of the models, while they are training or afterwards, through `tensorboard`. Enter this address into your browser, after the experiments have been started, to reach the tensorboard started by default.
```tensorboard
localhost:6007
```

The experiments described in the dissertation can be seen through the following Jupyternotebook:
```evaluation
code/evaluation/Evaluation.ipynb
```

## License and Copyright
This codebase is a fork. Two novel encoding methods were implemented in this fork of the codebase.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
