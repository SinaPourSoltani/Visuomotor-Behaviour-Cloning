# Visuomotor-Behaviour-Cloning

This project aims is to investigate the concepts behind Visuomotor and Behaviour Cloning. Visuo-motor addresses the problem of relating a visual input and a control signal while Behaviour Cloningis used to developed control strategies based on expert demonstrations. The inspiration for this project is taken from the paper *Self-Supervised Correspondence in Visuomotor Policy Learning* by Florence et al. [[1]](#1). We also use ResNet18 [[2]](#2). The UR5 URDF-model is from [sholtodouglas](https://github.com/sholtodouglas/ur5pybullet/tree/master/urdf).   

The project report, `Visuomotor-Behaviour-Cloning.pdf`, details the findings of the project.

The best stereo model accomplishing the goal of pushing the object (red lego) to the goal position (green plate) using only stereo images as input to the neural network:


![Best Stereo Model Results](stereo_model_best_demo.gif)

## Setup
Clone the project.

To create a virtual environment and install dependencies do: 

```sh
cd /path/to/Visuomotor-Behaviour-Cloning/

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```
The project has succesfully run on Ubuntu 20.04. Running the project on Ubuntu 18.04 you get problems with numpy requiring a different python version than the default one for Ubuntu 18.04. This can probably be circumvented in various ways.   

## Collecting data
Either use the data provided in [releases](https://github.com/SinaPourSoltani/Visuomotor-Behaviour-Cloning/releases) , here is both data for monocular and stereo setups. Or collect your own data.

To collect data for training, run main.py with flags `--episodes`, `--MaxSteps` and `--stereo_images` to decide the number of episodes to collect data from, the maximum amount of steps in an episode and whether to collect images from a monocular setup or stereo setup, e.g.

```sh
python3 main.py --episodes 500 --MaxSteps 250 --stereo_images True
```

As default images are saved to `./data/images/` while the .csv-file with ground truths for each image or image pair is saved to `./data/`. These paths can be changed with flags as well.

**Beware:** be careful with the collected data. With the current configuration about one in 100 episodes, the expert controller fails to put the object at the goal position before the maximum number of steps has been reached. After collecting data, check the .csv-file for any episodes that reaches maximum number of steps and either prune them directly or check the images.  

## Training

We trained our models in [Google Colaboratory](https://colab.research.google.com), however you should be able to train on your local machine as well. An environment with CUDA cores are probably recommended for this.

The provided notebook `Behaviour_Cloning_CNN.ipynb` can be used for training.

## Testing
To test models run main.py with flags `--episodes`, `--MaxSteps`, `--stereo_images`, `--test` and `--model_name`, e.g.

```sh
python3 main.py --episodes 500 --MaxSteps 250 --stereo_images True --test True --model_name ResNet18_epoch10_stereo_augment_unfrozen_from_15.pth
```

Models should be located in the same folder as `main.py`
If you trained your own models, replace the model name.

**Beware:** if you collected stereo data and trained on that data you will have a stereo model. This model will not be compatible with a monocular setup and vice versa. 

## Results
The results at the end of the project are presented in the following table. The different models were tested on 500 episodes with maximum steps set to 250. Performance is percentage of the 500 episodes, the robot succesfully pushes the object to the correct goal. The models referenced can be found in the [releases](https://github.com/SinaPourSoltani/Visuomotor-Behaviour-Cloning/releases). 

| I | Model | Loss | Performance |
|---|-------|:------:|:-------------:|
| **0** | Expert| --   |99.8 %       |
| **1** | Baseline| 0.0550 | 72.4 % |
| **2** | Unfreeze Layers| 0.0472 | 78.6 % |
| **3** | **2** + Normalize Output | 0.1899 | 78.0 %|
| **4** | **2** + Augmentation & Noise | 0.0272 | 84.6 % |
| **5** | **4** + Stereo | **0.0124** | 91.0 % |
| **6** | **4** + Dropout | 0.0232 | **91.8 %** |  

## References
<a id="1">[1]</a>
P. Florence, L. Manuelli, and R. Tedrake,  2019
*Self-Supervised Correspondence in Visuomotor Policy Learning*
e-print: 1909.06933 arXiv

<a id="2">[2]</a> 
Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun, 2015
*Deep Residual Learning for Image Recognition*
e-print: 1512.03385 arXiv
