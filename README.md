# Near-Optimal Collaborative Learning in Bandits

This repository is the official implementation of [Near-Optimal Collaborative Learning in Bandits] (to appear in the proceedings of NeurIPS'22). 

## Requirements

To install requirements (Python 3.8.5, using Conda to set up a virtual environment, and pip to install packages):

```setup
conda create -n near_optimal_federated python=3.8.5
conda activate near_optimal_federated
python3 -m pip install -r requirements.txt
conda deactivate
```

## Getting started

### Acquire a personal academic license for MOSEK (free)

Go to *https://www.mosek.com/products/academic-licenses/* to acquire a personal academic license, and follow instructions.

### Reproduce results from the paper

- Run command
```bash
bash commands.sh
```
### Run

Parameters can be changed in **params.py**, and are overridden by the parameter values stored in a JSON file at the path provided by params\_path.

### Compare the collaborative setting versus independently solving the bandit instances per population

To compute the corresponding complexity constants from the lower bound, run 

```
python3 test_useful.py
```

## Add new elements of code

- Add a new bandit by creating a new instance of class *FederatedAI* in file **algorithms.py**
- Add a new dataset by adding a few lines of code to file **data.py**
- Add new types of rewards by creating a new instance of class *problem* in file **problems.py**

## Results

Please refer to the paper.

## Contributing

All of the code is under MIT license. Everyone is most welcome to submit pull requests.
