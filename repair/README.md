### LLM Scripts and Results
This is located under `LLM` directory.

### HPO, AutoTrainer, Random Full Results
Due to a space limit of the paper, here we present full results of
- Table 3 in the paper: Please see `Table III.md` file.
- AJ values of RQ4 results: Please see `AJ.md` file.

### HPO, AutoTrainer, Random Raw Results
- The raw results of HPO and random can be found at `results` directory.
- The raw logs of AutoTrainer can be found at https://figshare.com/s/c93d3c14edbb472349b6.

### Dataset
- Artificial faults dataset using DeepCrime is available at https://figshare.com/s/e0f53137fd6550478049.
- DeepFD faults used in the study are located in `DFD` directory. Find the full list of their original faults at https://github.com/ArabelaTso/DeepFD.


## Source Code To Run ### HPO, AutoTrainer, Random Experiments
### Directories and Files
- `data`: contains a predefined set of hyperparameters, results (execution time, accuracy or other evaluation metrics) of running the faults and fixes.
- `DC_mutants`: contains subdirectories of each dataset of artificial faults. For example, `DC_mutants/MNIST` has Python files of the faults (mutants) and one Ray Tune file (`mnist_raytune.py`) to run HPO techniques for all mutants. The directory named `initial_params` has two files, which of each is mutants' parameters and original parameters, respectively.
- `DFD`: has nine real faults from DeepFD paper. Each subdirectory has `origin_raytune.py` file
to run HPO experiments. `gt.json` contains hyperparameters of the fixes by the developers.
- `scripts`: contains scripts for collecting accuracies (or other evaluation metrics), inserting `train_test_split` for splitting test set, and saving default Ray Tune parameters.


### Environments
Tested with Python 3.8.2
- tensorflow 2.3.4
- Keras 2.4.3

### How to run Ray Tune Python scripts
Decide target fault and run corresponding Ray Tune file.

For example,

```bash
cd DFD/31880720
python python origin_raytune.py --algo hebo --limit 20 --repeat 5 --top10 1
```
The options of the Ray Tune file are:
- `algo`: decides the HPO algorithm. It can be one of [hebo, bohb, random]
- `limit`: sets the maximum time limit.
- `repeat`: repeats the experiments N times.
- `top10`: now only supports 1.
