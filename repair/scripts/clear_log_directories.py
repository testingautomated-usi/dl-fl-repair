import os, subprocess


def deepcrime(dataset):
    dir_path = f"/home/hpo/DC_mutants/{dataset}"
    for _path in os.listdir(dir_path):
        if _path in ["initial_params", "holdout_py", "Datasets"]:
            continue

        path = os.path.join(dir_path, _path)
        if os.path.isdir(path):
            print(path)
            subprocess.run(["rm", "-rf", path])

deepcrime("MNIST")
deepcrime("CIFAR")
deepcrime("UNITY")