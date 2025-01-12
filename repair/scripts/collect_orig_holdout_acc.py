import os
import subprocess, time

def run(subject_paths, runs=5, cuda="2"):
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = cuda

    for path, pnum in sorted(subject_paths):
        print("="*30)
        print(path, pnum)
        for _ in range(runs):
            st_time = time.time()
            subprocess.run(["python", path], env=my_env, cwd=os.path.dirname(path))
            duration = time.time() - st_time
            
            with open(f"/home/hpo/data/DC_orig_training_time.txt", "a") as f:
                f.write(f"{pnum}|{duration:.2f}\n")

def deepfd():
    DeepFD_subjects_path = "/home/hpo/DFD/"
    subject_paths = []
    for dir1 in os.listdir(DeepFD_subjects_path):
        dir1_path = os.path.join(DeepFD_subjects_path, dir1)
        
        origin_path = os.path.join(dir1_path, 'repair_holdout.py')
        if os.path.exists(origin_path):
            subject_paths.append([origin_path, dir1])

        origin_path = os.path.join(dir1_path, 'origin_holdout.py')
        if os.path.exists(origin_path):
            subject_paths.append([origin_path, dir1])
    print(subject_paths)
    run(subject_paths, runs=5, cuda="0")


def deepcrime(dataset):
    DeepCrime_holdout_subjects_path = f"/home/hpo/DC_mutants/{dataset}/holdout_py/"
    subject_paths = []


    for dir1 in os.listdir(DeepCrime_holdout_subjects_path):
        if ".py" in dir1 and "raytune" not in dir1:
            if "lenet_change_learning_rate_mutated0_MP_False_0.001.py" not in dir1:
                continue
            dir1_path = os.path.join(DeepCrime_holdout_subjects_path, dir1)
            subject_paths.append([dir1_path, dir1])
            print(dir1_path)
    
    run(subject_paths, runs=5, cuda="0")

# deepcrime('MNIST')
# deepcrime('CIFAR')
deepcrime('UNITY')
# deepcrime('REUTERS')
# deepfd()