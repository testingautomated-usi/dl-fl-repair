import os
import subprocess, fileinput


def insert_to_line(filepath, text_to_search, new_text):
    with open(filepath, 'r') as f:
        for line in f:
            if "DC_orig_holdout_acc" in line:
                return

    for line in fileinput.FileInput(filepath, inplace=1):
        if text_to_search in line:
            line = line.replace(line, line + new_text)
        print(line, end="")


def deepcrime(dataset):
    DeepCrime_subjects_path = f"/home/hpo/DC_mutants/{dataset}/"
    DeepCrime_holdout_subjects_path = f"/home/hpo/DC_mutants/{dataset}/holdout_py/"
    subject_paths = []

    for dir1 in os.listdir(DeepCrime_subjects_path):
        if ".py" == dir1[-3:]:
            dir1_path = os.path.join(DeepCrime_subjects_path, dir1)
            subject_paths.append([dir1_path, dir1])

    if dataset == "MNIST":
        t = []
        t.append(['(y_test, num_classes)', '''    X_test_used, X_test_holdout, y_test_used, y_test_holdout = train_test_split(x_test, y_test, test_size=0.2, random_state=42)'''])
        t.append(['score[1])', '''
        score = model.evaluate(X_test_holdout, y_test_holdout)
        with open("/home/hpo/data/DC_orig_holdout_acc.txt", "a") as f:
            f.write(f"{os.path.basename(__file__)}|{score[1]:.2f}\\n")
'''])
        for path, pnum in sorted(subject_paths):
            print(pnum)
            subprocess.run(["cp", path, DeepCrime_holdout_subjects_path])
            for _t in t:
                insert_to_line(os.path.join(DeepCrime_holdout_subjects_path, pnum),
                            _t[0], _t[1])

    elif dataset == "CIFAR":
        t = []
        t.append(['y_train_orig = np.copy(y_train)', '''    X_test_used, X_test_holdout, y_test_used, y_test_holdout = train_test_split(x_test, y_test, test_size=0.2, random_state=42)'''])
        t.append(['model.save(model_location)', '''
        score = model.evaluate(X_test_holdout, y_test_holdout)
        with open("/home/hpo/data/DC_orig_holdout_acc.txt", "a") as f:
            f.write(f"{os.path.basename(__file__)}|{score[1]:.2f}\\n")
'''])
        for path, pnum in sorted(subject_paths):
            print(pnum)
            subprocess.run(["cp", path, DeepCrime_holdout_subjects_path])
            for _t in t:
                insert_to_line(os.path.join(DeepCrime_holdout_subjects_path, pnum),
                            _t[0], _t[1])

    elif dataset == "UNITY":
        t = []
        t.append(['random_state=42', '''    x_img_test_used, x_img_test_holdout, x_ha_test_used, x_ha_test_holdout, y_gaze_test_used, y_gaze_test_holdout = train_test_split(x_img_test, x_ha_test, y_gaze_test, test_size=0.2, random_state=42)'''])
        t.append(['print(score)', '''
        score = model.evaluate([x_img_test_holdout, x_ha_test_holdout], y_gaze_test_holdout)
        with open("/home/hpo/data/DC_orig_holdout_acc.txt", "a") as f:
            f.write(f"{os.path.basename(__file__)}|{score[1]:.2f}\\n")
        print(score)
'''])
        for path, pnum in sorted(subject_paths):
            print(pnum)
            subprocess.run(["cp", path, DeepCrime_holdout_subjects_path])
            for _t in t:
                insert_to_line(os.path.join(DeepCrime_holdout_subjects_path, pnum),
                            _t[0], _t[1])

#deepcrime('MNIST')
#deepcrime('CIFAR')
deepcrime('UNITY')