import os
import subprocess, fileinput


def remove(dataset):
    DeepCrime_subjects_path = f"/home/hpo/DC_mutants/{dataset}/"
    subject_paths = []

    for dir1 in os.listdir(DeepCrime_subjects_path):
        if ".py" == dir1[-3:]:
            dir1_path = os.path.join(DeepCrime_subjects_path, dir1)
            subject_paths.append([dir1_path, dir1])


    for path, pnum in sorted(subject_paths):
        print(pnum)

        lines = []
        with open(path, "r") as f:
            for line in f:
                if "model.save(model_location)" in line:
                    lines.append("\n")
                else:
                    lines.append(line)

        with open(path, "w") as f:
            for line in lines:
                f.write(line)

remove('UNITY')