import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Input,
    Concatenate,
    Conv2D,
    Flatten,
    Dense,
    MaxPool2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal, Constant, GlorotNormal
from tensorflow import math

import random, os, pickle, sys, json
import ray
from ray import tune
from ray.tune.suggest.hebo import HEBOSearch
from ray.tune.suggest.basic_variant import BasicVariantGenerator
import numpy as np
from ray.tune.schedulers import ASHAScheduler

from keras.optimizers import *

N_ITER = -1
EPOCHS_MULTIPLIER = random.uniform(1.1, 1.3)
TIME_LIMIT = int(sys.argv[2])
NUM_TRAINING = 1
ORIG_TRAINING_TIME_PATH = "/home/hpo/data/DC_orig_training_time.txt"


def angle_loss_fn(y_true, y_pred):
    x_p = math.sin(y_pred[:, 0]) * math.cos(y_pred[:, 1])
    y_p = math.sin(y_pred[:, 0]) * math.sin(y_pred[:, 1])
    z_p = math.cos(y_pred[:, 0])

    x_t = math.sin(y_true[:, 0]) * math.cos(y_true[:, 1])
    y_t = math.sin(y_true[:, 0]) * math.sin(y_true[:, 1])
    z_t = math.cos(y_true[:, 0])

    norm_p = math.sqrt(x_p * x_p + y_p * y_p + z_p * z_p)
    norm_t = math.sqrt(x_t * x_t + y_t * y_t + z_t * z_t)

    dot_pt = x_p * x_t + y_p * y_t + z_p * z_t

    angle_value = dot_pt / (norm_p * norm_t)
    angle_value = tf.clip_by_value(angle_value, -0.99999, 0.99999)

    loss_val = math.acos(angle_value)

    return tf.reduce_mean(loss_val, axis=-1)


def train(config):
    dataset_folder = "/home/hpo/DC_mutants/UNITY/Datasets"

    x_img = np.load(os.path.join(dataset_folder, "dataset_x_img.npy"))
    x_head_angles = np.load(
        os.path.join(dataset_folder, "dataset_x_head_angles_np.npy")
    )
    y_gaze_angles = np.load(
        os.path.join(dataset_folder, "dataset_y_gaze_angles_np.npy")
    )

    (
        x_img_train,
        x_img_test,
        x_ha_train,
        x_ha_test,
        y_gaze_train,
        y_gaze_test,
    ) = train_test_split(
        x_img, x_head_angles, y_gaze_angles, test_size=0.2, random_state=42
    x_img_test_used, x_img_test_holdout, x_ha_test_used, x_ha_test_holdout, y_gaze_test_used, y_gaze_test_holdout = train_test_split(x_img_test, x_ha_test, y_gaze_test, test_size=0.2, random_state=42)    )

    (
        x_img_test_used,
        x_img_test_holdout,
        x_ha_test_used,
        x_ha_test_holdout,
        y_gaze_test_used,
        y_gaze_test_holdout,
    ) = train_test_split(
        x_img_test, x_ha_test, y_gaze_test, test_size=0.2, random_state=42
    x_img_test_used, x_img_test_holdout, x_ha_test_used, x_ha_test_holdout, y_gaze_test_used, y_gaze_test_holdout = train_test_split(x_img_test, x_ha_test, y_gaze_test, test_size=0.2, random_state=42)    )

    # Build the model
    image_input = Input((36, 60, 1))
    head_pose_input = Input((2,))

    initialiser_normal = RandomNormal(mean=0.0, stddev=0.1)
    initialiser_const = Constant(value=0)
    initialiser_xavier = GlorotNormal(seed=None)

    for k, v in config.items():
        if v == "initialiser_normal":
            v = initialiser_normal
        elif v == "initialiser_const":
            v = initialiser_const
        elif v == "initialiser_xavier":
            v = initialiser_xavier
        elif v == "angle_loss_fn":
            v = angle_loss_fn
        elif v == "None":
            v = None
        config[k] = v

    x = Conv2D(
        filters=20,
        kernel_size=(5, 5),
        strides=1,
        padding="valid",
        kernel_initializer=config["ki1"],
        bias_initializer=config["bi1"],
        activation=config["act1"],
    )(image_input)

    x = MaxPool2D(strides=2, pool_size=2)(x)

    x = Conv2D(
        filters=50,
        kernel_size=(5, 5),
        strides=1,
        padding="valid",
        kernel_initializer=config["ki2"],
        bias_initializer=config["bi2"],
        activation=config["act2"],
    )(x)

    x = MaxPool2D(strides=2, pool_size=2)(x)

    x = Flatten()(x)
    # relu
    x = Dense(
        500,
        activation=config["act3"],
        kernel_initializer=config["ki3"],
        bias_initializer=config["bi3"],
        kernel_regularizer=config["kr3"]
    )(x)

    x = Concatenate()([head_pose_input, x])

    output = Dense(
        2,
        activation=config["act4"],
        kernel_initializer=config["ki4"],
        bias_initializer=config["bi4"],
    )(x)

    model = Model(inputs=[image_input, head_pose_input], outputs=output)

    if config["optimizer"] == "sgd":
        optimizer = SGD(lr=config["lr"])
    elif config["optimizer"] == "adam":
        optimizer = Adam(lr=config["lr"])
    elif config["optimizer"] == "rmsprop":
        optimizer = RMSprop(lr=config["lr"])
    elif config["optimizer"] == "adagrad":
        optimizer = Adagrad(lr=config["lr"])
    elif config["optimizer"] == "adadelta":
        optimizer = Adadelta(lr=config["lr"])
    elif config["optimizer"] == "adamax":
        optimizer = Adamax(lr=config["lr"])
    elif config["optimizer"] == "nadam":
        optimizer = Nadam(lr=config["lr"])

    model.compile(
        optimizer=optimizer,
        loss=config["loss"],
        metrics=[angle_loss_fn],
    )

    history = model.fit(
        [x_img_train, x_ha_train],
        y_gaze_train,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        shuffle=True,
        validation_split=0.1,
        verbose=0
    )

    _, val_loss = model.evaluate([x_img_test_used, x_ha_test_used], y_gaze_test_used)
    _, holdout_loss = model.evaluate(
        [x_img_test_holdout, x_ha_test_holdout], y_gaze_test_holdout
    )
    tune.report(mean_loss=val_loss, holdout_loss=holdout_loss)


def run_tune(algo, mutant, repeat):
    if algo is None:
        algo = "hebo"

    with open("/home/hpo/data/raytune_predefined_params.pkl", "rb") as f:
        _params = pickle.load(f)

    used_params = ["loss", "batch_size", "optimizer", "lr"]
    params = {}
    for p in used_params:
        if p not in _params:
            continue
        params[p] = _params[p]

    params["act1"] = params["act2"] = params["act3"] = params["act4"] = _params[
        "activation_function"
    ]
    params["ki1"] = params["ki2"] = params["ki3"] = params["ki4"] = _params[
        "kernel_initializer"
    ]
    params["bi1"] = params["bi2"] = params["bi3"] = params["bi4"] = _params[
        "kernel_initializer"
    ]


    with open("./initial_params/lenet.json", "r") as f:
        initial_params = json.load(f)
    with open(f"./initial_params/mutant_params.json", "r") as f:
        mutant_params = json.load(f)
        mutant_params = mutant_params[mutant]

    for k, v in mutant_params.items():
        if k in initial_params:
            initial_params[k] = v
        else:
            pass  # TODO

    with open(ORIG_TRAINING_TIME_PATH, "r") as f:
        train_times = []
        for line in f:
            pnum, train_time = line.strip().split("|")
            if pnum == mutant:
                train_times.append(float(train_time))

    epochs = [initial_params["epochs"]]
    initial_params = [initial_params]

    for x in range(5):
        val = int(initial_params[0]["epochs"] * (EPOCHS_MULTIPLIER**x))
        epochs.append(val)

        val = int(initial_params[0]["epochs"] / (EPOCHS_MULTIPLIER**x))
        if val > 0:
            epochs.append(val)

    epochs = tune.choice(list(set(epochs)))
    params["epochs"] = epochs

    for k, v in initial_params[0].items():
        if k == "lr" or "dropout" in k:
            continue
        if v not in params[k].categories:
            params[k] = tune.choice(params[k].categories + [v])
            initial_params[0][k] = v

    if algo == "hebo":
        search_alg = HEBOSearch(
            metric="mean_loss", mode="min", points_to_evaluate=initial_params
        )
    elif algo == "random":
        search_alg = BasicVariantGenerator(points_to_evaluate=initial_params)
    elif algo == "asha":
        search_alg = ASHAScheduler(
            time_attr="time_total_s", max_t=np.mean(train_times) * (EPOCHS_MULTIPLIER**4),
            grace_period=np.mean(train_times) / (EPOCHS_MULTIPLIER**4)
        )
    
    if algo in ["hebo", "random"]:
        analysis = tune.run(
            train,
            name=f"{mutant}_train_{algo}_iter_{N_ITER}_time_{TIME_LIMIT}",
            metric="mean_loss",
            mode="min",
            time_budget_s=np.mean(train_times) * TIME_LIMIT,
            num_samples=N_ITER,
            resources_per_trial={"cpu": 40, "gpu": 1},
            config=params,
            verbose=1,
            search_alg=search_alg,
            local_dir="./",
            raise_on_failed_trial=False,
        )
    else:
        analysis = tune.run(
            train,
            name=f"{mutant}_train_{algo}_iter_{N_ITER}_time_{TIME_LIMIT}",
            metric="mean_loss",
            mode="min",
            time_budget_s=np.mean(train_times) * TIME_LIMIT,
            num_samples=N_ITER,
            resources_per_trial={"cpu": 40, "gpu": 1},
            config=params,
            verbose=1,
            scheduler=search_alg,
            local_dir="./",
            raise_on_failed_trial=False,
        )
    print("Best hyperparameters found were: ", analysis.best_config)
    results_df = analysis.results_df
    results_df.to_csv(
        f"/home/hpo/results/DC_UNITY_|{mutant}|_{algo}_iter_{N_ITER}_time_{TIME_LIMIT}_repeat_{repeat}.csv"
    )


for mutant in os.listdir("./"):
    if (
        ".py" not in mutant
        or mutant in ["lenet.py", "lenet_raytune.py"]
        or not os.path.isfile(mutant)
    ):
        continue

    print("=" * 30)

    print(mutant)
    #NUM_STUDY_REPEAT = 1
    #for repeat in range(NUM_STUDY_REPEAT):
    #    run_tune(sys.argv[1], mutant, repeat)
