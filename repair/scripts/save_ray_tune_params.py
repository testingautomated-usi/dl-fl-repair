import pickle
from ray import tune

params = {    
    "loss": tune.choice([
        "mean_squared_error",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_logarithmic_error",
        "squared_hinge",
        "hinge",
        "categorical_hinge",
        "logcosh",
        "categorical_crossentropy",
        "sparse_categorical_crossentropy",
        "binary_crossentropy",
        "kullback_leibler_divergence",
        "poisson",            
    ]),
    "batch_size": tune.choice([16, 32, 64, 128, 256, 512]),
    "dropout": tune.uniform(0.0, 1.0),
    "optimizer": tune.choice([
        "sgd",
        "rmsprop",
        "adagrad",
        "adadelta",
        "adam",
        "adamax",
        "nadam"
    ]),
    "kernel_regularizer": tune.choice([
        "None",
        "l1",
        "l2",
        "l1_l2"
    ]),
    "kernel_initializer": tune.choice([
        "zeros",
        "ones",
        "constant",
        "random_normal",
        "random_uniform",
        "truncated_normal",
        "orthogonal",
        "lecun_uniform",
        "glorot_normal",
        "glorot_uniform",
        "he_normal",
        "lecun_normal",
        "he_uniform"
    ]),
    "lr": tune.uniform(1e-6, 10),
    "activation_function": tune.choice([
        "elu",
        "softmax",
        "selu",
        "softplus",
        "softsign",
        "relu",
        "tanh",
        "sigmoid",
        "hard_sigmoid",
        "exponential",
        "linear"
    ])
}

#with open("../DLMTRepair/raytune_predefined_params.pkl", "wb") as f:
#    pickle.dump(params, f)
with open("/home/hpo/data/raytune_predefined_params.pkl", "wb") as f:
    pickle.dump(params, f)