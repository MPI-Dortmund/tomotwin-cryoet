"""
Copyright (c) 2022 MPI-Dortmund
SPDX-License-Identifier: MPL-2.0

This file is subject to the terms of the Mozilla Public License, Version 2.0 (MPL-2.0).
The full text of the MPL-2.0 can be found at http://mozilla.org/MPL/2.0/.

For files that are Incompatible With Secondary Licenses, as defined under the MPL-2.0,
additional notices are required. Refer to the MPL-2.0 license for more details on your
obligations and rights under this license and for instructions on how secondary licenses
may affect the distribution and modification of this software.
"""

import argparse
import copy
import json
import os
import sys
from typing import Dict

import numpy as np
import optuna
import optuna.samplers as samplers
from importlib_metadata import version
from optuna.pruners import MedianPruner
from optuna.storages import RetryFailedTrialCallback
from optuna.trial import TrialState
from tqdm import tqdm

from tomotwin import train_main as tmain
from tomotwin.modules.common.distances import DistanceManager
from tomotwin.modules.common.preprocess import label_filename
from tomotwin.modules.networks.networkmanager import NetworkManager
from tomotwin.modules.training.LossPyML import LossPyML
from tomotwin.modules.training.argparse_ui import TrainingConfiguration
from tomotwin.modules.training.mrctriplethandler import MRCTripletHandler
from tomotwin.modules.training.torchtrainer import TorchTrainer, TripletDataset
from tomotwin.train_main import get_augmentations, generate_triplets

EPOCHS = 3
MAX_RETRY = 3
DISTANCE = "COSINE"
BASIC_OUTPUT = "out_train/optuna/"
PDB_PATH = "data/molmaps"
VOL_PATH = "data/validation/"
VALID_PATH = "data/validation/"
STUDY_NAME = "MYSTUDY"
PARAMS = None


class UnknownTargetError(Exception):
    pass


def generate_settings3(trial, optuna_conf: dict) -> Dict:
    def resolve_suggest(child_dict, parent_key=None, parent=None):
        for k, v in child_dict.items():
            if isinstance(v, dict):
                if k in ["suggest_int", "suggest_float", "suggest_categorical"]:
                    parent[parent_key] = getattr(trial, k)(parent_key, **v)
                else:
                    resolve_suggest(v, k, child_dict)

    optuna_conf_copy = copy.deepcopy(optuna_conf)
    resolve_suggest(optuna_conf_copy)
    return optuna_conf_copy

def generate_settings2(trial: optuna.Trial, optuna_conf: Dict) -> Dict:
    val = {}

    for suggest_func in optuna_conf:
        if suggest_func == "optuna":
            continue
        if suggest_func.startswith("c_"):
            for c_param in optuna_conf[suggest_func]:
                conditional_param = optuna_conf[suggest_func][c_param]
                c_var = conditional_param["condition_variable"]
                c_val = conditional_param["condition_value"]
                if c_var in val:
                    if val[c_var] == c_val:
                        val[c_param] = getattr(trial, suggest_func[2:])(
                            c_param, **conditional_param["params"]
                        )
        elif suggest_func == "constant":
            for param in optuna_conf[suggest_func]:
                sparam = optuna_conf[suggest_func][param]
                val[param] = sparam["value"]
        else:
            for param in optuna_conf[suggest_func]:
                sparam = optuna_conf[suggest_func][param]
                val[param] = getattr(trial, suggest_func)(param, **sparam)
    return val


def objective(trial: optuna.Trial) -> float:
    output_path = os.path.join(BASIC_OUTPUT, str(trial.number))
    print("TRIAL NUMBER:", trial.number)
    ########################
    # Generate Optuna parameters
    ########################
    settings = generate_settings3(trial, PARAMS)
    learning_rate = settings["train_config"]["learning_rate"]
    optimizer_name = settings["train_config"]["optimizer"]
    amsgrad = settings["train_config"].get("amsgrad", False)
    weight_decay = settings["train_config"]["weight_decay"]
    EPOCHS = settings["train_config"]["epochs"]
    BATCH_SIZE = settings["train_config"]["batchsize"]
    miner_conf = settings["train_config"].get("miner", None)
    patience = settings["train_config"]["patience"]

    print("NUMBER OF EPOCHS", EPOCHS)

    ########################
    # Init distance function
    ########################
    dm = DistanceManager()
    distance = dm.get_distance(DISTANCE)
    print("Use distance function", distance.name())



    ########################
    # Setup network
    ########################
    nw = NetworkManager()
    network = nw.create_network(settings)

    ############################
    # Check if restart is necessary
    ############################

    retry_trial_number = RetryFailedTrialCallback.retried_trial_number(trial)
    retry_output_path = os.path.join(BASIC_OUTPUT, str(retry_trial_number))
    retry_checkpoint_pth = os.path.join(retry_output_path, "latest.pth")
    retry_checkpoint_exists = os.path.isfile(retry_checkpoint_pth)
    checkpoint = None
    optuna_values = {}
    optuna_values["best"] = None
    optuna_values["best_loss"] = np.inf
    optuna_values["ret_value"] = None

    if retry_trial_number is not None and retry_checkpoint_exists:
        print("Restart trial", retry_trial_number)
        checkpoint = retry_checkpoint_pth
        output_path = retry_output_path

        if os.path.exists(os.path.join(output_path,"optuna_values.json")):
            with open(os.path.join(output_path,"optuna_values.json")) as json_file:
                optuna_values = json.load(json_file)
            print("Read the following optuna values:")
            print(optuna_values)

    os.makedirs(output_path, exist_ok=True)
    pth_log_out = os.path.join(output_path, "out.txt")
    pth_log_err = os.path.join(output_path, "err.txt")
    print("Redirecting stdout to", pth_log_out)
    print("Redirecting stderr to", pth_log_err)
    f = open(pth_log_out, "a")
    sys.stdout = f
    f = open(pth_log_err, "a")
    sys.stderr = f

    ########################
    # Setup datasets
    ########################
    print("RETRY NUMBER", retry_trial_number)
    print("RETRY PATH", retry_checkpoint_pth, " Exists?", retry_checkpoint_exists)

    tconf = TrainingConfiguration(
        pdb_path=PDB_PATH,
        volume_path=VOL_PATH,
        output_path=output_path,
        num_epochs=EPOCHS,
        max_neg=1,
        netconfig=None,
        checkpoint=checkpoint,
        validvolumes=VALID_PATH,
        distance=DISTANCE,
        save_after_improvement=False
    )
    train_triplets, test_triplets = generate_triplets(tconf)

    aug_dist = settings.get("aug_distance", 2)
    print("Use augmentation distance of", aug_dist)
    use_pdb_as_anchor = tconf.pdb_path is not None
    aug_anchor, aug_volumes = get_augmentations(aug_dist, use_pdb_as_anchor=use_pdb_as_anchor)

    train_ds = TripletDataset(
        training_data=train_triplets,
        handler=MRCTripletHandler(),
        augmentation_anchors=aug_anchor,
        augmentation_volumes=aug_volumes,
        label_ext_func=label_filename
    )

    test_ds = TripletDataset(
        training_data=test_triplets,
        handler=MRCTripletHandler(),
        label_ext_func=label_filename
    )

    ########################
    # Create trainer and start training
    ########################

    loss_func = tmain.get_loss_func(
        net_conf=settings["network_config"], train_conf=settings["train_config"], distance=distance
    )

    miner = tmain.get_miner(miner_conf)

    trainer = TorchTrainer(
        epochs=tconf.num_epochs,
        batchsize=int(BATCH_SIZE),
        learning_rate=learning_rate,
        network=network,
        criterion=LossPyML(
            loss_func=loss_func,
            miner=miner,
        ),
        workers=12,
        log_dir=os.path.join(tconf.output_path, "tensorboard"),
        training_data=train_ds,
        test_data=test_ds,
        output_path=tconf.output_path,
        checkpoint=tconf.checkpoint,
        optimizer=optimizer_name,
        amsgrad=amsgrad,
        weight_decay=weight_decay,
        patience=patience
    )
    trainer.set_network_config(settings["network_config"])

    train_loader, test_loader = trainer.get_train_test_dataloader()

    # Training Loop

    for epoch in tqdm(
        range(trainer.start_epoch, trainer.epochs),
        initial=trainer.start_epoch,
        total=trainer.epochs,
        desc=f"Epochs",
    ):
        print("EPOCH START:", trainer.start_epoch, epoch)
        trainer.current_epoch = epoch
        train_loss = trainer.epoch(train_loader=train_loader)
        print(f"Train Loss: {train_loss:.4f}.")

        # Validation
        if test_loader is not None:
            current_val_loss = trainer.validation_loss(test_loader)
            trainer.scheduler.step(current_val_loss)
            current_val_f1 = trainer.classification_f1_score(test_loader=test_loader)
            print(f"Validation Loss: {current_val_loss:.4f}.")
            print(f"Validation F1 Score: {current_val_f1:.4f}.")
            trainer.save_best_loss(current_val_loss, epoch)
            trainer.save_best_f1(current_val_f1, epoch)

        trainer.writer.flush()

        if trainer.output_path is not None:
            trainer.write_results_to_disk(trainer.output_path)

        if PARAMS["optuna"]["target"].upper() == "F1":
            trial.report(current_val_f1, epoch)
            optuna_values["ret_value"] = float(current_val_f1)
        elif PARAMS["optuna"]["target"].upper() == "VAL_LOSS":
            trial.report(current_val_loss, epoch)
            optuna_values["ret_value"] = float(current_val_loss)
        elif PARAMS["optuna"]["target"].upper() == "F1_AT_VAL_LOSS":
            if current_val_loss < optuna_values["best_loss"]:
                optuna_values["best_loss"] = float(current_val_loss)
                optuna_values["ret_value"] = float(current_val_f1)
            trial.report(optuna_values["ret_value"], epoch)
        else:
            raise UnknownTargetError(
                f"Target {PARAMS['optuna']['target']} is not known."
            )
        if optuna_values["best"] == None:
            optuna_values["best"] = optuna_values["ret_value"]
        else:
            if PARAMS["optuna"]["minmax"] == "maximize":
                if optuna_values["ret_value"] > optuna_values["best"]:
                    optuna_values["best"] = optuna_values["ret_value"]
            if PARAMS["optuna"]["minmax"] == "minimize":
                if optuna_values["ret_value"] < optuna_values["best"]:
                    optuna_values["best"] = optuna_values["ret_value"]

        with open(os.path.join(output_path,"optuna_values.json"), 'w') as fp:
            json.dump(optuna_values, fp)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return optuna_values["best"]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optuna training interface for TomoTwin",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-p",
        "--pdbpath",
        type=str,
        default=None,
        help="Path to PDB files that should be use for training",
    )
    parser.add_argument(
        "-v",
        "--volpath",
        type=str,
        required=True,
        help="Path to subtomogram volumes that should be used for training",
    )

    parser.add_argument(
        "--validvolumes",
        type=str,
        default=None,
        required=True,
        help="Path for validation volumes.",
    )

    parser.add_argument(
        "-o",
        "--outpath",
        type=str,
        required=True,
        help="All output files are written in that path.",
    )

    parser.add_argument(
        "-sn",
        "--name",
        type=str,
        required=True,
        help="name of the study.",
    )

    parser.add_argument(
        "-c",
        "--optuna_config",
        type=str,
        required=True,
        help="Path to optuna configuration",
    )

    parser.add_argument(
        "-n",
        "--ntrials",
        type=int,
        default=100,
        help="Number of trials",
    )

    return parser


def _main_():
    """
    Get arguments
    """
    global PDB_PATH
    global VOL_PATH
    global VALID_PATH
    global BASIC_OUTPUT
    global EPOCHS
    global PARAMS
    print("TomoTwin Version:", version("tomotwin-cryoet"))
    parser = create_parser()
    args = parser.parse_args()
    PDB_PATH = args.pdbpath
    VOL_PATH = args.volpath
    VALID_PATH = args.validvolumes
    STUDY_NAME = args.name
    ntrials = args.ntrials
    optuna_config = args.optuna_config
    if ntrials == -1:
        ntrials = None
    BASIC_OUTPUT = os.path.join(args.outpath, STUDY_NAME)

    """
    READ CONFIGURATION
    """

    f = open(optuna_config)
    PARAMS = json.load(f)
    f.close()

    """
    Setup study
    """
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    print(PARAMS["optuna"])

    heartbeat_intervall = None

    storage_url = PARAMS["optuna"]["STORAGE"]
    # f"sqlite:///{STUDY_NAME}.db"
    storage = None
    if PARAMS["optuna"]["RDB"] is True:

        retry_callback = None
        if PARAMS["optuna"]["retry"] is True:
            retry_callback = RetryFailedTrialCallback(
                max_retry=PARAMS["optuna"]["max_retry"]
            )

        pruner = None
        if PARAMS["optuna"]["prune"] is True:
            pruner = MedianPruner(n_warmup_steps=PARAMS["optuna"]["n_warmup"])

        if "heartbeat" in PARAMS["optuna"]:
            heartbeat_intervall = PARAMS["optuna"]["heartbeat"]
            print("Use heartbeat interval ", heartbeat_intervall)

        storage = optuna.storages.RDBStorage(
            url=storage_url,
            engine_kwargs={"pool_pre_ping": True},
            heartbeat_interval=heartbeat_intervall,
            failed_trial_callback=retry_callback,
        )
    else:
        storage = storage_url

    study = optuna.create_study(
        direction=PARAMS["optuna"]["minmax"],
        sampler=samplers.RandomSampler(),
        pruner=pruner,
        study_name=STUDY_NAME,
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=ntrials, gc_after_trial=True, n_jobs=1)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    open(os.path.join(BASIC_OUTPUT, "DONE.txt"), "a").close()


if __name__ == "__main__":
    _main_()
