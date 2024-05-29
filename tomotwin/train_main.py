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

import os
import random
import sys
from typing import List, Tuple, Dict, Callable

import numpy as np
import tqdm
from pytorch_metric_learning import miners, losses

from tomotwin.modules.common import distances, exceptions
from tomotwin.modules.common.distances import DistanceManager
from tomotwin.modules.common.preprocess import label_filename
from tomotwin.modules.common.utils import check_for_updates
from tomotwin.modules.networks.networkmanager import NetworkManager
from tomotwin.modules.training.LossPyML import LossPyML
from tomotwin.modules.training.argparse_ui import (
    TrainingArgParseUI,
    TrainingConfiguration,
)
from tomotwin.modules.training.filenamematchingtripletprovidernopdb import (
    FilenameMatchingTripletProviderNoPDB,
)
from tomotwin.modules.training.filenametripletprovider import (
    FilenameMatchingTripletProvider,
)
from tomotwin.modules.training.filepathtriplet import FilePathTriplet
from tomotwin.modules.training.mrctriplethandler import MRCTripletHandler
from tomotwin.modules.training.torchtrainer import TorchTrainer, TripletDataset
from tomotwin.modules.training.transforms import (
    AugmentationPipeline,
    AddNoise,
    Rotate,
    Shift,
    RotateFull,
    VoxelDropout,
)

try:
    from importlib_metadata import version
except ModuleNotFoundError:
    from importlib.metadata import version


def get_augmentations(
    aug_train_shift_distance: int = 2, use_pdb_as_anchor: bool = True
) -> Tuple[AugmentationPipeline, AugmentationPipeline]:
    """
    :param aug_train_shift_distance: Augmentation shifts
    :param use_pdb_as_anchor: If true, only pdbs are used as anchors.
    :return: Training and validation pipeline
    """
    aug_anchor = None
    if use_pdb_as_anchor:
        aug_anchor = AugmentationPipeline(
            # [(CenterCrop, 0.3), (AddNoise, 0.3), (Blur, 0.3), (Rotate, 0.5), (Shift, 0.3)]
            augs=[
                Rotate(axes=(0, 1)),
                VoxelDropout(ratio=(0.05, 0.2)),
                RotateFull(axes=(1, 2)),  # x-y rotations
            ],
            probs=[
                0.3,
                0.3,
                0.3,
            ],
        )
    aug_volumes = AugmentationPipeline(
        augs=[
            VoxelDropout(ratio=(0.05, 0.2)),
            RotateFull(axes=(1, 2)),
            Shift(
                axis=0,
                min_shift=-aug_train_shift_distance,
                max_shift=aug_train_shift_distance,
            ),
            Shift(
                axis=1,
                min_shift=-aug_train_shift_distance,
                max_shift=aug_train_shift_distance,
            ),
            Shift(
                axis=2,
                min_shift=-aug_train_shift_distance,
                max_shift=aug_train_shift_distance,
            ),
            AddNoise(sigma=(0, 0.3)),
        ],
        probs=[0.3, 0.3, 0.3, 0.3, 0.3, 0.9],
    )
    if aug_anchor is None:
        aug_anchor = aug_volumes
    return aug_anchor, aug_volumes


def train_test_split_anchor_positive(
    data: List[FilePathTriplet], split: float = 0.8
) -> Tuple[List, List]:
    """
    Create a train/test split of the data. It will make sure, that the anchor-positives pairs are
    unique in train and test split.
    """
    unique_pdbs = set()
    for fpt in data:
        unique_pdbs.add(fpt.anchor)
    unique_pdbs = list(unique_pdbs)
    unique_pdbs.sort()

    unique_volumes = set()
    for fpt in data:
        unique_volumes.add(fpt.positive)
    unique_volumes = list(unique_volumes)
    unique_volumes.sort()

    anchor_positives_pairs = []
    for pdb in unique_pdbs:
        for vol in unique_volumes:
            if os.path.splitext(os.path.basename(pdb))[0].upper() in vol.upper():
                anchor_positives_pairs.append((pdb, vol))
    np.random.shuffle(anchor_positives_pairs)
    split = int(split * len(anchor_positives_pairs))
    train_anchor_positive = anchor_positives_pairs[:split]
    valid_anchor_positive = anchor_positives_pairs[split:]

    train_triplets = []
    valid_triplets = []

    def is_in_subset(triplet, pairs: List):
        for p in pairs:
            if triplet.anchor == p[0] and triplet.positive == p[1]:
                return True
        return False

    for triplet in tqdm.tqdm(data, "Train/Valid split"):
        if is_in_subset(triplet, train_anchor_positive):
            train_triplets.append(triplet)
            continue

        if is_in_subset(triplet, valid_anchor_positive):
            valid_triplets.append(triplet)
    return train_triplets, valid_triplets


def generate_triplets(
    tconf: TrainingConfiguration,
) -> Tuple[List[FilePathTriplet], List[FilePathTriplet]]:
    """
    Generates the train/test triplets
    :param tconf: Training configuration
    :return: Two lists of FilePathTriplets. One for training, one for testing.
    """
    if tconf.pdb_path:
        tripletprov = FilenameMatchingTripletProvider(
            path_pdb=tconf.pdb_path,
            path_volume=tconf.volume_path,
            max_neg=tconf.max_neg,
            mask_pdb="**/*.mrc",
            mask_volumes="**/*.mrc",
        )
    else:
        tripletprov = FilenameMatchingTripletProviderNoPDB(
            path_volume=tconf.volume_path, mask_volumes="**/*.mrc"
        )

    master_list = tripletprov.get_triplets()
    if tconf.validvolumes is None:
        if len(master_list) == 0:
            print("No training data could be found")
            sys.exit(0)
        # Get triplets and make dataframes for sampling during training
        train_triplets, test_triplets = train_test_split_anchor_positive(
            master_list, split=0.8
        )
    else:
        print("Use validvolumes path for validation data.")
        train_triplets = master_list
        if tconf.pdb_path:
            tripletprov = FilenameMatchingTripletProvider(
                path_pdb=tconf.pdb_path,
                path_volume=tconf.validvolumes,
                max_neg=tconf.max_neg,
                mask_pdb="**/*.mrc",
                mask_volumes="**/*.mrc",
            )
        else:
            tripletprov = FilenameMatchingTripletProviderNoPDB(
                path_volume=tconf.validvolumes, mask_volumes="**/*.mrc"
            )
        test_triplets = tripletprov.get_triplets()

    return train_triplets, test_triplets


def get_loss_func(
    net_conf: Dict, train_conf: Dict, distance: distances.Distance
) -> Callable:
    """
    :param net_conf: Network confiiguration dict
    :param train_conf: Training configuration dict
    :param distance: Distance measure
    :return: Loss function
    """
    if train_conf["loss"] == "ArcFaceLoss":
        loss_func = losses.ArcFaceLoss(
            num_classes=train_conf["num_classes"],
            margin=train_conf["af_margin"],
            distance=distance,
            scale=train_conf["af_scale"],
            embedding_size=net_conf["output_channels"],
        )
    elif train_conf["loss"] == "SphereFaceLoss":
        loss_func = losses.SphereFaceLoss(
            num_classes=train_conf["num_classes"],
            margin=train_conf["sf_margin"],
            distance=distance,
            scale=train_conf["sf_scale"],
            embedding_size=net_conf["output_channels"],
        )

    elif train_conf["loss"] == "TripletLoss":
        loss_func = losses.TripletMarginLoss(
            margin=train_conf["tl_margin"], distance=distance
        )
    elif train_conf["loss"] == "MultiSimilarityLoss":
        loss_func = losses.MultiSimilarityLoss(
            distance=distance
        )
    else:
        raise exceptions.UnknownLoss("Specified loss not known")

    return loss_func


def get_miner(miner_conf: dict):
    miner = None
    if miner_conf is None:
        return miner

    if miner_conf['name'] == "TripletMarginMiner":
        miner = miners.TripletMarginMiner(
            margin=miner_conf["miner_margin"], type_of_triplets="semihard"
        )
    elif miner_conf['name'] == "DistanceWeightedMiner":
        miner = miners.DistanceWeightedMiner(
            cutoff=miner_conf["cutoff"],
            nonzero_loss_cutoff=miner_conf["nonzero_loss_cutoff"],
        )
    return miner



def _main_():
    seed = 17  # seed value
    np.random.seed(seed)
    random.seed(seed)

    ########################
    # Get configuration from user interface
    ########################

    ui = TrainingArgParseUI()

    ui.run()

    check_for_updates()

    tconf = ui.get_training_configuration()

    os.makedirs(tconf.output_path, exist_ok=True)

    pth_log_out = os.path.join(tconf.output_path, "out.txt")
    pth_log_err = os.path.join(tconf.output_path, "err.txt")
    print("Redirecting stdout to", pth_log_out)
    print("Redirecting stderr to", pth_log_err)
    f = open(pth_log_out, "a", encoding="utf-8")
    sys.stdout = f
    f = open(pth_log_err, "a", encoding="utf-8")
    sys.stderr = f
    print("TomoTwin Version:", version("tomotwin-cryoet"))


    ########################
    # Generate Triplets
    ########################
    train_triplets, test_triplets = generate_triplets(tconf)

    print("Number of train triplets:", len(train_triplets))
    print("Number of validation triplets:", len(test_triplets))

    ########################
    # Load config
    ########################
    nw = NetworkManager()
    config = nw.load_configuration(tconf.netconfig)

    ########################
    # Configure datasets
    ########################

    aug_args = {"use_pdb_as_anchor": tconf.pdb_path is not None}
    if "aug_train_shift_distance" in config["train_config"]:
        print("Use shift:", config["train_config"]["aug_train_shift_distance"])
        aug_args["aug_train_shift_distance"] = config["train_config"][
            "aug_train_shift_distance"
        ]

    aug_anchor, aug_volumes = get_augmentations(**aug_args)

    train_ds = TripletDataset(
        training_data=train_triplets,
        handler=MRCTripletHandler(),
        augmentation_anchors=aug_anchor,
        augmentation_volumes=aug_volumes,
        label_ext_func=label_filename,
    )

    test_ds = TripletDataset(
        training_data=test_triplets,
        handler=MRCTripletHandler(),
        label_ext_func=label_filename,
    )

    ########################
    # Init distance function
    ########################
    dm = DistanceManager()
    distance = dm.get_distance(tconf.distance)
    print("Use distance function", distance.name())

    ########################
    # Setup network
    ########################
    config["distance"] = distance.name()
    network = nw.create_network(config)

    ########################
    # Setup miners and loss
    ########################
    miner = get_miner(config["train_config"].get("miner", None))

    loss_func = get_loss_func(
        net_conf=config["network_config"],
        train_conf=config["train_config"],
        distance=distance,
    )

    ########################
    # Create trainer and start training
    ########################
    only_negative_labels = []
    if "only_negative_labels" in config["train_config"]:
        only_negative_labels = config["train_config"]["only_negative_labels"]
    trainer = TorchTrainer(
        epochs=tconf.num_epochs,
        batchsize=int(config["train_config"]["batchsize"]),
        learning_rate=config["train_config"]["learning_rate"],
        network=network,
        criterion=LossPyML(
            loss_func=loss_func, miner=miner, only_negative_labels=only_negative_labels
        ),
        workers=12,
        log_dir=os.path.join(tconf.output_path, "tensorboard"),
        training_data=train_ds,
        test_data=test_ds,
        output_path=tconf.output_path,
        checkpoint=tconf.checkpoint,
        optimizer=config["train_config"]["optimizer"],
        weight_decay=config["train_config"]["weight_decay"],
        patience=config["train_config"]["patience"],
        save_epoch_seperately=tconf.save_after_improvement,
    )
    trainer.set_seed(seed)
    config["window_size"] = tuple(train_ds.get_triplet_dimension())
    trainer.set_network_config(config)

    trainer.train()
    trainer.write_results_to_disk(tconf.output_path)


if __name__ == "__main__":
    _main_()
