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
import copy
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.distributed as tdist
import torch.nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

import tomotwin.modules.common.preprocess as pp
from tomotwin.modules.inference.volumedata import VolumeDataset
from tomotwin.modules.networks.networkmanager import NetworkManager


class Embedor(ABC):
    """
    Abstract Embedor
    """

    @abstractmethod
    def embed(self, volume_data: VolumeDataset) -> np.array:
        """Given a set of volumes, this function calculates a set of embeddings"""

class TorchVolumeDataset(Dataset):
    """Implementation for the volume dataset"""

    def __init__(self, volumes: VolumeDataset):
        self.volumes = volumes

    def __getitem__(self, item_index):
        vol = self.volumes[item_index]
        vol = vol.astype(np.float32)
        vol = pp.norm2(vol)
        vol = vol[np.newaxis]
        if torch.cuda.is_available():
            vol = vol.astype(np.float16)  # Gives a speedup, when its done already here. However, does not work on CPUs.
        torch_vol = torch.from_numpy(vol)
        input_triplet = {"volume": torch_vol}

        return input_triplet, item_index

    def __len__(self):
        return len(self.volumes)



class WrongVolumeDimensionException(Exception):
    """Raised when input dimensions are wrong"""


class TorchEmbedor(Embedor):
    """
    Embedor for PyTorch
    """

    def __init__(
            self,
            weightspth: str,
            batchsize: int,
            workers: int = 0,
    ) -> None:
        """Inits the embedor"""
        self.batchsize = batchsize
        self.workers = workers
        self.weightspth = weightspth
        self.tomotwin_config = None
        print("reading", self.weightspth)
        self.model = None
        self.load_weights_()

    def load_weights_(self):
        checkpoint = None
        if self.weightspth is not None:
            checkpoint = torch.load(self.weightspth)
            self.tomotwin_config = checkpoint["tomotwin_config"]
            print("Model config:")
            print(self.tomotwin_config)
        print(self.tomotwin_config)
        self.model = NetworkManager.create_network(self.tomotwin_config).get_model()
        print(type(self.model))
        before_parallel_failed = False

        if checkpoint is not None:
            try:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError as e:
                print(e)
                print("Load before failed")
                before_parallel_failed = True

        self.model = torch.nn.DataParallel(self.model)
        if before_parallel_failed:
            self.model.load_state_dict(checkpoint["model_state_dict"])

    def embed(self, volume_data: VolumeDataset) -> np.array:
        """Calculates the embeddings. The volumes showed have the dimension NxBSxBSxBS where N is the number of nu"""

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = TorchVolumeDataset(volumes=volume_data)

        volume_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batchsize,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True
        )

        if device.type == "cuda":
            torch.cuda.get_device_name()

        model = self.model.to(device)
        model.eval()
        volume_loader_tqdm = tqdm(volume_loader, desc="Calculate embeddings", leave=False)
        embeddings = []
        from torch.cuda.amp import autocast
        with torch.no_grad():
            for batch, _ in volume_loader_tqdm:
                vol = batch["volume"]
                subvolume = vol.to(device)
                with autocast():
                    subvolume_out = model.forward(subvolume).data.cpu()
                embeddings.append(subvolume_out)

        embeddings_np = []
        for emb in embeddings:
            embeddings_np.append(emb.numpy())
        embeddings = np.concatenate(embeddings_np)

        return embeddings


class TorchEmbedorDistributed(Embedor):
    """
    Embedor for PyTorch
    """

    def __init__(
            self,
            weightspth: str,
            batchsize: int,
            rank: int,
            world_size: int,
            workers: int = 0,
    ) -> None:
        """Inits the embedor"""
        tdist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
        tdist.barrier()
        self.rank = rank
        self.batchsize = batchsize
        self.workers = workers
        self.weightspth = weightspth
        self.tomotwin_config = None
        print("reading", self.weightspth)
        self.model = None
        self.load_weights_()


    def load_weights_(self):
        """
        Loads the model weights
        """
        checkpoint = None
        if self.weightspth is not None:
            checkpoint = torch.load(self.weightspth)
            self.tomotwin_config = checkpoint["tomotwin_config"]
            if self.rank == 0:
                print("Model config:")
                print(self.tomotwin_config)

        self.model = NetworkManager.create_network(self.tomotwin_config).get_model()
        if checkpoint is not None:
            try:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError:
                print("Load before failed")

        self.model.to(self.rank)

        self.model = torch.compile(self.model, mode="reduce-overhead")
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank])

    def get_unique_indicis(self, a):
        """
        got this idea from https://github.com/pytorch/pytorch/issues/36748

        Behaves the same as np.unique(x,return_index=True)
        """
        #
        unique, inverse = torch.unique(a, sorted=True, return_inverse=True)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
        return perm

    def embed(self, volume_data: VolumeDataset) -> np.array:
        """Calculates the embeddings. The volumes showed have the dimension NxBSxBSxBS"""

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        dataset = TorchVolumeDataset(volumes=volume_data)
        sampler_data = torch.utils.data.DistributedSampler(dataset, rank=self.rank, shuffle=False)
        volume_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batchsize,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=False,
            sampler=sampler_data,
            persistent_workers=False
        )

        self.model.eval()
        volume_loader_tqdm = tqdm(volume_loader, desc=f"Calculate embeddings ({self.rank})", leave=False)
        embeddings = []
        items_indicis = []

        with torch.no_grad():
            for batch, item_index in volume_loader_tqdm:
                subvolume = batch["volume"].to(self.rank)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    subvolume = self.model.forward(subvolume).type(torch.HalfTensor)
                subvolume = subvolume.data.cpu()
                items_indicis.append(copy.deepcopy(item_index.data.cpu()))
                embeddings.append(copy.deepcopy(subvolume.data.cpu()))
                del subvolume
        ## Sync items
        items_indicis = torch.cat(items_indicis)  # .to(self.rank)  # necessary because of nccl
        items_gather_list = None
        if self.rank == 0:
            items_gather_list = [torch.zeros_like(items_indicis) for _ in range(tdist.get_world_size())]
        tdist.barrier()
        tdist.gather(items_indicis,
                     gather_list=items_gather_list,
                     dst=0)
        tdist.barrier()

        if self.rank == 0:
            items_indicis = torch.cat(items_gather_list)
            unique_elements = self.get_unique_indicis(items_indicis)
            items_indicis = items_indicis[unique_elements]
        else:
            items_indicis = None
        ## Sync embeddings
        embeddings = torch.cat(embeddings)
        tdist.barrier()
        embeddings_gather_list = None
        if self.rank == 0:
            embeddings_gather_list = [torch.zeros_like(embeddings) for _ in range(tdist.get_world_size())]

        torch.cuda.empty_cache()
        tdist.gather(embeddings,
                     gather_list=embeddings_gather_list,
                     dst=0)

        if self.rank == 0:
            embeddings = torch.cat(embeddings_gather_list)
            embeddings = embeddings[unique_elements]
            embeddings = embeddings[torch.argsort(items_indicis)]  # sort embeddings after gathering
            embeddings = embeddings.data.cpu().numpy()
        else:
            return


        return embeddings
