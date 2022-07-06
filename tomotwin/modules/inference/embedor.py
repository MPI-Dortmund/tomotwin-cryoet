"""
MIT License

Copyright (c) 2021 MPI-Dortmund

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from abc import ABC, abstractmethod
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn

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
        ...


class TorchVolumeDataset(Dataset):
    """Implementation for the volume dataset"""

    def __init__(self, volumes: VolumeDataset):
        self.volumes = volumes

    def __getitem__(self, item_index):

        vol = self.volumes[item_index]
        vol = vol.astype(np.float32)
        vol = pp.norm(vol)
        vol = vol[np.newaxis]
        torch_vol = torch.from_numpy(vol)
        input_triplet = {"volume": torch_vol}

        return input_triplet

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
        print("reading", self.weightspth)
        network_manager = NetworkManager()
        checkpoint = torch.load(self.weightspth)
        self.tomotwin_config = checkpoint["tomotwin_config"]
        print("Model config:")
        print(self.tomotwin_config)
        self.model = network_manager.create_network(self.tomotwin_config).get_model()
        before_parallel_failed=False
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError:
            print("Load before failed")
            before_parallel_failed=True

        self.model = torch.nn.DataParallel(self.model)
        if before_parallel_failed:
            self.model.load_state_dict(checkpoint["model_state_dict"])

    def embed(self, volume_data: VolumeDataset) -> np.array:
        """Calculates the embeddings. The volumes showed have the dimension NxBSxBSxBS where N is the number of nu"""

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = TorchVolumeDataset(volumes=volume_data)
        dataset.device = device
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
            for batch in volume_loader_tqdm:
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
