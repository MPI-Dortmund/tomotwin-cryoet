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
import os
import sys
import tempfile
from argparse import ArgumentParser
from glob import glob
from types import SimpleNamespace

import mrcfile
import numpy as np
import pandas as pd
from scipy import ndimage as ndimg
from skimage import morphology

from tomotwin import embed_main as embed
from tomotwin import locate_main as locate
from tomotwin import map_main as tmap
from tomotwin.modules.inference.embed_ui import EmbedConfiguration, EmbedMode, DistrMode
from tomotwin.modules.inference.findmaxima_locator import FindMaximaLocator
from tomotwin.modules.inference.map_ui import MapMode, MapConfiguration
from tomotwin.modules.tools import median_embedding as median_tool
from tomotwin.modules.tools.tomotwintool import TomoTwinTool


class EmbeddingMaskTool(TomoTwinTool):
    """
    Tools to create ROI of embedding
    """

    def get_command_name(self) -> str:
        '''
        :return: Name of the command
        '''
        return "embedding_mask"

    def create_parser(self, parentparser: ArgumentParser) -> ArgumentParser:
        """
        :param parentparser: ArgumentPaser where the subparser for this tool needs to be added.
        :return: Argument parser that was added to the parentparser


        """

        parser = parentparser.add_parser(
            self.get_command_name(),
            help="(EXPERIMENTAL) Generates an ROI mask to speed up embeddings",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        subparsers = parser.add_subparsers(help="Subcommand help")

        parse_intensity = subparsers.add_parser("intensity",
                                                help="Estimates potential ROIs purely based on intensity values.",
                                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parse_median = subparsers.add_parser("median",
                                             help="Estimates the ROI based on the median embedding.",
                                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # setup general
        parse_intensity.add_argument(
            "-i",
            "--input",
            type=str,
            required=True,
            help="Tomogram file that needs to be embedded",
        )

        parse_intensity.add_argument(
            "-o", "--output", type=str, required=True, help="Output folder"
        )

        # setup median

        parse_median.add_argument(
            "-i",
            "--input",
            type=str,
            required=True,
            help="Tomogram file that needs to be embedded",
        )

        parse_median.add_argument(
            "-m",
            "--modelpth",
            type=str,
            required=True,
            help="Path to the tomotwin model",
        )

        parse_median.add_argument(
            "-o", "--output", type=str, required=True, help="Output folder"
        )

        parse_median.add_argument(
            "-s",
            "--stride",
            type=int,
            default=5,
            help="Stride of the sliding window. ",
        )

        parse_median.add_argument(
            "-b",
            "--batchsize",
            type=int,
            default=64,
            help="Batch size during calculating the embeddings",
        )

        parse_median.add_argument(
            "-t",
            "--threshold",
            type=float,
            default=0.3,
            help="Threshold between 0 - 1. As higher the threshold as more conservative mask is.",
        )

        parse_median.add_argument(
            "-d",
            "--dilation",
            type=int,
            default=1,
            help="Dilation radius. Add an additional",
        )

        parse_median.add_argument(
            "-p"
            "--padding",
            dest="padding",
            action="store_true",
            default=True,
            help="Add padding of half box size to the tomogram so that it is all included in the mask",
        )

        parse_median.add_argument(
        "--no-padding",
        dest="padding",
        action="store_false",
        help="Disable padding"
        )

        return parser

    def median_mode(self,
                    tomo_pth: str,
                    model_pth: str,
                    stride: int,
                    batch_size: int,
                    threshold: float,
                    dilation: float,
                    padding: bool 
                    ) -> np.array:
        '''
        Calculates a mask based on median embedding
        '''
        with tempfile.TemporaryDirectory() as tmp_pth:
            # Embed
            emb_out_pth = os.path.join(tmp_pth, "embed")

            print ('median_moade.padding = ', padding)
            conf = EmbedConfiguration(
                model_path=model_pth,
                volumes_path=tomo_pth,
                output_path=emb_out_pth,
                mode=EmbedMode.TOMO,
                batchsize=batch_size,
                stride=stride,
                zrange=None,
                maskpth=None,
                distr_mode=DistrMode.DDP,
                padding = padding
            )

            embed.start(conf)

            # Median embedding
            median_out_pth = os.path.join(tmp_pth, "median_emb")
            args = SimpleNamespace(input=glob(os.path.join(emb_out_pth, "*.temb"))[0], output=median_out_pth)
            mtool = median_tool.MedianTool()
            mtool.run(args)

            # Map
            map_out_pth = os.path.join(tmp_pth, "map")
            map_conf = MapConfiguration(
                reference_embeddings_path=glob(os.path.join(median_out_pth, "*.temb"))[0],
                volume_embeddings_path=glob(os.path.join(emb_out_pth, "*.temb"))[0],
                output_path=map_out_pth,
                mode=MapMode.DISTANCE,
                skip_refinement=True
            )
            tmap.run(map_conf)

            # Heatmap
            print("Calculate heatmap")
            map_output = pd.read_pickle(glob(os.path.join(map_out_pth, "*.tmap"))[0])
            raw_heatmap = FindMaximaLocator.to_volume(
                df=map_output,
                target_class=0,
                stride=(stride, stride, stride),
                window_size=map_output.attrs['window_size'],
            )
            raw_heatmap = raw_heatmap.astype(np.float32)
            heatmap = locate.scale_and_pad_heatmap(raw_heatmap,
                                                   stride=stride,
                                                   tomo_input_shape=map_output.attrs['tomogram_input_shape'])
            print("Binarize heatmap")
            # Binarize heatmap
            mask = np.logical_and(heatmap < threshold, heatmap > np.min(heatmap))

            if dilation != 0:
                mask = morphology.binary_dilation(
                    mask, morphology.ball(radius=dilation)
                )

            bin_mask = np.zeros_like(mask, dtype=np.float32)
            print(bin_mask.shape)
            bin_mask[mask] = 1

            return bin_mask

    def intensity_mode(self, img: np.array) -> np.array:
        '''
        Calculates a mask based on intensity heuristics.
        '''
        print("Background subtraction")
        filtered = ndimg.gaussian_filter(img, (10, 10, 10))
        background_removed = img - filtered

        print("Threshold estimation")
        blurred = ndimg.gaussian_filter(background_removed, (2, 2, 2))
        min_img = ndimg.minimum_filter(blurred, 5 * 2)

        hist = np.histogram(min_img, bins=256)
        b = np.argmax(hist[0])
        t = hist[1][b]
        print(f"Found  threshold: {t:2f}")
        mask = min_img < t
        return mask

    def run(self, args):
        """
        Runs the tools
        """


        print("Calculate mask")
        if sys.argv[2] == "median":
            print('args.padding = ', args.padding)
            mask = self.median_mode(tomo_pth=args.input,
                                    model_pth=args.modelpth,
                                    stride=args.stride,
                                    dilation=args.dilation,
                                    threshold=args.threshold,
                                    batch_size=args.batchsize,
                                    padding = args.padding)
            print(f"Masked out: {100 - np.sum(mask) * 100 / np.prod(mask.shape):.2f}%")
        elif sys.argv[2] == "intensity":
            print("Read data")
            with mrcfile.open(args.input) as mrc:
                img = mrc.data

            mask = self.intensity_mode(img)
            print(f"Masked out: {100 - np.sum(mask) * 100 / np.prod(mask.shape):.2f}%")
        print("Write results to disk")
        os.makedirs(args.output, exist_ok=True)
        with mrcfile.new(
                os.path.join(
                    args.output,
                    os.path.splitext(os.path.basename(args.input))[0] + "_mask.mrc",
                ),
                overwrite=True,
        ) as mrc:
            mrc.set_data(mask.astype(np.float32))
        print("Done")
