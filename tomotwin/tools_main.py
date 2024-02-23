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
import sys
from typing import List

from tomotwin.modules.tools.embedding_mask import EmbeddingMaskTool
from tomotwin.modules.tools.extract_reference import ExtractReference
from tomotwin.modules.tools.filter_embedding import FilterTool
from tomotwin.modules.tools.info import Info
from tomotwin.modules.tools.make_target import MakeTargetEmbeddings
from tomotwin.modules.tools.median_embedding import MedianTool
from tomotwin.modules.tools.scale_coordinates import ScaleCoordinates
from tomotwin.modules.tools.tomotwintool import TomoTwinTool
from tomotwin.modules.tools.umap import UmapTool


def get_tool_list() -> List[TomoTwinTool]:
    '''
    Create the list of supported tools
    :return: List of tools
    '''

    tools = []

    tools.append(ScaleCoordinates())
    tools.append(ExtractReference())
    tools.append(UmapTool())
    tools.append(Info())
    tools.append(MedianTool())
    tools.append(FilterTool())
    tools.append(EmbeddingMaskTool())
    tools.append(MakeTargetEmbeddings())

    return tools

def _main_():

    parser = argparse.ArgumentParser(
        description="TomoTwin Tools",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(help="sub-command help")

    tools = get_tool_list()

    tools = sorted(tools, key=lambda x: x.get_command_name())

    for tool in tools:
        tool.create_parser(subparsers)

    args = parser.parse_args()

    for tool in tools:
        if tool.get_command_name() in sys.argv[1]:
            tool.run(args)

if __name__ == "__main__":
    _main_()
