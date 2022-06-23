from typing import List
import sys
import argparse
from tomotwin.modules.tools.tomotwintool import TomoTwinTool

def get_tool_list() -> List[TomoTwinTool]:
    tools = []

    from tomotwin.modules.tools.extract_reference import ExtractReference
    extract_ref_tool = ExtractReference()
    tools.append(extract_ref_tool)

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
