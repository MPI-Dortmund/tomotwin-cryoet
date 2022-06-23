
from abc import ABC, abstractmethod
from argparse import ArgumentParser

class TomoTwinTool(ABC):

    @abstractmethod
    def get_command_name(self) -> str:
        '''
        :return: Command name
        '''
        pass

    @abstractmethod
    def create_parser(self, parentparser : ArgumentParser) -> ArgumentParser:
        '''
        :param parentparser: ArgumentPaser where the subparser for this tool needs to be added.
        :return: Argument parser that was added to the parentparser
        '''
        pass

    @abstractmethod
    def run(self, args):
        pass