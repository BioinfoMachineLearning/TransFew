import os
from typing import Union, List, Tuple, Optional, Callable

from torch_geometric.data import Dataset, Data


class GraphDataset(Dataset):
    def __init__(self, root: Optional[str] = None, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None, **kwargs):

        self.proteins_names = kwargs.get('proteins_names', [])
        super().__init__(root, transform, pre_transform, pre_filter)

    @staticmethod
    def find_files_dir(path):
        return  os.listdir(path)

    @property
    def raw_dir(self) -> str:
        return "D:/Workspace/python-3/TransFun2/data/raw"

    @property
    def processed_dir(self) -> str:
        return "D:/Workspace/python-3/TransFun2/data/processed"

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return ["AF-{}-F1-model_v4.pdb".format(i) for i in self.proteins_names]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ["{}.pt".format(i) for i in self.proteins_names]

    def download(self):
        available = [file.split("-")[1] for file in self.find_files_dir(self.raw_dir)]
        unavailabe = set(self.proteins_names) - set(available)
        print("Downloading {} proteins".format(len(unavailabe)))

    def process(self):
        unprocessed = [file.split(".")[0] for file in self.find_files_dir(self.processed_dir)]
        unprocessed = set(self.proteins_names) - set(unprocessed)
        print("{} unprocessed proteins out of {}".format(len(unprocessed), len(self.raw_file_names)))

        for protein in unprocessed:
            print("Processing {}".format(protein))

    def len(self) -> int:
        return 0

    def get(self, idx: int) -> Data:
        pass


kwargs = {
    "proteins_names": ["bbdfnk", "A0JNW5", "A0JP26", "A2A2Y4", "A5D8V7", "A7MD48", "O14503"]
}
graph = GraphDataset(**kwargs)
print(graph)
