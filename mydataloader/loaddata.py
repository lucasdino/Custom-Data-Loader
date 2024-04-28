import os
import glob
import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, dataset_names, datasets_path=os.getenv('DATASET_PATH')):
        self.file_paths = {}
        self.indices = []
        
        for dataset_name in dataset_names:
            data_path = os.path.join(datasets_path, dataset_name, "data", "*.parquet")
            for file_key, file_path in enumerate(glob.glob(data_path)):
                self.file_paths[file_key] = file_path
                
                num_rows = self.get_number_of_rows(file_path)
                for row_index in range(num_rows):
                    self.indices.append((file_key, row_index))


    def __len__(self):
        return len(self.indices)


    def __getitem__(self, idx):
        file_key, row_index = self.indices[idx]
        file_path = self.file_paths[file_key]
        return self.load_row(file_path, row_index)


    @staticmethod
    def get_number_of_rows(file_path):
        parquet_file = pq.ParquetFile(file_path)
        return parquet_file.metadata.num_rows


    def load_row(self, file_path, row_index):
        """
            Function that loads a specific row from a Parquet file and returns it as a list of strings.

            Inputs:
                file_path: (str) Path to the Parquet file
                row_index: (int) Index of the row to load from the Parquet file

            Returns:
                row_data: (list) List of strings containing the data from the specified row
        """
        df = pd.read_parquet(file_path, columns=['text'])
        row_data = df.iloc[row_index]['text']
        if isinstance(row_data, pd.Series):
            return row_data.tolist()
        else:
            return [row_data]



def get_dataloader(selected_datasets, datasets_path=DATASETS_PATH, batch_size=8, shuffle=True, num_workers=0):
    """
        Function that prompts for selection of which datasets to include, then creates a Pytorch Dataset and Dataloader. Returns the Dataloader.

        Inputs:
            batch_size:  (int) Specify batch_size param for the dataloader (how many examples are returned with each iteration of the dataloader)
            selected_datasets: (list) List of directories in the 'datasets' folder that will be included in the dataloader
            datasets_path: (str) Path to the datasets folder
            shuffle:     (boolean) Specifies whether the dataloader shuffles the data off start or not
            num_workers: (int) Sets number of subprocesses to create for loading. Set to 0 to only run one process in 'main'

        Returns dataloader
    """
    dataset = CustomDataset(selected_datasets, datasets_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)