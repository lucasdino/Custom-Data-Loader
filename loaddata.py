# Code that we'll import into our workflows to create a dataloader that can select data based on which datasets we specify we want to use.
# Will call the get_data_loader function to return our desired dataloader object.

# First, import dependencies
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import tkinter as tk
from tkinter import filedialog
import os

# Define the path to your datasets
DATASETS_PATH = "datasets"

def select_datasets(datasets_path):
    """
        Function that prompts user to select datasets from our datasets directory.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # This will store your selection outside the inner function
    selected_datasets = []

    # Browse the dataset directory to list all subdirectories
    dataset_dirs = next(os.walk(datasets_path))[1]
    
    # Create a new window for selection
    selection_window = tk.Toplevel(root)
    selection_window.title("Select Datasets")

    listbox = tk.Listbox(selection_window, selectmode='multiple', width=50, height=15)
    for dataset_dir in dataset_dirs:
        listbox.insert(tk.END, dataset_dir)
    listbox.pack()

    def confirm_selection():
        nonlocal selected_datasets  # This line is changed to reference the outer scope variable
        selections = listbox.curselection()
        selected_datasets = [dataset_dirs[i] for i in selections]
        selection_window.destroy()
        root.quit()

    confirm_button = tk.Button(selection_window, text="Confirm", command=confirm_selection)
    confirm_button.pack()

    root.mainloop()
    try:
        root.destroy()  # Ensure the root tkinter window is closed
    except:
        pass  # Window is already closed

    return selected_datasets  # This will now return the correct value

class CustomDataset(Dataset):
    def __init__(self, dataset_names, datasets_path):
        self.file_paths = {}  # Dictionary to store file paths keyed by an integer
        self.indices = []  # List of tuples (file_key, row_index)
        file_key = 0  # Initialize file key
        
        for dataset_name in dataset_names:
            data_path = os.path.join(datasets_path, dataset_name, "data")
            for file_name in os.listdir(data_path):
                if file_name.endswith('.parquet'):
                    file_path = os.path.join(data_path, file_name)
                    self.file_paths[file_key] = file_path  # Store file path in dictionary
                    
                    num_rows = self.get_number_of_rows(file_path)
                    for row_index in range(num_rows):
                        self.indices.append((file_key, row_index))  # Use file_key instead of file_path
                        
                    file_key += 1  # Increment file_key for the next file

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        file_key, row_index = self.indices[idx]
        file_path = self.file_paths[file_key]  # Lookup file path using file_key
        return self.load_row(file_path, row_index)

    @staticmethod
    def get_number_of_rows(file_path):
        parquet_file = pq.ParquetFile(file_path)
        return parquet_file.metadata.num_rows

    def load_row(self, file_path, row_index):
        # Load the specified columns of the Parquet file into a Pandas DataFrame
        df = pd.read_parquet(file_path, columns=['text'])
        # Select the specific row's 'text' column value
        # Convert the Series object to a list or a string
        row_data = df.iloc[row_index]['text']
        if isinstance(row_data, pd.Series):
            # Convert Series to list if multiple rows were somehow selected
            return row_data.tolist()
        else:
            # If it's a single value, you can return it directly, or as a single-element list
            # Depending on whether you expect to handle batching manually or not
            return [row_data]  # or just `return row_data` for direct string handling
        
def get_dataloader(batch_size, shuffle=True, num_workers=4):
    """
        Function that prompts for selection of which datasets to include, then creates a Pytorch Dataset and Dataloader. Returns the Dataloader.

        Inputs:
            batch_size:  (int) Specify batch_size param for the dataloader (how many examples are returned with each iteration of the dataloader)
            shuffle:     (boolean) Specifies whether the dataloader shuffles the data off start or not
            num_workers: (int) Sets number of subprocesses to create for loading. Set to 0 to only run one process in 'main'

        Returns dataloader
    """
    selected_datasets = select_datasets(DATASETS_PATH)
    dataset = CustomDataset(selected_datasets, DATASETS_PATH)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)