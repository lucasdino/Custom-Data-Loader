# Custom Data Loader
---  
Because I'll be reusing several datasets in workflows, I've created this code that will help me load in my desired data in multiple projects.

### How to use:
Navigate to the folder that this file is in. Ensure that you are on the correct virtual environment that you'll be using. Then, call 'pip install .'

This will install the files so that I can easily call on this in other files.


### About The Datasets:
The `datasets` directory has my various datasets that I can use. The `loaddata.py` code will be what I call to return a Pytorch `dataloader` object that I can use in my workflows. This code has a function called `get_dataloader` that prompts the user to specify which dataset to use, then returns a dataloader that efficiently stores and indexes Parquet files that can then be used to train a dataset.