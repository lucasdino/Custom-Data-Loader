# Custom Data Loader
---  
Because I'll be reusing several datasets in workflows, I've created this code that will help me load in my desired data in multiple projects.

The `datasets` directory has my various datasets that I can use. The `loaddata.py` code will be what I call to return a Pytorch `dataloader` object that I can use in my workflows. This code has a function called `get_dataloader` that prompts the user to specify which dataset to use, then returns a dataloader that efficiently stores and indexes Parquet files that can then be used to train a dataset.