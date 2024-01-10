# RLHF Data Generation

This repostory generates instruction tuning dataset from different datasets.  

```
src
    ├── afri_rlhf
    │   ├── __init__.py
    │   ├── data
    │   │   ├── __init__.py
    │   │   └── sources.py                # create Datasource class for your new dataset
    │   ├── prompt
    │   │   ├── __init__.py
    │   │   ├── templates.py               # Add template for your data 
    │   │   └── validation.py
    │   └── utils
    │       ├── __init__.py
    │       ├── language.py
    │       └── support.py
    └── create_rlhf_dataset.py
```

