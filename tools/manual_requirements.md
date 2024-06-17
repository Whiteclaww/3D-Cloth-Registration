## Kaolin version 0.15.0

Check the [Kaolin documentation](https://kaolin.readthedocs.io/en/v0.15.0/notes/installation.html "Installation - Kaolin documentation") on how to install.

I already added the necessary libraries in conda_requirements and pip_requirements.

Install kaolin inside 3D-Cloth-Registration, your project should have the following architecture:

```
.
├── __init__.py
├── main.py
├── main_test.py
├── modules
│   └── [...]
├── README.md
├── run.py
├── test_dataset
│   └── [...]
├── tests
│   └── [...]
└── tools
    ├── conda_requirements_nvidia.yml
    ├── conda_requirements.yml
    ├── kaolin
    │   └── [...]
    ├── manual_requirements.md
    ├── pip_requirements.txt
    └── [...]
```