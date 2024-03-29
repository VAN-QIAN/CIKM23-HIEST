# CIKM23-Rethinking Sensors Modeling: Hierarchical Information Enhanced Traffic Forecasting

# Note

This repo is for the code implementation of our submitted paper **Rethinking Sensors Modeling: Hierarchical Information Enhanced Traffic Forecasting**.

The readme file is updated as the libcity library is integrated now.

For a quick start guideline, please refer to the section [Quick Start](https://github.com/VAN-QIAN/CIKM23-HIEST#quick-start)

For the codes and details corresponding to our core contribution, please refer to the section [Anonymous Github version](https://github.com/VAN-QIAN/CIKM23-HIEST#anonymous-github-version)

If you find this repo useful, please cite it as follows,

```latex
@inproceedings{HIEST23,
  title={Rethinking Sensors Modeling:Hierarchical Information Enhanced Traffic Forecasting},
  author={Qian Ma, Zijian Zhang, Xiangyu Zhao, Haoliang Li, Hongwei Zhao, Yiqi Wang, Zitao Liu, and Wanyu Wang.},
  booktitle={Proceedings of the 32nd ACM International Conference on Information \& Knowledge Management},
  url = {https://doi.org/10.1145/3583780.3614910},
  doi = {10.1145/3583780.3614910},
  year={2023}
}

ACM Reference Format:
Qian Ma, Zijian Zhang, Xiangyu Zhao, Haoliang Li, Hongwei Zhao, Yiqi
Wang, Zitao Liu, and Wanyu Wang. 2023. Rethinking Sensors Modeling:
Hierarchical Information Enhanced Traffic Forecasting . In Proceedings
of the 32nd ACM International Conference on Information and Knowledge
Management (CIKM ’23), October 21–25, 2023, Birmingham, United Kingdom.
ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3583780.3614910
```



# Quick Start

## Acknowledgements

We refer to the code implementation of [lib-city](https://bigscity-libcity-docs.readthedocs.io/en/latest/get_started/quick_start.html)
Please also cite the following papers if you find the code useful.

```latex
@inproceedings{libcity,
  author = {Wang, Jingyuan and Jiang, Jiawei and Jiang, Wenjun and Li, Chao and Zhao, Wayne Xin},
  title = {LibCity: An Open Library for Traffic Prediction},
  year = {2021},
  isbn = {9781450386647},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3474717.3483923},
  doi = {10.1145/3474717.3483923},
  booktitle = {Proceedings of the 29th International Conference on Advances in Geographic Information Systems},
  pages = {145–148},
  numpages = {4},
  keywords = {Spatial-temporal System, Reproducibility, Traffic Prediction},
  location = {Beijing, China},
  series = {SIGSPATIAL '21}
}
```

## 1. Clone this repo

```
git clone https://github.com/VAN-QIAN/CIKM23-HIEST.git
```



## 2. Prepare your dataset

You can create a new folder "raw_data" under the root path and download a dataset from the collection [libcity](https://bigscity-libcity-docs.readthedocs.io/en/latest/tutorial/install_quick_start.html#download-one-dataset) under the new path.

Then simply add the mapping matrix "XXX.mor.py" into the folder of a dataset e.g. ,  $ROOT_PATH/raw_data/METR_LA/METR_LA.mor.py. 

You can utilize our proposed mapping matrix or generate one by the provided utils.

## 3. Create a config file

A simple configs.json file to indicate some hyperparameters, e.g., the number of global nodes, hidden_size,$\eta_{1,2,3,4}$

```json
{
	"n1": 1,
	"n2": 1,
	"n3": 1,
	"n4": 1,
	"global_nodes":15,
	"nhid":32
}
```



## 4. Execution

Under the root path with the run_model.py, the program should be executed properly.

```bash
python3 ./run_model.py --task traffic_state_pred --model HIEST --config configs --dataset METR_LA
```

You can refer to [the following section](https://github.com/VAN-QIAN/CIKM23-HIEST#53-execution) for more references.

# Anonymous Github version

The following parts are organized as follows,

1. The model files
2. The processed adjacency matrices and mapping matrices for datasets.
3. The utils for solving BCC
4. The visualization code.
5. The environment image preparation.



## 1. Model

Our model is under the path of ./code/HIEST.py.
We also provide an implementation of a Traffic-Transformer under the [guide of the lib-city](https://bigscity-libcity-docs.readthedocs.io/en/latest/developer_guide/implemented_models.html) 

These two models are for 'traffic-state-prediction', you can add them into the pipeline under the [instructions]((https://bigscity-libcity-docs.readthedocs.io/en/latest/developer_guide/implemented_models.html) ) provided by lib-city.

## 2. Processed Data

For the attributes self.adj_mx and self.Mor, they will be initialized with the processed adjacency matrix and mapping matrix. Please check the path settings to make it correspond with the dataset.

For the training datasets, you can refer to the [datasets collection of lib-city](https://bigscity-libcity-docs.readthedocs.io/en/latest/get_started/quick_start.html)

## 3. The utils for solving BCC

The utils for solving BCC are under the path of ./utils .

For the usage, you can refer to the visualization code under the path of ./code/visualization.py

## 4. The visualization code

Our visualization result is implemented by the [QGIS](https://qgis.org/en/site/).

The visualization code is used to generate the Geo_JSON file to be imported into the QGIS.

We generate a .json file for each regional/global node.

Then you can import them as follows:

![image-20230604094224902](./README.assets/image-20230604094224902.png)

You can also search and install the QuickMap services to add the base map.

![image-20230604094519919](./README.assets/image-20230604094519919.png)

## 5. Running environment

The running environment aligns with the [requirements of lib-city](https://github.com/LibCity/Bigscity-LibCity/blob/master/requirements.txt)

We are glad to share the following guide for build environment to ease reproducibility.

We implement the customized environment with [singularity](https://docs.sylabs.io/guides/3.7/user-guide/index.html) image for better execution.

If you are using Docker, the key idea should be similar with our implementation.

The singularity official documentation will provide the quick start-up with installation steps.

*All of the following scripts are executed on the **root path** of lib-city!*

### 5.1 Base image

As we refer to the implementation of the lib-city, we follow their basic pytorch major version of 1.7.1 with cuda11.0.

A good practice is to use a dev version of the PyTorch base image from the official docker registry

```sh
# https://hub.docker.com/layers/pytorch/pytorch/1.7.1-cuda11.0-cudnn8-devel/images/sha256-f0d0c1b5d4e170b4d2548d64026755421f8c0df185af2c4679085a7edc34d150?context=explore
singularity pull docker://pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
```

If everything goes well, you will see the following INFO when pulling the base image

![image-20230525102152806](./README.assets/image-20230525102152806.png)

Once the downloading is done, you will get a **SIF image** with the suffix **.sif(like pytorch_1.7.1-cuda11.0-cudnn8-devel.sif)** on your local machine. This will be used as a base image in the following steps.

### 5.2 Install Requirements

1. Create a definition file(named HIEST.def) as follows,

```sh
#Bootstrap is used to specify the agent,where the base image from,here localimage means to build from a local image
Bootstrap: localimage
## This is something like 'From' in DOCKERFILE to indicate the base image
From: ./pytorch_1.7.1-cuda11.0-cudnn8-devel.sif

# %files can be used to copy files from host into the image
# like 'COPY' in DOCKERFILE
# Here we copy the requirements.txt into the image, then we can use it to install the required dependencies.
%files
    ./Bigscity-LibCity/requirements.txt /opt

# %post is used to build the new image
# Usage is same to shell.Here we used pip to install dependencies.
%post
    pip install -r /opt/requirements.txt
    pip install protobuf==3.20.0 #to solve some warning we met
 
#% environment is used to set env_variables once the image starts
# These lines are necessary to load cuda
%environment
    export PATH=$PATH:/usr/local/cuda-11.0/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/lib64:/usr/lib/x86_64-linux-gnu
```

2. Now execute the following command to build the image

```sh
## still on the root path
singularity build HIEST.sif HIEST.def
```

You will see the following INFO when building the new image

![image-20230525102417208](./README.assets/image-20230525102417208.png)

If nothing is wrong after creating SIF file, then you will get the image file **HIEST.sif** on the root path.

### 5.3 Execution

Now the environment is ready, and all the code should be able to execute properly now.

Here are the slurm script and command for reference

```shell
# cd libs-city
# pwd
# this is the key command,remember to add the '--nv' option
singularity exec --nv ../HIEST.sif python3 ./run_model.py --task traffic_state_pred
--model HIEST --dataset METR_LA
```

```shell
#!/bin/bash
## if you are using SBATCH,pls remeber to add proper command,such as
# SPARTION

singularity exec --nv ../HIEST.sif python3 ./run_model.py --task traffic_state_pred
--model HIEST --dataset METR_LA
```

