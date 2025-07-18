# More Than Routing: Joint GPS and Route Modeling for Refine Trajectory Representation Learning

## Latest News

**📅 19/02/2025: WE RELEASE THE EXPANDED VERSIONS OF BOTH DATASETS AND FIXED THE ISSUES WITH THE PREVIOUS DATASET [HERE](https://drive.google.com/file/d/1w1fMautspOMG2B0Z2_mI16hsxRbRFywu/view?usp=drive_link). Unzip password is ‘dataset’**

**📅 18/04/2024: FULL [SLIDE](https://www.slideshare.net/slideshow/more-than-routing-joint-gps-and-route-modeling-for-refine-trajectory-representation-learning/269319522) IS UPDATED.**

**📅 18/03/2024: A SHORT [VIDEO](https://www.youtube.com/watch?v=IA3quCF0LWM) OF PRESENTATION AT WWW2024.**

## About

This is a Pytorch implementation of **JGRM** as described in the paper [More Than Routing: Joint GPS and Route Modeling for Refine Trajectory Representation Learning](https://dl.acm.org/doi/10.1145/3589334.3645644).

![image](https://github.com/mamazi0131/JGRM/blob/main/framework.png)

## Requirement
We build this project by Python 3.7.12 with the following packages:
```
torch==1.7.1
torch-geometric==2.3.1
scikit-learn==1.0.2
pandas==1.3.5
pickleshare==0.7.5
shapely==2.0.1
faiss-cpu==1.7.4
```

## Dataset
We recently revisited the project and discovered that the previously released dataset could not be imported due to transmission issues. We have re-uploaded and verified the relevant data, and further supplemented the dataset with a 200k version to facilitate comparison for everyone.

You can access our data [here](https://drive.google.com/file/d/1w1fMautspOMG2B0Z2_mI16hsxRbRFywu/view?usp=drive_link) **2025.06.21 UPDATED**.

The folder contains a total of two datasets, Chengdu and Xi'an. Each folder contains 7 files with the following file directory:
```
|----Chengdu\
|    |----chengdu_1101_1115_data_sample10w.pkl      # Training data
|    |----chengdu_1101_1115_data_seq_evaluation.pkl # Evaluation data
|    |----transition_prob_mat.npy                   # The transition matrix is obtained from the full dataset, a choice for initializing the road network
|    |----init_w2v_road_emb.pt                      # The section embedding is obtained from the word2vec
|    |----edge_geometry.csv                         # Topology of the road section
|    |----edge_features.csv                         # Attributes of the road section
|    |----line_graph_edge_idx.npy                   # Adjacency matrix
```

## Cite
If you have any questions related to the code or the paper, feel free to email mazhipeng1024@my.swjtu.edu.cn.
```
@inproceedings{10.1145/3589334.3645644,
  author = {Ma, Zhipeng and Tu, Zheyan and Chen, Xinhai and Zhang, Yan and Xia, Deguo and Zhou, Guyue and Chen, Yilun and Zheng, Yu and Gong, Jiangtao},
  title = {More Than Routing: Joint GPS and Route Modeling for Refine Trajectory Representation Learning},
  year = {2024},
  isbn = {9798400701719},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3589334.3645644},
  doi = {10.1145/3589334.3645644},
  booktitle = {Proceedings of the ACM on Web Conference 2024},
  pages = {3064–3075},
  numpages = {12},
  location = {, Singapore, Singapore, },
  series = {WWW '24} }
```
