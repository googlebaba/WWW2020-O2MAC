# WWW2020-O2MAC

WWW2020-One2Multi Graph Autoencoder for Multi-view Graph Clustering

The source code is based on [ARGA](https://github.com/Ruiqi-Hu/ARGA)

paper: http://www.shichuan.org/doc/83.pdf

# Contact
Shaohua Fan, Email:fanshaohua92@163.com


# Preprocessed-Data
Preprocessed DBLP can be found in: https://pan.baidu.com/s/1Qr2e97MofXsBhUvQqgJqDg password：6b3h

Preprocessed IMDB can be found in: https://pan.baidu.com/s/199LoAr5WmL3wgx66j-qwaw password:qkec

Google dirve: https://drive.google.com/file/d/1uF64x8IgiqQMUeepRDOmY6KSkWKtfGjz/view?usp=sharing
# Requirements
pip -r requirements.txt
# Running the model
python O2MAC.py

# Note
Set input_view (Informative view) in settings.py as 0 for ACM and 1 for DBLP.

# Reference
@inproceedings{

> author = {Shaohua Fan, Xiao Wang, Chuan Shi, Emiao Lu, Ken Lin, Bai Wang},
 
> title = {One2Multi Graph Autoencoder for Multi-view Graph Clustering},
 
> booktitle = {Proceedings of The Web Conference 2020 (WWW ’20)},

> year = {2020}, 

> publisher = {ACM},

> address = {Taipei,Taiwan},
 
}
