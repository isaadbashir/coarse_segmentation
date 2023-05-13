# A Novel Framework for Coarse-Grained Semantic Segmentation of Whole-Slide Images
Raja Muhammad Saad Bashir, Muhammad Shaban, Shan E Ahmed Razaa, Nasir M. Rajpoot

Semantic segmentation is requires alot of pixel-wise annotations for training data while gathering pixel-wise annotations is tidious and expensive. Coarse segmentation solves by making it easy to annotate the large images using caorse masks.

![Framework](https://isaadbashir.github.io/portfolio/assets/img/publication_preview/coarsenet.png)

## Datasets Preparation
Create course masks by running the `create_coarse_masks.py` script from pre_processing

## Training
For training run the `train.py` after configuring different variable in the `config.py`

# Citation
If you find this project useful, please consider citing:

```
@inproceedings{bashir2022novel,
  title={A Novel Framework for Coarse-Grained Semantic Segmentation of Whole-Slide Images},
  author={Bashir, Raja Muhammad Saad and Shaban, Muhammad and Raza, Shan E Ahmed and Khurram, Syed Ali and Rajpoot, Nasir},
  booktitle={Medical Image Understanding and Analysis: 26th Annual Conference, MIUA 2022, Cambridge, UK, July 27--29, 2022, Proceedings},
  pages={425--439},
  year={2022},
  organization={Springer}
}
```





