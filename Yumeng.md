## 预处理数据
### 1. 生成voxel的mesh
处理单个文件
```
python voxel_surface.py path/to/file.glb -r 200
```
处理整个文件夹
```
### 测试一下
python voxel_surface.py mesh/drawers_test -r 200
### 实际运行
python voxel_surface.py mesh/glb_part -r 200
```

### 2. 根据PartCrafter的要求预处理partnet-mobility
```
CUDA_VISIBLE_DEVICES=7 \
	python datasets/preprocess/preprocess_partnet.py
```