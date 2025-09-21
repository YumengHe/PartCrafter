## 预处理数据
pip install -U "trimesh[easy]" pillow pygltflib


### 1. 处理partnet-mobility的mesh，obj+mtl -> glb
```
python merge_to_glb.py mesh/partnet_test mesh/partnet_glb
python merge_to_glb.py mesh/partnet_test mesh/partnet_glb --single 40147
```

### 2. 生成voxel的mesh
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
处理subfolder
```
python voxel_surface.py mesh/partnet_glb --subfolder -r 100
```

### 3. 根据PartCrafter的要求预处理partnet-mobility
```
CUDA_VISIBLE_DEVICES=7 python datasets/preprocess/preprocess_partnet.py --input mesh/partnet_glb --output preprocessed_data
```