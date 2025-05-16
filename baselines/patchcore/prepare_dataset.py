import shutil
import os


### example dataset
# # convert example dataset to mvtec format
# _path = '/home/tri/multi-level-anomaly/baseline/patchcore-inspection/my_data/class_based/example/level_0_train'
# normals = os.listdir(_path)
# for i in normals:
#     os.makedirs(os.path.join(_path, i + '_', 'train'), exist_ok=True)
#     shutil.move(os.path.join(_path, i), os.path.join(_path, i + '_', 'train', 'good'))
#     os.rename(os.path.join(_path, i + '_'), os.path.join(_path, i))

# # Only keep 500 samples for each class
# _path = '/home/tri/multi-level-anomaly/baseline/patchcore-inspection/my_data/class_based/example/level_0_train'
# normals = os.listdir(_path)
# for i in normals:
#     samples = os.listdir(os.path.join(_path, i, 'train', 'good'))
#     samples.sort()
#     for sample in samples[500:]:
#         os.remove(os.path.join(_path, i, 'train', 'good', sample))


### diabetic-retinopathy dataset
# # convert diabetic-retinopathy dataset to mvtec format
# _path = '/home/tri/multi-level-anomaly/baseline/patchcore-inspection/my_data/severity-based/diabetic-retinopathy/level_0_train'
# os.makedirs(os.path.join(_path + '_', 'diabetic-retinopathy', 'train'), exist_ok=True)
# shutil.move(_path, os.path.join(_path + '_', 'diabetic-retinopathy', 'train', 'good'))
# os.rename(_path + '_', _path)

# # Only keep 1000 samples for each class
# _path = '/home/tri/multi-level-anomaly/baseline/patchcore-inspection/my_data/severity-based/diabetic-retinopathy/level_0_train'
# normals = os.listdir(_path)
# for i in normals:
#     samples = os.listdir(os.path.join(_path, i, 'train', 'good'))
#     samples.sort()
#     for sample in samples[1000:]:
#         os.remove(os.path.join(_path, i, 'train', 'good', sample))


### covid19 dataset
# # convert covid19 dataset to mvtec format
# _path = '/home/tri/multi-level-anomaly/baseline/patchcore-inspection/my_data/severity-based/covid19/level_0_train'
# os.makedirs(os.path.join(_path + '_', 'covid', 'train'), exist_ok=True)
# shutil.move(_path, os.path.join(_path + '_', 'covid', 'train', 'good'))
# os.rename(_path + '_', _path)
