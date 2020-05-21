import os
import sys
import cv2 as cv
import numpy as np
import mxnet as mx
import gluoncv as gcv
from glob import glob
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints

# 加载检测器
ctx = mx.gpu()
detector_name = "ssd_512_mobilenet1.0_coco"
detector = get_model(detector_name, pretrained=True, ctx=ctx)

estimator_name = "simple_pose_resnet18_v1b"
estimator = get_model(estimator_name, pretrained='ccd24037', ctx=ctx)

detector.reset_class(classes=['person'], reuse_weights={'person':'person'})

detector.hybridize()
estimator.hybridize()

def estimate(frame):
    frame = mx.nd.array(cv.cvtColor(frame, cv.COLOR_BGR2RGB)).astype('uint8')

    x, img = gcv.data.transforms.presets.ssd.transform_test(frame, short=512)
    x = x.as_in_context(ctx)
    class_IDs, scores, bounding_boxs = detector(x)

    pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs, output_shape=(128, 96), ctx=ctx)

    # 只检测一个人
    if len(upscale_bbox) == 1:
        pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs, output_shape=(128, 96), ctx=ctx)
        return estimator(pose_input).asnumpy().flatten()

    return False

# 保存所有样本内容
X_samples = []
y_samples = []

DATA_DIR = 'data'
for img_name in glob(os.path.join(DATA_DIR, '*', '*.jpg')):
    print(img_name)
    img = cv.imread(img_name)

    # 检测人物
    X = estimate(img)
    if isinstance(X, bool):
        # 人数不对
        continue
    # print(X.shape)
    
    X_samples.append(X)
    y_samples.append(os.path.basename(os.path.dirname(img_name)).split('_')[0])
    
np.save(os.path.join('features', 'X_samples.npy'), np.array(X_samples))
np.save(os.path.join('features', 'y_samples.npy'), np.array(y_samples))
