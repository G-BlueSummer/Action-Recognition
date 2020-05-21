import argparse, time, os
import cv2 as cv
import mxnet as mx
import numpy as np
import gluoncv as gcv
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints
from joblib import load

# 读取参数
parser = argparse.ArgumentParser()
parser.add_argument('--video', default=0)
args = parser.parse_args()

fps_time = 0

# 设置模型
ctx = mx.gpu()

detector_name = "ssd_512_mobilenet1.0_coco"
detector = get_model(detector_name, pretrained=True, ctx=ctx)

estimator_name = "simple_pose_resnet18_v1b"
estimator = get_model(estimator_name, pretrained='ccd24037', ctx=ctx)

detector.reset_class(classes=['person'], reuse_weights={'person':'person'})

detector.hybridize()
estimator.hybridize()

le = load(os.path.join('model', 'label.joblib'))
pca = load(os.path.join('model', 'pca.joblib'))
cf = load(os.path.join('model', 'classifier.joblib'))

# 视频读取
cap = cv.VideoCapture(args.video)

ret, frame = cap.read()
while ret:
    
    frame = mx.nd.array(cv.cvtColor(frame, cv.COLOR_BGR2RGB)).astype('uint8')

    x, img = gcv.data.transforms.presets.ssd.transform_test(frame, short=512)
    x = x.as_in_context(ctx)
    class_IDs, scores, bounding_boxs = detector(x)

    pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs, output_shape=(128, 96), ctx=ctx)

    # 检测到人时
    if len(upscale_bbox) > 0:
        predicted_heatmap = estimator(pose_input)
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
        img = cv_plot_keypoints(img, pred_coords, confidence, class_IDs, bounding_boxs, scores)

        # 动作识别
        X = predicted_heatmap.asnumpy().flatten().reshape((len(upscale_bbox), -1))
        X = pca.transform(X)
        action = le.inverse_transform(cf.predict(X))


    cv_plot_image(img, 
        upperleft_txt=f"FPS:{(1.0 / (time.time() - fps_time)):.2f}", upperleft_txt_corner=(10,25),
        left_txt_list=action)
    fps_time = time.time()
    
    # ESC键退出
    if cv.waitKey(1) == 27:
        break

    ret, frame = cap.read()

cv.destroyAllWindows()
cap.release()