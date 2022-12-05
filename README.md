<h1>SEGMENTATION</h1>
<h2>Load library</h2>
<p><i>
from mrcnn.main import *<br>
from mrcnn.utils import *<br>
import warnings<br>
%matplotlib inline <br>
from mrcnn.config import Config
</i></p>

<h2>Load pretrained weights</h2>
<p><i>
model_path = 'C:/TEMP/Mask-R-CNN/V3/Mask_RCNN/logs/cell20180618T0605/mask_rcnn_cell_0040.h5'
</i></p>

<h2>Load data</h2>
<p><i>
size=400<br><br>

images = sorted((glob.glob("./MaskRCNN/train/*crop.png")))<br>
labels = sorted((glob.glob("./MaskRCNN/train/*label.png")))<br><br>

images_val = sorted((glob.glob("./MaskRCNN/val/*crop.png")))<br>
labels_val = sorted((glob.glob("./MaskRCNN/val/*label.png")))
</i></p>

<h2>Configuration for the prediction</h2>
<p><i>
class InferenceConfig(CellConfig):<br>
    GPU_COUNT = 1<br>
    IMAGES_PER_GPU = 1<br>
    IMAGE_RESIZE_MODE = "pad64"<br>
    DETECTION_MAX_INSTANCES = 1000<br>
    DETECTION_MIN_CONFIDENCE = 0.8<br>
    DETECTION_NMS_THRESHOLD = 0.20<br>
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)<br><br>
    
    POST_NMS_ROIS_INFERENCE=10000
</i></p>

<h2>Load Model</h2>
<p><i>
inference_config = InferenceConfig()<br>
model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)<br>
model.load_weights(model_path, by_name=True)<br>
</i></p>

<h2>Prediction on training data</h2>
<p><i>
train_images = sorted(glob.glob('./train/*crop.png'))<br>
train_masks = sorted(glob.glob('./train/*label.png'))<br>
</i></p>
<img src="https://github.com/vpNjit/segmentation/blob/main/seg/seg1.png">

<h2>Prediction on test data</h2>
<p><i>
test_images = sorted(glob.glob('./data/RGB_2/*.png'))<br>
train_masks = sorted(glob.glob('./train/*label.png'))<br><br>

%time<br>
for image in test_images:<br>
    plt.figure(figsize=(20,10),dpi=100)<br>
    im = imageio.imread(image)<br>
    plt.subplot(1,2,1)<br>
    plt.imshow(im)<br>
    ax2=plt.subplot(1,2,2)<br>
    r=results = model.detect([im], verbose=1)[0]<br>
    visualize.display_instances_new(im, r['rois'], r['masks'], r['class_ids'], r['scores'], ax=ax2)<br>
</i></p>
<img src="https://github.com/vpNjit/segmentation/blob/main/seg/seg2.png">
