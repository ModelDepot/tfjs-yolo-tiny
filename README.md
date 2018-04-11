<a href='https://modeldepot.io/mikeshi/tiny-yolo-in-javascript'> <img src='https://img.shields.io/badge/ModelDepot-Pre--trained_Model-3d9aff.svg'/> </a>

# ⚡️ Fast In-Browser Object Detection 👀

Detect objects in images right in your browser using [Tensorflow.js](https://js.tensorflow.org/)! Currently takes ~800ms
to analyze each frame on Chrome MBP 13" mid-2014.

Supports [`Tiny YOLO`](https://pjreddie.com/darknet/yolo/), as of right now,
 [`tfjs`](https://github.com/tensorflow/tfjs) does not have
support to run any full YOLO models (and your user's computers probably
can't handle it either).

## Demo

[Check out the Live Demo](https://modeldepot.github.io/tfjs-yolo-tiny-demo/)

(You can only get so far with 1 FPS)

![yolo person detection](https://github.com/ModelDepot/tfjs-yolo-tiny/raw/master/assets/demo.gif)

## Install

### Yarn
    yarn add tfjs-yolo-tiny
### Or NPM
    npm install tfjs-yolo-tiny

## Usage Example
```javascript
import yolo, { downloadModel } from 'tfjs-yolo-tiny';

const model = await downloadModel();
const inputImage = webcam.capture();

const boxes = await yolo(inputImage, model);

// Display detected boxes
boxes.forEach(box => {
  const {
    top, left, bottom, right, classProb, className,
  } = box;

  drawRect(left, top, right-left, bottom-top, `${className} ${classProb}`)
});
```
## API Docs

### yolo(input, model, classProbThreshold, iouThreshold, filterBoxesThreshold)

#### Args

Param | Type | Default | Description
-- | -- | -- | --
input | tf.Tensor | - | Expected shape (1, 416, 416, 3) Tensor representing input image (RGB 416x416)
model | tf.Model | - | Tiny YOLO tf.Model
classProbThreshold | Number | 0.4 | Don't return detections below a certain class probability.
iouThreshold | Number | 0.4 | Don't return boxes that have more intersection over union than a more likely box. See [non max suppression](https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression).
filterBoxesThreshold | Number | 0.01 | Don't return boxes that have a box_prob * class_prob of less than this threshold.

#### Returns

Returns an array of objects.

Property | Type | Description
-- | -- | --
top | Number | Pixels from top of image where bounding box starts
left | Number | Pixels from left of image where bounding box starts
bottom | Number | Pixels from top of image where box ends.
right | Number | Pixels from left of image where box ends.
classProb | Number | Probability of the class in the bounding box.
className | String | Human name of the class.

### downloadModel(url)

#### Args

Param | Type | Default | Description
-- | -- | -- | --
url | string | See DEFAULT_MODEL_LOCATION | Tiny YOLO Model config path. See [tf.loadModel](https://js.tensorflow.org/api/0.8.0/#loadModel)

#### Returns

Returns a `Promise` that can resolve to a `tf.Model`.
