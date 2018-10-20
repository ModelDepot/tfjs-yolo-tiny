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

### yolo(input, model, options)

#### Args

Param | Type | Default | Description
-- | -- | -- | --
input | tf.Tensor | - | Expected shape (1, 416, 416, 3) Tensor representing input image (RGB 416x416)
model | tf.Model | - | Tiny YOLO tf.Model
[options] | Object | See Below | Optional, Additional Configs

If you're using a custom Tiny YOLO model or want to adjust the default
filtering cutoffs, you may do so by passing an additional options
object.

Example: `yolo(inputImage, model, { classProbThreshold: 0.8 });`

Option | Type | Default | Description
-- | -- | -- | --
| [options.classProbThreshold] | <code>Number</code> | <code>0.4</code> | Filter out classes below a certain threshold |
| [options.iouThreshold] | <code>Number</code> | <code>0.4</code> | Filter out boxes that have an IoU greater than this threadhold (refer to tf.image.nonMaxSuppression) |
| [options.filterBoxesThreshold] | <code>Number</code> | <code>0.01</code> | Threshold to filter out box confidence * class confidence |
| [options.maxBoxes] | <code>Number</code> | <code>2048</code> | Number of max boxes to return, refer to tf.image.nonMaxSuppression. Note: The model itself can only return so many boxes. |
| [options.yoloAnchors] | <code>tf.Tensor</code> | <code>See src/postprocessing.js</code> | (Advanced) Yolo Anchor Boxes, only needed if retraining on a new dataset |
| [options.width] | <code>Number</code> | <code>416</code> | (Advanced) If your model's input width is not 416, only if you're using a custom model |
| [options.height] | <code>Number</code> | <code>416</code> | (Advanced) If your model's input height is not 416, only if you're using a custom model |
| [options.numClasses] | <code>Number</code> | <code>80</code> | (Advanced) If your model has a different number of classes, only if you're using a custom model |
| [options.classNames] | <code>Array.&lt;String&gt;</code> | <code>See src/coco_classes.js</code> | (Advanced) If your model has non-MSCOCO class names, only if you're using a custom model |

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

# Contributing

PR's are more than welcome! Perf improvement or better test coverage
are probably the two biggest areas of immediate need. If you have thoughts
on extensibility as well, feel free to open an issue!

## Install Dependencies
```
yarn install
```

## Run Tests

If you're running tests, make sure to `yarn add @tensorflow/tfjs@0.7.0`
so that you you don't get tfjs package not found errors. If you're developing,
make sure to remove tfjs as a dependency, as it'll start using the
local version of `tfjs` intead of the peer version.

Note: Test coverage is poor, definitely don't rely on them to catch your errors.

```
yarn test
```

## Build

```
yarn build
```

Or during development, use watch mode, you can use the [demo app](https://github.com/ModelDepot/tfjs-yolo-tiny-demo)
to test out changes.

```
yarn watch
```
