import * as tf from '@tensorflow/tfjs';

import {
  yolo_boxes_to_corners,
  yolo_head,
  yolo_filter_boxes,
  YOLO_ANCHORS,
} from './postprocess';
import class_names from './coco_classes';

const DEFAULT_INPUT_DIM = 416;

const DEFAULT_MAX_BOXES = 2048; // TODO: There is a limit to tiny-yolo, need to check the model.
const DEFAULT_FILTER_BOXES_THRESHOLD = 0.01;
const DEFAULT_IOU_THRESHOLD = 0.4;
const DEFAULT_CLASS_PROB_THRESHOLD = 0.4
const DEFAULT_MODEL_LOCATION =
  'https://raw.githubusercontent.com/MikeShi42/yolo-tiny-tfjs/master/model2.json';

/**
 * Downloads a tf.Model, defaults to a MSCOCO trained Tiny YOLO model
 * @param {String} url Tiny YOLO Model URL
 */
export async function downloadModel(url = DEFAULT_MODEL_LOCATION) {
  return await tf.loadModel(url);
}

/**
 * Given an input image and model, outputs bounding boxes of detected
 * objects with class labels and class probabilities.
 * @param {tf.Tensor} input Expected shape (1, 416, 416, 3)
 * Tensor representing input image (RGB 416x416)
 * @param {tf.Model} model Tiny YOLO model to use
 * @param {Object} [options] Override options for customized
 * models or performance
 * @param {Number} [options.classProbThreshold=0.4] Filter out classes
 * below a certain threshold
 * @param {Number} [options.iouThreshold=0.4] Filter out boxes that
 * have an IoU greater than this threadhold (refer to tf.image.nonMaxSuppression)
 * @param {Number} [options.filterBoxesThreshold=0.01] Threshold to
 * filter out box confidence * class confidence
 * @param {Number} [options.maxBoxes=2048] Number of max boxes to
 * return, refer to tf.image.nonMaxSuppression. Note: The model
 * itself can only return so many boxes.
 * @param {tf.Tensor} [options.yoloAnchors=See src/postprocessing.js]
 * (Advanced) Yolo Anchor
 * Boxes, only needed if retraining on a new dataset
 * @param {Number} [options.width=416] (Advanced) If your model's input width is not 416, only if you're using a custom model
 * @param {Number} [options.height=416] (Advanced) If your model's input height is not 416, only if you're using a custom model
 * @param {Number} [options.numClasses=80] (Advanced) If your model has a different number of classes, only if you're using a custom model
 * @param {Array<String>} [options.classNames=See src/coco_classes.js] (Advanced) If your model has non-MSCOCO class names, only if you're using a custom model
 * @returns {Array<Object>} An array of found objects with `className`,
 * `classProb`, `bottom`, `top`, `left`, `right`. Positions are in pixel
 * values. `classProb` ranges from 0 to 1. `className` is derived from
 * `options.classNames`.
 */
async function yolo(
  input,
  model,
  {
    classProbThreshold = DEFAULT_CLASS_PROB_THRESHOLD,
    iouThreshold = DEFAULT_IOU_THRESHOLD,
    filterBoxesThreshold = DEFAULT_FILTER_BOXES_THRESHOLD,
    yoloAnchors = YOLO_ANCHORS,
    maxBoxes = DEFAULT_MAX_BOXES,
    width: widthPx = DEFAULT_INPUT_DIM,
    height: heightPx = DEFAULT_INPUT_DIM,
    numClasses = 80,
    classNames = class_names,
  } = {},
) {
  const outs = tf.tidy(() => { // Keep as one var to dispose easier
    const activation = model.predict(input);

    const [box_xy, box_wh, box_confidence, box_class_probs ] =
      yolo_head(activation, yoloAnchors, numClasses);

    const all_boxes = yolo_boxes_to_corners(box_xy, box_wh);

    let [boxes, scores, classes] = yolo_filter_boxes(
      all_boxes, box_confidence, box_class_probs, filterBoxesThreshold);

    // If all boxes have been filtered out
    if (boxes == null) {
      return null;
    }

    const width = tf.scalar(widthPx);
    const height = tf.scalar(heightPx);

    const image_dims = tf.stack([height, width, height, width]).reshape([1,4]);

    boxes = tf.mul(boxes, image_dims);

    return [boxes, scores, classes];
  });

  if (outs === null) {
    return [];
  }

  const [boxes, scores, classes] = outs;

  const indices = await tf.image.nonMaxSuppressionAsync(boxes, scores, maxBoxes, iouThreshold)

  // Pick out data that wasn't filtered out by NMS and put them into
  // CPU land to pass back to consumer
  const classes_indx_arr = await classes.gather(indices).data();
  const keep_scores = await scores.gather(indices).data();
  const boxes_arr = await boxes.gather(indices).data();

  tf.dispose(outs);
  indices.dispose();

  const results = [];

  classes_indx_arr.forEach((class_indx, i) => {
    const classProb = keep_scores[i];
    if (classProb < classProbThreshold) {
      return;
    }

    const className = classNames[class_indx];
    let [top, left, bottom, right] = [
      boxes_arr[4 * i],
      boxes_arr[4 * i + 1],
      boxes_arr[4 * i + 2],
      boxes_arr[4 * i + 3],
    ];

    top = Math.max(0, top);
    left = Math.max(0, left);
    bottom = Math.min(heightPx, bottom);
    right = Math.min(widthPx, right);

    const resultObj = {
      className,
      classProb,
      bottom,
      top,
      left,
      right,
    };

    results.push(resultObj);
  });

  return results;
}

export default yolo;
