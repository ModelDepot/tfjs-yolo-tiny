import * as tf from '@tensorflow/tfjs';

import {
  yolo_boxes_to_corners,
  yolo_head,
  yolo_filter_boxes,
  YOLO_ANCHORS,
} from './postprocess';
import class_names from './coco_classes';

const INPUT_DIM = 416;

const MAX_BOXES = 2048; // TODO: There is a limit to tiny-yolo, need to check the model.
const DEFAULT_FILTER_BOXES_THRESHOLD = 0.01;
const DEFAULT_IOU_THRESHOLD = 0.4;
const DEFAULT_CLASS_PROB_THRESHOLD = 0.4
const DEFAULT_MODEL_LOCATION =
  'https://raw.githubusercontent.com/MikeShi42/yolo-tiny-tfjs/master/model2.json';

export async function downloadModel(url = DEFAULT_MODEL_LOCATION) {
  return await tf.loadModel(url);
}

export default async function yolo(
  input,
  model,
  classProbThreshold = DEFAULT_CLASS_PROB_THRESHOLD,
  iouThreshold = DEFAULT_IOU_THRESHOLD,
  filterBoxesThreshold = DEFAULT_FILTER_BOXES_THRESHOLD
) {
  const outs = tf.tidy(() => { // Keep as one var to dispose easier
    const activation = model.predict(input);

    const [box_xy, box_wh, box_confidence, box_class_probs ] =
      yolo_head(activation, YOLO_ANCHORS, 80);

    const all_boxes = yolo_boxes_to_corners(box_xy, box_wh);

    let [boxes, scores, classes] = yolo_filter_boxes(
      all_boxes, box_confidence, box_class_probs, filterBoxesThreshold);

    // If all boxes have been filtered out
    if (boxes == null) {
      return null;
    }

    const width = tf.scalar(INPUT_DIM);
    const height = tf.scalar(INPUT_DIM);

    const image_dims = tf.stack([height, width, height, width]).reshape([1,4]);

    boxes = tf.mul(boxes, image_dims);

    return [boxes, scores, classes];
  });

  if (outs === null) {
    return [];
  }

  const [boxes, scores, classes] = outs;

  const indices = await tf.image.nonMaxSuppressionAsync(boxes, scores, MAX_BOXES, iouThreshold)

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

    const className = class_names[class_indx];
    let [top, left, bottom, right] = [
      boxes_arr[4 * i],
      boxes_arr[4 * i + 1],
      boxes_arr[4 * i + 2],
      boxes_arr[4 * i + 3],
    ];

    top = Math.max(0, top);
    left = Math.max(0, left);
    bottom = Math.min(416, bottom);
    right = Math.min(416, right);

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
