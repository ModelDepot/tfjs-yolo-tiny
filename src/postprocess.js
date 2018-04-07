// Heavily derived from YAD2K (https://github.com/allanzelener/YAD2K)
import * as tf from '@tensorflow/tfjs';

import class_names from './coco_classes';

export const YOLO_ANCHORS = tf.tensor2d([
  [0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434],
  [7.88282, 3.52778], [9.77052, 9.16828],
]);

// Note: Returns [null] * 3 if all boxes are filtered out
export async function yolo_filter_boxes(
  boxes,
  box_confidence,
  box_class_probs,
  threshold
) {
  const box_scores = tf.mul(box_confidence, box_class_probs);
  const box_classes = tf.argMax(box_scores, -1);
  const box_class_scores = tf.max(box_scores, -1);

  const prediction_mask = tf.greaterEqual(box_class_scores, tf.scalar(threshold));

  const mask_arr = await prediction_mask.data();

  const indices_arr = [];
  for (let i=0; i<mask_arr.length; i++) {
    const v = mask_arr[i];
    if (v) {
      indices_arr.push(i);
    }
  }

  if (indices_arr.length == 0) {
    return [null, null, null];
  }

  const indices = tf.tensor1d(indices_arr);

  return [
    tf.gather(boxes.reshape([mask_arr.length, 4]), indices),
    tf.gather(box_class_scores.flatten(), indices),
    tf.gather(box_classes.flatten(), indices),
  ];
}

/**
 * Given XY and WH tensor outputs of yolo_head, returns corner coordinates.
 * @param {tf.Tensor} box_xy Bounding box center XY coordinate Tensor
 * @param {tf.Tensor} box_wh Bounding box WH Tensor
 * @returns {tf.Tensor} Bounding box corner Tensor
 */
export function yolo_boxes_to_corners(box_xy, box_wh) {
  const two = tf.tensor1d([2.0]);
  const box_mins = tf.sub(box_xy, tf.div(box_wh, two));
  const box_maxes = tf.add(box_xy, tf.div(box_wh, two));

  const dim_0 = box_mins.shape[0];
  const dim_1 = box_mins.shape[1];
  const dim_2 = box_mins.shape[2];
  const size = [dim_0, dim_1, dim_2, 1];

  return tf.concat([
    box_mins.slice([0, 0, 0, 1], size),
    box_mins.slice([0, 0, 0, 0], size),
    box_maxes.slice([0, 0, 0, 1], size),
    box_maxes.slice([0, 0, 0, 0], size),
  ], 3);
}

/**
 * Filters/deduplicates overlapping boxes predicted by YOLO. These
 * operations are done on CPU as AFAIK, there is no tfjs way to do it
 * on GPU yet.
 * @param {TypedArray} boxes Bounding box corner data buffer from Tensor
 * @param {TypedArray} scores Box scores data buffer from Tensor
 * @param {Number} iouThreshold IoU cutoff to filter overlapping boxes
 */
export function non_max_suppression(boxes, scores, iouThreshold) {
  // Zip together scores, box corners, and index
  const zipped = [];
  for (let i=0; i<scores.length; i++) {
    zipped.push([
      scores[i], [boxes[4*i], boxes[4*i+1], boxes[4*i+2], boxes[4*i+3]], i,
    ]);
  }
  // Sort by descending order of scores (first index of zipped array)
  const sorted_boxes = zipped.sort((a, b) => b[0] - a[0]);

  const selected_boxes = [];

  // Greedily go through boxes in descending score order and only
  // return boxes that are below the IoU threshold.
  sorted_boxes.forEach(box => {
    let add = true;
    for (let i=0; i < selected_boxes.length; i++) {
      // Compare IoU of zipped[1], since that is the box coordinates arr
      // TODO: I think there's a bug in this calculation
      const cur_iou = box_iou(box[1], selected_boxes[i][1]);
      if (cur_iou > iouThreshold) {
        add = false;
        break;
      }
    }
    if (add) {
      selected_boxes.push(box);
    }
  });

  // Return the kept indices and bounding boxes
  return [
    selected_boxes.map(e => e[2]),
    selected_boxes.map(e => e[1]),
    selected_boxes.map(e => e[0]),
  ];
}

// Convert yolo output to bounding box + prob tensors
export function yolo_head(feats, anchors, num_classes) {
  const num_anchors = anchors.shape[0];

  const anchors_tensor = tf.reshape(anchors, [1, 1, num_anchors, 2]);

  let conv_dims = feats.shape.slice(1, 3);

  // For later use
  const conv_dims_0 = conv_dims[0];
  const conv_dims_1 = conv_dims[1];

  let conv_height_index = tf.range(0, conv_dims[0]);
  let conv_width_index = tf.range(0, conv_dims[1]);
  conv_height_index = tf.tile(conv_height_index, [conv_dims[1]])

  conv_width_index = tf.tile(tf.expandDims(conv_width_index, 0), [conv_dims[0], 1]);
  conv_width_index = tf.transpose(conv_width_index).flatten();

  let conv_index = tf.transpose(tf.stack([conv_height_index, conv_width_index]));
  conv_index = tf.reshape(conv_index, [conv_dims[0], conv_dims[1], 1, 2])
  conv_index = tf.cast(conv_index, feats.dtype);

  feats = tf.reshape(feats, [conv_dims[0], conv_dims[1], num_anchors, num_classes + 5]);
  conv_dims = tf.cast(tf.reshape(tf.tensor1d(conv_dims), [1,1,1,2]), feats.dtype);

  let box_xy = tf.sigmoid(feats.slice([0,0,0,0], [conv_dims_0, conv_dims_1, num_anchors, 2]))
  let box_wh = tf.exp(feats.slice([0,0,0, 2], [conv_dims_0, conv_dims_1, num_anchors, 2]))
  const box_confidence = tf.sigmoid(feats.slice([0,0,0, 4], [conv_dims_0, conv_dims_1, num_anchors, 1]))
  const box_class_probs = tf.softmax(feats.slice([0,0,0, 5],[conv_dims_0, conv_dims_1, num_anchors, num_classes]));

  box_xy = tf.div(tf.add(box_xy, conv_index), conv_dims);
  box_wh = tf.div(tf.mul(box_wh, anchors_tensor), conv_dims);

  return [ box_xy, box_wh, box_confidence, box_class_probs ];
}

function box_intersection(a, b) {
  const w = Math.min(a[3], a[3]) - Math.max(a[1], b[1]);
  const h = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  if (w < 0 || h < 0) {
    return 0;
  }
  return w * h;
}

function box_union(a, b) {
  const i = box_intersection(a, b);
  return (a[3] - a[1]) * (a[2] - 0) + (b[3] - b[1]) * (b[2] - b[0]) - i;
}

function box_iou(a, b) {
  return box_intersection(a, b) / box_union(a, b);
}
