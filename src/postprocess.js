// Heavily derived from YAD2K (https://github.com/allanzelener/YAD2K)
import * as tf from '@tensorflow/tfjs';

export const YOLO_ANCHORS = tf.tensor2d([
  [0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434],
  [7.88282, 3.52778], [9.77052, 9.16828],
]);

export function yolo_filter_boxes(
  boxes,
  box_confidence,
  box_class_probs,
  threshold
) {
  const box_scores = tf.mul(box_confidence, box_class_probs);
  const box_classes = tf.argMax(box_scores, -1);
  const box_class_scores = tf.max(box_scores, -1);

  // Many thanks to @jacobgil
  // Source: https://github.com/ModelDepot/tfjs-yolo-tiny/issues/6#issuecomment-387614801
  const prediction_mask = tf.greaterEqual(box_class_scores, tf.scalar(threshold)).as1D();

  const N = prediction_mask.size
  // linspace start/stop is inclusive.
  const all_indices = tf.linspace(0, N - 1, N).toInt();
  const neg_indices = tf.zeros([N], 'int32');
  const indices = tf.where(prediction_mask, all_indices, neg_indices);

  return [
    tf.gather(boxes.reshape([N, 4]), indices),
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

export function box_intersection(a, b) {
  const w = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  const h = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  if (w < 0 || h < 0) {
    return 0;
  }
  return w * h;
}

export function box_union(a, b) {
  const i = box_intersection(a, b);
  return (a[3] - a[1]) * (a[2] - a[0]) + (b[3] - b[1]) * (b[2] - b[0]) - i;
}

export function box_iou(a, b) {
  return box_intersection(a, b) / box_union(a, b);
}
