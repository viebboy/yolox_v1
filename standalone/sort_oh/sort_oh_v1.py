# written by Dat Tran (datran@axon.com)
from filterpy.kalman import KalmanFilter as BaseKalmanFilter
import numpy as np
from scipy.optimize import linear_sum_assignment
from loguru import logger

DEFAULT_PARAMS = {
    'estimate_box_ratio': False,
    'initial_uncertainty_factor': 10,
    'process_noise_factor': 0.01,
    'max_age_threshold': 22,
    'max_age': 20,
    'min_hits': 3,
    'min_iou_in_step1': 0.3,
    'min_iou_in_step2': 0.3,
    'min_cover_percentage': 0.3,
    'target_occlusion_threshold': 0.35,
    'object_occlusion_threshold': 0.75,
    'outside_percentage_threshold': 0.5,
}

class KalmanFilter:
    """
    implementation of modified Kalman Filter
    """
    def __init__(self, tracker_id, img_height, img_width, cur_bbox, prev_bbox, kwargs):
        self.img_height = img_height
        self.img_width = img_width
        self.img_area = img_height * img_width
        self.id = tracker_id

        # bookkeeping parameters
        self.age = 0
        self.time_since_update = 0
        self.time_since_observed = 0
        self.confidence = 0
        self.kwargs = kwargs
        self.cur_bbox = cur_bbox

        if kwargs['estimate_box_ratio']:
            self.filter = BaseKalmanFilter(dim_x=8, dim_z=4)
            # state transition matrix
            self.filter.F = np.array(
                [
                    [1, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )

            # state project matrix
            self.filter.H = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                ]
            )

            # adjust covariance matrix to give high uncertainty to initial
            # velocity
            self.filter.P[4:, 4:] *= kwargs['initial_uncertainty_factor']

            # process noise
            self.filter.Q[4:, 4:] *= kwargs['process_noise_factor']
            self.filter.Q[-1, -1] *= kwargs['process_noise_factor']

            if prev_bbox is None:
                self.filter.x[:4] = self.box_to_state(cur_bbox)
                # additional scaling if no prev box exists
                self.filter.P *= kwargs['initial_uncertainty_factor']
            else:
                cur_state = self.box_to_state(cur_bbox)
                prev_state = self.box_to_state(prev_bbox)
                self.filter.x[:4] = cur_state

                # if there is prev box, we can provide initial velocity
                self.filter.x[4:] = cur_state - prev_state

        else:
            self.filter = BaseKalmanFilter(dim_x=7, dim_z=4)
            # state transition matrix
            self.filter.F = np.array(
                [
                    [1, 0, 0, 0, 1, 0, 0,],
                    [0, 1, 0, 0, 0, 1, 0,],
                    [0, 0, 1, 0, 0, 0, 1,],
                    [0, 0, 0, 1, 0, 0, 0,],
                    [0, 0, 0, 0, 1, 0, 0,],
                    [0, 0, 0, 0, 0, 1, 0,],
                    [0, 0, 0, 0, 0, 0, 1,],
                ]
            )

            # state project matrix
            self.filter.H = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                ]
            )

            # adjust covariance matrix to give high uncertainty to initial
            # velocity
            self.filter.P[4:, 4:] *= kwargs['initial_uncertainty_factor']

            # process noise
            self.filter.Q[4:, 4:] *= kwargs['process_noise_factor']
            self.filter.Q[-1, -1] *= kwargs['process_noise_factor']

            if prev_bbox is None:
                self.filter.x[:4] = self.box_to_state(cur_bbox)
                # additional scaling if no prev box exists
                self.filter.P *= kwargs['initial_uncertainty_factor']

            else:
                cur_state = self.box_to_state(cur_bbox)
                prev_state = self.box_to_state(prev_bbox)
                self.filter.x[:4] = cur_state

                # if there is prev box, we can provide initial velocity
                self.filter.x[4:] = cur_state[:3] - prev_state[:3]


    def update(self, bbox):
        self.time_since_update = 0
        if bbox is None:
            # if no observation, adjust the velocity of area and aspect ratio
            self.filter.x[6] /= 2
            if self.kwargs['estimate_box_ratio']:
                self.filter.x[7] /= 2
            self.filter.update(None)
            self.cur_bbox = self.state_to_box(self.filter.x[:4])
        else:
            self.filter.update(self.box_to_state(bbox))
            self.time_since_observed = 0
            self.cur_bbox = bbox

    def predict(self):
        # check if velocity update leads to negative area or aspect ratio
        # adjust to 0 if needed
        if self.filter.x[6] + self.filter.x[2] <= 0:
            self.filter.x[6] *= 0.0

        if self.kwargs['estimate_box_ratio']:
            if self.filter.x[7] + self.filter.x[3] <= 0:
                self.filter.x[7] = 0.0

        # calling predict
        self.filter.predict()

        # predicted position
        self.cur_bbox = self.state_to_box(self.filter.x[:4])

        # compute some statistics
        self.age += 1
        self.time_since_update += 1
        self.time_since_observed += 1
        return self.cur_bbox

    def update_confidence(self, avg_area):
        # predicted area
        area = (self.cur_bbox[2] - self.cur_bbox[0]) * (self.cur_bbox[3] - self.cur_bbox[1])
        # compute confidence score of this tracklet
        self.confidence = min(1, self.age / (self.time_since_observed * 10) * (area / avg_area))

    def current_position(self):
        return self.cur_bbox


    def box_to_state(self, bbox):
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2

        # relative area
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        aspect_ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
        return np.array([x_center, y_center, area, aspect_ratio]).reshape(4, 1)

    def state_to_box(self, state_vec):
        x_center, y_center, area, aspect_ratio = state_vec.flatten().tolist()

        # h * w = area, w / h = aspect_ratio
        width = np.sqrt(area * aspect_ratio)
        height = area / width
        x0 = x_center - width / 2
        x1 = x_center + width / 2
        y0 = y_center - height / 2
        y1 = y_center + height / 2
        return np.array([x0, y0, x1, y1])

    def extended_position(self):
        width_ext = np.minimum(1.2, self.time_since_observed * 0.3)
        height_ext = np.minimum(0.5, self.time_since_observed * 0.1)
        width_ext = (self.cur_bbox[2] - self.cur_bbox[0]) * width_ext
        height_ext = (self.cur_bbox[3] - self.cur_bbox[1]) * height_ext
        x0 = self.cur_bbox[0] - width_ext/2
        x1 = self.cur_bbox[2] + width_ext/2
        y0 = self.cur_bbox[1] - height_ext/2
        y1 = self.cur_bbox[3] + height_ext/2
        return np.array([x0, y0, x1, y1])


class SORT_OH:
    def __init__(
        self,
        img_height,
        img_width,
        kwargs,
    ):
        self.kwargs = kwargs
        self.img_height = img_height
        self.img_width = img_width

        # internal bookkeeping
        # key is the tracker_id
        # value is an instance of KalmanTracker
        self.trackers = {}
        self.frame_count = 0
        self.tracker_count = 0
        self.prev_unmatched = []
        self.pprev_unmatched = []


    @logger.catch
    def update(self, detections):
        """
        detections must be a numpy array of size N x 5
        the first 4 columns are the x0, y0, x1, y1 coorindates
        """

        self.frame_count += 1

        # make prediction of the tracklets' positions even if there is no
        # detections
        all_pos = self._predict()

        if detections is None or detections.shape[0] == 0:
            #print('no detection, returning None from tracker')
            return None, all_pos

        #print(f'before matching: all tracker ids: {list(self.trackers.keys())}')

        # perform matching
        (
            matched_pairs,
            occluded_tracker_ids,
            unmatched_tracker_ids,
            unmatched_detection_indices
        ) = self._match_boxes_to_trackers(detections)

        #print(f'matched pairs: {matched_pairs}')
        #print(f'occluded_tracker_ids: {occluded_tracker_ids}')
        #print(f'unmatched_tracker_ids: {unmatched_tracker_idsw}')
        #print(f'unmatched_detection_indices: {unmatched_detection_indices}')

        # update the matched trackers
        for tracker_id, det_idx in matched_pairs:
            self.trackers[tracker_id].update(detections[det_idx][:4].flatten())

        # calling update for occluded trackers
        # calling update without any associated bbox
        # the velocity of area and aspect ratio is reduced by half
        for tracker_id in occluded_tracker_ids:
            self.trackers[tracker_id].update(None)

        # now we create new trackers
        unmatched_detections = [detections[idx][:4].flatten() for idx in unmatched_detection_indices]
        self._create_new_trackers(unmatched_detections)

        # remove dead tracklets
        self._remove_dead_tracklets()

        target_ids = [None for _ in range(len(detections))]
        for target_id, detection_idx in matched_pairs:
            target_ids[detection_idx] = target_id

        return target_ids, all_pos

    def _remove_dead_tracklets(self):
        id_to_remove = []
        for tracker_id, tracker in self.trackers.items():
            if tracker.time_since_update > np.minimum(
                self.kwargs['max_age_threshold'], self.kwargs['max_age'] + tracker.age / 10):
                id_to_remove.append(tracker_id)
            """
            if tracker.time_since_update > self.kwargs['max_age']:
                id_to_remove.append(tracker_id)
            """

        for id in id_to_remove:
            del self.trackers[id]

    def _create_new_trackers(self, detections):
        if self.frame_count <= self.kwargs['min_hits']:
            # if still in the first few frames
            for det in detections:
                self.tracker_count += 1
                self.trackers[f'{self.tracker_count}'] = KalmanFilter(
                    f'{self.tracker_count}',
                    self.img_height,
                    self.img_width,
                    det,
                    None,
                    self.kwargs
                )
        else:
            # if not initial frames, then check prev_unmatched and
            # prev_prev_unmatched
            if len(self.prev_unmatched) > 0 and len(self.pprev_unmatched) > 0 and len(detections) > 0:
                iou_matrix1 = self._compute_iou(self.pprev_unmatched, self.prev_unmatched)
                pprev_indices, prev_indices1 = linear_sum_assignment(-iou_matrix1)
                iou_matrix2 = self._compute_iou(self.prev_unmatched, detections)
                prev_indices2, indices = linear_sum_assignment(-iou_matrix2)

                # convert to list
                pprev_indices = pprev_indices.flatten().tolist()
                prev_indices1 = prev_indices1.flatten().tolist()
                prev_indices2 = prev_indices2.flatten().tolist()
                indices = indices.flatten().tolist()

                prev_indices = []
                for idx in prev_indices1:
                    if idx in prev_indices2:
                        prev_indices.append(idx)

                current_to_remove = []
                prev_to_remove = []

                for prev_idx in prev_indices:
                    idx_idx = prev_indices1.index(prev_idx)
                    pprev_idx = pprev_indices[idx_idx]
                    idx_idx = prev_indices2.index(prev_idx)
                    current_idx = indices[idx_idx]

                    # create new tracker
                    self.tracker_count += 1
                    self.trackers[f'{self.tracker_count}'] = KalmanFilter(
                        f'{self.tracker_count}',
                        self.img_height,
                        self.img_width,
                        detections[current_idx],
                        self.prev_unmatched[prev_idx],
                        self.kwargs
                    )
                    current_to_remove.append(current_idx)
                    prev_to_remove.append(prev_idx)

                current_to_remove.sort()
                prev_to_remove.sort()

                for idx in current_to_remove[::-1]:
                    detections.pop(idx)
                for idx in prev_to_remove[::-1]:
                    self.prev_unmatched.pop(idx)

            self.prev_unmatched = detections
            self.pprev_unmatched = self.prev_unmatched


    def _match_boxes_to_trackers(self, detections):
        """
        cascade matching
        trackers: a list of 2-element tuples: (tracker_id, bbox)
        detections: a Nx5 numpy array of coorindates and confidence score
        """

        matched_pairs = []
        unmatched_tracker_ids = []
        unmatched_detection_indices = []
        occluded_tracker_ids = []

        if len(self.trackers.keys()) == 0 or len(detections) == 0:
            unmatched_detection_indices = list(range(len(detections)))
            return (
                matched_pairs,
                occluded_tracker_ids,
                unmatched_tracker_ids,
                unmatched_detection_indices
            )

        """
        first step: match all detections to all tracklets
        """
        # get the bboxes of trackers first
        tracker_bboxes = []
        tracker_ids = []
        for id, tracker in self.trackers.items():
            tracker_bboxes.append(tracker.current_position().flatten())
            tracker_ids.append(id)

        # get the detection bboxes
        det_bboxes = [item[:4].flatten() for item in detections]

        # compute IoU matrix as the cost matrix
        iou_matrix = self._compute_iou(tracker_bboxes, det_bboxes)

        # hungarian matching
        tracker_indices, detection_indices = linear_sum_assignment(-iou_matrix)
        #print(f'matched tracker indices after matching: {tracker_indices}')

        # construct matched pairs of: tracker_id, box_index
        # taking into account minimum IoU score
        for tracklet_idx, detection_idx in zip(tracker_indices, detection_indices):
            if iou_matrix[tracklet_idx, detection_idx] >= self.kwargs['min_iou_in_step1']:
                matched_pairs.append((tracker_ids[tracklet_idx], detection_idx))
            else:
                #print(f'minimum IoU doesnt satisfy: {iou_matrix[tracklet_idx, detection_idx]}')
                unmatched_tracker_ids.append(tracker_ids[tracklet_idx])
                unmatched_detection_indices.append(detection_idx)

        # also find unmatched tracklets and detections
        for idx in range(len(tracker_ids)):
            if idx not in tracker_indices:
                unmatched_tracker_ids.append(tracker_ids[idx])

        for idx in range(len(detections)):
            if idx not in detection_indices:
                unmatched_detection_indices.append(idx)

        """
        after step1: we have matched pairs, unmatched tracklets and unmatched detections

        now perform step2 in matching if unmatched tracklets and unmatched detections are non empty:
            - first we need to find potentially occluded targets
            the bounding boxes of these targets are enlarged with respect to the time they have not been observed
            these enlarged bounding boxes are matched with unmatched detections
        """

        if len(unmatched_detection_indices) > 0 and len(unmatched_tracker_ids) > 0:
            # first find occluded trackers
            occluded_tracker_ids = self._find_occluded_trackers(unmatched_tracker_ids)
            # remove the occluded tracker from unmatched trackers
            for id in occluded_tracker_ids:
                unmatched_tracker_ids.remove(id)

            # there are occluded trackers
            # extend their bounding boxes and match them with unmatched detections
            if len(occluded_tracker_ids) > 0:
                unmatched_detections = [detections[idx].flatten()[:4] for idx in unmatched_detection_indices]
                ext_iou_matrix = self._compute_extended_iou(occluded_tracker_ids, unmatched_detections)
                tracker_indices, detection_indices = linear_sum_assignment(-ext_iou_matrix)
                trk_to_remove = []
                det_to_remove = []
                # add those matched trackers and detections to the matched_pair
                for tracker_idx, det_idx in zip(tracker_indices, detection_indices):
                    if ext_iou_matrix[tracker_idx, det_idx] >= self.kwargs['min_iou_in_step2']:
                        matched_pairs.append(
                            (
                                occluded_tracker_ids[tracker_idx],
                                unmatched_detection_indices[det_idx]
                            )
                        )
                        trk_to_remove.append(occluded_tracker_ids[tracker_idx])
                        det_to_remove.append(unmatched_detection_indices[det_idx])

                # remove them from the list of occluded trackers and unmatched detections
                for id in trk_to_remove:
                    occluded_tracker_ids.remove(id)
                for idx in det_to_remove:
                    unmatched_detection_indices.remove(idx)

        """
        now at this stage, we should have:
            - matched pairs
            - occluded trackers
            - unmatched trackers
            - unmatched detections
        """
        return matched_pairs, occluded_tracker_ids, unmatched_tracker_ids, unmatched_detection_indices

    def _compute_extended_iou(self, occluded_tracker_ids, unmatched_detections):
        iou_matrix = np.zeros((len(occluded_tracker_ids), len(unmatched_detections)))
        for idx1, tracker_id in enumerate(occluded_tracker_ids):
            for idx2, det_bbox in enumerate(unmatched_detections):
                ext_tracker_bbox = self.trackers[tracker_id].extended_position()
                tracker_bbox = self.trackers[tracker_id].current_position()
                iou_matrix[idx1, idx2] = self._ext_iou(ext_tracker_bbox, tracker_bbox, det_bbox)
        return iou_matrix

    def _ext_iou(self, ext_tracker_bbox, tracker_bbox, detection_bbox):
        trk_w = tracker_bbox[2] - tracker_bbox[0]
        trk_h = tracker_bbox[3] - tracker_bbox[1]
        xx1 = np.maximum(detection_bbox[0], ext_tracker_bbox[0])
        xx2 = np.minimum(detection_bbox[2], ext_tracker_bbox[2])
        w = np.maximum(0., xx2 - xx1)
        if w == 0:
            return 0

        yy1 = np.maximum(detection_bbox[1], ext_tracker_bbox[1])
        yy2 = np.minimum(detection_bbox[3], ext_tracker_bbox[3])
        h = np.maximum(0., yy2 - yy1)
        if h == 0:
            return 0

        wh = w * h
        area_det = (detection_bbox[2] - detection_bbox[0]) * (detection_bbox[3] - detection_bbox[1])
        area_trk = (tracker_bbox[2] - tracker_bbox[0]) * (tracker_bbox[3] - tracker_bbox[1])
        o = wh / (area_det + area_trk - wh)
        return o

    def _find_occluded_trackers(self, unmatched_tracker_ids):
        """
        given a list of tracker ids, return those that are occluded
        """

        occluded_tracker_ids = []
        if len(unmatched_tracker_ids) == 0:
            return occluded_tracker_ids

        all_tracker_ids = list(self.trackers.keys())

        # first we need to compute cover percentage between the unmatched trackers
        # and all trackers
        cp_matrix = np.zeros((len(unmatched_tracker_ids), len(all_tracker_ids)))
        for idx1 in range(len(unmatched_tracker_ids)):
            for idx2 in range(len(all_tracker_ids)):
                id1 = unmatched_tracker_ids[idx1]
                id2 = all_tracker_ids[idx2]
                if id1 != id2:
                    box1 = self.trackers[id1].current_position()
                    box2 = self.trackers[id2].current_position()
                    cp_matrix[idx1, idx2] = self._cover_percentage(box1, box2)

        # find the maximum cover percentage for each unmatched tracker
        CP = np.max(cp_matrix, axis=1).flatten()
        for cp, tracker_id in zip(CP, unmatched_tracker_ids):
            # target-target occlusion condition
            if (cp >= self.kwargs['min_cover_percentage'] and
                self.trackers[tracker_id].confidence > self.kwargs['target_occlusion_threshold']):
                occluded_tracker_ids.append(tracker_id)
            # target-object occlusion condition
            elif self.trackers[tracker_id].confidence > self.kwargs['object_occlusion_threshold']:
                occluded_tracker_ids.append(tracker_id)

        return occluded_tracker_ids


    def _compute_iou(self, boxes1, boxes2):
        iou_matrix = np.zeros((len(boxes1), len(boxes2)))
        for idx1, box1 in enumerate(boxes1):
            for idx2, box2 in enumerate(boxes2):
                iou_matrix[idx1, idx2] = self._iou(box1, box2)

        return iou_matrix

    def _iou(self, box1, box2):
        xx1 = np.maximum(box1[0], box2[0])
        xx2 = np.minimum(box1[2], box2[2])
        w = np.maximum(0., xx2 - xx1)
        if w == 0:
            return 0
        yy1 = np.maximum(box1[1], box2[1])
        yy2 = np.minimum(box1[3], box2[3])
        h = np.maximum(0., yy2 - yy1)
        if h == 0:
            return 0
        wh = w * h
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        o = wh / (area1 + area2 - wh)
        return o


    def _predict(self):
        """
        make predictions of the tracklet positions in the current frame
        """
        total_area = 0
        to_remove = []
        all_pos = []
        for tracker_id, tracklet in self.trackers.items():
            pos = tracklet.predict()
            #print(f'position of {tracker_id} is : {pos}')
            if np.any(np.isnan(pos)):
                #print('to remove because of nan')
                to_remove.append(tracker_id)
            elif self._compute_outside_percentage(pos) > self.kwargs['outside_percentage_threshold']:
                #print('to remove because of outside percentage')
                to_remove.append(tracker_id)
            else:
                total_area += (pos[2] - pos[0]) * (pos[3] - pos[1])
                all_pos.append((tracker_id, *pos))

        # remove NaN tracklets
        for tracker_id in to_remove:
            del self.trackers[tracker_id]

        # compute average area
        if len(self.trackers.keys()) > 0:
            avg_area = total_area / len(self.trackers.keys())
            # update confidence for each tracklet
            for _, tracklet in self.trackers.items():
                tracklet.update_confidence(avg_area)

        return all_pos


    def _compute_outside_percentage(self, bbox):
        # bbox: x0, y0, x1, y1
        out_x = 0
        out_y = 0
        if bbox[0] < 0:
            out_x = -bbox[0]
        if bbox[2] > self.img_width:
            out_x = bbox[2] - self.img_width
        if bbox[1] < 0:
            out_y = -bbox[1]
        if bbox[3] > self.img_height:
            out_y = bbox[3] - self.img_height
        out_a = out_x * (bbox[3] - bbox[1]) + out_y * (bbox[2] - bbox[0])
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        return out_a / area


    def _cover_percentage(self, bbox1, bbox2):
        """
        compute cover percentage of bbox1 by bbox2
        """
        xx1 = np.maximum(bbox1[0], bbox2[0])
        yy1 = np.maximum(bbox1[1], bbox2[1])
        xx2 = np.minimum(bbox1[2], bbox2[2])
        yy2 = np.minimum(bbox1[3], bbox2[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]))
        return o


