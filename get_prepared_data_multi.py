# coding=utf-8
"""Given the final dataset or the anchor dataset, compile prepared data."""

import argparse
import json
import os
import operator
import pickle
import numpy as np
from tqdm import tqdm

def do_All_Prepared(o_bboxes,drop_frame=12):#12 or 10
    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def convert_bbox(bbox):
        x, y, w, h = bbox
        return [x, y, x + w, y + h]

    def get_feet(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, y2)


    def get_frame_data(o_bboxes):
        # bboxes = filter_neg_boxes(o_bboxes)
        bboxes = o_bboxes
        frame_data = {}  # frame_idx -> data
        for one in bboxes:
            if one["frame_id"] not in frame_data:
                frame_data[one["frame_id"]] = []
            frame_data[one["frame_id"]].append(one)

        return frame_data

    def filter_neg_boxes(bboxes):
        new_bboxes = []
        for bbox in bboxes:
            x, y, w, h = bbox["bbox"]
            coords = x, y, x + w, y + h
            bad = False
        for o in coords:
            if o < 0:
                bad = True
        if not bad:
            new_bboxes.append(bbox)
        return new_bboxes
    
    obs_length = 8
    class2classid = {"Person": 0,"Car": 1,}
    videoname = 1
    # traj_path = os.path.join('.', f'./traj_2.5fps/{videoname}')
    # person_box_path = os.path.join('.', f'./anno_person_box/{videoname}')
    # other_box_path = os.path.join('.', f'./anno_other_box/{videoname}')
    
    drop_frame = drop_frame
    
    # multi-future pred starts at 124/102
    # we want the obs to be 3.2 sec long
    if drop_frame == 12:
        frame_range = (40, 125) # range(40, 125, 12)
        start_frame, end_frame = 40, 125
    else:
        frame_range = (32, 103),  # range(32, 103, 10)
        start_frame, end_frame = 32, 103
    
    # 1. first pass, get the needed frames
    frame_data = get_frame_data(o_bboxes)
    frame_idxs = sorted(frame_data.keys())
    # print(frame_idxs)
    # assert frame_idxs[0] == 0
    needed_frame_idxs = frame_idxs[start_frame::drop_frame]
    
    assert len(needed_frame_idxs) > obs_length, (needed_frame_idxs, start_frame)
    obs_frame_idxs = needed_frame_idxs[:obs_length]
    
    # 2. gather data for each frame_idx, each person_idx
    traj_data = []  # [frame_idx, person_idx, x, y]
    person_box_data = {}  # (frame_idx, person_id) -> boxes
    other_box_data = {}  # (frame_idx, person_id) -> other boxes + boxclasids
    obs_x_agent_traj = []
    for frame_idx in obs_frame_idxs:
        box_list = frame_data[frame_idx]
        # filter out negative boxes
        box_list.sort(key=operator.itemgetter("track_id"))
        for i, box in enumerate(box_list):
            class_name = box["class_name"]
            track_id = box["track_id"]
            is_x_agent = box["is_x_agent"]
            bbox = convert_bbox(box["bbox"])
            if class_name == "Person":
                new_frame_idx = frame_idx - start_frame
                person_key = "%d_%d" % (new_frame_idx, track_id)

                x, y = get_feet(bbox)
                traj_data.append((new_frame_idx, float(track_id), x, y))
                if int(is_x_agent) == 1:
                    obs_x_agent_traj.append((new_frame_idx, float(track_id), x, y))

                person_box_data[person_key] = bbox

                all_other_boxes = [convert_bbox(box_list[j]["bbox"])
                                    for j in range(len(box_list)) if j != i]
                all_other_boxclassids = [class2classid[box_list[j]["class_name"]]
                                            for j in range(len(box_list)) if j != i]

                other_box_data[person_key] = (all_other_boxes, all_other_boxclassids)

    # now we save all the multi future paths for all agent.
    multifuture_data = {}  # videoname -> {"x_agent_traj", "all_boxes"}
    frame_data = get_frame_data(o_bboxes)
    frame_idxs = sorted(frame_data.keys())
    # assert frame_idxs[0] == 0
    # 1. first pass, get the needed frames
    needed_frame_idxs = frame_idxs[start_frame::drop_frame]

    # assert len(needed_frame_idxs) > obs_length, (needed_frame_idxs, start_frame)
    pred_frame_idxs = needed_frame_idxs[obs_length:]

    x_agent_traj = []
    all_boxes = []
    for frame_idx in pred_frame_idxs:
        box_list = frame_data[frame_idx]
        box_list.sort(key=operator.itemgetter("track_id"))
        for i, box in enumerate(box_list):
            class_name = box["class_name"]
            track_id = box["track_id"]
            is_x_agent = box["is_x_agent"]
            bbox = convert_bbox(box["bbox"])

            new_frame_idx = frame_idx - start_frame
            if int(is_x_agent) == 1:
                x, y = get_feet(bbox)
                x_agent_traj.append((new_frame_idx, track_id, x, y))

            all_boxes.append((new_frame_idx, class_name, is_x_agent,
                            track_id, bbox))
    multifuture_data[videoname] = {
        "x_agent_traj": x_agent_traj, # future
        "all_boxes": all_boxes,
        "obs_traj": obs_x_agent_traj,
    }
    # print(traj_data)
    # print(multifuture_data)
    # print('end get_prepared_data_multifuture')
    return traj_data, person_box_data, other_box_data, multifuture_data

# do_All_Prepared(o_bboxes)