import numpy as np
from scipy.optimize import linear_sum_assignment

def _match_scores(scores, composition, optional_object_score_threshold=0.25):
    
    # build score matrix:
    # rows -> objects, columns -> class (repeated if multiple instances can be present)
    match_mat = []
    for mask_class, (min_instances, max_instances) in composition.items():
        match_mat.append(scores.take([mask_class] * max_instances, axis=1))
    match_mat = np.concatenate(match_mat, axis=1)
    
    # find optimal match
    ri, ci = linear_sum_assignment(match_mat, True)
    
    possibilities_checked = 0    
    res_objs = []
    res_cls = []
    
    for mask_class, (min_instances, max_instances) in composition.items():
        
        # get all matchings to current class
        ri_slice = ri[(ci >= possibilities_checked) & (ci < possibilities_checked + max_instances)]
        ci_slice = ci[(ci >= possibilities_checked) & (ci < possibilities_checked + max_instances)]
        score_slice = match_mat[ri_slice, ci_slice]
        
        # not enough matches to satisfy minimal mumber of instances
        n_matches = len(score_slice)
        if n_matches < min_instances:
            return None
        
        # sort matches ascending by score
        # take at least min_instances + optional instances with score over threshold
        idxs = np.argsort(score_slice)[::-1]
        idxs = idxs[(np.arange(n_matches) < min_instances) |
                    (score_slice[idxs] >= optional_object_score_threshold)]
    
        res_objs.extend(ri_slice[idxs])
        res_cls.extend([mask_class] * len(idxs))
    
        possibilities_checked += n_matches
        
    return res_objs, res_cls


def _resolve_subobjects(parent_class, scores, possible_compositions,
                       optional_object_score_threshold=0.25, parent_override_thresh=2.0):
    
    # TODO: cleaner handling of edge cases
    
    if not int(parent_class) in possible_compositions:
        raise ValueError(f'must provide a possible subobject composition for parent class {parent_class}')
    
    # get matching for proposed parent class
    main_match = _match_scores(scores, possible_compositions[int(parent_class)], optional_object_score_threshold)
    if main_match is not None:
        res_objs, res_cls = main_match
        score_i = np.mean(scores[res_objs, res_cls])
        main_hypothesis = res_objs, res_cls, score_i
    else:
        main_hypothesis = None
        
    # check matchings for alternative parent classes
    alt_hypothesis = None    
    for cls, comp in possible_compositions.items():
        if cls == parent_class:
            continue
            
        match = _match_scores(scores, comp, optional_object_score_threshold)
        
        if match is None:
            continue
        
        res_objs, res_cls = match
        score_i = np.mean(scores[res_objs, res_cls])
        
        # accept an alternative parent class if it has a much higher score
        if main_hypothesis is None or score_i > main_hypothesis[2] * parent_override_thresh:
            # we have no alternative or this one is even better
            if alt_hypothesis is None or score_i > alt_hypothesis[2]:
                alt_hypothesis = res_objs, res_cls, score_i, cls
    
    if main_hypothesis is None and alt_hypothesis is None:
        return None
    
    final_parent_class = parent_class if alt_hypothesis is None else alt_hypothesis[3]
    subobject_assignment = ((main_hypothesis if alt_hypothesis is None else alt_hypothesis)[0],
                            (main_hypothesis if alt_hypothesis is None else alt_hypothesis)[1])
    
    return final_parent_class, subobject_assignment


def postproc_multimask(inst, possible_compositions,
                       optional_object_score_threshold=0.25, parent_override_thresh=2.0,
                       cls_offset = {1:1, 2:3}):
    # TODO: remove cls_offset?
    
    boxes = list(inst.pred_boxes)
    boxes = [tuple(box.cpu().numpy()) for box in boxes]

    masks = list(inst.pred_masks)
    masks = [mask.cpu().numpy() for mask in masks]

    scores = list(inst.scores)
    scores = [score.cpu().numpy() for score in scores]

    classes = list(inst.pred_classes)
    labels = [cls.cpu().numpy() for cls in classes]

    if len(boxes) == 0:
        return [], None
    
    #boxes, scores, labels, masks = check_iou(boxes, scores, labels, masks)
        
    tensorboxes = np.stack(list(boxes))

    x_centroids = tensorboxes[:,0] + (tensorboxes[:,2] - tensorboxes[:,0])//2
    y_centroids = tensorboxes[:,1] + (tensorboxes[:,3] - tensorboxes[:,1])//2

    pts = np.stack([x_centroids, y_centroids], axis=1)

    things = []
    excluded_singles = []
    for n in range(len(boxes)):
        if int(labels[n]) in possible_compositions and scores[n] > 0.:
            thing = {'box': boxes[n], 'class': labels[n], 'score': scores[n], 'objects':[], 'id':n}

            ll = [boxes[n][0],boxes[n][1]]
            ur = [boxes[n][2],boxes[n][3]]
                        
            inidx = np.where(np.all(np.logical_and(ll <= pts, pts <= ur), axis=1) == True)
            
            # collect mean mask scores for each class
            mask_scores = []
            accepted_box_idxs = []
            
            for j, boxidx in enumerate(inidx[0]):
                if scores[boxidx] > 0.9 and labels[boxidx] == 0:
                    fullmask = masks[n][:,int(boxes[boxidx][1]):int(boxes[boxidx][3]),int(boxes[boxidx][0]):int(boxes[boxidx][2])]
                    thresh_mask = masks[boxidx][1][int(boxes[boxidx][1]):int(boxes[boxidx][3]),int(boxes[boxidx][0]):int(boxes[boxidx][2])] > 0.
                    
                    k = np.mean(fullmask[:,thresh_mask],axis=(1))
                    mask_scores.append(k)
                    accepted_box_idxs.append(j)
            
            if len(mask_scores) < 1:
                continue
                
            mask_scores = np.stack(mask_scores, axis=0)

            res = _resolve_subobjects(int(labels[n]), mask_scores, possible_compositions,
                                      optional_object_score_threshold=optional_object_score_threshold,
                                      parent_override_thresh=parent_override_thresh)
            
            if res is None:
                continue # TODO: handle
                
            final_parent_class, subobject_assignment = res
            
            thing['class'] = final_parent_class
            
            for obj, cls in zip(*subobject_assignment):
                
                off = 0 if int(labels[n]) not in cls_offset else cls_offset[int(labels[n])]
                
                
                subthing = {'box': boxes[inidx[0][accepted_box_idxs[obj]]], 'class': cls-off,
                            'testscore': mask_scores[obj,cls], 'score': scores[n]}
                thing['objects'].append(subthing)
            things.append(thing)
    
    fullmask = np.zeros(masks[0].shape[1:])
    for n in range(len(boxes)):
        if labels[n] == 0 and scores[n] > 0.9:# and n not in excluded_singles:
            fullmask[int(boxes[n][1]):int(boxes[n][3]),int(boxes[n][0]):int(boxes[n][2])][masks[n][1][int(boxes[n][1]):int(boxes[n][3]),int(boxes[n][0]):int(boxes[n][2])] > 0.5] = n+1
            
            thing = {'box': boxes[n], 'class': 0, 'score': scores[n], 'id':n}
            things.append(thing)
    
    return things, fullmask
