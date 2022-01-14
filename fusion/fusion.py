import numpy as np
import os
import logging
def bb_to_area(bb):
    """
    width : float Rectangle width
    height : float Rectangle height
    :param bb: xmin    ymin    xmax    ymax
    :return:
    """
    width = bb[2] - bb[0]
    height = bb[3] - bb[1]
    area = width * height
    return area


def point_in_rectangle(point, box):
    top_left, bottom_right = tuple(box[:2]), tuple(box[2:])
    point_x = point[0]
    point_y = point[1]
    x_cond = (point_x >= top_left[0]) and (point_x <= bottom_right[0])
    y_cond = (point_y >= top_left[1]) and (point_y <= bottom_right[1])
    return x_cond and y_cond


def new_score2(mask, bb, score):
    # detect gan inference based on a bb
    if score <= 0.1:
        return score

    Area = bb_to_area(bb)
    box = np.array(bb, dtype=int)
    mask_bb = mask[box[1]:box[3], box[0]:box[2]]
    mask_bb[mask_bb < 255] = 0
    N = np.sum(np.sum(mask_bb / 255.0, axis=1), axis=0)
    M = (Area - N) * score  # other pixels remain default score
    prob = (M + N) / Area  # normalize the score
    return min(prob, 1.0)


def new_score(mask, bb, score):
    if score <= 0.1:
        return score

    # detect gan inference based on a bb
    Area = bb_to_area(bb)

    box = np.array(bb, dtype=int)
    mask_bb = mask[ box[0]:box[1] , box[2]:box[3] ]
    mask_bb[mask_bb < 255] = 0
    N = np.sum(np.sum(mask_bb/255, axis=1), axis=0)
    # X, Y = np.where(mask > 0)
    # N = 0
    # for x, y in zip(X.tolist(), Y.tolist()):
    #     point = (y, x)  # gan pixel
    #     if (point_in_rectangle(point, bb)):
    #         N += 1  # calculate number of pixel inside the bounding box
    surplus = 0.00  # extra area due to over approximation from bounding box
    score2 = (N + score * Area) / (2 * (1.0 - surplus) * Area)
    if score < 0.5 and score2 > score:
        print("[+] improvement ")
    return min(score2, 1.0)


def save_tree(tree, folder, indexFileDict):
    '''
    :param tree: updated TreeMemory after fusion
    :param folder: result output directory
    :param indexFileDict: a dictionary that maps tree index to filename
    :return: create txt files contain detection results in the output folder
    '''
    if not tree or not tree.size or not tree.payload:
        return
    if tree.left:
        save_tree(tree.left, folder, indexFileDict)

    out_file = os.path.join(folder, indexFileDict[tree.index])
    if len(tree.payload) % tree.size == 0 and tree.size > 0:
        incr = len(tree.payload) // tree.size
        pedestrians = [tree.payload[i:incr + i] for i in range(0, len(tree.payload), incr)]
        # print('writing ', tree.index)


        # person 0.471781 0 13 174 244
        all_objects = []
        for obj in pedestrians:
            conf, bb = obj[0], obj[1:]
            if len(bb) != 4:
                continue
            bb = list(map(str, map(int, bb)))  # list of string
            item = ['person', str(conf)] + bb
            item = " ".join(item)
            all_objects.append(item)

        all_objects = "\n".join(all_objects)

        with open(out_file, 'w+') as file:
            file.write(all_objects)
    # else:
    #     with open(out_file, 'w+') as file:
    #         pass

    if tree.right:
        save_tree(tree.right, folder, indexFileDict)

def fuse(root, index, ped_mask):
    '''
    :param root: TreeMemory
    :param index: fileIndex
    :param ped_mask: pedestrain mask from semantic
    :return: Tree node will be updated
    '''
    node = root.search(index)
    if node and len(node.payload) > 0 and len(node.payload) % node.ncol == 0:
        incr = len(node.payload) // node.size
        value = [node.payload[i:incr + i] for i in range(0, len(node.payload), incr)]
        for i, pred in enumerate(value):
            score, bb = pred[0], pred[1:]
            # FIXME choose between new_score and new_score2
            updateScore = new_score2(ped_mask, bb, score)
            # update node information
            node.payload[i * node.ncol] = updateScore

            if updateScore > score and score < 0.5:
                logging.debug("[+ fuse] found improvement %d" % index)
    else:
        # root.insert(index, 0, [])
        logging.error('[+ fuse] index = %d not found'% index)
        return index
