import pathlib
from collections import defaultdict 
from .converter import read_seq, read_vbb
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import io
from collections import namedtuple
import os 
import csv 

BB = namedtuple('BoundingBox', 'orig width height conf')
Detection = namedtuple('Detection','id image boxes tag')

def read_files(path):
    folder = pathlib.Path(path)
    for item in folder.glob('*.txt'):
        result = defaultdict(list)
        empty = True
        with open(item) as csvfile:
            for line in csvfile.readlines():
                line = line.strip()
                data = line.split(',')
                data = list(map(float, data))
                if data[-1] >= 0.1:
                    result[int(data[0])].append(data[1:])
                    empty = False
        if not empty:
            yield item.name.split('.')[0], result

def read_block(name, frame, img, vbb_data):
    labels = vbb_data['frames'][frame]
    boxes = []
    for label in labels:
        x0, y0, width, height = label['pos']
        org = (x0, y0)
        conf = 0.5
        bb = BB(orig=org, width=width, height=height, conf=conf)
        boxes.append(bb)

    detection = Detection(id=f"{name}_{frame}", image=img, boxes=boxes, tag='red')
    return detection
        
def decodeResult(results, target, video_dir, annotation_dir, name):
    video_file = video_dir.format(target, name)
    if os.path.exists(video_file):
        images, _ = read_seq(video_file)
    annotation = annotation_dir.format(target, name)
    vbb_data = read_vbb(annotation)
    for frame, detections in results.items():
        # print(f"[+] name = {name} frame = {frame} : detection = {detections}")
        img = None 
        if os.path.exists(video_file):
            buffer = images[frame - 1]
            img = Image.open(io.BytesIO(buffer))
        results = []
        for person in detections:
            x0, y0, width, height, conf = person
            org = (x0, y0)
            bb = BB(orig=org, width=width, height=height, conf=conf)
            results.append(bb)
        detection = Detection(id=f"{name}_{frame}", image=img, boxes=results, tag='green')
        groundTruth = read_block(name, frame, img, vbb_data)
        yield detection, groundTruth

def show_image(detection):
    for res, gt in detection:
        print(res.id)
        plt.axis('off')
        plt.imshow(res.image, origin='upper')
        ax = plt.gca()
        for bb in res.boxes:
            rect = Rectangle(bb.orig, bb.width, bb.height, alpha = min(bb.conf, 1.0), color = res.tag)
            ax.add_patch(rect)
        
        for bb in gt.boxes:
            rect = Rectangle(bb.orig, bb.width, bb.height, alpha = min(bb.conf, 1.0), color = gt.tag)
            ax.add_patch(rect)
        plt.show()

def formatTxt(detection:Detection, isGroundTruth:bool):
    out = []
    for bb in detection.boxes:
        x0, y0 = bb.orig
        w, h = bb.width, bb.height
        conf = bb.conf

        x1 = int(x0 + w)
        y1 = int(y0 + h)
        x0, y0 = int(x0), int(y0)

        conf = "{:.4f}".format(min(1.0, conf))
        _res = ['person', conf, x0, y0, x1, y1] if not isGroundTruth else ['person', x0, y0, x1, y1]
        out.append(_res)
    return out



def save_results(target, video_dir, annotation_dir):
    os.makedirs('results/gt/%s' % target, exist_ok=True)
    os.makedirs('results/dt/%s' % target, exist_ok=True)



    for name, results in read_files(target):
        per_image_results = decodeResult(results, target, video_dir, annotation_dir, name)
        for res, gt in per_image_results:
            norm_res = formatTxt(res, False)
            norm_gt = formatTxt(gt, True)
            print(res.id, norm_res)
            baseFile = "{}.txt".format(res.id)
            with open("results/dt/%s/%s" % (target, baseFile), 'w+') as file:
                writer = csv.writer(file, delimiter=' ')
                writer.writerows(norm_res)

            with open("results/gt/%s/%s" % (target, baseFile), 'w+') as file:
                writer = csv.writer(file, delimiter=' ')
                writer.writerows(norm_gt)

            

def sampleImage():
    for name, results in read_files(target):
        per_image_results = decodeResult(results, video_dir, annotation_dir, name)
        show_image(per_image_results)

if __name__ == "__main__":
    target = 'set08'
    video_dir = "/home/redwan/Desktop/MashukBhai/InferenceFusionPaper2020/dataset/caltech/analysis/{}/{}.seq"
    annotation_dir = "/home/redwan/Desktop/MashukBhai/InferenceFusionPaper2020/dataset/caltech/data/annotations/{}/{}.vbb"
    
    save_results(target, video_dir, annotation_dir)
        # show_image(groundTruth)
        