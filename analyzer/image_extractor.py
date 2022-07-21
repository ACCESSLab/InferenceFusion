from pathlib import Path
import os
from converter import read_seq

def read_txt_filenames(path):
    for file in Path(path).glob('*.txt'):
        basename = file.name.split('.')[0]
        parts = basename.split('_')
        yield {'set':parts[0], 'video':parts[1], 'frame':parts[2]}

def extract_image(file):
    video_root = '/home/redwan/Desktop/MashukBhai/InferenceFusionPaper2020/dataset/caltech/analysis'
    video_file = os.path.join(video_root, file['set'], file['video'] + '.seq')
    images, _ = read_seq(video_file)
    frame_id = int(file['frame']) - 1
    return images[frame_id]


if __name__ == '__main__':
    path = '/home/redwan/PycharmProjects/caltechBenchmark/results/F-DNN/F-DNN_dt'
    outdir = '/home/redwan/PycharmProjects/caltechBenchmark/results/F-DNN/test_images'
    os.makedirs(outdir, exist_ok=True)
    for file in read_txt_filenames(path):
        image_name = '_'.join(file.values()) + '.png'
        print(file)
        img = extract_image(file)
        with open(os.path.join(outdir, image_name), 'wb+') as f:
            f.write(img)

