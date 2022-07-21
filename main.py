from analyzer import read_files, decodeResult, formatTxt
import yaml
import os
import csv

def writeFile(name, data):
    with open(name, 'w+') as file:
        writer = csv.writer(file, delimiter=' ')
        writer.writerows(data)


def generateTxtResults(params, model):
    output = os.path.join(params['dirs']['output'], model)
    gt_path = os.path.join(output, '%s_gt' % model )
    dt_path = os.path.join(output, '%s_dt' % model )

#     create output dirs to avoid error
    os.makedirs(gt_path)
    os.makedirs(dt_path)


    video_dir = params['dirs']['video']
    annotation_dir = params['dirs']['annotation']

    for target in params['data']['folders']:
        target_path = os.path.join(params['data']['root'], target)
        for name, results in read_files(target_path):
            per_image_results = decodeResult(results, target, video_dir, annotation_dir, name)
            for res, gt in per_image_results:
                det_txt = formatTxt(res, False)
                gt_txt = formatTxt(gt, True)
                print(res.id, det_txt)
                baseFile = "{}_{}.txt".format(target, res.id)

                det_filename = os.path.join(dt_path, baseFile)
                gt_filename = os.path.join(gt_path, baseFile)

                writeFile(det_filename, det_txt)
                writeFile(gt_filename, gt_txt)




if __name__ == '__main__':
    modelName = "SDS-RCNN"
    with open('config/%s.yml' % modelName ) as file:
        params = yaml.load(file, Loader=yaml.Loader)
    generateTxtResults(params, modelName)
