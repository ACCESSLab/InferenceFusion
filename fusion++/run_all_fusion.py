import yaml
import subprocess
import shlex
import os
import json

class ExecuteFusion:
    def __init__(self, pkl, detection_dir, label_dir, img_dir, out_dir):
        self.command = "python3 fusion++.py --pkl {} --detection_dir {} \
        --label_dir {} --img_dir {} --out_dir {}".format(pkl, detection_dir, label_dir,
                                                         img_dir, out_dir)

    def run(self):
        process = subprocess.Popen(shlex.split(self.command), stdout=subprocess.PIPE)
        while True:
            output = process.stdout.readline().decode("utf-8")
            if output == '' and process.poll() is not None:
                break
            if output:
                print("[+ Evaluation]: ", output.strip())
            if "terminated" in output:
                break
        rc = process.poll()
        # print("[-] finished subprocess ", command)
        return rc

    def __repr__(self):
        return self.command
class mAP_calculate:
    def __init__(self, gt_path, det_path):
        self.command = "mAP --gt {} --dt {} --color_out 0".format(gt_path, det_path)
    def run(self):
        process = subprocess.Popen(shlex.split(self.command), stdout=subprocess.PIPE)
        result = {}
        while True:
            output = process.stdout.readline().decode("utf-8")
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                if line.startswith("[AP for person]"):
                    AP = line.split(':')[-1].strip()
                    result['AP'] = float(AP)
                if line.startswith("[Log Miss Rate for person]"):
                    MR = line.split(':')[-1].strip()
                    result['MR'] = float(MR)
                # print("[+ Evaluation]: ", output.strip())
            if "terminated" in output:
                break
        rc = process.poll()
        # print("[-] finished subprocess ", command)
        return result


with open('fusion_param.yaml') as file:
    parameter = yaml.load(file, yaml.FullLoader)
def mAP_calculate_with_fusion():
    global_path = 'all_results/{}/{}/{}'
    save_json_dic = {}
    for dataset, payload in parameter['dataset'].items():
        # dataset > cityperson
        img_dir = payload['image']
        label_dir = payload['label']
        save_json_dic[dataset] = {}
        for network, detection_dir in payload['networks'].items():
            save_json_dic[dataset][network] = {}
            for sem_network, pkl in payload['semantic_map'].items():
                out_dir = global_path.format(dataset, network, sem_network)
                mAP_execute = mAP_calculate(label_dir,out_dir)
                save_json_dic[dataset][network][sem_network] = mAP_execute.run()
                print(save_json_dic[dataset][network][sem_network])
                # print(execute)

    with open("all_fusion_results.json", 'w+') as file:
        json.dump(save_json_dic, file, indent=4)
def run_all_fusion():
    global_path = 'all_results/{}/{}/{}'

    for dataset, payload in parameter['dataset'].items():
        # dataset > cityperson
        img_dir = payload['image']
        label_dir = payload['label']
     
        for network, detection_dir in payload['networks'].items():
          
            for sem_network, pkl in payload['semantic_map'].items():
                out_dir = global_path.format(dataset, network, sem_network)
              
                execute = ExecuteFusion(pkl, detection_dir, label_dir, img_dir, out_dir)
                execute.run()
              
 

def map_baseline():
    save_json_dic = {}
    for dataset, payload in parameter['dataset'].items():
        # dataset > cityperson
        img_dir = payload['image']
        label_dir = payload['label']
        save_json_dic[dataset] = {}
        for network, detection_dir in payload['networks'].items():
            mAP_execute = mAP_calculate(label_dir,detection_dir)
            save_json_dic[dataset][network] = mAP_execute.run()
            
         
    with open("all_baseline_results.json", 'w+') as file:
        json.dump(save_json_dic, file, indent=4)


if __name__ == '__main__':
    map_baseline()


    




