#include <iostream>
#include "EvaluationMetrics.h"
#include <boost/program_options.hpp>
using namespace boost::program_options;



int main(int argc, char **argv) {


    auto desc = std::make_unique<options_description>("Options");
    desc->add_options()
            ("help,h", "Help screen")
            ("gt", value<std::string>(), "dir for ground truth")
            ("dt", value<std::string>(), "dir for detection results")
            ("iou", value<double>()->default_value(0.5), "default iou threshold value 0.5 ")
            ("color_out", value<bool>()->default_value(true), "default std outs are highlighted with a color ")
            ("class", value<std::string>()->default_value("person"), "default class label for computing AP & MR is person")
            ("ext", value<std::string>()->default_value(".txt"), "default extension .txt");
    auto vm = make_unique<variables_map>();
    store(parse_command_line(argc, argv, *desc), *vm);
    notify(*vm);

    if(vm->count("help"))
    {
        std::cout<<*desc<<std::endl;
        return 1;
    }

    string groundTruth, detection, ext;
    double iou;
    auto vms = *vm.get();
    groundTruth = vms["gt"].as<std::string>();
    detection = vms["dt"].as<std::string>();
    ext = vms["ext"].as<std::string>();
    iou = vms["iou"].as<double>();

    string HIGHLIGHT_START  = (vms["color_out"].as<bool>()) ? "\033[1m\033[36m" :"";
    string HIGHLIGHT_END    = (vms["color_out"].as<bool>()) ? "\033[0m" :"";

    std::cout << HIGHLIGHT_START << "Evaluation Metrics started !" << HIGHLIGHT_END << std::endl;
    EvaluationMetrics eval(groundTruth, detection, ext, iou);
    auto jobjects = eval.combine_results();
    eval.compute(jobjects);

    string label = vms["class"].as<std::string>();


    cout << HIGHLIGHT_START << "[total tp]: "<< HIGHLIGHT_END << eval.total_tp() << endl;
    cout << HIGHLIGHT_START << "[total fp]: "<< HIGHLIGHT_END << eval.total_fp() << endl;
    cout << HIGHLIGHT_START << "[AP for " << label <<"]: "<< HIGHLIGHT_END << eval.compute_ap(label) << endl;
    cout << HIGHLIGHT_START << "[Log Miss Rate for person]: "<< HIGHLIGHT_END << eval.log_miss_rate() << endl;

    return 0;
}
