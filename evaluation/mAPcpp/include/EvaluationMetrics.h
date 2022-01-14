//
// Created by redwan on 12/6/21.
//

#ifndef MAP_EVALUATIONMETRICS_H
#define MAP_EVALUATIONMETRICS_H
#include "FileParser.h"
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include <iomanip>
#include <functional>
#include <numeric>
#include <future>

#define MIN_CONF_SCORE (0.5)


using namespace std;


template<>
struct std::hash<shared_ptr<Object>>
{
    std::size_t operator()(shared_ptr<Object> const& s) const noexcept
    {
        string str_name = s->filename;
        for (int i = 0; i < 4; ++i) {
            str_name += to_string(s->box[i]);
        }
        return std::hash<std::string>{}(str_name);
    }
};


class EvaluationMetrics{
    using vec = vector<shared_ptr<Object>>;
    using OBJS = unordered_map<string, vec>;
    using JOIN_OBJS = unordered_map<string, pair<vec, vec>>;
public:
    EvaluationMetrics(const string& gt, const string& dt, const string& ext, double iou):iou_threshold_(iou)
    {
        detector_ = make_unique<FileParser>(dt, ext);
        ground_truth_ = make_unique<FileParser>(gt, ext);
    }

    JOIN_OBJS combine_results()
    {

        OBJS gtObjects, dtObjects;
        for (auto &obj: ground_truth_->read())
            gtObjects[obj->filename].push_back(obj);
        for (auto &obj:detector_->read())
            dtObjects[obj->filename].push_back(obj);

        cout << "[Detection]: total file = " << dtObjects.size() << endl;
        cout << "[Ground truth]: total file = " << gtObjects.size() << endl;

        return combine_results(gtObjects, dtObjects);
    }

    size_t total_tp()
    {
        return tp_.size();
    }

    size_t total_fp()
    {
        return fp_.size();
    }

    size_t size()
    {
        return total_index_;
    }

    double log_miss_rate()
    {
        return 100.0 * fu_lamr_.get();
    }

    void compute(JOIN_OBJS& data)
    {


        int idx = 0;

        // Calculate the AP for each class

        for(auto & it : data) {
            auto detections = it.second.second;
            auto groundTruth = it.second.first;

            for(auto cat:categories_)
            {

                // find detection results for corresponding file
                unordered_set<std::size_t> visited;

                for (auto detection: detections)
                {
                    // assign detection-results to ground truth object if any
                    // open ground-truth with that file_id


                    double ovmax = -1;
                    weak_ptr<Object> gt_match;
                    // find detection-results of that class
                    if (cat.compare(detection->category) == 0)
                    {

                        if(detection->conf_score < MIN_CONF_SCORE) continue;

                        // paralle implementation of iou
                        vector< future<double> >parallel;
                        // Assign detection-results to ground-truth objects
                        for(auto gt_compare:groundTruth){
                            if (gt_compare->category.compare(detection->category) == 0
                                && gt_compare->filename.compare(detection->filename) == 0
                            )
                            {
                                double iou = compute_iou(gt_compare, detection);
                                if(iou > ovmax)
                                {
                                    ovmax = iou;
                                    gt_match = gt_compare;
                                }
                            }

                        }
                    // ------------------------------compute tp/fp---------------------------------------------

                        if(ovmax >= iou_threshold_)
                        {
                            // if this gt match never used before
                            if (std::shared_ptr<Object> spt = gt_match.lock())
                            {
                                size_t hval = hash<shared_ptr<Object>>{}(spt);
                                if(visited.count(hval) == 0)
                                {
                                    tp_[idx] = 1;
                                    if(count_true_positives_.count(cat) == 0)
                                        count_true_positives_[cat] = 0;
                                    count_true_positives_[cat] += 1;
                                    visited.insert(hval);
//                                cout << count_true_positives[cat] << endl;
                                }
                                else
                                    fp_[idx] = 1;
                            }
                            else
                                cerr << "weak_ptr not found !" << endl;
                        }
                        else
                            fp_[idx] = 1;

                    }
                    // increment index here
                    ++idx;
                    // end of detection
                }
                // end of gt
            }

            // end of data
        }
        total_index_ = idx;
    }

    /* @brief this function will compute average precision for a specific class \par
     * this function will also asynchronously invoke log average miss rate function \par
     * @param class_name: class label for the object
     */

    double compute_ap(const string& class_name)
    {
        size_t total_length = size();
        auto cum_sum = [](vector<size_t>& cfm)
        {
            double cumsum = 0;
            for (int i = 0; i < cfm.size(); ++i) {
                size_t val = cfm[i];
                cfm[i] += cumsum;
                cumsum += val;
            }
        };

        // compute log average miss rate in advance so that we can just return it when it called later
        std::promise<vector<double>> promise_recall, promise_precision;
        std::future<vector<double>> fu_recall = promise_recall.get_future();
        std::future<vector<double>> fu_precision = promise_precision.get_future();
        fu_lamr_ = std::async(std::launch::async, &EvaluationMetrics::async_compute_log_average_miss_rate, this, std::ref(fu_recall), std::ref(fu_precision));


        vector<size_t> tp, fp;

        for (int i = 0; i < total_length; ++i) {
            if(tp_.count(i) != 0)
                tp.push_back(tp_[i]);
            if(fp_.count(i) != 0)
                fp.push_back(fp_[i]);
        }


        while (fp.size() < total_length)
            fp.push_back(0);

        while (tp.size() < total_length)
            tp.push_back(0);

        sort(tp.begin(), tp.end(), greater<size_t>());
        sort(fp.begin(), fp.end(), less<size_t>());
        cum_sum(fp);
        cum_sum(tp);
        // -------------------------------------compute precision/recall----------------------------------

        vector<double> recall, precision;
        copy(tp.begin(), tp.end(), back_inserter(recall));
        copy(tp.begin(), tp.end(), back_inserter(precision));
        for (int i = 0; i < total_length; ++i)
        {
            recall[i] = double (tp[i]) / double (gt_counter_per_class_[class_name]);
            precision[i] = double (tp[i]) / double (fp[i] + tp[i]);
        }

        // update promises
        promise_precision.set_value(precision);
        promise_recall.set_value(recall);

        return 100.0 * compute_voc_ap(recall, precision);

    }

protected:

    double compute_voc_ap(const vector<double>& rec_in, const vector<double>& prec_in) const
    {
        vector<double> rec, prec;

        // insert 0 at the beginning for both recall and precision
        rec.push_back(0.0);
        prec.push_back(0.0);

        copy(rec_in.begin(), rec_in.end(), back_inserter(rec));
        copy(prec_in.begin(), prec_in.end(), back_inserter(prec));

        // insert 1 at the end of recall but 0 at the end of precision
        rec.push_back(1.0);
        prec.push_back(0.0);
        /*
         * This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
         */
        for (int i = prec.size() - 2; i >=0 ; --i) {
            prec[i] = max(prec[i], prec[i+1]);
        }
        // This part creates a list of indexes where the recall changes
        vector<double> i_list;
        for (int i = 1; i < rec.size(); ++i) {
            if(rec[i] != rec[i-1])
                i_list.push_back(i);
        }
        double ap = 0.0;
        for(auto i: i_list)
            ap += ((rec[i]-rec[i-1])*prec[i]);

        return ap;
    }
    /*
     * log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.

        output:
                lamr | log-average miss rate

        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
     */

    double compute_log_average_miss_rate(const vector<double>& recall, const vector<double>& precision) const
    {
        assert(precision.size() > 0 && "precision size needs to be greater than zero");
        vector<double> fppi(precision.size()), mr(recall.size());
        std::transform(precision.begin(), precision.end(), fppi.begin(), [](const double& val){return 1- val;});
        std::transform(recall.begin(), recall.end(), mr.begin(), [](const double& val){return 1- val;});

        vector<double> fppi_tmp, mr_tmp;
        fppi_tmp.push_back(-1.0);
        mr_tmp.push_back(1.0);

        copy(fppi.begin(), fppi.end(), back_inserter(fppi_tmp));
        copy(mr.begin(), mr.end(), back_inserter(mr_tmp));

        vector<double> ref{0.01, 0.01778279, 0.03162278, 0.05623413, 0.1, 0.17782794, 0.31622777, 0.56234133, 1.};

        for (int i = 0; i < 9; ++i) {
            double ref_i = ref[i];
            //np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
            int index = -1;
            for (int j = 0; j < fppi_tmp.size(); ++j) {
                if(fppi_tmp[j] <= ref_i)
                    index = j;
            }
            // log(0) is undefined, so we use the np.maximum(1e-10, ref)
            ref[i] = mr_tmp[index] != 0 ? mr_tmp[index] : 1e-10;
        }


        vector<double> log_ref(ref.size());
        // compute log(ref)
        std::transform(ref.begin(), ref.end(), log_ref.begin(), [](const double& val){return log(val);});

        // compute mean(log_ref)
        double sum = std::accumulate(log_ref.begin(),log_ref.end(), 0.0);
        double mean = sum / log_ref.size();

        // log miss rate
        double lamr = exp(mean);

        return lamr;

    }

    double async_compute_log_average_miss_rate(std::future<vector<double>>& recall, std::future<vector<double>>& precision) const
    {
        return compute_log_average_miss_rate(recall.get(), precision.get());;
    }

    double compute_iou(shared_ptr<Object>& gt, shared_ptr<Object> dt) const
    {
        auto bbgt = gt->box;
        auto bb = dt->box;

        array<double, 4> bi = {max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]),
                               min(bb[3], bbgt[3])};
        double iw = bi[2] - bi[0] + 1;
        double ih = bi[3] - bi[1] + 1;
        if (iw > 0 && ih > 0) {
            double ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                     + 1) * (bbgt[3] - bbgt[1] + 1) -
                        iw * ih;
            double ov = iw * ih / ua;
            return ov;
        }
        return -1.0;
    }


    /* @brief group ground truth box and corresponding detection results
     * each file has multiple objects but filename should be unique since it represents which image we are comparing.
     * This grouping will allow us to remove mismatch between ground truths and detections
     * @param grObjects: all ground truth objects
     * @pram dtObjects: all detection objects
     */
    JOIN_OBJS combine_results(const OBJS& gtObjects, OBJS& dtObjects)
    {
        // use unordered_map to join objects
        JOIN_OBJS results;
        // find common objects
        for(auto it = gtObjects.begin(); it != gtObjects.end(); ++it)
        {
            // check whether ground truth filename exists in detection results
            // in some cases, detector might not detect objects even if it exists in ground truth
            if(dtObjects.find(it->first) != dtObjects.end())
            {
                // we are interested in those results that are comparable with ground truth
                auto detections = dtObjects[it->first];

                for(auto &obj:it->second)
                {
                    categories_.insert(obj->category);
                    if(gt_counter_per_class_.count(obj->category) == 0)
                        gt_counter_per_class_[obj->category] = 1;
                    else
                        gt_counter_per_class_[obj->category] += 1;
                }

                if(!detections.empty())
                {                    // a perfect match is found, store it
                    results[it->first] = make_pair(it->second, detections);
                }
            }
        }
        return results;
    }

private:
    double iou_threshold_;
    unique_ptr<FileParser> detector_, ground_truth_;
    unordered_set<string> categories_;
    unordered_map<string, long> gt_counter_per_class_;
    unordered_map<long, long> tp_, fp_;
    unordered_map<string, long> count_true_positives_;
    size_t total_index_;
    std::future<double> fu_lamr_;
};

#endif //MAP_EVALUATIONMETRICS_H
