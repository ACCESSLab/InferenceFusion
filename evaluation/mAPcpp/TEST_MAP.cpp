//
// Created by redwan on 12/6/21.
//

#include <gtest/gtest.h>
#include "EvaluationMetrics.h"

TEST(mAPCppTest, TotalTruePositive) {
    string groundTruth("../data/gt"), detection("../data/dt"), ext(".txt");
    EvaluationMetrics eval(groundTruth, detection, ext);
    auto jobjects = eval.combine_results();
    eval.compute(jobjects);
    ASSERT_EQ (1629, eval.total_tp());
}

TEST (mAPCppTest, TotalFalsePositive) {
    string groundTruth("../data/gt"), detection("../data/dt"), ext(".txt");
    EvaluationMetrics eval(groundTruth, detection, ext);
    auto jobjects = eval.combine_results();
    eval.compute(jobjects);
    ASSERT_EQ (39850, eval.total_fp());
}

TEST (mAPCppTest, RecallParameters) {
    string groundTruth("../data/gt"), detection("../data/dt"), ext(".txt");
    EvaluationMetrics eval(groundTruth, detection, ext);
    auto jobjects = eval.combine_results();
    eval.compute(jobjects);
    ASSERT_EQ (84.05, eval.compute_ap("person"));
//    ASSERT_EQ (1629, eval.compute_recall());
}



TEST (mAPCppTest, sizeTest) {
    string groundTruth("../data/gt"), detection("../data/dt"), ext(".txt");
    EvaluationMetrics eval(groundTruth, detection, ext);
    auto jobjects = eval.combine_results();
    eval.compute(jobjects);
    ASSERT_EQ (41479, eval.size());
}
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}