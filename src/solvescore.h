//
// Created by richardzvonek on 11/21/20.
//

#ifndef ANOII2020_SOLVESCORE_H
#define ANOII2020_SO

struct SolveScore {
  SolveScore(const double falsePositives,
             const double falseNegatives,
             const double truePositives,
             const double trueNegatives) :
      falsePositives_(falsePositives),
      falseNegatives_(falseNegatives),
      truePositives_(truePositives),
      trueNegatives_(trueNegatives),
      accuracy_(static_cast<double> (truePositives + trueNegatives) / static_cast<double>(truePositives + trueNegatives + falsePositives + falseNegatives)),
      sensitivity_(static_cast<double>(truePositives) / static_cast<double>(truePositives + falseNegatives)),
      f1Score_(2.f * (accuracy_ * sensitivity_) / (accuracy_ + sensitivity_)) {
  }
  
  SolveScore(const SolveScore &) = default;
  
  double getAccuracy() const {
    return accuracy_;
  }
  
  double getSensitivity() const {
    return sensitivity_;
  }
  
  double getF1Score() const {
    return f1Score_;
  }
  
  double getFalsePositives() const {
    return falsePositives_;
  }
  
  double getFalseNegatives() const {
    return falseNegatives_;
  }
  
  double getTruePositives() const {
    return truePositives_;
  }
  
  double getTrueNegatives() const {
    return trueNegatives_;
  }

private:
  const double falsePositives_;
  const double falseNegatives_;
  const double truePositives_;
  const double trueNegatives_;
  const double accuracy_;
  const double sensitivity_;
  const double f1Score_;
};

#endif //ANOII2020_SOLVESCORE_H
