//
// Created by richardzvonek on 11/21/20.
//

#ifndef ANOII2020_TRAININPUTSET_H
#define ANOII2020_TRAININPUTSET_H


#include <vector>
#include <opencv2/core/mat.hpp>

#include "space.h"
#include "inputset.h"

class TrainInputSet : public InputSet {
public:
  TrainInputSet(const std::string &filename, const std::array<Space, SPACES_COUNT> &spaces);
  
  typedef std::pair<LoadedData, uint8_t> InputPair;
  
  const std::vector<InputPair> &getInputSet() const;

private:
  std::vector<InputPair> inputSet_;
};


#endif //ANOII2020_TRAININPUTSET_H