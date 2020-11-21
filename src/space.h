//
// Created by richardzvonek on 11/21/20.
//

#ifndef ANOII2020_SPACE_H
#define ANOII2020_SPACE_H

#include <cstdint>

const uint16_t SPACES_COUNT = 56;
const cv::Size SPACE_SIZE(80, 80);

struct Space {
  Space(int x01, int y01, int x02, int y02, int x03, int y03, int x04, int y04, int occup) :
      x01(x01),
      y01(y01),
      x02(x02),
      y02(y02),
      x03(x03),
      y03(y03),
      x04(x04),
      y04(y04),
      occup(occup) {}
  
  Space() :
      x01(0),
      y01(0),
      x02(0),
      y02(0),
      x03(0),
      y03(0),
      x04(0),
      y04(0),
      occup(0) {}
  
  Space(const Space &) = default;
  
  int x01, y01, x02, y02, x03, y03, x04, y04, occup;
};

#endif //ANOII2020_SPACE_H
