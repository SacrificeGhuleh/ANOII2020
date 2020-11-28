#include "trainedsolver.h"
#include "netdef.h"


template
class TrainedDlibSolver<AlexNet>;

template
class TrainedDlibSolver<LeNet>;

template
class TrainedDlibSolver<VGG19>;
