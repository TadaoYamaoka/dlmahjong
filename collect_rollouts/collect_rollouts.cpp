#include "../../cmajiang/src_cpp/game.h"

#include "cxxopts.hpp"

#include <iostream>

int main(int argc, char** argv)
{
    cxxopts::Options options("collect_rollouts", "Collect rollouts");

    const auto result = options.parse(argc, argv);

    Game game;
    game.kaiju();
    game.qipai();

    std::cout << game.shoupai_(0).toString() << std::endl;

    return 0;
}
