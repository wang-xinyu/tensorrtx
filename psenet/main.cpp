#include "psenet.h"

int main(int argc, char** argv)
{
    PSENet psenet(1200, 640, 0.90, 6, 4);

    if (argc == 2 && std::string(argv[1]) == "-s")
    {
        std::cout << "Serializling Engine" << std::endl;
        psenet.serializeEngine();
        return 0;
    }
    else if (argc == 2 && std::string(argv[1]) == "-d")
    {
        psenet.init();
        std::vector<std::string> files;
        for (int i = 0; i < 10; i++)
            files.emplace_back("test.jpg");
        for (auto file : files)
        {
            std::cout << "Detect " << file << std::endl;
            psenet.detect(file);
        }

        return 0;
    }
    else
    {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./psenet -s  // serialize model to plan file" << std::endl;
        std::cerr << "./psenet -d  // deserialize plan file and run inference" << std::endl;
        return -1;
    }
}
