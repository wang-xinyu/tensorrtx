#include "psenet.h"

int main(int argc, char **argv)
{
    PSENet psenet(1600, 0.9, 6, 4);

    if (argc == 2 && std::string(argv[1]) == "-s")
    {
        std::cout << "Serializling Engine" << std::endl;
        psenet.serializeEngine();
        return 0;
    }
    else if (argc == 2 && std::string(argv[1]) == "-d")
    {
        psenet.init();
        vector<string> files = {"test.jpg", "test.jpg", "test.jpg"};
        for (int i = 0; i < files.size(); i++)
        {
            std::cout << "Detect " << files[i] << std::endl;
            psenet.detect(files[i]);
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
