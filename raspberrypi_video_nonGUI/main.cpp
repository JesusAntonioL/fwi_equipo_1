#include <QCoreApplication>
#include <QTimer>
#include <QString>
#include "LeptonThread.h"

QString tcpIp = "10.22.244.159";  // Default IP

void printUsage(char *cmd) {
    char *cmdname = basename(cmd);
    printf("Usage: %s [OPTION]...\n"
           " -h      display this help and exit\n"
           " -cm x   select colormap\n"
           " -tl x   select type of Lepton\n"
           " -ss x   SPI bus speed [MHz] (10 - 30)\n"
           " -min x  override minimum value for scaling\n"
           " -max x  override maximum value for scaling\n"
           " -d x    log level (0-255)\n"
           " -ip x   destination IP for TCP transmission\n"
           "", cmdname);
}

int main(int argc, char **argv) {
    QCoreApplication app(argc, argv);

    int typeLepton = 3;
    int spiSpeed = 20;
    int rangeMin = -1;
    int rangeMax = -1;
    int loglevel = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            loglevel = std::atoi(argv[++i]) & 0xFF;
        } else if (strcmp(argv[i], "-tl") == 0 && i + 1 < argc) {
            typeLepton = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-ss") == 0 && i + 1 < argc) {
            spiSpeed = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-min") == 0 && i + 1 < argc) {
            rangeMin = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-max") == 0 && i + 1 < argc) {
            rangeMax = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-ip") == 0 && i + 1 < argc) {
            tcpIp = argv[++i];
        }
    }

    LeptonThread *leptonThread = new LeptonThread();
    leptonThread->setTcpTarget(tcpIp);
    leptonThread->setLogLevel(loglevel);
    leptonThread->useLepton(typeLepton);
    leptonThread->useSpiSpeedMhz(spiSpeed);
    
    if (rangeMin >= 0) leptonThread->useRangeMinValue(rangeMin);
    if (rangeMax >= 0) leptonThread->useRangeMaxValue(rangeMax);

    leptonThread->start();

    return app.exec();
}
