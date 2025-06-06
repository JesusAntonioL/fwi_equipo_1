#ifndef TEXTTHREAD
#define TEXTTHREAD

#include <QThread>
#include <QString>
#include <vector>
#include <QTcpSocket>
#include <cstdint>

// Make sure these are defined
#define PACKET_SIZE 164
#define PACKET_SIZE_UINT16 (PACKET_SIZE / 2)
#define PACKETS_PER_FRAME 60
#define FRAME_SIZE_UINT16 (PACKET_SIZE_UINT16 * PACKETS_PER_FRAME)

class LeptonThread : public QThread
{
    Q_OBJECT

public:
    LeptonThread();
    ~LeptonThread();

    void setLogLevel(uint16_t);
    void useLepton(int);
    void useSpiSpeedMhz(unsigned int);
    void useRangeMinValue(uint16_t);
    void useRangeMaxValue(uint16_t);
    void setTcpTarget(const QString& ip);
    void run();

private:
    void sendMaxTempsTcp(const std::vector<std::vector<uint8_t>> &matrix);
    void log_message(uint16_t level, std::string msg);  // <-- REQUIRED declaration

    QString tcpIp;
    quint16 tcpPort;
    QTcpSocket *tcpSocket = nullptr;
    bool sendOverTcp = true;

    uint16_t loglevel;
    int typeLepton;
    unsigned int spiSpeed;
    bool autoRangeMin;
    bool autoRangeMax;
    uint16_t rangeMin;
    uint16_t rangeMax;
    int myImageWidth;
    int myImageHeight;

    uint8_t result[PACKET_SIZE * PACKETS_PER_FRAME];
    uint8_t shelf[4][PACKET_SIZE * PACKETS_PER_FRAME];  // <-- use 4 for Lepton 3.x

    std::vector<std::vector<float>> tempMatrix;         // <-- REQUIRED declaration
    std::vector<std::vector<uint8_t>> maxTemps;
};

#endif
