#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <cstring>
#include <QTcpSocket>
#include <QBuffer>
#include <QDataStream>
#include "LeptonThread.h"
#include "SPI.h"
#include "Lepton_I2C.h"
#include "QCoreApplication"
#include <cstdlib>

LeptonThread::LeptonThread() : QThread() {
    loglevel = 0;
    typeLepton = 3;
    myImageWidth = 160;
    myImageHeight = 120;
    spiSpeed = 20 * 1000 * 1000;
    autoRangeMin = true;
    autoRangeMax = true;
    rangeMin = 30000;
    rangeMax = 32000;
    tempMatrix.resize(120, std::vector<float>(160, 0.0f));
    maxTemps.resize(120, std::vector<uint8_t>(160, 0));
}

LeptonThread::~LeptonThread() {}

void LeptonThread::setLogLevel(uint16_t newLoglevel) {
    loglevel = newLoglevel;
}

void LeptonThread::useLepton(int newTypeLepton) {
    typeLepton = newTypeLepton;
    myImageWidth = (typeLepton == 3) ? 160 : 80;
    myImageHeight = (typeLepton == 3) ? 120 : 60;
    tempMatrix.resize(myImageHeight, std::vector<float>(myImageWidth, 0.0f));
}

void LeptonThread::useSpiSpeedMhz(unsigned int newSpiSpeed) {
    spiSpeed = newSpiSpeed * 1000 * 1000;
}

void LeptonThread::useRangeMinValue(uint16_t newMinValue) {
    autoRangeMin = false;
    rangeMin = newMinValue;
}

void LeptonThread::useRangeMaxValue(uint16_t newMaxValue) {
    autoRangeMax = false;
    rangeMax = newMaxValue;
}

void LeptonThread::setTcpTarget(const QString& ip) {
    tcpIp = ip;
    tcpPort = 5005;
    sendOverTcp = true;
}

void LeptonThread::run() {
    SpiOpenPort(0, spiSpeed);

    const float TEMP_MIN_C = 20.0f;
    const float TEMP_MAX_C = 150.0f;
    const float TEMP_RANGE = TEMP_MAX_C - TEMP_MIN_C;
    bool htopLaunched = false;

    while (true) {
        if(!htopLaunched){system("lxterminal -e htop &");
            htopLaunched=true;
        }
        int resets = 0;
        int segmentNumber = -1;

        for (int j = 0; j < PACKETS_PER_FRAME; j++) {
            read(spi_cs0_fd, result + j * PACKET_SIZE, PACKET_SIZE);
            int packetNumber = result[j * PACKET_SIZE + 1];
            if (packetNumber != j) {
                j = -1;
                resets++;
                usleep(1000);
                if (resets >= 750) {
                    SpiClosePort(0);
                    lepton_reboot();
                    usleep(750000);
                    SpiOpenPort(0, spiSpeed);
                    resets = 0;
                }
                continue;
            }
            if (typeLepton == 3 && packetNumber == 20) {
                segmentNumber = (result[j * PACKET_SIZE] >> 4) & 0x0f;
                if (segmentNumber < 1 || segmentNumber > 4) {
                    log_message(10, "[ERROR] Wrong segment number " + std::to_string(segmentNumber));
                    break;
                }
            }
        }

        if (typeLepton == 3 && (segmentNumber < 1 || segmentNumber > 4)) {
            continue;
        }

        if (typeLepton == 3) {
            memcpy(shelf[segmentNumber - 1], result, PACKET_SIZE * PACKETS_PER_FRAME);
            if (segmentNumber != 4) continue;
        } else {
            memcpy(shelf[0], result, PACKET_SIZE * PACKETS_PER_FRAME);
        }

        if (autoRangeMin || autoRangeMax) {
            uint16_t minValue = autoRangeMin ? 65535 : rangeMin;
            uint16_t maxValue = autoRangeMax ? 0 : rangeMax;
            for (int seg = 0; seg < 4; seg++) {
                for (int i = 0; i < FRAME_SIZE_UINT16; i++) {
                    if (i % PACKET_SIZE_UINT16 < 2) continue;
                    uint16_t value = (shelf[seg][i * 2] << 8) + shelf[seg][i * 2 + 1];
                    if (value == 0) continue;
                    if (autoRangeMax && value > maxValue) maxValue = value;
                    if (autoRangeMin && value < minValue) minValue = value;
                }
            }
            rangeMin = minValue;
            rangeMax = maxValue;
        }

        float scale = 255.0f / (rangeMax - rangeMin);

        for (int seg = 0; seg < 4; seg++) {
            int rowOffset = seg * 30;
            for (int i = 0; i < FRAME_SIZE_UINT16; i++) {
                if (i % PACKET_SIZE_UINT16 < 2) continue;

                uint16_t raw = (shelf[seg][i * 2] << 8) + shelf[seg][i * 2 + 1];
                float tempK = raw * 0.01f;
                float tempC = tempK - 273.15f;

                int col, row;
                if (typeLepton == 3) {
                    col = (i % PACKET_SIZE_UINT16) - 2 +
                          (myImageWidth / 2) * ((i % (PACKET_SIZE_UINT16 * 2)) / PACKET_SIZE_UINT16);
                    row = i / PACKET_SIZE_UINT16 / 2 + rowOffset;
                } else {
                    col = (i % PACKET_SIZE_UINT16) - 2;
                    row = i / PACKET_SIZE_UINT16;
                }

                if (row >= 0 && row < myImageHeight && col >= 0 && col < myImageWidth) {
                    tempMatrix[row][col] = tempC;
                }
            }
        }

        for (int y = 0; y < 120; ++y) {
            for (int x = 0; x < 160; ++x) {
                float val = tempMatrix[y][x];
                float tempC = std::max(TEMP_MIN_C, std::min(TEMP_MAX_C, val));
                maxTemps[y][x] = static_cast<uint8_t>(255.0f * (tempC - TEMP_MIN_C) / TEMP_RANGE);
            }
        }

        sendMaxTempsTcp(maxTemps);
        QCoreApplication::processEvents();
    }

    SpiClosePort(0);
}

void LeptonThread::sendMaxTempsTcp(const std::vector<std::vector<uint8_t>> &matrix) {
    if (!sendOverTcp) return;

    // If no socket, or not connected, try reconnecting
    if (!tcpSocket || tcpSocket->state() != QAbstractSocket::ConnectedState) {
        if (tcpSocket) {
            tcpSocket->disconnectFromHost();
            tcpSocket->deleteLater();
        }

        tcpSocket = new QTcpSocket();
        tcpSocket->connectToHost(tcpIp, tcpPort);
        if (!tcpSocket->waitForConnected(500)) {
            std::cerr << "[TCP] ❌ Failed to connect to " << tcpIp.toStdString() << ":" << tcpPort << std::endl;
            delete tcpSocket;
            tcpSocket = nullptr;
            return;
        }

        std::cout << "[TCP] ✅ Reconnected to " << tcpIp.toStdString() << std::endl;
    }

    // Compose payload
    QByteArray payload;
    payload.resize(160 * 120);
    for (int y = 0; y < 120; ++y) {
        memcpy(payload.data() + y * 160, matrix[y].data(), 160);
    }

    // Send header (payload length)
    QByteArray header;
    QDataStream stream(&header, QIODevice::WriteOnly);
    stream << quint32(payload.size());

    // Try writing both header and payload
    qint64 sent1 = tcpSocket->write(header);
    qint64 sent2 = tcpSocket->write(payload);
    tcpSocket->flush();

    // Check if full payload was sent
    if (sent1 != header.size() || sent2 != payload.size()) {
        std::cerr << "[TCP] Incomplete send, resetting socket.\n";
        tcpSocket->disconnectFromHost();
        tcpSocket->deleteLater();
        tcpSocket = nullptr;
    }
}

void LeptonThread::log_message(uint16_t level, std::string msg) {
    if (level <= loglevel) {
        std::cerr << msg << std::endl;
    }
}
