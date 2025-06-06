TEMPLATE = app

# Only the required Qt modules
QT += core network
CONFIG += console c++11
CONFIG -= app_bundle

TARGET = raspberrypi_video

# Paths
RPI_LIBS = ../raspberrypi_libs
LEPTONSDK = leptonSDKEmb32PUB

# Build SDK first
PRE_TARGETDEPS += sdk
QMAKE_EXTRA_TARGETS += sdk sdkclean
sdk.commands = make -C $${RPI_LIBS}/$${LEPTONSDK}
sdkclean.commands = make -C $${RPI_LIBS}/$${LEPTONSDK} clean

# Include directories
DEPENDPATH += .
INCLUDEPATH += . $${RPI_LIBS}

# Output paths
DESTDIR = .
OBJECTS_DIR = gen_objs
MOC_DIR = gen_mocs

# Only include necessary sources and headers
HEADERS += LeptonThread.h \
           SPI.h \
           Lepton_I2C.h

SOURCES += main.cpp \
           LeptonThread.cpp \
           SPI.cpp \
           Lepton_I2C.cpp

# Link SDK library
unix:LIBS += -L$${RPI_LIBS}/$${LEPTONSDK}/Debug -lLEPTON_SDK

# Clean up
unix:QMAKE_CLEAN += -r $(OBJECTS_DIR) $${MOC_DIR}