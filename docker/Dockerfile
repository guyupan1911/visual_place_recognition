FROM ubuntu:20.04

ENV WORK_SPACE = /root/projects
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    gdb git cmake g++ libgtk2.0-dev && \
    mkdir -p $WORK_SPACE/visual_palce_recognition/

RUN git clone https://github.com/opencv/opencv.git && \
    cd opencv && \
    mkdir build && \
    cd build && \
    cmake .. - && \
    make -j8 && \
    make install

RUN git clone https://github.com/rmsalinas/DBow3.git && \
    cd DBow3 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j8 && \
    make install


