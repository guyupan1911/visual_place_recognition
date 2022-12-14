function abspath() {
    # generate absolute path from relative path
    # $1     : relative filename
    # return : absolute path
    if [ -d "$1" ]; then
        # dir
        (cd "$1"; pwd)
    elif [ -f "$1" ]; then
        # file
        if [[ $1 = /* ]]; then
            echo "$1"
        elif [[ $1 == */* ]]; then
            echo "$(cd "${1%/*}"; pwd)/${1##*/}"
        else
            echo "$(pwd)/$1"
        fi
    fi
}

VPR_DIR=$(abspath "..")

docker run \
    -dit \
    --rm \
    --net=host \
    --name=visual_palce_recognition \
    --device=/dev/dri \
    --group-add video \
    --volume=/tmp/.X11-unix \
    --env="DISPLAY=$DISPLAY" \
    -v ${VPR_DIR}:/root/projects/visual_palce_recognition/ \
    autox:visual_place_reognition \
    /bin/bash 