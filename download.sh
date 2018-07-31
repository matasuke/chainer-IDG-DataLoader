#!bin/sh

return_yes_or_no(){

    _ANSWER=

    while :
    do
        if [ "`echo -n`" = "-n" ]; then
            echo "$@\c"
        else
            echo -n "$@"
        fi
        read _ANSWER
        case "$_ANSWER" in
            [yY] | yes | YES | Yes) return 0 ;;
            [nN] | no | NO | No ) return 1 ;;
            * ) echo "type yes or no."
        esac
    done
}

echo 'download MSCOCO images ?(yes/no)'
images=`return_yes_or_no`
echo 'download MSCOCO official caption dataset? (yes/no)'
mscoco=`return_yes_or_no`
echo 'download STAIR captions? (yes/no)'
stair=`return_yes_or_no`

if [ -d data/images/original ]; then
        mkdir -p data/images/original
fi

if $images; then
        # download official images
        echo 'downloading MSCOCO images...'
        if [ ! -d data/images/original/train2014 ]; then
                curl -# http://images.cocodataset.org/zips/train2014.zip > \
                        data/images/original/train2014.zip
                unzip data/images/original/train2014.zip -d data/images/original/
                rm data/images/original/train2014.zip
        fi

        if [ -d data/images/original/val2014 ]; then
                curl -# http://images/cocodataset.org/zips/val2014.zip > \
                        data/images/original/val2014.zip
                unzip data/images/original/val2014.zip -d data/images/original/
                rm data/images/original/val2014.zip
        fi

        if [ -d data/images/original/test2014 ]; then
                curl -# http://images/cocodataset.org/zips/test2014.zip > \
                        data/images/original/test2014.zip
                unzip data/images/original/test2014.zip -d data/images/original/
                rm data/images/original/test2014.zip
        fi
fi

if [ ! -d data/captions/original ]; then
        mkdir -p data/captions/original \
                 data/captions/formatted/MSCOCO_captions/ \
                 data/captions/formatted/STAIR_captions/ \
                 data/captions/converted/MSCOC_captions/ \
                 data/captions/converted/STAIR_captions/
fi

if $mscoco; then
        # download official captions
        if [ ! -d data/captions/original/MSCOCO_captions_en ]; then
                echo 'downloading MSCOCO Captions...'
                curl -# http://images.cocodataset.org/annotations/annotations_trainval2014.zip > \
                        data/captions/original/annotations_trainval2014.zip
                unzip data/captions/original/annotations_trainval2014.zip -d data/captions/original
                rm data/captions/original/annotations_trainval2014.zip
                mv data/captions/original/annotations data/captions/original/MSCOCO_captions_en
        fi
fi

if $stair; then
        # download stair capions
        if [ ! -d data/captions/original/STAIR_captions/stair_captions_v1.2 ]; then
                echo 'downloading STAIR Captions...'
                curl -sL https://github.com/STAIR-Lab-CIT/STAIR-captions/tarball/master/stair_captions_v1.2.tar.gz > \
                        data/captions/original/STAIR_captions.tar.gz
                tar -zxvf data/captions/original/STAIR_captions.tar.gz -C data/captions/original
                mv data/captions/original/STAIR-Lab-CIT-STAIR-captions-6ac656e data/captions/original/STAIR_captions
                rm data/captions/original/STAIR_captions.tar.gz
                tar -zxvf data/captions/original/STAIR_captions/stair_captions_v1.2.tar.gz \
                        -C data/captions/original/STAIR_captions/
                rm data/captions/original/STAIR_captions/stair_captions_v1.2.tar.gz
        fi
fi

echo 'finished.'
