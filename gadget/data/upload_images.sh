current_time=$(date "+%Y.%m.%d-%H.%M.%S")

zip -r /home/aaeon/gadget/data/0_$current_time.zip /home/aaeon/gadget/data/image_archive/sensor/gadget-sensor-avt/0
gsutil -m cp /home/aaeon/gadget/data/0_$current_time.zip gs://lmi-nordson-models/data/0

zip -r /home/aaeon/gadget/data/1_$current_time.zip /home/aaeon/gadget/data/image_archive/sensor/gadget-sensor-avt/1
gsutil -m cp /home/aaeon/gadget/data/1_$current_time.zip gs://lmi-nordson-models/data/1

zip -r /home/aaeon/gadget/data/2_$current_time.zip /home/aaeon/gadget/data/image_archive/sensor/gadget-sensor-avt/2
gsutil -m cp /home/aaeon/gadget/data/2_$current_time.zip gs://lmi-nordson-models/data/2

rm /home/aaeon/gadget/data/0_$current_time.zip
rm /home/aaeon/gadget/data/1_$current_time.zip
rm /home/aaeon/gadget/data/2_$current_time.zip
