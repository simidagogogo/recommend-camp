
HADOOP_CMD="/usr/local/share/hadoop-2.10.1/bin/hadoop"
STREAM_JAR_PATH="/usr/local/share/hadoop-2.10.1/share/hadoop/tools/lib/hadoop-streaming-2.10.1.jar"

INPUT_FILE_PATH_1="/The_Man_of_Property.txt"
#INPUT_FILE_PATH_1="/1.data"

OUTPUT_PATH="/output"

$HADOOP_CMD fs -rm -r -skipTrash $OUTPUT_PATH

# Step 1.
$HADOOP_CMD jar $STREAM_JAR_PATH \
    -input $INPUT_FILE_PATH_1 \
    -output $OUTPUT_PATH \
    -mapper "python map_new.py" \
    -reducer "python red_new.py" \
    -file ./map_new.py \
    -file ./red_new.py

