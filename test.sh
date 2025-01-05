nvidia-docker run \
  -v /home/star/Yanjun/RL-VLM-F/softgym:/workspace/softgym \
  -v /home/star/anaconda3:/workspace/anaconda3 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -it xingyu/softgym:latest bash
