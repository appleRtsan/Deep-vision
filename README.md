# Deep-vision
NSYSU CSE318 深度視覺 DEEP VISION


# Install guide
Please follow:
1. https://docs.docker.com/compose/gpu-support/
2. https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Usage
**Run as a daemon**
```
docker-compose up -d
```

**Stop compose this daemon**
```
docker-compose stop
```

**Get your jupyter token**
```
docker logs scc-tf
```

**Other configuration**
Please refer:
1. https://docs.docker.com/compose/compose-file/compose-file-v3/