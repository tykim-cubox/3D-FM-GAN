from pathlib import Path
path = list(Path("/home/aiteam/tykim/generative/gan/3D-FM-GAN/data/ffhq/detections2").glob('*.txt'))

print(len(path))

detect = []
for i in path:
  lnd = open(i, 'r')
  check = lnd.read()
  if len(check) != 60:
    detect.append(i)
  lnd.close()