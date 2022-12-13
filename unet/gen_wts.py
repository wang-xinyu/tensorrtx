import torch
import sys
import struct

def main():
  device = torch.device('cpu')
  state_dict = torch.load(sys.argv[1], map_location=device)

  f = open("unet.wts", 'w')
  f.write("{}\n".format(len(state_dict.keys())))
  for k, v in state_dict.items():
    print('key: ', k)
    print('value: ', v.shape)
    vr = v.reshape(-1).cpu().numpy()
    f.write("{} {}".format(k, len(vr)))
    for vv in vr:
      f.write(" ")
      f.write(struct.pack(">f", float(vv)).hex())
    f.write("\n")
  f.close()

if __name__ == '__main__':
  main()

