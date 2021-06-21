import sys


# python test_main.py dr=0.1 mo=333

if __name__ == '__main__':
    # print(len(sys.argv))

    if len(sys.argv) == 2:
        print("Insufficient arguments")
        sys.exit()
    
    # parameters: keys and default values
    params = {
        "bs": 32,
        "height": 80, 
        "width": 80,
        "depth": 32,
        "epoch_num": 500,
        "l_rate": 1e-4,
        "network": "unet3plus",
        "precision": 32,
        "gpu_num": 4,
        "dropout": 0.3,
        "model_channels": 32,
    }

    # set parameter values from command line arguments
    key = ""
    for i, arg in enumerate(sys.argv):
        # print("{} is {}".format(i, arg))
        if i==0:
            continue
        elif i%2 == 1:
            key = arg
        else:
            if key in ("b"): key = "bs"
            elif key in ("h", "H", "Height"): key = "height"
            elif key in ("w", "W", "Width"): key = "width"
            elif key in ("d", "D", "Depth"): key = "depth"
            elif key in ("e", "ep", "epoch"): key = "epoch_num"
            elif key in ("l"): key = "l_rate"
            elif key in ("n", "ne", "net"): key = "network"
            elif key in ("p", "pr", "pre", "prec"): key = "precision"
            elif key in ("g", "gp", "gpu"): key = "gpu_num"
            elif key in ("dr", "drop", "do"): key = "dropout"
            elif key in ("m", "mo", "model", "mc", "c", "ch", "channel", "channels"): key = "model_channels"

            if key in ("bs", "height", "width", "depth", "epoch_num", "precision", "gpu_num", "model_channels"):
                params[key] = int(arg)
            elif key in ("l_rate", "dropout"):
                params[key] = float(arg)
            else:
                params[key] = arg
    
    # print parameter values
    # print(params)
    for k, v in params.items():
        print("{} = {}".format(k,v))








# bs = 32
# Height = 80
# Width = Height
# Depth = 32
# epoch_num = 500
# l_rate = 1e-4
# network = "unet3plus"
# precision = 32
# gpu_num = 4
# dropout = 0.3
# model_channels = 32

