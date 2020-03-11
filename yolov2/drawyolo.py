import sys
sys.path.append('../')
from pycore.tikzeng import *
import math

network = []

f = open("./arch_yolov3.txt", "r")
for x in f:
    layer = {}
    input_1 = x.split()
    layer["id"] = input_1[0]
    layer["type"] = input_1[1]

    if("->" in x):
        input_2 = x.split("->")[1].split()
        layer["height"] = float(input_2[0])
        layer["depth"] = float(input_2[2])

        #handle filter size of 1024 where no space is present after x
        if( not input_2[3].isalpha()): 
            layer["width"]= float(input_2[3][1:])
        else:
            layer["width"]= float(input_2[4])
    
    if("res" in x):
        layer["from"] = str(int(input_1[2])-1)
    
    if("route" in x):
        layer["c1"] = input_1[2]
        try:
            layer["c2"] = input_1[3]
        except:
            layer["c2"]=None

    network.append(layer)

WIDTH_SCALE=100
HEIGHT_SCALE=10

arch2 = [    
    to_head( '..' ),
    to_cor(),
    to_begin()]

to_east="(0,0,0)"
yolos = []

for i, layer in enumerate(network):
    c = []
    l = None
    if (layer["type"] == "conv" ):
        if(network[i-1]["type"]!="conv" ): # and network[i+1]["type"]=="conv"
            l = to_Conv(layer["id"],"",str(int(layer["width"])), offset="(1,0,0)", to=to_east, height=layer["height"]/HEIGHT_SCALE,depth=layer["depth"]/HEIGHT_SCALE, width=layer["width"]/WIDTH_SCALE)
            if(i - 1 > 0):
                if(to != None): c.append(to_connection(to,layer["id"]))
        elif(network[i+1]["type"]!="conv"):
            l = to_Conv(layer["id"],str(int(layer["height"])),str(int(layer["width"])), offset="(0,0,0)", to=to_east, height=layer["height"]/HEIGHT_SCALE,depth=layer["depth"]/HEIGHT_SCALE, width=layer["width"]/WIDTH_SCALE)
        else:
            l = to_Conv(layer["id"],"",str(int(layer["width"])), offset="(0,0,0)", to=to_east, height=layer["height"]/HEIGHT_SCALE,depth=layer["depth"]/HEIGHT_SCALE, width=layer["width"]/WIDTH_SCALE)
    if ( layer["type"] == "upsample"):
        #if(network[i-1]["type"]!="conv" ): # and network[i+1]["type"]=="conv"
        l = to_UnPool(layer["id"], offset="(1,0,0)", to=to_east, height=layer["height"]/HEIGHT_SCALE,depth=layer["depth"]/HEIGHT_SCALE, width=layer["width"]/WIDTH_SCALE)
        if(i - 1 > 0):
            if(to != None): 
                c.append(to_connection(to,layer["id"]))
                c.append(to_Upscale(layer["id"], to))
    if (layer["type"] == "max"):   
        l = to_Pool(layer["id"], offset="(0,0,0)", to=to_east, height=layer["height"]/HEIGHT_SCALE,depth=layer["depth"]/HEIGHT_SCALE, width=0.1)

    if(layer["type"] == "res"):
        l = to_ResAdd(layer["id"], to)
        if(to != None): c.append(to_connection(to,layer["id"]))
        if(network[int(layer["from"])]["type"]=="res"):
            fromheight = network[int(layer["from"])-1]["height"]/(HEIGHT_SCALE*2)+ int(layer["id"][-1])
            if(to != None): c.append(to_skip_ball(str(int(layer["from"])-1), layer["id"], fromheight))
        else:
            fromheight = network[int(layer["from"])]["height"]/(HEIGHT_SCALE*2)+ int(layer["id"][-1])
            if(to != None): c.append(to_skip_ball(layer["from"], layer["id"], fromheight))
    
    if(layer["type"]=="yolo"):
        prev_layer = network[int(layer["id"])-1]
        new_x = str(2)#len(network)-int(layer["id"])
        new_y = str(10-int(layer["id"][-1]))
        offset = "("+new_x+",-"+new_y+",0)"
        yolo = to_SoftMax(layer["id"], str(int(prev_layer["height"])), to = "("+network[-2]["id"]+"-east)", offset=offset,height=prev_layer["height"]/HEIGHT_SCALE,depth=prev_layer["depth"]/HEIGHT_SCALE, width=float(42)/WIDTH_SCALE)
        yolos.append(to_connection_yolo(to,layer["id"]))
        yolos.append(yolo)
        to=None
    
    if(layer["type"] == "route"):
        fromheight = 5 #network[int(layer["id"])-1]["height"]/(HEIGHT_SCALE*2)
        l = to_ResConcat(layer["id"], to_east)
        c.append(to_skip_ball(layer["c1"],layer["id"],fromheight))
        if(layer["c2"] is not None): 
            id = layer["c2"]
            if (network[int(id)]["type"]=="res"): id = str(int(id)-1)
            c.append(to_skip_ball(id,layer["id"],fromheight + int(layer["id"][:1])))
    

    if(l is not None):
        to = layer["id"]
        to_east = "("+to+"-east)"
        arch2.append(l)
        if c is not None: 
            for e in c:
                arch2.append(e)
yolos.reverse()
for e in yolos:
    arch2.append(e)

#c = to_connection(network[i-1]["id"],layer["id"]),
#arch2.append(c)


arch2.append(to_end())

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_Conv("conv1", 512, 64, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2 ),
    to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)"),
    to_Conv("conv2", 128, 64, offset="(1,0,0)", to="(pool1-east)", height=32, depth=32, width=2 ),
    to_connection( "pool1", "conv2"),
    to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=28, depth=28, width=1),
    to_SoftMax("soft1", 10 ,"(3,0,0)", "(pool1-east)", caption="SOFT"  ),
    to_connection("pool2", "soft1"),
    to_end()
    ]
    
def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch2, namefile + '.tex' )
    pass

if __name__ == '__main__':
    main()
