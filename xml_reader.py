# 将labelimg 生成的xml文件转txt
import xml.dom.minidom
import os

path = "./1/"
file_list = os.walk(path)


# for _,_,i in file_list:
#     for file in i:
#         xml_name = path + file
#         txt_name = path + file.split('.')[0] + '.bmp.txt'
#         document_tree = xml.dom.minidom.parse(xml_name)
#         file_handle=open(txt_name,mode='w')
#         name_nodes = document_tree.getElementsByTagName("name")
#         for i in range(len(name_nodes)):
#
#             name = name_nodes[i].childNodes[0].data
#             xmin = document_tree.getElementsByTagName("xmin")[i].childNodes[0].data
#             ymin = document_tree.getElementsByTagName("ymin")[i].childNodes[0].data
#             xmax = document_tree.getElementsByTagName("xmax")[i].childNodes[0].data
#             ymax = document_tree.getElementsByTagName("ymax")[i].childNodes[0].data
#         #     print("{},{},{},{},{}".format(name, (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin,ymax)))
#             file_handle.write("{},{},{},{},{},{},{},{},{}\n".format(xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax,name))
#         file_handle.close()


"""生成train_list.txt"""
file_list = os.walk("./dataset/pzt_carrier/train_images/")
file_handle = open("./dataset/pzt_carrier//train_list.txt",mode='w')
for _,_,i in file_list:
    for file in i:
        file_handle.write("{}\n".format(file.split('.')[0] +'.'+ file.split('.')[1]))
file_handle.close()