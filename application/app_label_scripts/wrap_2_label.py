'''
    wrp文件写出标签。
    注意：
        1）在geomagic运行环境下：C:\Users\Public\Documents\Geomagic\Geomagic Wrap 2015\macros
        2）修改注意替换文件！

'''

# active描述的是选中的，而非显示的
import sys
# print('Python: {}'.format(sys.version)) #Python: 2.7.3
if sys.version_info[0] >= 3:
    from geomagic.api.v3 import *
else:
    from geomagic.api.v2 import *
import geomagic.app.v2
for m in geomagic.app.v2.execStrings: exec m in locals(), globals()
import geoapiall
for m in geoapiall.modules: exec "from %s import *" % m
import os

# 测试区域
import pickle

def write_dict_to_binary(data, filename):
  """
  将字典写入到二进制文件。
  Args:
    要写入的字典。
    filename:  文件名（包括路径）。
  """
  try:
    with open(filename, 'wb') as f:  # 'wb' 表示以二进制写入模式打开文件
      pickle.dump(data, f)
    print("success write")
  except Exception as e:
    print("wrong write!")

# data = {'book':[i for i in range(10000)], 'note':[i*2 for i in range(10000)]}
# filename = r"E:\LabelScripts\testdata\wraptest\patht.pth"
# write_dict_to_binary(data, filename)

# 0 外部参数
# classlist = ["wigwam","background"]
classlist = ["background", "building", "wigwam",
           "car",
           "vegetation",
           "farmland",
           "shed",
           "stockpiles",
           "bridge",
           "pole",
           "others",
           "grass"]
data_type = ".wrp"
inputPath = r"E:\LabelScripts\testdata\wraptest\27DATA19\wrp"
outputPath = r"E:\LabelScripts\testdata\wraptest\27DATA19"  # 自动生成pth和obj文件夹
outputPath_pth = os.path.join(outputPath,'pth')
outputPath_obj = os.path.join(outputPath,'obj')
bool_write_obj = True  # 写出obj文件
if not os.path.exists(outputPath_pth):
    os.makedirs(outputPath_pth)
if not os.path.exists(outputPath_obj):
    os.makedirs(outputPath_obj)
faceIDs = dict()  # 记录所有类别的标签

# 1 遍历所有wrp文件
for fileNameEpx in os.listdir(inputPath):
    # 检查文件扩展名
    fileName, ext = os.path.splitext(fileNameEpx)
    if ext != data_type:
        continue
    inputFile = os.path.join(inputPath, fileNameEpx)
    binaryLabelFile = os.path.join(outputPath_pth, fileName + '.pth')
    # print("binaryLabelFile = " + binaryLabelFile)

    # 2 写出obj文件
    # # 4 逐个类别处理
    # activeModel = geoapp.getActiveModel()  # 所有mesh？
    # allmesh = geoapp.getMesh(activeModel)
    # meshNum = allmesh.numTriangles
    # print(meshNum)  # 【面数量】-总数量

    # 打开模型
    geoapp.openFile(inputFile)
    mdls = geoapp.getModels()
    # 3 逐个模型处理
    for md in mdls:
        if md == mdls[0]:
            continue  # 跳过“全局”
        print("Models = " + md.name)
        # print("Model type = " +str(type(md)))
        mesh = geoapp.getMesh(md)
        # print("meshnnn" + str(mesh.numTriangles) + " triangles")
        # getActiveTriangleSelection(md)

        # 4 逐个类别处理
        activeModel = geoapp.getActiveModel()  # 所有mesh？
        allmesh = geoapp.getMesh(activeModel)
        meshNum = allmesh.numTriangles
        print("1 meshNum: "+str(meshNum))  # 【面数量】-总数量
        # faceGroup = ""
        for classNamei in classlist:
            xc = geoapp.getSelectionByName(activeModel, classNamei)  # 【可解析标签字符串】
            # print(faceIDs)
            # print("-------------")
            if xc == None:
                continue
            faceIDs[classNamei] = list(xc)
            # print(len(faceIDs[id]))
            print('label: '+str(classNamei)+', face nums: '+str(len(faceIDs[classNamei])))
            meshNum -= xc.numSelected
        if meshNum != 0:
            print("[warn]: left meshes"+str(meshNum))
        # 5 写出group列表
        write_dict_to_binary(faceIDs, binaryLabelFile)

        # 6 写出obj
        if bool_write_obj:
            obj_out_path = os.path.join(outputPath_obj, fileName)
            if not os.path.exists(obj_out_path):
                os.makedirs(obj_out_path)
            obj_out_file = os.path.join(obj_out_path, fileName + '.obj')
            if os.path.exists(obj_out_file):
                print("obj has been writed before, will not rewirte here!!")
            else:
                filewriter = FileWrite()
                filewriter.filename = obj_out_file
                filewriter.mesh = allmesh
                filewriter.run()

        # inputFile = os.path.join(inputPath, fileNameEpx)
        # outputPathOBJ = outputPath + fileName
        # if not os.path.exists(outputPathOBJ):
        #     os.mkdir(outputPathOBJ)
        # filewriter = FileWrite()
        # filewriter.filename = outputPathOBJ + "\\" + fileName + ".obj"
        # filewriter.mesh = geoapp.getMesh(activeModel)
        # filewriter.run()


#--------------------下面不用-----------------------
