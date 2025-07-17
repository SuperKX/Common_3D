'''
    wrp文件写出标签。
    注意：
        1）在geomagic运行环境下：C:\Users\Public\Documents\Geomagic\Geomagic Wrap 2015\macros
        2）修改注意替换文件！
        3) API文档：C:\Program Files\Geomagic\Geomagic Foundation 2015\help\scriptingV1
    数据说明：
        1）代码中获取到面片编号，是从1开始计数
        2）面片返回的顶点索引，同样从1开始计数。
        3）filewriter写出的obj文件，
            代码中点索引对应的写出顶点对应一致（记得编号减1）；
            对应的mesh编号打乱了，推测写出obj的时候面片索引有过修改。
        4）库文件的头
            C:\Program Files\Geomagic\Geomagic Foundation 2015\geoapi\fileio\__init__.py
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
path_folder = r"E:\LabelScripts\testdata\27data19_20250717"
inputPath = path_folder+ r"\wrp"
outputPath =path_folder  # 自动生成pth和obj文件夹

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
        # print(dir(activeModel))  # xuek 2025.07.11

        allmesh = geoapp.getMesh(activeModel)
        meshNum = allmesh.numTriangles
        print("1 meshNum: "+str(meshNum))  # 【面数量】-总数量


        # test
        """
        getVertexIndex(self, int f, int version, int which) -> int
        f索引从1开始
        version 只跟点顺序有关。
        getVertexIndex(self, OrientedTriangle iabc, int which) -> int
        """
        # for facei in range(1,11):  # 写出面索引、点索引及坐标 （索引均从1开始技术）
        #     print('face ' + str(facei))
        #     for verti in [0, 1, 2]:
        #         point_idx = mesh.getVertexIndex(facei, 0, verti)
        #         print(point_idx)
        #         coord = mesh.getCoordinate(point_idx)
        #         print(str(coord.x())+" "+str(coord.y())+" "+str(coord.z()))
        #     print('')

        # writer =  FileWrite()
        # writer.filename = r"E:\LabelScripts\testdata\wraptest\27DATA19\obj\out.obj",
        # mesh.write(writer)  # 错误，待排查

        # print(allmesh.getNumMeshInstances())  # mesh数量??
        # print(str(classID.getNumMeshInstances()))  # classID未定义

        # print("md INFO =\n")
        # print(dir(md))
        # print("mesh INFO =\n" )
        # print(dir(mesh))
        # print("allmesh INFO =\n")
        # print(dir(allmesh))


        # faceGroup = ""
        for classNamei in classlist:
            xc = geoapp.getSelectionByName(activeModel, classNamei)  # 【可解析标签字符串】
            # print(faceIDs)
            # print("-------------")
            if xc == None:
                continue
            faceIDs[classNamei] = list(xc)


            #for it in xc:  # 临时输出所有索引
            #   print(it)


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
                print("obj has been writed before, will not rewrite here!!")
            else:
                filewriter = FileWrite()
                filewriter.filename = obj_out_file
                filewriter.mesh = mesh  # allmesh  # mesh
                filewriter.run()

            dic_out_file = os.path.join(obj_out_path, fileName + '_old_face_dict.pth')
            if not os.path.exists(dic_out_file):
                face_indices = []  # 用于存储所有面对应的顶点索引的列表
                for i in range(mesh.numTriangles):
                    v1 = mesh.getVertexIndex(i+1, 0, 0)
                    v2 = mesh.getVertexIndex(i+1, 0, 1)
                    v3 = mesh.getVertexIndex(i+1, 0, 2)
                    face_indices.append((v1, v2, v3))  # 将顶点索引作为一个元组添加到列表中
                    # print(v1)
                print("=============  finish __ face_id collect")

                write_dict_to_binary(face_indices, dic_out_file)

                # print("------------")
                # print(dir(filewriter))
                # print("------------")
                # property_names = filewriter.getPropertyNames()
                # print( property_names)
                # print("------------")
                # mesh_obj = filewriter.getProperty("mesh")
                # property_names = mesh_obj.getPropertyNames()
                # print("mesh_obj properties:", property_names)
                # print("------------")
                # for facei in range(3, 5):  # 写出面索引、点索引及坐标 （索引均从1开始计数）
                #     print('face ' + str(facei))
                #     for verti in [0, 1, 2]:
                #         point_idx = mesh.getVertexIndex(facei, 0, verti)
                #         print(point_idx)
                #         point_idx = filewriter.mesh.getVertexIndex(facei, 0, verti)
                #         print(point_idx)
                #     print('')
print('wrp process finished')
''''
其他功能：
mesh = geoapp.getMesh(md)
Vector3D = mesh.getCoordinate(0)  错误，待确认
'''