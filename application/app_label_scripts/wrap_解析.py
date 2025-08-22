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
        5）wrp文件缩放后，写出obj可能丢失点。
        6）wrp文件说明，wrp中面片id是初始id编号，如果删减过后id不会更新

    # 打标签解决方法：
        1 生成obj文件，及标签索引，在obj层面打标签
        2 直接分割成多个mesh写出。
        3 直接分割mesh再点云化处理。

    # API记录：
        class CreatePointsFromMesh-mesh顶点转点云
        class CreateMesh    创建mesh-可能没纹理
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
    print("dict.pth success write")
  except Exception as e:
    print("dict.pth wrong write!")

# 1) 批量处理
def process_wrp(inputPath, outputPath, log_file):
    '''
    wrp处理主函数
    inputPath       一个场景输入  # r"D:\DATASET\BIMTwins\WRP\batch1\39PTY2"
    outputPath      一个场景的输出  # r"D:\DATASET\BIMTwins\WRP\batch1_out\39PTY2_out"
    log_file        log.txt地址  # r"D:\DATASET\BIMTwins\WRP\batch1_out\39PTY2_out\log.txt"
    '''
    log_str = ""
    outputPath_pth = os.path.join(outputPath, 'pth')
    outputPath_obj = os.path.join(outputPath, 'obj')
    bool_write_obj = True  # 写出obj文件
    if not os.path.exists(outputPath_pth):
        os.makedirs(outputPath_pth)
    if not os.path.exists(outputPath_obj):
        os.makedirs(outputPath_obj)

    # 1 遍历所有wrp文件
    for fileNameEpx in os.listdir(inputPath):
        faceIDs = dict()  # 记录所有类别的标签
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
            log_str += "------ mesh info -------" + "\n"
            print("------ mesh info -------")
            print("Models = " + md.name)
            log_str += "Models = " + md.name + "\n"
            # print("Model type = " +str(type(md)))
            mesh = geoapp.getMesh(md)
            # print("mesh_name = " + str(mesh.name))  # 空
            print("points_num = " + str(mesh.numPoints))
            log_str += "points_num = " + str(mesh.numPoints) + "\n"
            print("faces_num = " + str(mesh.numTriangles))
            log_str += "faces_num = " + str(mesh.numTriangles) + "\n"
            # getActiveTriangleSelection(md)

            # 优化：比例尺错误处理
            if mesh.maxBoxCoord.x() - mesh.minBoxCoord.x() < 10:
                trans = Transform3D()
                trans.setScaleTransform(1000)  # 放大1000倍
                mesh.transform(trans)
                print('[warn]wrong scale size,1000* process!')
                log_str += '[warn]wrong scale size,1000* process!' + "\n"

            # 4 逐个类别处理
            activeModel = geoapp.getActiveModel()  # 所有mesh？
            # print(dir(activeModel))  # xuek 2025.07.11

            allmesh = geoapp.getMesh(activeModel)
            meshNum_rest = allmesh.numTriangles

            # print(allmesh.getNumMeshInstances())  # mesh数量??
            # print(str(classID.getNumMeshInstances()))  # classID未定义

            # faceGroup = ""
            log_str += "----label--------faces_num----\n"
            for classNamei in classlist:
                xc = geoapp.getSelectionByName(activeModel, classNamei)  # 【可解析标签字符串】
                # print(faceIDs)
                # print("-------------")
                if xc == None:
                    continue
                faceIDs[classNamei] = list(xc)

                # test-----------------------
                if False:
                    """
                    getVertexIndex(self, int f, int version, int which) -> int
                    f索引从1开始
                    version 只跟点顺序有关。
                    getVertexIndex(self, OrientedTriangle iabc, int which) -> int
                    """
                    print('class ' + str(classNamei))
                    for iii in faceIDs[classNamei][0:6]:
                        print(iii)
                    # for it in range(1, 6):  # 写出面索引、点索引及坐标 （索引均从1开始技术）
                    #     facei = faceIDs[classNamei][it]
                    #     print('face ' + str(facei))
                    #     for verti in [0, 1, 2]:
                    #         point_idx = mesh.getVertexIndex(facei, 0, verti)
                    #         print(point_idx)
                    #     for verti in [0, 1, 2]:
                    #         point_idx = mesh.getVertexIndex(facei, 0, verti)
                    #         # print(point_idx)
                    #         coord = mesh.getCoordinate(point_idx)
                    #         print(str(coord.x()) + " " + str(coord.y()) + " " + str(coord.z()))
                    #     print('')
                    # for it in xc:  # 临时输出所有索引
                    #   print(it)
                # test-----------------------

                # print(len(faceIDs[id]))
                print('label: ' + str(classNamei) + ', face nums: ' + str(len(faceIDs[classNamei])))
                log_str += "    " + str(classNamei) + " " * (14 - len(classNamei)) + str(
                    len(faceIDs[classNamei])) + "\n"
                meshNum_rest -= xc.numSelected
            if meshNum_rest != 0:
                print("[warn]: left meshes" + str(meshNum_rest))
                log_str += "[warn]: left meshes" + str(meshNum_rest) + "\n"
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
                    log_str += "[warn]obj has been writed before, will not rewrite here!\n"
                else:
                    filewriter = FileWrite()
                    filewriter.filename = obj_out_file
                    filewriter.mesh = mesh  # allmesh  # mesh
                    filewriter.run()
                    log_str += "[out]write obj success!\n"

                # 7 写出原始标签顺序
                dic_out_file = os.path.join(obj_out_path, fileName + '_old_face_dict.pth')
                if not os.path.exists(dic_out_file):
                    face_indices = []  # 用于存储所有面对应的顶点索引的列表
                    print("xuek test   "+str(mesh.numTriangles))
                    for i in range(mesh.numTriangles):

                        # test
                        # if i>=2343 and i<=2346:
                        #     print("------" + str(i))
                        #     # abc = OrientedTriangle(i, 0)
                        #     # a = m.getVertexIndex(abc, 0)
                        #     # b = m.getVertexIndex(abc, 1)
                        #     # c = m.getVertexIndex(abc, 2)
                        #     #
                        #     # print(a)
                        #     # print(b)
                        #     # print(c)
                        #     # print("22")
                        #     v1 = mesh.getVertexIndex(i + 1, 0, 0)
                        #     v2 = mesh.getVertexIndex(i + 1, 0, 1)
                        #     v3 = mesh.getVertexIndex(i + 1, 0, 2)
                        #     # face_indices.append((v1, v2, v3))  # 将顶点索引作为一个元组添加到列表中
                        #     print(v1)
                        #     print(v2)
                        #     print(v3)
                        # else:
                        #     continue
                        # if i>=2344 and i<=2360:
                        #     continue
                        # elif i>2360: #2376，2344，2345
                        #     print(i)

                        v1 = mesh.getVertexIndex(i + 1, 0, 0)
                        v2 = mesh.getVertexIndex(i + 1, 0, 1)
                        v3 = mesh.getVertexIndex(i + 1, 0, 2)
                        face_indices.append((v1, v2, v3))  # 将顶点索引作为一个元组添加到列表中
                        # print(v1)
                    # print("finish __ face_id collect")

                    write_dict_to_binary(face_indices, dic_out_file)
                    log_str += "[out]write old_face_inices success!\n"
            log_str += '\n'

    print('[finish]wrp process finished')
    log_str += '[finish]wrp process finished\n'

    # 写出log
    with open(log_file, 'w') as f:  # 追加模式
        f.write(log_str)
        print("write log finish")

# 2) 单个处理
def process_wrp_one():
    '''
    单文件测试
    '''
    folder = r'E:\DATA\baddata\batch2-wrong\01JXY_WRONG'
    fileName = r'Tile_0006_0014'

    wrp_file = os.path.join(folder,fileName+'.wrp')
    binaryLabelFile = os.path.join(folder,fileName+'.pth')
    obj_out_path = os.path.join(folder,fileName)
    bool_overlap = True  # 是否覆盖文件(已经写出的obj、pth是否重复写出)

#-----------------------------------------------
    faceIDs = dict()  # 记录所有类别的标签
    inputFile = wrp_file
    # 打开模型
    geoapp.openFile(inputFile)
    mdls = geoapp.getModels()
    # 3 逐个模型处理
    for md in mdls:
        if md == mdls[0]:
            continue  # 跳过“全局”
        print("------ mesh info -------")
        print("Models = " + md.name)
        # print("Model type = " +str(type(md)))
        mesh = geoapp.getMesh(md)
        # print("mesh_name = " + str(mesh.name))  # 空
        print("points_num = " + str(mesh.numPoints))
        print("faces_num = " + str(mesh.numTriangles))
        # getActiveTriangleSelection(md)

        # 优化：比例尺错误处理
        if mesh.maxBoxCoord.x() - mesh.minBoxCoord.x() < 10:
            trans = Transform3D()
            trans.setScaleTransform(1000)  # 放大1000倍
            mesh.transform(trans)
            print('[warn]wrong scale size,1000* process!')

        # 4 逐个类别处理
        activeModel = geoapp.getActiveModel()  # 所有mesh？
        # print(dir(activeModel))  # xuek 2025.07.11

        allmesh = geoapp.getMesh(activeModel)
        meshNum_rest = allmesh.numTriangles

        # print(allmesh.getNumMeshInstances())  # mesh数量??
        # print(str(classID.getNumMeshInstances()))  # classID未定义

        # faceGroup = ""
        for classNamei in classlist:
            xc = geoapp.getSelectionByName(activeModel, classNamei)  # 【可解析标签字符串】


            # print(faceIDs)
            # print("-------------")
            if xc == None:
                continue
            faceIDs[classNamei] = list(xc)  # 存在老标签。

            # print(len(faceIDs[id]))
            print('label: ' + str(classNamei) + ', face nums: ' + str(len(faceIDs[classNamei])))

            meshNum_rest -= xc.numSelected
        if meshNum_rest != 0:
            print("[warn]: left meshes" + str(meshNum_rest))
        # 5 写出group列表
        write_dict_to_binary(faceIDs, binaryLabelFile)


        # bool_overlap = True  # 是否覆盖文件
        if True:
            # 6 写出obj
            if not os.path.exists(obj_out_path):
                os.makedirs(obj_out_path)
            obj_out_file = os.path.join(obj_out_path, fileName + '.obj')
            if (not bool_overlap) and os.path.exists(obj_out_file):  # 已存在，且不覆盖文件
                print("obj has been writed before, will not rewrite here!!")
            else:
                filewriter = FileWrite()
                filewriter.filename = obj_out_file
                filewriter.mesh = mesh  # allmesh  # mesh
                filewriter.run()

            # 7 写出原始标签顺序
            dic_out_file = os.path.join(obj_out_path, fileName + '_old_face_dict.pth')
            if (not bool_overlap) and os.path.exists(dic_out_file):  # 已存在，且不覆盖文件
                print("'_old_face_dict.pth' has been writed before, will not rewrite here!!")
            else:
                face_indices = []  # 用于存储所有面对应的顶点索引的列表

                # test
                # print("xuek test   " + str(mesh.numTriangles))
                # testi = 230999
                # if mesh.isValidTriangle(testi):
                #     print("valide "+str(testi))
                # else:
                #     print("invalid "+str(testi))
                # print("xuek test end 1 ")
                # v1= mesh.getVertexIndex(2344 + 1, 0, 0)
                # print("xuek test end  ")

                is_deleted_meshids =True  # 是否处理由于删除面片导致的索引对应不上？
                if is_deleted_meshids:
                    num_mesh = mesh.numTriangles
                    i=0
                    while len(face_indices) < mesh.numTriangles:
                        if not mesh.isValidTriangle(i + 1):  # 当前面片编号无效
                            i += 1
                            continue
                        else:
                            # 处理
                            v1 = mesh.getVertexIndex(i + 1, 0, 0)
                            v2 = mesh.getVertexIndex(i + 1, 0, 1)
                            v3 = mesh.getVertexIndex(i + 1, 0, 2)
                            face_indices.append((v1, v2, v3))  # 将顶点索引作为一个元组添加到列表中
                            i += 1
                    if num_mesh > mesh.numTriangles:
                        print("[warn]:mesh may deleted before: nums = "+ str(num_mesh-mesh.numTriangles))

                else:  # 暂时不用，不判断面片编号是否正确
                    for i in range(mesh.numTriangles):
                        # test
                        # if i>=2343 and i<=2346:
                        #     print("------" + str(i))
                        #     # abc = OrientedTriangle(i, 0)
                        #     # a = m.getVertexIndex(abc, 0)
                        #     # b = m.getVertexIndex(abc, 1)
                        #     # c = m.getVertexIndex(abc, 2)
                        #     #
                        #     # print(a)
                        #     # print(b)
                        #     # print(c)
                        #     # print("22")
                        #     v1 = mesh.getVertexIndex(i + 1, 0, 0)
                        #     v2 = mesh.getVertexIndex(i + 1, 0, 1)
                        #     v3 = mesh.getVertexIndex(i + 1, 0, 2)
                        #     # face_indices.append((v1, v2, v3))  # 将顶点索引作为一个元组添加到列表中
                        #     print(v1)
                        #     print(v2)
                        #     print(v3)
                        # else:
                        #     continue
                        # if i>=2344 and i<=2360:
                        #     continue
                        # elif i>2360: #2376，2344，2345
                        #     print(i)
                        # print("xuek test   num" + str(i))
                        v1 = mesh.getVertexIndex(i + 1, 0, 0)
                        v2 = mesh.getVertexIndex(i + 1, 0, 1)
                        v3 = mesh.getVertexIndex(i + 1, 0, 2)
                        face_indices.append((v1, v2, v3))  # 将顶点索引作为一个元组添加到列表中
                        # print(v1)

                # print("finish __ face_id collect")

                write_dict_to_binary(face_indices, dic_out_file)



# 0 外部参数
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


if True:
    # 处理单个文件
    process_wrp_one()
else:
    # 批量处理文件
    path_root = r"D:\DATASET\BIMTwins\WRP\batch2"
    out_root = r"D:\DATASET\BIMTwins\WRP\out\batch2"
    for scene_name in os.listdir(path_root):
        # if scene_name != '39PTY2':
        #     continue
        path_folder = os.path.join(path_root, scene_name)  # r"D:\DATASET\BIMTwins\WRP\batch1\39PTY2"
        if os.path.isdir(path_folder):
            inputPath = path_folder
            outputPath = os.path.join(out_root, scene_name + '_out')
            log_file = outputPath + r"\log.txt"

            process_wrp(inputPath, outputPath, log_file)




