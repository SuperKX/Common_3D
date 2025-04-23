import json
import os
import WindowReasoning as wr

if __name__ == '__main__':

    # 输入参数
    testMode = True  # 是否输出中间文件

    # # 图片所在文件夹
    # inputImagesFolder = r'H:\LOD3\LOD3SAutoExe\code\LayoutReasoning\limian20250225\testdata\70pic\img70'
    # # json文件地址
    # inputJsonPath = r'H:\LOD3\LOD3SAutoExe\code\LayoutReasoning\limian20250225\testdata\70pic\r_without_reason.json'  # result_50
    # # 临时文件夹(不写出可不给)
    # tempFolder = r'H:\LOD3\LOD3SAutoExe\code\LayoutReasoning\limian20250225\testdata\70pic\result70'
    # outputJsonPath = r'H:\LOD3\LOD3SAutoExe\code\LayoutReasoning\limian20250225\testdata\70pic\result70\r_without_reason_out.json'

    # 108pic
    # 图片所在文件夹
    inputImagesFolder = r'H:\TestData\layoutReasoning\0org_images\识别率测试数据108'
    # json文件地址
    inputJsonPath = r'H:\TestData\layoutReasoning\1json\result108_noreason.json'  #
    # 临时文件夹(不写出可不给)
    tempFolder = r'H:\TestData\layoutReasoning\v0421\temp'
    outputJsonPath = r'H:\TestData\layoutReasoning\v0421\result108_reason_out.json'


    # 解析json
    with open(inputJsonPath, 'r', encoding='utf-8') as f:
        inputJson = json.load(f)

    # [主函数]
    wr.layoutResaoning(inputJson, inputImagesFolder, tempFolder, testMode)

    # 写出Json
    with open(outputJsonPath, "w") as f:
        json.dump(inputJson, f)

    # # 验证生成文件
    # if testMode:
    #     with open(outputJsonPath, 'r', encoding='utf-8') as f_out:
    #         inputJson_out = json.load(f_out)
    #     tempFolder_out = os.path.join(tempFolder, '2_val')
    #     wr.initLayout(inputJson_out, inputImagesFolder, tempFolder_out, True)


