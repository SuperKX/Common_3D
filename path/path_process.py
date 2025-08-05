import os
import shutil

# 在字符串列表中找到包含该字符串的索引
def find_str_in_strlist(stri, strlist):
    '''
    在字符串列表中找到包含该字符串位置,不是字符串
    '''
    str_ids = []
    for i, stritem in enumerate(strlist):
        if stri in stritem:
            str_ids.append(i)
    return str_ids

def get_all_files(folder_path):
    '''
    获取所有文件
    '''
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files


def creat_folder_structure():
    '''
    创建跟输入目录同样结构的输出目录.
    '''
    return

def copy_rename_move(source_file, new_name, destination_folder):
    '''
    文件本地复制,并改名,然后移动到指定位置.
    source_file 文件名
    new_name 新名
    destination_folder 目标文件夹
    '''
    try:
        # 确保目标文件夹存在
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        # 构建新文件的完整路径
        new_file_path = os.path.join(destination_folder, new_name)

        # 复制文件并改名
        shutil.copy2(source_file, new_file_path)
        print(f"文件 {source_file} 已成功复制并重命名为 {new_name}，并移动到 {destination_folder}")
    except FileNotFoundError:
        print(f"错误: 源文件 {source_file} 未找到!")
    except PermissionError:
        print("错误: 没有足够的权限执行此操作!")
    except Exception as e:
        print(f"错误: 发生了一个未知错误: {e}")


def change_filename_in_multifolders(input_folder, output_folder, filename):
    '''
    遍历多级目录input_folder,将名为filename的文件以其所在文件夹名的方式命名.
    '''
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    _,ext = os.path.splitext(filename)

    for root, dirs, files in os.walk(input_folder):
        if filename not in files:
            continue
        new_path = root.replace(input_folder, output_folder)
        folder_to, new_filename = os.path.split(new_path)
        if not os.path.exists(folder_to):
            os.mkdir(folder_to)
        file_from = os.path.join(root,filename) # '/home/input/avename.pcd'
        filename_to = new_filename+ext
        copy_rename_move(file_from, filename_to, folder_to)
    return

def get_files_by_format(folder_path, formats=None):
    """
    获取指定文件夹下特定格式的文件/文件夹列表,并排序
    1) 不指定formats,则读入所有文件\文件夹; 只读文件,则['.*']??代验证

    :param folder_path: 文件夹路径
    :param formats: 文件格式列表，如 ['.txt', '.jpg'],如果不指定格式,则不传入
    :return: 符合格式的文件列表
    """
    file_list = []
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"错误: 文件夹 {folder_path} 不存在。")
    for root, dirs, files in os.walk(folder_path):
        # 1 所有文件\文件夹
        if formats == None:
            file_list.extend(dirs)
            file_list.extend(files)
        # 2 所有格式文件
        elif formats == ['.*']:
            file_list.extend(dirs)
        # 3 指定格式文件
        else:
            for file in files:
                if any(file.lower().endswith(fmt) for fmt in formats):
                    file_list.append(file)
        if len(file_list) == 0:
            print(f'文件夹中未找到指定目标!')
    return sorted(file_list)


if __name__ == '__main__':
    # change_filename_in_multifolders test
    input_folder = r'/home/xuek/桌面/TestData/input/staticmap_huangk_0415'
    output_folder = r'/home/xuek/桌面/TestData/input/staticmap_test/staticmap_test_0415_test'
    filename = r'map_ror.pcd'
    change_filename_in_multifolders(input_folder, output_folder, filename)
