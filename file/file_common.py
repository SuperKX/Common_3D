'''
文件处理相关函数
'''
def add_to_file(str_to_write, filepath):
    """
        文件分批写入
    """
    with open(filepath, 'a') as f:  # 追加模式
        f.write(str_to_write)


def copy_first_n_lines(input_file, output_file, n):
    """逐行读取复制前n行，内存高效"""
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for i, line in enumerate(f_in):
            if i >= n:
                break
            f_out.write(line)
