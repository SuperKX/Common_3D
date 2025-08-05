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

def replace_file_content(filename, start_line, new_content):
    """
    指定行，替换字符串。
    Args:
        filename (str): The path to the file.
        start_line (int): The line number (1-based) from which to start replacing.
        new_content (str): The new content to write to the file.
    """

    try:
        with open(filename, "r") as f:
            lines = f.readlines()  # Read all lines into a list

        # Validate start_line
        if not (1 <= start_line <= len(lines) + 1):  #Line 1 is the first line
            raise ValueError(f"start_line must be between 1 and {len(lines) + 1}")


        # Modify the lines list. index  = line_number -1
        if start_line <= len(lines):
             lines[start_line-1:] = new_content.splitlines(keepends=True)  #Splits the content into lines, preserves line endings
        else: # append
             lines.extend(new_content.splitlines(keepends=True))

        with open(filename, "w") as f:
            f.writelines(lines)  # Write the modified lines back to the file

        print(f"Successfully replaced content in {filename} starting from line {start_line}.")

    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
    except ValueError as e:
        print(f"Error: Invalid start_line: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")