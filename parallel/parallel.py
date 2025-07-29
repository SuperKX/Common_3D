# 并行处理函数


import concurrent.futures
from tqdm import tqdm

def parallel_process(function, args):
    '''
    并行处理批量数据。
    function 函数名
    args    传入的参数列表,每一组参数是一个元组，eg：[(para1,para2,para3),(...),...]
    return  返回结果列表，如果没有返回值，则返回None列表, eg: [result1,...]
    '''
    results = []
    # print(f"初始线程数: {threading.active_count()}")
    # 使用 ThreadPoolExecutor 创建线程池
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # print(f"线程池创建后线程数: {threading.active_count()}")  # 应该增加
        # 提交所有任务到线程池
        future_to_item = {executor.submit(function, *arg): arg for arg in args}
        # print(f"任务提交后线程数: {threading.active_count()}")  # 应该为1(主)+3(工作线程)
        # 获取完成的任务结果
        for future in tqdm(concurrent.futures.as_completed(future_to_item)):
            try:
                result = future.result()
                results.append(result)
                # print(f"完成处理：{future_to_item[future]}")
            except Exception as e:
                item = future_to_item[future]
                print(f"处理项目 {item} 时出错: {e}")
    return results



