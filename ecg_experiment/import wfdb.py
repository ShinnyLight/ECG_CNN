import wfdb
import os
import matplotlib.pyplot as plt

source_dirs = [
    r"D:\um\7023\brench dataset\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\records100",
    r"D:\um\7023\brench dataset\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\records500"
]
save_dir = r"D:\um\7023\ecg_experiment\ecg_images"

# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

# 遍历每个 source_dir
for src_dir in source_dirs:
    for folder in os.listdir(src_dir):
        folder_path = os.path.join(src_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.endswith(".dat"):
                record_name = file[:-4]
                full_record_path = os.path.join(folder_path, record_name)
                try:
                    # 读取 ECG 信号记录
                    record = wfdb.rdrecord(full_record_path)
                    # 使用 wfdb 自带的绘图函数生成 matplotlib 图像
                    fig = wfdb.plot_wfdb(record=record, title=record_name, return_fig=True)
                    # 保存为 .png
                    fig.savefig(os.path.join(save_dir, f"{record_name}.png"))
                    plt.close(fig)  # 释放内存
                    print(f"✅ Saved {record_name}.png")
                except Exception as e:
                    print(f"❌ Failed to process {record_name}: {e}")
