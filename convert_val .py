import os
from midi2audio import FluidSynth
import soundfile as sf

def convert_midi_to_audio(midi_dir, audio_dir):
    # 创建一个FluidSynth实例
    fs = FluidSynth()

    # 获取目录中所有的MIDI文件
    midi_files = [f for f in os.listdir(midi_dir) if f.endswith('.midi') or f.endswith('.mid')]

    # 将每个MIDI文件转换为音频
    for midi_file in midi_files:
        # 构造MIDI文件的完整路径
        midi_path = os.path.join(midi_dir, midi_file)

        # 构造输出音频文件的完整路径
        audio_file = os.path.splitext(midi_file)[0] + '.wav'
        audio_path = os.path.join(audio_dir, audio_file)

        # 将MIDI文件转换为音频
        fs.midi_to_audio(midi_path, audio_path)

        # 打印成功消息
        print(f"Successfully converted {midi_file} to {audio_file}")

# 设置目录
midi_dir = '/root/autodl-tmp/midi-model-main/data/gen/2lc'  # MIDI文件的路径
audio_dir = '/root/autodl-tmp/midi-model-main/data/gen/2lc'  # 请替换为你想要保存音频文件的路径

# 调用函数将所有的MIDI文件转换为音频
convert_midi_to_audio(midi_dir, audio_dir)
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

def plot_mfcc(mfccs,file):

    # 绘制MFCC图
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()

    # 保存图像
    plt.savefig(file)
def standardize_audio_files_length(audio_dir, target_length_sec):
    #fs = FluidSynth()
    # 获取目录中所有的音频文件
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

    # 将每个音频文件的长度标准化
    for audio_file in audio_files:
        # 构造音频文件的完整路径
        audio_path = os.path.join(audio_dir, audio_file)

        # 加载音频文件
        x , sr = librosa.load(audio_path)

        # 设置目标长度
        target_length = sr * target_length_sec  # target_length_sec秒

        # 如果音频太长，裁剪它
        if len(x) > target_length:
            x = x[:target_length]

        # 如果音频太短，填充它
        elif len(x) < target_length:
            x = np.pad(x, (0, target_length - len(x)))

        # 确保x的长度等于target_length
        assert len(x) == target_length
        

        # 保存音频
        sf.write(audio_path, x, sr)

        # 保存标准化后的音频
        #librosa.output.write_wav(audio_path, x, sr)

        # 打印成功消息
        print(f"Successfully standardized the length of {audio_file}")
        
standardize_audio_files_length("/root/autodl-tmp/midi-model-main/data/gen/2lc", 30)
#standardize_audio_files_length("/root/autodl-tmp/midi-model-main/gen", 30)

import numpy as np
import librosa
from scipy.special import rel_entr

def calculate_kl_divergence(audio_path1, audio_path2, n_mfcc=13, n_bins=30):
    # 加载音频文件
    y1, sr1 = librosa.load(audio_path1)
    y2, sr2 = librosa.load(audio_path2)

    # 提取MFCC特征
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=n_mfcc)
    plot_mfcc(mfcc1,"mfcc1.png")
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=n_mfcc)
    plot_mfcc(mfcc2,"mfcc2.png")

    # 计算MFCC特征的概率分布
    hist1, _ = np.histogram(mfcc1.flatten(), bins=n_bins, density=True)
    hist2, _ = np.histogram(mfcc2.flatten(), bins=n_bins, density=True)

    # 计算KL散度
    kl_divergence = np.sum(rel_entr(hist1+1e-10, hist2+1e-10))

    return kl_divergence

# 调用函数计算KL散度
audio_path1 = '/root/autodl-tmp/midi-model-main/data/gen/2lc/c.wav'  
audio_path2 = '/root/autodl-tmp/midi-model-main/data/Val_ori/A Touch Of Blues - Bobby Blue Bland.wav'  
kl_divergence1 = calculate_kl_divergence(audio_path1, audio_path2)
audio_path1 = '/root/autodl-tmp/midi-model-main/data/gen/2lc/1.wav'  
#audio_path2 = '/root/autodl-tmp/midi-model-main/data/Val_ori/9 To 5 - Dolly Parton.wav'  
kl_divergence2 = calculate_kl_divergence(audio_path1, audio_path2)
audio_path1 = '/root/autodl-tmp/midi-model-main/data/gen/2lc/output.wav'  
#audio_path2 = '/root/autodl-tmp/midi-model-main/data/Val_ori/9 To 5 - Dolly Parton.wav'  
kl_divergence3 = calculate_kl_divergence(audio_path1, audio_path2)
print(f"The KL divergence avg of the midi prompt is: {(kl_divergence1+kl_divergence2+kl_divergence3)/3}")
