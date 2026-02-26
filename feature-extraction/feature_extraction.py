import librosa
import numpy as np
from scipy.fftpack import dct

# ---- 可视化依赖（保存图片，适合无GUI环境）----
import matplotlib
matplotlib.use("Agg")   #生成PNG图片，用于非交互式后端
import matplotlib.pyplot as plt


# preemphasis config
alpha = 0.97

# Enframe config
frame_len = 400      # 25ms if fs=16kHz
frame_shift = 160    # 10ms if fs=16kHz
fft_len = 512

# Mel filter config
num_filter = 23
num_mfcc = 12


def plot_feature_heatmap(feats, title, out_png, x_label="Frame index", y_label="Feature dim"):
    """
    将特征矩阵画成热力图并保存。
    feats: shape (num_frames, feat_dim)
    为了符合“纵轴=特征维度，横轴=时间”，这里会转置成 (feat_dim, num_frames) 再绘制。
    """
    feats_show = feats.T  # (feat_dim, num_frames)

    plt.figure(figsize=(16, 4))
    plt.imshow(feats_show, aspect="auto", origin="lower", interpolation="nearest")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def preemphasis(signal, coeff=alpha):
    """perform preemphasis on the input signal."""
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def enframe(signal, frame_len=frame_len, frame_shift=frame_shift, win=np.hamming(frame_len)):
    """Enframe with Hamming window function."""
    num_samples = signal.size
    num_frames = np.floor((num_samples - frame_len) / frame_shift) + 1
    frames = np.zeros((int(num_frames), frame_len))
    for i in range(int(num_frames)):
        frames[i, :] = signal[i * frame_shift:i * frame_shift + frame_len]
        frames[i, :] = frames[i, :] * win
    return frames


def get_spectrum(frames, fft_len=fft_len):
    """Get magnitude spectrum using FFT."""
    cFFT = np.fft.fft(frames, n=fft_len)
    valid_len = int(fft_len / 2) + 1
    spectrum = np.abs(cFFT[:, 0:valid_len])
    return spectrum


def hz2mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel2hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def fbank(spectrum, fs, num_filter=num_filter, fft_len=fft_len):
    """Get log mel filter bank feature from spectrum.
    Returns: (num_frames, num_filter) and already applies log.
    """
    num_frames, num_bins = spectrum.shape  # num_bins = fft_len/2 + 1

    low_hz = 0.0
    high_hz = fs / 2.0

    low_mel = hz2mel(low_hz)
    high_mel = hz2mel(high_hz)

    mel_points = np.linspace(low_mel, high_mel, num_filter + 2)
    hz_points = mel2hz(mel_points)

    bins = np.floor((fft_len + 1) * hz_points / fs).astype(int)

    fbanks = np.zeros((num_filter, num_bins), dtype=np.float64)

    for m in range(1, num_filter + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        if right > num_bins - 1:
            right = num_bins - 1

        for k in range(left, center):
            fbanks[m - 1, k] = (k - left) / (center - left)

        for k in range(center, right):
            fbanks[m - 1, k] = (right - k) / (right - center)

    mel_energy = np.dot(spectrum, fbanks.T)
    mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)
    log_mel = np.log(mel_energy)

    return log_mel


def mfcc(fbank_feats, num_mfcc=num_mfcc):
    """Get mfcc feature from log mel filter bank feature.
    Returns: (num_frames, num_mfcc)
    """
    mfcc_feats = dct(fbank_feats, type=2, axis=1, norm='ortho')[:, :num_mfcc]
    return mfcc_feats


def write_file(feats, file_name):
    """Write the feature to file (plain text)."""
    with open(file_name, 'w') as f:
        (row, col) = feats.shape
        for i in range(row):
            f.write('[')
            for j in range(col):
                f.write(str(feats[i, j]) + ' ')
            f.write(']\n')


def main():
    wav, fs = librosa.load('02-feature-extraction_test.wav', sr=None)

    # 1) 预加重
    signal = preemphasis(wav)

    # 2) 分帧 + 加窗
    frames = enframe(signal)

    # 3) 幅度谱
    spectrum = get_spectrum(frames)

    # 4) 23维 log-Mel FBank
    fbank_feats = fbank(spectrum, fs=fs)

    # 5) 12维 MFCC
    mfcc_feats = mfcc(fbank_feats)

    # 6) 保存纯文本
    write_file(fbank_feats, 'test.fbank')
    write_file(mfcc_feats, 'test.mfcc')

    # 7) 可视化保存
    plot_feature_heatmap(
        fbank_feats,
        title="Log-Mel Filter Bank (23 dims)",
        out_png="fbank.png",
        y_label="Mel filter index"
    )
    plot_feature_heatmap(
        mfcc_feats,
        title="MFCC (12 dims)",
        out_png="mfcc.png",
        y_label="MFCC coefficient index"
    )

    print("Done. Saved: test.fbank, test.mfcc, fbank.png, mfcc.png")


if __name__ == '__main__':
    main()