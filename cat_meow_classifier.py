"""
貓叫聲規則式意圖分類器 (v0 - 無需訓練模型的雛形版本)

依據聲學研究線索設計的規則:
- 正向情緒 → 基頻走勢上升、時長較短
- 壓力/負向情緒 → 基頻走勢下降、平均基頻偏低、時長較長
- 呼嚕(滿足) → 極低頻、週期性強、噪音程度低
- 嘶聲/低吼(警戒防禦) → 噪音程度高(諧噪比低、過零率高)
- 顫音/短促上揚(打招呼) → 時長很短、基頻快速調變

需要套件: librosa, numpy, soundfile
安裝: pip install librosa numpy soundfile --break-system-packages

注意: 刻意不使用 librosa.pyin/librosa.yin 這類會觸發 numba JIT 編譯的函式,
因為第一次編譯會瞬間吃到 300-500MB 記憶體,在 Render 免費方案(512MB)上會被
系統 SIGKILL 砍掉。改用手寫的簡易自相關音高偵測,純 numpy 運算,記憶體足夠輕量。
"""

import numpy as np
import librosa


def _estimate_pitch_track(y: np.ndarray, sr: int, frame_size: int = 2048, hop: int = 512,
                           fmin: float = 60.0, fmax: float = 1500.0) -> np.ndarray:
    """簡易逐幀自相關音高偵測,不依賴 numba,回傳每一幀的頻率(0 代表無聲/雜訊)"""
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)

    pitches = []
    for start in range(0, max(1, len(y) - frame_size), hop):
        frame = y[start:start + frame_size]
        if len(frame) < frame_size:
            break

        # 能量太低直接判無聲,省掉不必要的運算
        if np.sqrt(np.mean(frame ** 2)) < 0.01:
            pitches.append(0.0)
            continue

        frame = frame - np.mean(frame)
        # 用 FFT 算自相關(比暴力迴圈快,純 numpy 不觸發 numba)
        corr = np.fft.irfft(np.abs(np.fft.rfft(frame, n=2 * frame_size)) ** 2)
        corr = corr[:frame_size]

        if max_lag >= len(corr):
            pitches.append(0.0)
            continue

        segment = corr[min_lag:max_lag]
        if len(segment) == 0 or np.max(segment) <= 0:
            pitches.append(0.0)
            continue

        peak_lag = min_lag + int(np.argmax(segment))
        confidence = segment.max() / (corr[0] + 1e-9)

        if confidence < 0.3:  # 週期性太弱,當作無聲/雜訊處理
            pitches.append(0.0)
        else:
            pitches.append(sr / peak_lag)

    return np.array(pitches)


def extract_features(audio_path: str) -> dict:
    """從音檔擷取分類所需的聲學特徵"""
    y, sr = librosa.load(audio_path, sr=None)

    duration = librosa.get_duration(y=y, sr=sr)

    pitch_track = _estimate_pitch_track(y, sr)
    f0_voiced = pitch_track[pitch_track > 0]

    if len(f0_voiced) < 3:
        # 抓不到穩定音高,通常代表噪音型聲音(嘶聲)或錄音太短
        mean_f0 = 0.0
        f0_trend = 0.0
    else:
        mean_f0 = float(np.mean(f0_voiced))
        # 走勢: 用首尾兩段的平均差,正值=上升,負值=下降
        n = len(f0_voiced)
        head = np.mean(f0_voiced[: max(1, n // 3)])
        tail = np.mean(f0_voiced[-max(1, n // 3):])
        f0_trend = float(tail - head)

    # 噪音程度: 過零率越高、頻譜平坦度越高 → 越接近嘶聲/噪音而非純音
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    # 週期性強度(抓呼嚕的規律震動): 用自相關的峰值強度當代理指標
    autocorr = librosa.autocorrelate(y, max_size=sr // 2)
    periodicity = float(np.max(autocorr[sr // 50:]) / (np.max(autocorr) + 1e-6))

    return {
        "duration": duration,
        "mean_f0": mean_f0,
        "f0_trend": f0_trend,
        "noisiness": (zcr + flatness) / 2,
        "periodicity": periodicity,
    }


def classify_intent(features: dict) -> dict:
    """規則式判斷,回傳最可能的意圖 + 簡短說明"""
    duration = features["duration"]
    mean_f0 = features["mean_f0"]
    f0_trend = features["f0_trend"]
    noisiness = features["noisiness"]
    periodicity = features["periodicity"]

    # 1. 呼嚕聲: 極低頻、規律性強、聲音持續較長
    if mean_f0 < 100 and periodicity > 0.4 and duration > 1.0:
        return {"intent": "滿足放鬆", "message": "牠現在很放鬆,可能覺得很滿足喔"}

    # 2. 嘶聲/低吼: 噪音程度高,抓不到穩定音高
    if noisiness > 0.35 and mean_f0 < 50:
        return {"intent": "警戒防禦", "message": "牠現在有點警戒或不高興,先給牠一點空間"}

    # 3. 打招呼/親近: 時長短、基頻快速上升
    if duration < 0.5 and f0_trend > 30:
        return {"intent": "打招呼", "message": "牠在跟你打招呼,聽起來很開心看到你"}

    # 4. 不安焦慮: 時長長、基頻偏低、走勢下降
    if duration > 1.0 and f0_trend < -20:
        return {"intent": "不安焦慮", "message": "牠聽起來有點緊張或不舒服,可以觀察一下環境"}

    # 5. 討東西(食物/關注): 基頻走勢上升、時長中等
    if f0_trend > 0 and 0.4 <= duration <= 1.2:
        return {"intent": "討東西", "message": "牠可能肚子餓了,或想引起你的注意"}

    # 都不符合明顯規則 → 給預設但標低信心
    return {"intent": "無法判斷", "message": "這次的叫聲不太明顯,再錄一次試試看"}


def analyze(audio_path: str) -> dict:
    """對外的主要接口:輸入音檔路徑,回傳分類結果"""
    features = extract_features(audio_path)
    result = classify_intent(features)
    result["features"] = features  # 保留原始特徵方便除錯/日後訓練模型用
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python cat_meow_classifier.py <音檔路徑>")
        sys.exit(1)

    output = analyze(sys.argv[1])
    print(f"判斷意圖: {output['intent']}")
    print(f"訊息: {output['message']}")
    print(f"原始特徵: {output['features']}")