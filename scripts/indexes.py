from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#Common Emotions 6 Emotions
#Ravdess Emotions 6 emotions


RAVDESS_RAW_DIR= PROJECT_ROOT / "data" / "raw" / "ravdess"
RAVDESS_INDEX_DIR= PROJECT_ROOT / "data" / "ravdess_index.csv"

RAVDESS_EMOTION_MAP = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}


RAVDESS_IDS= {1,3,4,5,6,7} 
#because out model trains both the radvess and cremad which have 6 overlapping emotions


def build_ravdess_index():
    if not RAVDESS_RAW_DIR.exists():
        print(f"[RAVDESS] Folder not found: {RAVDESS_RAW_DIR}")
        return

    rows = []

    for wav_path in RAVDESS_RAW_DIR.rglob("*.wav"):
        fname = wav_path.stem  # e.g. "03-01-05-01-02-01-12"
        parts = fname.split("-")
        if len(parts) != 7:
            print(f"[RAVDESS] Skipping weird filename: {wav_path}")
            continue

        try:
            emotion_id = int(parts[2])  
        except ValueError:
            print(f"[RAVDESS] Cannot parse emotion from: {wav_path}")
            continue

        # Skip calm (2) and surprised (8)
        if emotion_id not in RAVDESS_IDS:
            continue

        emotion = RAVDESS_EMOTION_MAP[emotion_id]

        # Tag whether this file is speech or song
        path_str = str(wav_path)
        if "Audio_Speech_Actors_01-24" in path_str:
            subset = "speech"
        elif "Audio_Song_Actors_01-24" in path_str:
            subset = "song"
        else:
            subset = "unknown"

        rows.append({
            "path": str(wav_path.resolve()),
            "emotion": emotion,
            "emotion_id": emotion_id,
            "dataset": "ravdess",
            "subset": subset,
        })

    if not rows:
        print("[RAVDESS] No matching wav files found! Check folder + filenames.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(RAVDESS_INDEX_DIR, index=False)
    print(f"[RAVDESS] Indexed {len(df)} files ")




RAVDESS_VIDEO_RAW_DIR= PROJECT_ROOT / "data" / "raw" / "ravdess"
RAVDESS_VIDEO_INDEX_DIR= PROJECT_ROOT / "data" / "ravdess_video_index.csv"




def build_ravdess_video_index():
    if not RAVDESS_VIDEO_RAW_DIR.exists():
        print(f"[RAVDESS-VIDEO] Folder not found: {RAVDESS_VIDEO_RAW_DIR}")
        return

    rows = []

    for vid_path in RAVDESS_VIDEO_RAW_DIR.rglob("*.mp4"):
        fname = vid_path.stem  
        parts = fname.split("-")
        if len(parts) != 7:
            print(f"[RAVDESS-VIDEO] Skipping weird filename: {vid_path}")
            continue

        try:
            emotion_id = int(parts[2]) 
        except ValueError:
            print(f"[RAVDESS-VIDEO] Cannot parse emotion from: {vid_path}")
            continue

        if emotion_id not in RAVDESS_IDS:
            continue

        emotion = RAVDESS_EMOTION_MAP[emotion_id]

        path_str = str(vid_path)
        if "Video_Song_Actor" in path_str:
            subset = "video_song"
        elif "Video_Speech_Actor" in path_str:
            subset = "video_speech"
        else:
            subset = "video_unknown"

        rows.append({
            "path": str(vid_path.resolve()),
            "emotion": emotion,
            "emotion_id": emotion_id,
            "dataset": "ravdess_video",
            "subset": subset,
        })

    if not rows:
        print("[RAVDESS-VIDEO] No matching mp4 files found! Check folder + filenames.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(RAVDESS_VIDEO_INDEX_DIR, index=False)
    print(f"[RAVDESS-VIDEO] Indexed {len(df)} files ")






CREMAD_RAW_DIR =  PROJECT_ROOT / "data" / "raw" / "cremad" / "AudioWAV"
CREMAD_INDEX_DIR =  PROJECT_ROOT / "data" / "cremad_index.csv"

CREMAD_EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def build_cremad_index():
    if not CREMAD_RAW_DIR.exists():
        print(f"[CREMA-D] Folder not found: {CREMAD_RAW_DIR}")
        return

    rows = []

    for wav_path in CREMAD_RAW_DIR.rglob("*.wav"):
        fname = wav_path.stem     
        parts = fname.split("_")
        if len(parts) < 3:
            print(f"[CREMA-D] Skipping weird filename: {wav_path}")
            continue

        emotion_code = parts[2]    
        emotion = CREMAD_EMOTION_MAP.get(emotion_code)
        if emotion is None:
            print(f"[CREMA-D] Unknown emotion code '{emotion_code}' in {wav_path}")
            continue

        # Match RAVDESS IDs for 6 emotions:
        # neutral=1, happy=3, sad=4, angry=5, fearful=6, disgust=7
        if emotion == "neutral":
            emotion_id = 1
        elif emotion == "happy":
            emotion_id = 3
        elif emotion == "sad":
            emotion_id = 4
        elif emotion == "angry":
            emotion_id = 5
        elif emotion == "fearful":
            emotion_id = 6
        elif emotion == "disgust":
            emotion_id = 7
        else:
            print(f"[CREMA-D] Unexpected emotion '{emotion}' in {wav_path}")
            continue

        rows.append({
            "path": str(wav_path.resolve()),
            "emotion": emotion,
            "emotion_id": emotion_id,
            "dataset": "cremad",
        })

    if not rows:
        print("[CREMA-D] No wav files indexed! Check folder + filenames.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(CREMAD_INDEX_DIR, index=False)
    print(f"[CREMA-D] Indexed {len(df)} files ")




def main():
    print("Building RAVDESS Audio index (6 emotions)")
    build_ravdess_index()
    
    print("Building RAVDESS Video Index (6 emotions)")
    build_ravdess_video_index()

    print("Building CREMA-D index (6 emotions)")

    build_cremad_index()
    print("Done.")


if __name__ == "__main__":
    main()






