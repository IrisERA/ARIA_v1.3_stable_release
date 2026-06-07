"""
Download conversation datasets for MK1 training.
Run: py -3.11 mk1/download_conversations.py

Requires: py -3.11 -m pip install datasets
"""

import os

os.makedirs("data", exist_ok=True)

def save(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    mb = os.path.getsize(path) / 1024 / 1024
    print(f"Saved {path} ({mb:.1f} MB)")

# ── DailyDialog ────────────────────────────────────────────────────────────
try:
    from datasets import load_dataset
    print("Downloading DailyDialog...")
    ds = load_dataset("daily_dialog", split="train+validation+test", trust_remote_code=True)
    lines = []
    for item in ds:
        dialog = item["dialog"]
        for i in range(0, len(dialog) - 1, 2):
            user = dialog[i].strip()
            reply = dialog[i+1].strip() if i+1 < len(dialog) else ""
            if user and reply:
                lines.append(f"User: {user}")
                lines.append(f"MK1-H: {reply}")
        lines.append("")
    save("data/dailydialog.txt", "\n".join(lines))
except Exception as e:
    print(f"DailyDialog failed: {e}")

# ── BlendedSkillTalk ───────────────────────────────────────────────────────
try:
    print("Downloading BlendedSkillTalk...")
    ds = load_dataset("blended_skill_talk", split="train", trust_remote_code=True)
    lines = []
    for item in ds:
        convos = item.get("free_messages", [])
        guided = item.get("guided_messages", [])
        combined = []
        for a, b in zip(convos, guided):
            combined.append(a.strip())
            combined.append(b.strip())
        for i in range(0, len(combined) - 1, 2):
            if combined[i] and combined[i+1]:
                lines.append(f"User: {combined[i]}")
                lines.append(f"MK1-H: {combined[i+1]}")
        lines.append("")
    save("data/blended.txt", "\n".join(lines))
except Exception as e:
    print(f"BlendedSkillTalk failed: {e}")

# ── Empathetic Dialogues ───────────────────────────────────────────────────
try:
    print("Downloading EmpatheticDialogues...")
    ds = load_dataset("empathetic_dialogues", split="train", trust_remote_code=True)
    lines = []
    seen = set()
    for item in ds:
        conv_id = item.get("conv_id", "")
        if conv_id in seen:
            continue
        seen.add(conv_id)
        utterance = item.get("utterance", "").strip()
        context = item.get("context", "").strip()
        if context and utterance:
            lines.append(f"User: {context}")
            lines.append(f"MK1-H: {utterance}")
            lines.append("")
    save("data/empathetic.txt", "\n".join(lines))
except Exception as e:
    print(f"EmpatheticDialogues failed: {e}")

# ── Wizard of Wikipedia ────────────────────────────────────────────────────
try:
    print("Downloading Wizard of Wikipedia...")
    ds = load_dataset("wiki_dialog", split="train", trust_remote_code=True)
    lines = []
    count = 0
    for item in ds:
        turns = item.get("turns", [])
        for i in range(0, len(turns) - 1, 2):
            u = turns[i].get("utterance", "").strip()
            r = turns[i+1].get("utterance", "").strip() if i+1 < len(turns) else ""
            if u and r:
                lines.append(f"User: {u}")
                lines.append(f"MK1-H: {r}")
        lines.append("")
        count += 1
        if count >= 50000:
            break
    save("data/wiki_dialog.txt", "\n".join(lines))
except Exception as e:
    print(f"WizardOfWikipedia failed: {e}")

print("\nAll done! Delete mk1/data_cache.pt then retrain.")
