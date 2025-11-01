# ai_module.py — AI Manager with Feedback & Persistent Data for Melli v2.0

import torch
import pickle
import os
import json
from train_model import ChannelNet
import re
import random

class AIManager:
    def __init__(self, 
                 model_path: str = "models/model.pt", 
                 vectorizer_path: str = "models/vectorizer.pkl", 
                 save_file: str = "data/melli_data.json", 
                 feedback_file: str = "data/feedback.json"):
        self.save_file = save_file
        self.feedback_file = feedback_file

        # Load previous logs and mood
        self._load_data()
        self._load_feedback()

        # Load vectorizer
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        # Load model
        input_size = len(self.vectorizer.get_feature_names_out())
        self.model = ChannelNet(input_size)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    # -----------------------------
    # Persistent storage
    # -----------------------------
    def _load_data(self):
        if os.path.exists(self.save_file):
            with open(self.save_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.user_logs = data.get("user_logs", {})
                self.mood = data.get("mood", 0)
        else:
            self.user_logs = {}
            self.mood = 0

    def _save_data(self):
        os.makedirs(os.path.dirname(self.save_file), exist_ok=True)
        data = {
            "user_logs": self.user_logs,
            "mood": self.mood
        }
        with open(self.save_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    # -----------------------------
    # Feedback logging
    # -----------------------------
    def _load_feedback(self):
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                self.feedback_log = json.load(f)
        else:
            self.feedback_log = []

    def log_feedback(self, message: str, suggested_name: str, reaction: str):
        """Log dev feedback for later model training."""
        self.feedback_log.append({
            "message": message,
            "suggested_name": suggested_name,
            "reaction": reaction
        })
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        with open(self.feedback_file, "w", encoding="utf-8") as f:
            json.dump(self.feedback_log, f, ensure_ascii=False, indent=4)

    # -----------------------------
    # Channel prediction
    # -----------------------------
    def should_create_channel(self, message: str) -> bool:
        if not message.strip():
            return False
        X = self.vectorizer.transform([message]).toarray()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            pred = self.model(X_tensor)
            return pred.item() > 0.5

    def suggest_channel_name(self, message: str) -> str:
        stop_words = {"a", "an", "the", "for", "we", "should", "can", "let’s", "have", "place", "make", "me", "melli"}
        words = [w for w in re.sub(r"[^a-z0-9\s]", "", message.lower()).split() if w not in stop_words]
        if not words:
            return "new-channel"
        # Use past feedback to improve naming
        for fb in self.feedback_log[::-1]:  # check most recent feedback first
            if fb["reaction"] == "up" and fb["message"].lower() == message.lower():
                return fb["suggested_name"]
        return "-".join(words[:3])[:90]

    # -----------------------------
    # Mood system
    # -----------------------------
    def adjust_mood(self, success: bool):
        if success:
            self.mood += 1
        else:
            self.mood -= 1
        self.mood = max(-10, min(10, self.mood))
        self._save_data()

    def get_personality_response(self) -> str:
        positive_base = [
            "Yay! I did it! :3",
            "A treat?! Thanks! 🍉",
            "I’m the best, right? 😸",
            "Feeling awesome! 😎"
        ]
        negative_base = [
            "Ouch! That hurt… 😿",
            "Slap! Not nice… 😵",
            "I’ll do better next time… 🙁",
            "Why did you do that?! 😾"
        ]
        if self.mood > 5:
            positive_base.append("I feel unstoppable! 💪")
        if self.mood < -5:
            negative_base.append("I’m feeling grumpy… 😾")
        return random.choice(positive_base if self.mood >= 0 else negative_base)

    # -----------------------------
    # User logs
    # -----------------------------
    def log_user_message(self, user_id: int, message: str):
        uid = str(user_id)
        if uid not in self.user_logs:
            self.user_logs[uid] = []
        self.user_logs[uid].append(message)
        self._save_data()

    def delete_user_data(self, user_id: int):
        uid = str(user_id)
        if uid in self.user_logs:
            del self.user_logs[uid]
            self._save_data()
