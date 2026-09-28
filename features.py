from __future__ import annotations

import numpy as np
import pandas as pd

SESSION_GAP_S = 30 * 60
PLATFORM_MAP = {"desktop": "web", "web": "web", "android": "android", "ios": "ios", "iphone": "ios"}
EVENT_NAMES = [
    "search_results_view", "item_view", "photo_swipe", "seller_page_view",
    "contact_phone_show", "contact_chat_open", "contact_message_sent",
    "favorite_add", "login",
]
CONTACT_EVENTS = ["contact_phone_show", "contact_chat_open", "contact_message_sent"]


def load_data(data_dir: str = "data"):
    dates = ["cookie_created_at", "window_start_ts", "window_end_ts"]
    train = pd.read_csv(f"{data_dir}/train.csv", parse_dates=dates)
    test = pd.read_csv(f"{data_dir}/test.csv", parse_dates=dates)
    events = pd.read_csv(f"{data_dir}/events.csv.gz", parse_dates=["event_ts"])
    return train, test, events
