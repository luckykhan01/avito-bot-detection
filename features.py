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


def clean_events(events: pd.DataFrame) -> pd.DataFrame:
    ev = events.drop_duplicates().copy()
    ev["platform"] = ev["platform"].str.lower().map(PLATFORM_MAP)
    return ev.sort_values(["cookie_id", "event_ts", "eid"], kind="mergesort").reset_index(drop=True)


def events_in_window(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    ev = events.merge(meta[["cookie_id", "window_start_ts", "window_end_ts"]], on="cookie_id")
    mask = (ev.event_ts >= ev.window_start_ts) & (ev.event_ts < ev.window_end_ts)
    return ev[mask].drop(columns=["window_start_ts", "window_end_ts"])


def basic_features(events_clean: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    ev = events_in_window(events_clean, meta)
    g = ev.groupby("cookie_id")
    f = pd.DataFrame({"n_events": g.size(), "item_nunique": g.item_id.nunique()})
    f = meta[["cookie_id"]].merge(f, left_on="cookie_id", right_index=True, how="left")
    return f.fillna(0).reset_index(drop=True)
