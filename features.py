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


def _safe_div(a, b):
    return np.where(b > 0, a / np.maximum(b, 1e-9), np.nan)


def timing_features(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev[["cookie_id", "event_ts"]].copy()
    ev["dt"] = ev.groupby("cookie_id").event_ts.diff().dt.total_seconds()
    ev["hour"] = ev.event_ts.dt.hour
    ev["minute_bucket"] = ev.event_ts.dt.floor("min")
    g = ev.groupby("cookie_id")

    f = pd.DataFrame({
        "duration_s": (g.event_ts.max() - g.event_ts.min()).dt.total_seconds(),
        "dt_min": g.dt.min(),
        "dt_median": g.dt.median(),
        "dt_mean": g.dt.mean(),
        "dt_std": g.dt.std(),
        "dt_q10": g.dt.quantile(0.1),
        "dt_share_le2": g.dt.apply(lambda x: (x.dropna() <= 2).mean()),
        "dt_share_le10": g.dt.apply(lambda x: (x.dropna() <= 10).mean()),
        "n_sessions": g.dt.apply(lambda x: 1 + (x > SESSION_GAP_S).sum()),
        "hours_active": g.hour.nunique(),
        "night_share": g.hour.apply(lambda h: h.between(0, 5).mean()),
        "first_hour": g.hour.first(),
        "max_events_per_min": ev.groupby(["cookie_id", "minute_bucket"]).size().groupby("cookie_id").max(),
    })
    f["dt_cv"] = _safe_div(f.dt_std, f.dt_mean)
    f["dt_unique_ratio"] = g.dt.apply(lambda x: x.dropna().nunique() / max(len(x.dropna()), 1))
    return f


def activity_features(ev: pd.DataFrame) -> pd.DataFrame:
    g = ev.groupby("cookie_id")
    counts = pd.crosstab(ev.cookie_id, ev.event_name).reindex(columns=EVENT_NAMES, fill_value=0)
    n = counts.sum(axis=1)

    f = pd.DataFrame({"n_events": n})
    for c in EVENT_NAMES:
        f[f"share_{c}"] = counts[c] / n
    f["n_item_view"] = counts["item_view"]
    f["n_search"] = counts["search_results_view"]
    f["n_contacts"] = counts[CONTACT_EVENTS].sum(axis=1)
    f["has_login"] = (counts["login"] > 0).astype(int)
    f["swipes_per_view"] = _safe_div(counts["photo_swipe"], counts["item_view"])
    f["contacts_per_view"] = _safe_div(f.n_contacts, counts["item_view"])
    f["favs_per_view"] = _safe_div(counts["favorite_add"], counts["item_view"])
    f["views_per_search"] = _safe_div(counts["item_view"], counts["search_results_view"])
    f["n_event_types"] = (counts > 0).sum(axis=1)
    return f.join(g.platform.first().rename("platform"))


def diversity_features(ev: pd.DataFrame) -> pd.DataFrame:
    g = ev.groupby("cookie_id")
    f = pd.DataFrame({
        "item_nunique": g.item_id.nunique(),
        "cat_nunique": g.item_category.nunique(),
        "loc_nunique": g.item_location.nunique(),
        "query_nunique": g.search_query.nunique(),
        "n_query_events": g.search_query.count(),
        "n_item_events": g.item_id.count(),
        "share_pro_seller": g.seller_type.apply(lambda s: (s == "pro").mean() if s.notna().any() else np.nan),
    })
    f["item_repeat_ratio"] = _safe_div(f.n_item_events, f.item_nunique)
    f["query_unique_ratio"] = _safe_div(f.query_nunique, f.n_query_events)
    f["loc_per_event"] = _safe_div(f.loc_nunique, g.item_location.count())
    f["cat_per_event"] = _safe_div(f.cat_nunique, g.item_category.count())
    f["top_cat_share"] = g.item_category.apply(
        lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else np.nan)

    s = ev.loc[ev.search_page.notna(), ["cookie_id", "search_query", "search_page"]].copy()
    s["page_step"] = s.groupby(["cookie_id", "search_query"]).search_page.diff()
    gs = s.groupby("cookie_id")
    f = f.join(pd.DataFrame({
        "page_max": gs.search_page.max(),
        "page_mean": gs.search_page.mean(),
        "page_share_ge5": gs.search_page.apply(lambda p: (p >= 5).mean()),
        "page_step1_share": gs.page_step.apply(lambda d: (d.dropna() == 1).mean() if d.notna().any() else np.nan),
    }))
    return f


def pointer_features(ev: pd.DataFrame) -> pd.DataFrame:
    web = ev[ev.platform == "web"]
    g = web.groupby("cookie_id")
    f = pd.DataFrame({
        "pointer_share": g.pointer_x.apply(lambda x: x.notna().mean()),
        "pointer_x_mean": g.pointer_x.mean(),
        "pointer_x_std": g.pointer_x.std(),
        "pointer_y_std": g.pointer_y.std(),
    })
    p = web.dropna(subset=["pointer_x"])
    gp = p.groupby("cookie_id")
    f["pointer_unique_ratio"] = gp.apply(
        lambda d: len(d[["pointer_x", "pointer_y"]].drop_duplicates()) / len(d), include_groups=False)
    f["pointer_edge_share"] = gp.apply(
        lambda d: ((d.pointer_x <= 0) | (d.pointer_x >= 1919) | (d.pointer_y <= 0) | (d.pointer_y >= 1080)).mean(),
        include_groups=False)
    return f


def meta_features(meta: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    first = ev.groupby("cookie_id").event_ts.min()
    m = meta.set_index("cookie_id")
    return pd.DataFrame({
        "cookie_age_h": (m.window_start_ts - m.cookie_created_at).dt.total_seconds() / 3600,
        "age_at_first_event_h": (first.reindex(m.index) - m.cookie_created_at).dt.total_seconds() / 3600,
        "created_in_window": (m.cookie_created_at >= m.window_start_ts).astype(int),
    })


def ua_features(ev: pd.DataFrame) -> pd.DataFrame:
    ua = ev.groupby("cookie_id").user_agent.agg(["first", "nunique"])
    s = ua["first"].str.lower()
    return pd.DataFrame({
        "ua_nunique": ua["nunique"],
        "ua_automation": s.str.contains("headless|python|curl|wget|bot|spider|scrapy|selenium").astype(int),
        "ua_chrome_ver": s.str.extract(r"chrome/(\d+)")[0].astype(float),
        "ua_is_app": s.str.startswith("avito/").astype(int),
    })


def build_features(events_clean: pd.DataFrame, meta: pd.DataFrame, with_ua: bool = False) -> pd.DataFrame:
    ev = events_in_window(events_clean, meta)
    blocks = [activity_features(ev), timing_features(ev), diversity_features(ev),
              pointer_features(ev), meta_features(meta, ev)]
    if with_ua:
        blocks.append(ua_features(ev))
    f = pd.concat(blocks, axis=1)
    f = meta[["cookie_id"]].merge(f, left_on="cookie_id", right_index=True, how="left")
    f["n_events"] = f["n_events"].fillna(0)
    f["platform"] = f["platform"].fillna("none").astype("category")
    return f.reset_index(drop=True)
