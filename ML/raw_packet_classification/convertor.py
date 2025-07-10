# ──────────────────────────────────────────────────────────────────────────────
# 1)  Dataset generator  (≈ 10 000 rows, 80 % valid / 20 % invalid)
# ──────────────────────────────────────────────────────────────────────────────
import struct, random, csv, os
from ipaddress import ip_address
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def ip_to_int(ip_str):                       # "10.10.10.20" → 0x0A0A0A14
    return int(ip_address(ip_str))

def int_to_ip(i):                            # helper (not strictly needed)
    return str(ip_address(i))

def rand_ip():                               # generate 10.10.x.y
    return f"10.10.{random.randint(0,255)}.{random.randint(0,255)}"

def build_packet(src_ip, dst_ip, src_port, dst_port, teid):
    # IPv4 header (20 B)
    ip_hdr = struct.pack(
        "!BBHHHBBHII",
        0x45, 0, 20+8+8,                     # v=4, IHL=5, totalLen=36
        random.randint(0,65535),             # identification
        0x4000,                              # flags+fragOffset (DF)
        64, 17, 0,                           # TTL, proto=UDP, checksum=0
        ip_to_int(src_ip),
        ip_to_int(dst_ip)
    )
    # UDP header (8 B)
    udp_hdr = struct.pack("!HHHH", src_port, dst_port, 8+8, 0)
    # GTP header (8 B)
    gtp_hdr = struct.pack("!BBH", 0x30, 0xff, 20) + struct.pack("!I", teid)
    pkt = ip_hdr + udp_hdr + gtp_hdr
    return (pkt + b"\x00"*(40-len(pkt)))     # pad to exactly 40 B

def generate_csv(path="ann_training_data.csv",
                 n_rows=10_000, valid_ratio=0.8):
    rows, n_valid = [], int(n_rows*valid_ratio)
    # ─ valid samples ─
    for _ in range(n_valid):
        s_ip, d_ip = rand_ip(), rand_ip()
        teid = random.randint(1, 0xFFFFF)
        pkt  = build_packet(s_ip, d_ip, 2152, 2152, teid)
        rows.append([pkt.hex(), s_ip, d_ip, 2152, 2152, teid])
    # ─ invalid samples ─
    for _ in range(n_rows - n_valid):
        s_ip, d_ip = rand_ip(), rand_ip()
        teid = random.randint(1, 0xFFFFF)
        choice = random.random()
        if choice < 0.5:                     # mismatch IP
            wrong_s_ip = rand_ip()
            pkt = build_packet(wrong_s_ip, d_ip, 2152, 2152, teid)
        elif choice < 0.75:                  # mismatch port
            wrong_s_port = random.randint(1024,60000)
            pkt = build_packet(s_ip, d_ip, wrong_s_port, 2152, teid)
        else:                                # TEID = 0
            pkt = build_packet(s_ip, d_ip, 2152, 2152, 0)
        rows.append([pkt.hex(), s_ip, d_ip, 2152, 2152, 0])
    # write
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["HexData","SrcIP","DstIP","SrcPort","DstPort","TEID"])
        csv.writer(f).writerows(rows)
    print(f"CSV saved → {os.path.abspath(path)}  ({len(rows)} rows)")

generate_csv()          # comment out if CSV already exists
# ──────────────────────────────────────────────────────────────────────────────
# 2)  Data pipeline  (hex → float tensor, label = TEID)
# ──────────────────────────────────────────────────────────────────────────────
def hex_to_byte_float(hexstr):
    return np.frombuffer(bytes.fromhex(hexstr), dtype=np.uint8) / 255.0  # (40,)

def ip_to_bytes(ip_str):
    return np.frombuffer(struct.pack("!I", ip_to_int(ip_str)), np.uint8) / 255.0  # (4,)

def build_feature_vector(row):
    pkt   = hex_to_byte_float(row.HexData)            # 40
    sip   = ip_to_bytes(row.SrcIP)                    # 4
    dip   = ip_to_bytes(row.DstIP)                    # 4
    sport = np.array([row.SrcPort/65535.0], np.float32)
    dport = np.array([row.DstPort/65535.0], np.float32)
    return np.concatenate([pkt, sip, dip, sport, dport])  # 40+4+4+1+1 = 50

df = pd.read_csv("ann_training_data.csv")
X  = np.vstack(df.apply(build_feature_vector, axis=1).to_numpy())
y  = df.TEID.to_numpy(np.float32).reshape(-1,1)       # regression target

# Split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ──────────────────────────────────────────────────────────────────────────────
# 3)  Build & train Keras model
# ──────────────────────────────────────────────────────────────────────────────
model = keras.Sequential([
    layers.Input(shape=(50,)),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(1)                     # linear output (regression)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=20, batch_size=128)

# ──────────────────────────────────────────────────────────────────────────────
# 4)  Post‑training usage
# ──────────────────────────────────────────────────────────────────────────────
def predict_teid(raw_hex, src_ip, dst_ip, src_port, dst_port):
    vec = build_feature_vector(
        pd.Series([raw_hex, src_ip, dst_ip, src_port, dst_port],
                  index=["HexData","SrcIP","DstIP","SrcPort","DstPort"]))
    pred = model.predict(vec.reshape(1,-1))[0,0]
    # heuristic: if prediction < 0.5 → treat as invalid
    return int(round(pred)) if pred > 0.5 else 0

# quick sanity check on one row
print("Predicted TEID on first row:",
      predict_teid(df.HexData[0], df.SrcIP[0], df.DstIP[0],
                   df.SrcPort[0], df.DstPort[0]))
