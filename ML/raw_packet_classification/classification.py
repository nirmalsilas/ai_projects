# =======================================================================
# 0)  Imports
# =======================================================================
import os, random, struct, csv
import numpy as np
import pandas as pd
from ipaddress import ip_address
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------
# 1)  Utility helpers
# -----------------------------------------------------------------------
def ip_to_int(ip_str):            # "10.10.10.20" -> 0x0A0A0A14
    return int(ip_address(ip_str))

def rand_ip():                    # e.g., 10.10.123.45
    return f"10.10.{random.randint(0,255)}.{random.randint(0,255)}"

def build_40byte_packet(src_ip, dst_ip, src_port, dst_port, teid):
    """Return exactly‑40‑byte raw header (IP+UDP+GTP)."""
    # IPv4 header (20 B)
    ip_hdr = struct.pack(
        "!BBHHHBBHII",
        0x45, 0, 20+8+8,                 # Version/IHL, TOS, TotalLen
        random.randint(0, 65535),        # Identification
        0x4000,                          # Flags+FragmentOffset (DF)
        64, 17, 0,                       # TTL, Protocol=UDP, Checksum=0
        ip_to_int(src_ip),
        ip_to_int(dst_ip)
    )
    # UDP header (8 B)
    udp_hdr = struct.pack("!HHHH", src_port, dst_port, 8+8, 0)
    # GTP header (8 B): flags=0x30, msgType=0xFF, length=20
    gtp_hdr = struct.pack("!BBH", 0x30, 0xFF, 20) + struct.pack("!I", teid)
    packet = ip_hdr + udp_hdr + gtp_hdr
    return packet + b"\x00" * (40 - len(packet))

# -----------------------------------------------------------------------
# 2)  Dataset generator
# -----------------------------------------------------------------------
def generate_csv(path="ann_training_data.csv",
                 n_rows=10_000, valid_ratio=0.8):
    rows = []
    n_valid = int(n_rows * valid_ratio)

    # ---- valid rows ----
    for _ in range(n_valid):
        s_ip, d_ip = rand_ip(), rand_ip()
        teid = random.randint(1, 0xFFFFF)
        pkt  = build_40byte_packet(s_ip, d_ip, 2152, 2152, teid)
        rows.append([pkt.hex(), s_ip, d_ip, 2152, 2152, teid, 1])

    # ---- invalid rows ----
    for _ in range(n_rows - n_valid):
        s_ip, d_ip = rand_ip(), rand_ip()
        teid = random.randint(1, 0xFFFFF)
        case = random.random()
        if case < 0.5:                         # mismatched source IP
            wrong_s_ip = rand_ip()
            pkt = build_40byte_packet(wrong_s_ip, d_ip, 2152, 2152, teid)
        elif case < 0.75:                      # mismatched source port
            wrong_s_port = random.randint(1024, 60000)
            pkt = build_40byte_packet(s_ip, d_ip, wrong_s_port, 2152, teid)
        else:                                  # TEID = 0
            pkt = build_40byte_packet(s_ip, d_ip, 2152, 2152, 0)
        rows.append([pkt.hex(), s_ip, d_ip, 2152, 2152, 0, 0])

    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["HexData", "SrcIP", "DstIP", "SrcPort", "DstPort", "TEID", "Label"])
        csv.writer(f).writerows(rows)
    print(f"✓  CSV generated: {os.path.abspath(path)}  ({len(rows)} rows)")

# (Re‑)generate CSV once; comment out if you already have data
generate_csv()

# -----------------------------------------------------------------------
# 3)  Data loading & preprocessing
# -----------------------------------------------------------------------
def hex_to_byte_float(hexstr):
    return np.frombuffer(bytes.fromhex(hexstr), np.uint8) / 255.0   # (40,)

def ip_to_bytes(ip_str):
    return np.frombuffer(struct.pack("!I", ip_to_int(ip_str)), np.uint8) / 255.0  # (4,)

def build_feature_vec(row):
    pkt   = hex_to_byte_float(row.HexData)          # 40 floats
    sip   = ip_to_bytes(row.SrcIP)                  # 4
    dip   = ip_to_bytes(row.DstIP)                  # 4
    sport = np.array([row.SrcPort / 65535.0], np.float32)
    dport = np.array([row.DstPort / 65535.0], np.float32)
    return np.concatenate([pkt, sip, dip, sport, dport])  # 40+4+4+1+1 = 50

df      = pd.read_csv("ann_training_data.csv")
X       = np.vstack(df.apply(build_feature_vec, axis=1).to_numpy())
y       = df.Label.to_numpy(np.float32).reshape(-1, 1)

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------------------------------------------
# 4)  Build & train Keras model (binary classifier)
# -----------------------------------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(50,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")   # binary output
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

history = model.fit(X_tr, y_tr,
                    validation_data=(X_val, y_val),
                    epochs=15, batch_size=128)

print("\nValidation accuracy:",
      f"{history.history['val_accuracy'][-1]*100:.2f}%")

# -----------------------------------------------------------------------
# 5)  Helper: predict validity (0 / 1)
# -----------------------------------------------------------------------
def is_valid_packet(raw_hex, src_ip, dst_ip, src_port, dst_port, thresh=0.5):
    vec = build_feature_vec(
        pd.Series([raw_hex, src_ip, dst_ip, src_port, dst_port],
                  index=["HexData", "SrcIP", "DstIP", "SrcPort", "DstPort"]))
    prob = model.predict(vec.reshape(1, -1), verbose=0)[0, 0]
    return 1 if prob >= thresh else 0

# Quick test on the first row
sample = df.iloc[0]
print("Sample label:", sample.Label,
      "→ model prediction:", is_valid_packet(
          sample.HexData, sample.SrcIP, sample.DstIP,
          sample.SrcPort, sample.DstPort))
