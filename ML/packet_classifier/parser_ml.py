from scapy.all import rdpcap, IP, UDP, SCTP, Raw
import pandas as pd

# Read packets from pcap file
packets = rdpcap("capture.pcap")

data = []

for packet in packets:
    if not packet.haslayer(IP):
        continue  # Skip non-IP packets

    ip_layer = packet[IP]

    # Extract Common IP Header Fields
    packet_info = {
        "src_ip": ip_layer.src,
        "dst_ip": ip_layer.dst,
        "protocol": ip_layer.proto,
        "ttl": ip_layer.ttl,
        "header_length": ip_layer.ihl,
        "total_length": ip_layer.len,
        "identification": ip_layer.id,
        "flags": ip_layer.flags,
        "fragment_offset": ip_layer.frag,
        "checksum": ip_layer.chksum,
    }

    # UDP Handling
    if packet.haslayer(UDP):
        udp_layer = packet[UDP]
        packet_info.update({
            "udp_src_port": udp_layer.sport,
            "udp_dst_port": udp_layer.dport,
            "udp_length": udp_layer.len,
            "udp_checksum": udp_layer.chksum
        })

    # Check for GTP-U (User Plane Traffic)
    if udp_layer.dport == 2152 and packet.haslayer(Raw):
        raw_payload = bytes(packet[Raw])
        if len(raw_payload) >= 8:  # GTP-U header must be at least 8 bytes
            teid = int.from_bytes(raw_payload[4:8], byteorder='big')
            packet_info.update({"teid" : teid}) 
            

    # SCTP Handling (Control Plane Traffic)
    elif packet.haslayer(SCTP):
        sctp_layer = packet[SCTP]
        packet_info.update({
            "sctp_src_port": sctp_layer.sport,
            "sctp_dst_port": sctp_layer.dport,
            "classification": "cplane packet"
        })

    data.append(packet_info)

# Convert to DataFrame and Save as CSV
df = pd.DataFrame(data)
df.to_csv("classified_packets.csv", index=False)

print("✅ Packet classification saved: classified_packets.csv")
