inputs :
    ip.src_ip
    ip.dst_ip
    ip.protocol
    ip.ttl
    ip.header_length
    ip.total_length
    ip.identification
    ip.flags
    ip.fragment_offset
    ip.checksum
    if udp:
        udp.src_port
        udp.dst_port
        udp.length
        udp.checksum
    if sctp:
        sctp.src_port
        sctp.dst_port
    if gtp:
        gtpu.version
        gtpu.protocol_type
        gtpu.extension_header_flag
        gtpu.sequence_number_flag
        gtpu.npdu_flag
        gtpu.message_type
        gtpu.length
        gtpu.teid
        gtpu.sequence_number
        gtpu.npdu_number
        gtpu.next_extension_header_type

    application ip :
        user_plane ip
        cplane_ip


hidden layers:
    if gtpu packet dst ip is not same as uplane application ip 
    if sctp packet dst ip is not same as cplane app ip 
    length mismatch 
    checksum error 
    



output:
    not_valid packet
    uplane packet with teid id =x
    cplane packet 

    