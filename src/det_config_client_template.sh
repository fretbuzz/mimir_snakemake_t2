{
    "plugins": {
        "http": {
            "target": TARGETIP,
            "port": 8080
        },
        "google_docs": {
            "target": "SERVER",
            "port": 8080
        },
        "dns": {
            "key": "google.com",
            "target": TARGETIP,
            "port": 53
        },
        "gmail": {
            "username": "dataexfil@gmail.com",
            "password": "CrazyNicePassword",
            "server": "smtp.gmail.com",
            "port": 587
        },
        "tcp": {
            "target": TARGETIP,
            "port": 6969
        },
        "tcp_ipv6": {
            "target": "::1",
            "port": 6969,
        },
        "udp_ipv6": {
            "target": "::1",
            "port": 6969,
        },
        "udp": {
            "target": TARGETIP,
            "port": 6969
        },
        "twitter": {
            "username": "PaulWebSec",
            "CONSUMER_TOKEN": "XXXXXXXXXXX",
            "CONSUMER_SECRET": "XXXXXXXXXXX",
            "ACCESS_TOKEN": "XXXXXXXXXXX",
            "ACCESS_TOKEN_SECRET": "XXXXXXXXXXX"
        },
        "icmp": {
            "target": TARGETIP
        },
        "slack": {
            "api_token": "xoxb-XXXXXXXXXXX",
            "chan_id": "XXXXXXXXXXX",
            "bot_id": "<@XXXXXXXXXXX>:"
        },
        "smtp": {
            "target": TARGETIP,
            "port": 25
        },
        "ftp": {
            "target": TARGETIP,
            "port": 21
        },
        "sip": {
            "target": TARGETIP,
            "port": 5060
        },
        "wifi": {
            "interface": "wlan0mon"
        },
        "github_gist": {
            "username": "email@gmail.com",
            "password": "XXXXXXXXXXXXXXXXXXXXXXXXX"
        }
    },
    "AES_KEY": "THISISACRAZYKEY",
    "max_time_sleep": 10,
    "min_time_sleep": 1,
    "max_bytes_read": 400,
    "min_bytes_read": 300,
    "compression": 1
}