import os

import requests

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


def send_discord_notification(message: str):
    data = {"content": message}
    response = requests.post(DISCORD_WEBHOOK_URL, json=data)
    if response.status_code == 204:
        print("Notification sent successfully!")
    else:
        print(f"Failed to send notification: {response.status_code}, {response.text}")
