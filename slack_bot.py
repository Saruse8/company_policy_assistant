
import os
import logging

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from vector_rag import ask_vector_llm

load_dotenv()

app = App(token=os.environ["SLACK_BOT_TOKEN"])

CHANNEL_NAME = "polici_assistant_sara"


@app.event("message")
def answer_policy_question(event, say, client):
    # Ignore messages from bots and system events
    if event.get("bot_id") or event.get("subtype"):
        return

    # Only respond in the designated testing channel
    channel_id = event.get("channel")
    if event.get("channel_type") != "channel":
        return

    try:
        channel_info = client.conversations_info(channel=channel_id)

        if channel_info["channel"]["name"] != CHANNEL_NAME:
            return

        question = event.get("text", "").strip()
        if not question:
            return

        result = ask_vector_llm(question)

        say(
            f"*Answer:*\n{result['answer']}\n\n"
            f"*Relevant policy:* {result['policy']}"
        )

    except Exception:
        logging.exception("Error answering policy question")
        say("Sorry, I couldn't answer your question. Please try again.")


if __name__ == "__main__":
    print("Company Policy Bot is starting...")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()

