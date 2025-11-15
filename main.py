import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

load_dotenv()

from model import APP_NAME, root_agent  # noqa: E402

BOT_TOKEN = os.getenv("TGBOT_TOKEN")

session_service = InMemorySessionService()
runner = Runner(agent=root_agent, session_service=session_service, app_name=APP_NAME)


async def process_user_message(user_id, user_text):
    session_key = f"session_{user_id}"

    session = await session_service.get_session(
        app_name=APP_NAME, user_id=str(user_id), session_id=session_key
    )

    if session is None:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=str(user_id), session_id=session_key
        )

    query = types.Content(role="user", parts=[types.Part(text=user_text)])
    response_texts = []

    async for event in runner.run_async(
        user_id=str(user_id), session_id=session.id, new_message=query
    ):
        if event.content and event.content.parts:
            text = event.content.parts[0].text
            if text and text != "None":
                response_texts.append(text)

    return "\n".join(response_texts)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text
    response = await process_user_message(user_id, user_text)
    if response:
        await update.message.reply_text(response)


if __name__ == "__main__":
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot started...")
    
    app.run_polling()
