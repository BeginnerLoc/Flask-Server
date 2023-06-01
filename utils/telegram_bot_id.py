import asyncio
from telegram import Bot

async def get_chat_ids(token):
    bot = Bot(token=token)
    updates = await bot.get_updates()
    chat_ids = {}
    for update in updates:
        chat_id = update.effective_chat.id
        chat_name = update.effective_chat.title or update.effective_chat.first_name
        if update.effective_chat.type == 'private':
            user_id = update.effective_chat.id
            chat_ids[user_id] = {'name': chat_name, 'type': 'User'}
        else:
            group_id = update.effective_chat.id
            chat_ids[group_id] = {'name': chat_name, 'type': 'Group'}
    return chat_ids

# Replace 'TOKEN' with your bot token
bot_token = '5959752019:AAHZvf9E64dnXrYDPsX97ePMcoGz-t88KEw'

# Create an event loop
loop = asyncio.get_event_loop()

# Call the function to retrieve chat IDs, names, and types
chat_ids = loop.run_until_complete(get_chat_ids(bot_token))

# Print the chat IDs, names, and types
for chat_id, chat_info in chat_ids.items():
    print(f"Chat ID: {chat_id}, Name: {chat_info['name']}, Type: {chat_info['type']}")


