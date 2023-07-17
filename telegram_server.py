import asyncio
from telegram import Bot, Update
import textwrap
import pymongo
import base64
import io
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_fixed

bot_token = '6060060457:AAGRyic-1HVFcUy1dSEsdLMJo0rB9Mvz0y0'

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def reply_to_message(bot: Bot, update: Update):
    user_id = update.effective_user.id
    message_text = update.message.text.lower()
    chat_id = update.effective_chat.id
    response_text = ""
    chat_name = update.effective_user.first_name 

    if not message_text.startswith('/'):
        return

    if message_text == '/start':
        response_text = "Welcome, " + chat_name + " I am SafetyManager Bot!"

    elif message_text == '/commandlist':
        response_text = textwrap.dedent("""\
        /Commandlist - Use this to view the command list menu
        
        -- Breaches -- \n
        /breachList - Use this to see the last 20 incidents that had occurred. (i.e. /breachList) \n
        /breach <Number> - Use this to view an incident using the incident ID. (i.e. /breach 100) \n
        /resolvebreach <Number> <Action> - Use this to resolve an incident using the incident ID. (i.e. /resolve 100 Verbal Warning was given.) \n\n
        -- Incidents -- \n
                                        
        """)

    elif message_text == '/breachlist':
        breaches = await breachList()
        if breaches:
            response_text = "Last 20 Breaches:\n\n"
            for breach in breaches:
                breach_id = breach.get("breach_id", "")
                name = breach.get("name", "")
                breach_type = ", ".join(breach.get("breach_type", []))
                timestamp = breach.get("timestamp", "")
                locations = breach.get("location", "")
                case_resolved = breach.get("case_resolved", "")

                response_text += textwrap.dedent(f"""\
                Breach ID: {breach_id}
                Name: {name}
                Breach Type: {breach_type}
                Timestamp: {timestamp}
                Location: {locations}
                Case Resolved: {case_resolved}\n
                """)

        else:
            response_text = "No breaches found."

    elif message_text.startswith('/breach'):
        # Extract the incident ID from the message text
        incident_id = int(message_text.split(' ')[-1])
        document = await search_mongo_id("breaches", incident_id)
        if document :
            name = document["name"]
            encoded_image = document["image"]
            breach_type = document["breach_type"]
            timestamp = document["timestamp"]
            locations = document["location"]
            breach_id = document["breach_id"]
            case_resolved = document["case_resolved"]

            breach_type_text = ''
            if 'NO-Hardhat' in breach_type and 'NO-Safety Vest' in breach_type:
                breach_type_text = 'Not wearing hardhat and safety vest'
            elif 'NO-Hardhat' in breach_type:
                breach_type_text = 'Not wearing hardhat'
            elif 'NO-Safety Vest' in breach_type:
                breach_type_text = 'Not wearing safety vest'
            else:
                breach_type_text = ', '.join(breach_type)

            response_text = textwrap.dedent(f"""\
            Breach ID: {breach_id}
            Name: {name}
            Breach Type: {breach_type_text}
            Timestamp: {timestamp}
            Location: {locations}
            Case Resolved: {case_resolved}""")


            if encoded_image:
                decoded_image = base64.b64decode(encoded_image)
                image_io = io.BytesIO(decoded_image)
                image = Image.open(image_io)

                # Convert the image to JPEG format
                jpeg_image_io = io.BytesIO()
                image.save(jpeg_image_io, 'JPEG')
                jpeg_image_io.seek(0)

                # Send the image via the Telegram message
                await bot.send_photo(chat_id=chat_id, photo=jpeg_image_io, caption=response_text)
                return

        else:
            response_text = "The ID that you have input is invalid"

    elif message_text.startswith('/resolvebreach'):
         incident_id = int(message_text.split(' ')[-1])


    else:
        response_text = f"Invalid Action or Command. Please use /Commandlist to get a list of valid commands."

    if response_text:
        await bot.send_message(chat_id=chat_id, text=response_text)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def search_mongo_id(type, id):
    # Connect to MongoDB
    client = pymongo.MongoClient("mongodb+srv://Astro:enwVEQqCyk9gYBzN@c290.5lmj4xh.mongodb.net/")
    db = client["construction"]

    if type == "breaches":
        collection = db["breach_images"]
    elif type == "hazards":
        collection = db["hazards_1"]
    else:
        return None

    document = collection.find_one({"breach_id": id})
    if document:
        return document
    else:
        return None  # No document found

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def breachList():
    try:
        client = pymongo.MongoClient("mongodb+srv://Astro:enwVEQqCyk9gYBzN@c290.5lmj4xh.mongodb.net/")
        db = client["construction"]
        collection = db["breach_images"]
        breaches = collection.find().sort([("timestamp", pymongo.DESCENDING)]).limit(20)
        return list(breaches)
    except pymongo.errors.ConnectionFailure as e:
        print(f"Connection failed: {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def main():
    bot = Bot(token=bot_token)
    offset = 0 

    while True:
        try:
            updates = await bot.get_updates(offset=offset, timeout=10)
            for update in updates:
                offset = max(offset, update.update_id) + 1
                await reply_to_message(bot, update)

            await asyncio.sleep(1)
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Try again later!")


if __name__ == '__main__':
    asyncio.run(main())
