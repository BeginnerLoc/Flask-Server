import asyncio
from telegram import Bot, Update
import textwrap
import pymongo
import base64
import io
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_fixed
import logging
import time
import datetime

bot_token = '6060060457:AAGRyic-1HVFcUy1dSEsdLMJo0rB9Mvz0y0'

@retry(stop=stop_after_attempt(10), wait=wait_fixed(2)) 
async def reply_to_message(bot: Bot, update: Update):
    user_id = update.effective_user.id #Get the user id of tele user
    message_text = update.message.text.lower() #Get user input and convert to lower case
    chat_id = update.effective_chat.id #Chatid of the tele user
    response_text = "" #Default response text
    chat_name = update.effective_user.first_name #Username 
    logging.info(f"User {chat_name} typed {message_text}") #Used to log the message from tele user 

    # Ignore any message that does not start with '/' 
    if not message_text.startswith('/'):
        return

    # Welcome Message
    if message_text == '/start':
        response_text = "Welcome, " + chat_name + "\nYour chat id is " + str(chat_id) +"\nUse the /commandlist to view all commands"

    # Connectivity test
    elif message_text == '/ping': #For testing connective purposes only
        start_time = time.time()
        await bot.send_message(chat_id=chat_id, text="Pinging...")
        end_time = time.time()
        ping_time = end_time - start_time
        response_text = f"Pong! Bot latency: {ping_time:.2f} seconds"

    # Command List 
    elif message_text == '/commandlist':
        response_text = textwrap.dedent("""\
        Command Syntax for SafetyManager Bot \n
        /Commandlist - Use this to view the command list menu
        
        -- Breaches -- 
        /breachList - Use this to see the last 20 incidents that had occurred. (i.e. /breachList) \n
        /breach <Number> - Use this to view an incident using the incident ID. (i.e. /breach 100) \n
        /resolveBreach <Number> <Action> - Use this to resolve an incident using the incident ID. (i.e. /resolve 100 Verbal Warning was given.) \n
        /unresolvedbreach - Use this to view the unresolved breaches. \n
                              
        -- Incidents --
        /hazardList - Use this to see the last 20 hazards that had occurred. (i.e. /hazardList) \n
        /hazard <Number> - Use this to view a hazard using the hazard ID. (i.e. /hazard 100) \n
        /resolveHazard - Use this to resolve a hazard using the hazard ID. (i.e. /resolve 100 Hazard was removed.)\n    
        /unresolvedHazard - Use this to view the unresolved hazards. 
        """)

    # View the last 20 breaches that had happened
    elif message_text == '/breachlist':
        breaches = await breachList()
        if breaches:
            response_text = "Last 20 Breaches:\n\n"
            for breach in breaches:
                breach_id = breach.get("breach_id", "")
                name = breach.get("worker_name", "")
                breach_type = breach.get("description", [])
                datetime = breach.get("datetime", "")
                convertedTime = datetime.strftime("%d-%m-%y %H:%M:%S")
                resolved = breach.get("case_resolved", "")
                if resolved:
                    resolve = "Resolved"
                else:
                    resolve = "Unresolved"
                if "no-helmet" in breach_type:
                    description = "No Helmet"
                elif "no-vest" in breach_type:
                    description = "No Safety Vest"
                else:
                    description = "None"
                
                response_text += textwrap.dedent(f"""\
                Breach ID: {breach_id} - {resolve}
                Name: {name} 
                Breach Type: {description} 
                Time of breach: {convertedTime}\n
                """)

        else:
            response_text = "No breaches found."

    # View the list of unresolved breaches that had happened
    elif message_text == '/unresolvedbreach':
        breachesCount = 0
        breaches = await viewUnresolved("breaches")
        if breaches:
            response_text = "Unresolved Breaches:\n\n"
            for breach in breaches:
                breach_id = breach.get("breach_id", "")
                name = breach.get("worker_name", "")
                breach_type = breach.get("description", "")
                datetime = breach.get("datetime", "")
                convertedTime = datetime.strftime("%d-%m-%y %H:%M:%S")

                if "no-helmet" in breach_type:
                    description = "No Helmet"
                elif "no-vest" in breach_type:
                    description = "No Safety Vest"
                else:
                    description = "None"

                response_text += textwrap.dedent(f"""\
                Breach ID: {breach_id}
                Name: {name}
                Breach Type: {description}
                Time of breach: {convertedTime}\n
                """)
                breachesCount += 1
        if breachesCount == 0:
            response_text = "Congrats! There is no unresolved cases for now!."

    #View the specific breach with the specific ID
    elif message_text.startswith('/breach '):
        # Extract the incident ID from the message text
        incident_id_str = message_text.split(' ')[-1]
        try:
            incident_id = int(incident_id_str)
            document = await search_mongo_id("breaches", incident_id)
        except ValueError:
            response_text = "The ID that you have input is invalid"
        
        if document :
            name = document["worker_name"]
            encoded_image = document["evidence_photo"]
            description = document["description"]
            timestamp = document["datetime"]
            cut_timestamp = str(timestamp)[:-7]
            locations = document["location"]
            breach_id = document["breach_id"]
            case_resolved = document["case_resolved"]
            case_resolved_time = document["case_resolved_time"]
            case_resolution = document["case_resolution"]
                
            if case_resolved_time is not None:
                formatted_time = str(case_resolved_time)
                cut_time = formatted_time[:-7]
            else:
                cut_time = ""


            response_text = textwrap.dedent(f"""\
            Breach ID: {breach_id}
            Name: {name}
            Breach Type: {description}
            Timestamp: {cut_timestamp}
            Location: {locations}
            Case Resolved: {case_resolved}
            Resolved Time: {cut_time} 
            Case Resolution: {case_resolution} """)

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

    # Resolve the breaches using /resolvebreach <Number> <Action> which would be updated in the database
    elif message_text.startswith('/resolvebreach '):
        # Split the message text by space
        command_parts = message_text.split(' ')

        # Check if the command has at least three parts
        if len(command_parts) >= 3:
            incident_id = int(command_parts[1])
            action = ' '.join(command_parts[2:])  # Join the remaining parts as the action

            document = await search_mongo_id("breaches", incident_id)
            if document :
                resolved = await changeResolved("breaches", incident_id, action)
                if resolved: 
                    response_text = "Incident id " + str(incident_id) + " has been marked as resolved with the action of '" + action +"'"
                else:
                    response_text = "Incident id: " + str(incident_id) + " could not have been resolved"
            else:
                response_text = "The Incident id you have entered is invalid or not in the Database!"
        else:
            response_text = "Invalid command format. Please use /resolveBreach <Number> <Action>."

    # View the last 20 hazards that had happened
    elif message_text == '/hazardlist':
        hazards = await hazardList()
        if hazards:
            response_text = "Last 20 Hazards:\n\n"
            for hazard in hazards:
                hazard_id = hazard.get("hazard_id", "")
                hazard_type = hazard.get("item")
                case_resolved = hazard.get("case_resolved", "")
                if case_resolved:
                    resolve = "Resolved"
                else:
                    resolve = "Unresolved"
                datetime = hazard.get("timestamp", "")
                convertedTime = datetime.strftime("%d-%m-%y %H:%M:%S")
                response_text += textwrap.dedent(f"""\
                Hazard ID: {hazard_id} - {resolve}
                Hazard Type: {hazard_type}
                Time of breach: {convertedTime}\n
                """)

        else:
            response_text = "No breaches found."
    
    # View the specific hazard with the specific ID
    elif message_text.startswith('/hazard '):
        # Extract the incident ID from the message text
        hazard_id = int(message_text.split(' ')[-1])
        document = await search_mongo_id("hazards", hazard_id)
        if document :
            name = document["item"]
            encoded_image = document["image"]
            timestamp = document["timestamp"]
            cut_timestamp = str(timestamp)[:-7]
            locations = document["location"]
            hazard_id = document["hazard_id"]
            case_resolved = document["case_resolved"]
            case_resolved_time = document["case_resolved_time"]
            case_resolution = document["case_resolution"]
                
            if case_resolved_time is not None:
                formatted_time = str(case_resolved_time)
                cut_time = formatted_time[:-7]
            else:
                cut_time = ""

            response_text = textwrap.dedent(f"""\
            Hazard ID: {hazard_id}
            Hazard Type: {name}
            Timestamp: {cut_timestamp}
            Location: {locations}
            Case Resolved: {case_resolved}
            Resolved Time: {case_resolved_time}
            Case Resolution: {case_resolution} """)

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

    # Resolve the hazard using /resolvehazard <Number> <Action> which would be updated in the database
    elif message_text.startswith('/resolvehazard '):
        # Split the message text by space
        command_parts = message_text.split(' ')

        # Check if the command has at least three parts
        if len(command_parts) >= 3:
            hazard_id = int(command_parts[1])
            action = ' '.join(command_parts[2:])  # Join the remaining parts as the action

            document = await search_mongo_id("hazards", hazard_id)
            if document :
                resolved = await changeResolved("hazards", hazard_id, action)
                if resolved: 
                    response_text = "Hazard id " + str(hazard_id) + " has been marked as resolved with the action of '" + action +"'"
                else:
                    response_text = "Hazard id: " + str(hazard_id) + " could not have been resolved"
            else:
                response_text = "The Hazard id you have entered is invalid or not in the Database!"
        else:
            response_text = "Invalid command format. Please use /resolveHazard <Number> <Action>."

    # View the list of unresolved hazards that had happened
    elif message_text == '/unresolvedhazard':
        hazardCount = 0
        hazards = await viewUnresolved("hazards")
        if hazards:
            response_text = "Unresolved Hazards:\n\n"
            for hazard in hazards:
                hazard_id = hazard.get("hazard_id", "")
                hazard_type = hazard.get("item")
                case_resolved = hazard.get("case_resolved", "")

                response_text += textwrap.dedent(f"""\
                Breach ID: {hazard_id}
                Hazard Type: {hazard_type}
                Case Resolved: {case_resolved}\n
                """)
                hazardCount += 1
        if hazardCount == 0:
            response_text = "Congrats! There is no unresolved cases for now!"

    # If command is invalid, tell user command is not valid
    else:
        response_text = f"Invalid Action or Command. Please use /Commandlist to get a list of valid commands."

    #If there is a response_text, use bot to send message to user
    if response_text:
        await bot.send_message(chat_id=chat_id, text=response_text)


@retry(stop=stop_after_attempt(10), wait=wait_fixed(2))
async def search_mongo_id(type, id):
    # Connect to MongoDB
    client = pymongo.MongoClient("mongodb+srv://Astro:enwVEQqCyk9gYBzN@c290.5lmj4xh.mongodb.net/")
    db = client["construction"]

    if type == "breaches":
        collection = db["db_breaches_2"]
        document = collection.find_one({"breach_id": id})
    elif type == "hazards":
        collection = db["hazards_1"]
        document = collection.find_one({"hazard_id": id})
    else:
        return None

    
    if document:
        return document
    else:
        return None  # No document found


@retry(stop=stop_after_attempt(10), wait=wait_fixed(2))
async def viewUnresolved(type):
    client = pymongo.MongoClient("mongodb+srv://Astro:enwVEQqCyk9gYBzN@c290.5lmj4xh.mongodb.net/")
    db = client["construction"]

    if type == "breaches":
        collection = db["db_breaches_2"]
    elif type == "hazards":
        collection = db["hazards_1"]
    else:
        return None
    document = collection.find({"case_resolved": False}).sort([("timestamp", pymongo.DESCENDING)]).limit(20)

    if document:
        return document
    else:
        return None  # No document found


@retry(stop=stop_after_attempt(10), wait=wait_fixed(2))
async def changeResolved(type, id, reason):
        # Connect to MongoDB
    client = pymongo.MongoClient("mongodb+srv://Astro:enwVEQqCyk9gYBzN@c290.5lmj4xh.mongodb.net/")
    db = client["construction"]

    if type == "breaches":
        collection = db["db_breaches_2"]
    elif type == "hazards":
        collection = db["hazards_1"]
    else:
        return False  # Return False if type is invalid
    try:
        timeNow = datetime.datetime.now()
        if type == "breaches":
            # Update the document with the provided id
            collection.update_one(
                {"breach_id": id},
                {"$set": {"case_resolved": True, "case_resolution": reason, "case_resolved_time": timeNow},
                 }
            )
        elif type == "hazards":
            collection.update_one(
                {"hazard_id": id},
                {"$set": {"case_resolved": True, "case_resolution": reason, "case_resolved_time": timeNow},
                }
            )
        client.close()
        return True  # Return True if the update is successful
    except pymongo.errors.ConnectionFailure as e:
        print(f"Connection failed: {e}")
        return False  # Return False if the update fails


@retry(stop=stop_after_attempt(10), wait=wait_fixed(2))
async def breachList():
    try:
        client = pymongo.MongoClient("mongodb+srv://Astro:enwVEQqCyk9gYBzN@c290.5lmj4xh.mongodb.net/")
        db = client["construction"]
        collection = db["db_breaches_2"]
        breaches = collection.find().sort([("timestamp", pymongo.DESCENDING)]).limit(20)
        return list(breaches)
    except pymongo.errors.ConnectionFailure as e:
        print(f"Connection failed: {e}")
        return None


@retry(stop=stop_after_attempt(10), wait=wait_fixed(2))
async def hazardList():
    try:
        client = pymongo.MongoClient("mongodb+srv://Astro:enwVEQqCyk9gYBzN@c290.5lmj4xh.mongodb.net/")
        db = client["construction"]
        collection = db["hazards_1"]
        hazards = collection.find().sort([("timestamp", pymongo.DESCENDING)]).limit(20)
        return list(hazards)
    except pymongo.errors.ConnectionFailure as e:
        print(f"Connection failed: {e}")
        return None


@retry(stop=stop_after_attempt(10), wait=wait_fixed(2))
async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
            logging.exception("Error occurred during update processing.")
            print("Try again later!")


if __name__ == '__main__':
    asyncio.run(main())
