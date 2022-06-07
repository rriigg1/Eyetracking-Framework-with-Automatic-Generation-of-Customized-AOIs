logged_messages = set()

def log_error(message):
    """
    Ensures that each message is only logged once.
    """
    global logged_messages
    if (message not in logged_messages):
        print(message)
        logged_messages.add(message)

def clear_log():
    """
    Clears logged message so that they can be printed again.
    """
    global logged_messages
    logged_messages = set()

def remove_error(message):
    """
    Removes a single message from the list so that it can be printed again.
    """
    global logged_messages
    if (message in logged_messages):
        logged_messages.remove(message)
